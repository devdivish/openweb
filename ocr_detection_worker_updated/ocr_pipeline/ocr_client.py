"""OpenAI-compatible async OCR client with multi-backend routing.

Works with any server exposing /v1/chat/completions (vLLM, SGLang,
LMDeploy, llama.cpp server, etc.).

The client holds configuration for EVERY backend declared in
`config.ocr_backends`, so the caller (orchestrator) can route each
individual image to the most appropriate model based on a detected
Unicode script — e.g. Arabic/Urdu -> dots.ocr, everything else -> Qwen.

Concurrency is controlled by a single asyncio.Semaphore so the caller
can fan out page-level parallelism while the OCR servers see at most
`ocr_concurrency` in-flight requests in total — letting vLLM/SGLang
batch them efficiently on the H200s.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
from PIL import Image

from .config import Config
from .utils import encode_image_b64, fit_image_for_ocr


_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)

log = logging.getLogger(__name__)


@dataclass
class _Backend:
    """Per-backend resolved configuration.

    `models` holds one or more model-name aliases served at the SAME
    `base_url`. Multiple names are the llama-swap pattern: every request
    hits the same endpoint, but the `model` field in the payload tells
    llama-swap which replica (instance/GPU) to route to. Rotating the
    name across concurrent requests therefore spreads load evenly across
    replicas without any client-side URL management.

    With a single model name this behaves exactly like before.
    """
    name: str
    base_url: str
    models: List[str]
    prompt: str
    api_key: str
    # Per-backend image caps — inherit from global ocr.max_image_* when
    # the backend config doesn't override. Different VLMs have different
    # input constraints: dots.ocr is typically stricter than Qwen-VL.
    max_image_pixels: Optional[int]
    max_image_long_side_px: Optional[int]
    # Live rotation state. `_healthy` is the pool actually picked from;
    # `_cursor` is the round-robin index. We use cursor+modulo over a
    # mutable list (not itertools.cycle) so mark_unhealthy / set_healthy
    # take effect on the very next rotation.
    #
    # Concurrency note: asyncio is single-threaded and every access here
    # happens synchronously between awaits, so no lock is needed. The
    # update-and-read sequence inside `next_model` is atomic from the
    # event-loop's perspective.
    _healthy: List[str] = field(init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError(
                f"backend {self.name!r}: must declare at least one model "
                f"(set `model: <name>` or `models: [<name>, ...]`)"
            )
        # Start optimistic: every configured alias is healthy until a
        # probe or a live request proves otherwise.
        self._healthy = list(self.models)

    def next_model(self) -> str:
        """Return the next model alias in round-robin order.

        Rotates through the CONFIGURED `models` list in order, skipping
        any entry that's been demoted via `mark_unhealthy`. Indexing
        against the stable list (rather than the shrinking healthy pool)
        keeps rotation predictable: demoting `models[k]` means the next
        pick is `models[k+1]`, not whatever was at position k in a
        shrunken list.

        If every alias has been demoted we fall back to the full
        configured list — always better to keep trying (the replica may
        have just recovered) than to fail every OCR call outright.
        """
        n = len(self.models)
        if not self._healthy:
            # Fallback path: every alias demoted. Cycle through the raw
            # list so repeated retries don't all hit the same dead one.
            m = self.models[self._cursor % n]
            self._cursor = (self._cursor + 1) % n
            return m

        healthy_set = set(self._healthy)
        # Search forward from `_cursor`, linearly, for the next healthy
        # alias. n <= small (4 for llama-swap setups) so this is cheap.
        for i in range(n):
            idx = (self._cursor + i) % n
            candidate = self.models[idx]
            if candidate in healthy_set:
                self._cursor = (idx + 1) % n
                return candidate
        # Unreachable given the empty-check above.
        return self.models[0]

    def set_healthy(self, healthy_models: List[str]) -> None:
        """Replace the healthy pool (called by OCRClient.probe_models).

        Order is preserved from the configured `models` list so the
        rotation order stays predictable across probes — we don't want
        the round-robin to jump around just because the server returned
        aliases in a different order.
        """
        wanted = set(healthy_models)
        preserved = [m for m in self.models if m in wanted]
        self._healthy = preserved
        self._cursor = 0

    def mark_unhealthy(self, model: str) -> None:
        """Remove `model` from the healthy pool so future `next_model()`
        calls skip it. No-op if the model was already demoted. Safe to
        call from multiple coroutines concurrently."""
        try:
            self._healthy.remove(model)
        except ValueError:
            pass  # already demoted or never in pool

    def healthy_models(self) -> List[str]:
        """Snapshot of the current healthy pool (for logging / tests)."""
        return list(self._healthy)


class OCRClient:
    """Single shared client. Instantiate once per pipeline run.

    Usage (from orchestrator)::

        async with OCRClient(cfg) as ocr:
            text = await ocr.ocr_image(img, backend="dots_ocr")
            text = await ocr.ocr_image(img)                 # uses default
            text = await ocr.ocr_image(img, max_tokens=256) # probe pass

    For routing, call `resolve_backend_for_script(script)` to map a
    detected Unicode script name to a backend name.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_tokens = int(cfg.ocr.max_output_tokens)
        self.temperature = float(cfg.ocr.temperature)
        self.timeout = float(cfg.ocr.request_timeout_s)
        self.max_retries = int(cfg.ocr.max_retries)
        self.backoff = float(cfg.ocr.retry_backoff_s)

        # Global image-size caps (fallback for backends that don't set
        # their own). vLLM / Qwen-VL / dots.ocr reject oversized images
        # with HTTP 400. 0 / null disables the cap on that axis.
        global_max_pixels = int(cfg.ocr.get("max_image_pixels") or 0) or None
        global_max_long_side = int(cfg.ocr.get("max_image_long_side_px") or 0) or None

        # Load every backend declared in config — we may need any of them.
        # Per-backend image caps override the globals above; this is how
        # dots.ocr gets a tighter cap than qwen in the same run.
        #
        # Model aliasing: backends may declare EITHER a single `model:` or
        # a list `models: [...]`. Multi-entry lists are for llama-swap (or
        # any server that routes by model name) — requests rotate across
        # the aliases so concurrent calls hit different replicas.
        self._backends: Dict[str, _Backend] = {}
        for name, be in cfg.ocr_backends.items():
            be_max_pixels = int(be.get("max_image_pixels") or 0) or global_max_pixels
            be_max_long_side = (
                int(be.get("max_image_long_side_px") or 0) or global_max_long_side
            )
            models = self._resolve_models(name, be)
            if len(models) > 1:
                log.info(
                    "backend %s: rotating across %d model aliases (%s)",
                    name, len(models), ", ".join(models),
                )
            self._backends[name] = _Backend(
                name=name,
                base_url=be.base_url.rstrip("/"),
                models=models,
                prompt=be.prompt,
                api_key=be.get("api_key", "EMPTY"),
                max_image_pixels=be_max_pixels,
                max_image_long_side_px=be_max_long_side,
            )

        # Default backend when no routing hint is available.
        self.default_backend: str = cfg.ocr.active_backend

        # Script routing rules (may be disabled).
        routing = cfg.ocr.get("script_routing") or {}
        self.routing_enabled: bool = bool(routing.get("enabled", False))
        self.fallback_backend: str = (routing.get("fallback_backend")
                                      or self.default_backend)
        self.by_script: Dict[str, str] = dict(routing.get("by_script") or {})
        self.detection_mode: str = (
            (routing.get("detection_mode") or "both").lower()
        )
        # Cheaper token cap for probe OCR calls (script-sniff pass).
        self.probe_max_tokens: int = int(
            routing.get("probe_max_tokens") or 512
        )

        self._semaphore = asyncio.Semaphore(int(cfg.parallelism.ocr_concurrency))
        self._http: Optional[httpx.AsyncClient] = None

        # Alias-hop: when a request fails with network/timeout/5xx, the
        # current alias is demoted and the image is retried on the next
        # rotated alias, up to this many total attempts per call.
        # 1 = feature disabled (current alias only, no hop).
        self.alias_hop_attempts: int = int(cfg.ocr.get("alias_hop_attempts") or 2)

        # Health-check config (may be absent; defaults are conservative).
        hc = cfg.ocr.get("health_check") or {}
        self.health_check_enabled: bool = bool(hc.get("enabled", False))
        self.health_probe_timeout_s: float = float(hc.get("probe_timeout_s") or 10)
        self.health_probe_before_batch: bool = bool(
            hc.get("probe_before_batch", True)
        )
        self.health_min_healthy: int = int(hc.get("min_healthy") or 1)

    async def __aenter__(self) -> "OCRClient":
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            http2=False,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._http is not None:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_models(name: str, be) -> List[str]:
        """Read `models: [...]` or `model: <name>` from a backend config
        and return a non-empty list. Preserves order (important — it's
        the rotation order). Strips blanks and de-dupes while preserving
        first occurrence.
        """
        raw = be.get("models")
        if raw:
            candidates: List[str] = list(raw)
        else:
            single = be.get("model")
            candidates = [single] if single else []

        seen = set()
        out: List[str] = []
        for m in candidates:
            if not m:
                continue
            m = str(m).strip()
            if not m or m in seen:
                continue
            seen.add(m)
            out.append(m)
        if not out:
            raise ValueError(
                f"ocr_backends.{name}: no model declared — set either "
                f"`model: <name>` or `models: [<name1>, <name2>, ...]`"
            )
        return out

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def resolve_backend_for_script(self, script: Optional[str]) -> str:
        """Return the backend name to use for an image whose dominant
        script is `script` (or None if unknown)."""
        if not self.routing_enabled or script is None:
            return self.fallback_backend
        return self.by_script.get(script, self.fallback_backend)

    def backends(self) -> List[str]:
        return list(self._backends.keys())

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def probe_models(
        self, timeout_s: Optional[float] = None
    ) -> Dict[str, Dict[str, bool]]:
        """Probe every model alias on every backend with a tiny chat
        completion and update each backend's healthy pool in place.

        Rationale: with llama-swap in front of N replicas, a specific
        replica can go down while its sibling aliases keep serving. We
        want to detect that BEFORE a batch starts so dead aliases never
        enter rotation — otherwise every Nth request would fail and
        waste the alias-hop budget.

        The probe sends `{"max_tokens": 1}` text-only so it's cheap for
        the GPU (no image tokens, single output token). Probes run in
        parallel across aliases AND backends.

        Returns `{backend_name: {alias: is_healthy}}` for logging.

        Safe to call repeatedly; idempotent. Re-invocation can also
        RECOVER an alias that was previously demoted but has come back.

        Single-alias backends are reported as healthy without probing
        (nothing to fall back to if the probe says "down" — the user
        would rather see a real request failure than silent skipping).
        """
        if self._http is None:
            raise RuntimeError("OCRClient must be used as async context manager")

        t_s = float(timeout_s if timeout_s is not None else self.health_probe_timeout_s)
        report: Dict[str, Dict[str, bool]] = {}

        async def _probe_backend(be: _Backend) -> None:
            # Trivial backend: nothing to probe.
            if len(be.models) == 1:
                report[be.name] = {be.models[0]: True}
                return

            probe_tasks = [
                self._probe_alias(be, m, t_s) for m in be.models
            ]
            oks = await asyncio.gather(*probe_tasks)
            status = {m: bool(ok) for m, ok in zip(be.models, oks)}
            report[be.name] = status

            healthy = [m for m, ok in status.items() if ok]
            be.set_healthy(healthy)

            n_ok = len(healthy)
            n_total = len(be.models)
            per_alias = " ".join(
                f"{m}={'ok' if ok else 'DOWN'}" for m, ok in status.items()
            )
            if n_ok == 0:
                log.error(
                    "health: backend %s has ZERO healthy aliases — rotation "
                    "will fall back to the full configured list (%d aliases) "
                    "and every request may fail until a replica recovers. [%s]",
                    be.name, n_total, per_alias,
                )
            elif n_ok < n_total:
                log.warning(
                    "health: backend %s -> %d/%d aliases healthy [%s]",
                    be.name, n_ok, n_total, per_alias,
                )
            else:
                log.info(
                    "health: backend %s -> %d/%d aliases healthy",
                    be.name, n_ok, n_total,
                )

            if 0 < n_ok < self.health_min_healthy:
                log.warning(
                    "health: backend %s has fewer healthy aliases (%d) than "
                    "min_healthy=%d — performance will degrade",
                    be.name, n_ok, self.health_min_healthy,
                )

        await asyncio.gather(*[_probe_backend(be) for be in self._backends.values()])
        return report

    async def _probe_alias(self, be: _Backend, model: str,
                           timeout_s: float) -> bool:
        """Single-alias liveness probe. True iff the server answers 2xx
        to a trivial chat completion within `timeout_s`."""
        url = f"{be.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {be.api_key}"}
        try:
            resp = await self._http.post(
                url, json=payload, headers=headers,
                timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
            )
            if resp.status_code < 400:
                return True
            # 4xx/5xx from the server — treat as down. For multi-alias
            # llama-swap setups a 4xx on probe usually means "this alias
            # isn't loaded"; a 5xx means "loaded but unhealthy". Either
            # way we don't want it in rotation.
            log.debug(
                "probe %s/%s -> HTTP %d: %s",
                be.name, model, resp.status_code,
                (resp.text or "")[:200].replace("\n", " "),
            )
            return False
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            log.debug("probe %s/%s unreachable: %s", be.name, model, e)
            return False
        except Exception as e:
            # Defensive: never let a probe bubble up and abort the run.
            log.debug("probe %s/%s raised %s", be.name, model, e)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ocr_image(self, image: Image.Image,
                        backend: Optional[str] = None,
                        max_tokens: Optional[int] = None,
                        prompt_override: Optional[str] = None) -> str:
        """OCR a single PIL image.

        Args:
            image: the PIL image to OCR.
            backend: backend name from `ocr_backends`; if None uses
                     `default_backend` (i.e. ocr.active_backend).
            max_tokens: override `ocr.max_output_tokens`. Used by the
                        probe-OCR pass to keep the script-detection call cheap.
            prompt_override: replace the backend's default prompt.
        """
        be = self._backends[backend or self.default_backend]

        # Downscale to fit THIS backend's limits. Caps are per-backend:
        # dots.ocr gets a tighter cap than qwen when configured that way.
        # Skipping this is what caused the 400 Bad Request regression on
        # full-page renders after script routing started sending them to
        # dots.ocr — the new probe-then-route path can pick a stricter
        # backend than the one that originally accepted the image.
        original_size = image.size
        image = fit_image_for_ocr(
            image,
            max_pixels=be.max_image_pixels,
            max_long_side_px=be.max_image_long_side_px,
        )
        if image.size != original_size:
            log.info(
                "downscaled image for OCR: %dx%d -> %dx%d (backend=%s)",
                original_size[0], original_size[1],
                image.size[0], image.size[1], be.name,
            )

        data_url = encode_image_b64(image, fmt="PNG")
        prompt = prompt_override or be.prompt
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Alias hopping: if the first alias fails with a transport-level
        # error (unreachable / timeout) or a 5xx, that replica is probably
        # down — demote it and retry the SAME image on the next alias.
        # 4xx errors are NOT hopped because they signal a real problem
        # with the request (image too large, bad prompt, etc.), and the
        # next replica would just reject it the same way.
        #
        # Bounded by `alias_hop_attempts`; we also stop if we'd revisit
        # an alias we've already tried in this call (the pool cycled).
        max_attempts = max(1, self.alias_hop_attempts)
        tried: List[str] = []
        last_exc: Optional[BaseException] = None

        for attempt in range(max_attempts):
            model_name = be.next_model()
            if model_name in tried:
                # We've cycled through every healthy alias at least once.
                break
            tried.append(model_name)
            try:
                return await self._chat_with_image(
                    be, model_name, prompt, data_url, tokens,
                    img_size=image.size,
                )
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if 400 <= status < 500:
                    # Not a liveness issue — propagate immediately so
                    # callers (e.g. _probe_then_route) can handle the
                    # 4xx on this image, not pretend the alias is dead.
                    raise
                # 5xx: server thought the alias was up but something
                # broke inside. Treat as a transient replica problem.
                last_exc = e
                be.mark_unhealthy(model_name)
                log.warning(
                    "alias %s/%s returned %d; demoting and hopping "
                    "(attempt %d/%d, tried=%s)",
                    be.name, model_name, status, attempt + 1,
                    max_attempts, tried,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exc = e
                be.mark_unhealthy(model_name)
                log.warning(
                    "alias %s/%s unreachable (%s); demoting and hopping "
                    "(attempt %d/%d, tried=%s)",
                    be.name, model_name, type(e).__name__,
                    attempt + 1, max_attempts, tried,
                )

        assert last_exc is not None, "alias-hop loop exited without trying any alias"
        raise last_exc

    async def ocr_many(self, images: List[Image.Image],
                       backends: Optional[List[Optional[str]]] = None
                       ) -> List[str]:
        """OCR many images concurrently, respecting ocr_concurrency.

        `backends[i]` selects the backend for `images[i]`. Pass None for
        entries that should use `default_backend`. If `backends` is None
        everything uses the default.
        """
        if not images:
            return []
        if backends is None:
            backends = [None] * len(images)
        assert len(backends) == len(images), \
            "backends list length must match images length"
        tasks = [asyncio.create_task(self.ocr_image(img, backend=be))
                 for img, be in zip(images, backends)]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _chat_with_image(self, be: _Backend, model: str, prompt: str,
                               data_url: str, max_tokens: int,
                               img_size: Optional[tuple] = None) -> str:
        if self._http is None:
            raise RuntimeError("OCRClient must be used as async context manager")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {be.api_key}"}

        async with self._semaphore:
            data = await self._post_with_retries(
                be, model, payload, headers, img_size=img_size,
            )

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            log.error("Unexpected OCR response shape from %s/%s: %s",
                      be.name, model, data)
            raise RuntimeError(
                f"Malformed OCR response from {be.name}/{model}: {e}"
            ) from e

    async def _post_with_retries(self, be: _Backend, model: str,
                                 payload: dict, headers: dict,
                                 img_size: Optional[tuple] = None) -> dict:
        """POST to the OCR server with exponential backoff + jitter.

        Retries on: network errors, timeouts, HTTP 5xx. 4xx errors (bad
        request, auth, etc.) fail immediately since retrying won't help,
        but we extract the server's error body so the cause is visible
        in logs (image-too-large, aspect-ratio out of bounds, etc.).

        Logs both success and failure with the backend + model name so a
        run log captures which replica served each request — useful for
        diagnosing load imbalance or flagging specific replicas that
        produce worse output.
        """
        import time as _time   # local import: avoids polluting module namespace
        url = f"{be.base_url}/chat/completions"
        attempts = self.max_retries + 1
        last_exc: Optional[BaseException] = None

        for i in range(attempts):
            t0 = _time.perf_counter()
            try:
                resp = await self._http.post(url, json=payload, headers=headers)
                if 500 <= resp.status_code < 600:
                    log.warning("%s (%s) returned %s (attempt %d/%d)",
                                be.base_url, model, resp.status_code,
                                i + 1, attempts)
                    resp.raise_for_status()
                resp.raise_for_status()
                elapsed_ms = int((_time.perf_counter() - t0) * 1000)
                size_str = (
                    f"{img_size[0]}x{img_size[1]}" if img_size else "?"
                )
                # Success line — same field layout as the 4xx error log
                # below, so grepping "backend=... model=..." yields every
                # request (success + failure).
                log.info(
                    "OCR ok (backend=%s model=%s img=%s) in %dms",
                    be.name, model, size_str, elapsed_ms,
                )
                return resp.json()

            except httpx.HTTPStatusError as e:
                last_exc = e
                status = e.response.status_code
                if 400 <= status < 500:
                    # Extract the server's error body — vLLM usually
                    # tells you exactly why (e.g. image pixel count
                    # exceeds model max_pixels). Truncate to keep logs
                    # sane.
                    try:
                        body = e.response.text or ""
                    except Exception:
                        body = "<unreadable>"
                    body = body.strip().replace("\n", " ")
                    if len(body) > 600:
                        body = body[:600] + "... [truncated]"
                    size_str = (
                        f"{img_size[0]}x{img_size[1]}" if img_size else "?"
                    )
                    log.error(
                        "OCR %s HTTP %d (backend=%s model=%s img=%s): %s",
                        url, status, be.name, model, size_str, body,
                    )
                    raise
            except _RETRYABLE as e:
                last_exc = e
                log.warning("OCR request to %s (%s) failed (attempt %d/%d): %s",
                            be.name, model, i + 1, attempts, e)

            if i < attempts - 1:
                delay = min(30.0, self.backoff * (2 ** i))
                delay = random.uniform(0, delay)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
