"""Configuration loader.

Loads the YAML configuration file into a nested `dict`-like object that
supports both attribute and dict access. All tunables live in the YAML
file; nothing here is hard-coded from the caller's perspective.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """Dict subclass with attribute-style access and safe `.get` nesting."""

    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for k, v in data.items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | os.PathLike) -> Config:
    """Load and validate a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail fast on missing/invalid required fields."""
    required_top = ("ocr", "ocr_backends", "parallelism", "extractors", "output")
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"config.yaml missing top-level key '{key}'")

    backend = cfg.ocr.get("active_backend")
    if backend not in cfg.ocr_backends:
        raise ValueError(
            f"ocr.active_backend='{backend}' not found in ocr_backends "
            f"(available: {list(cfg.ocr_backends.keys())})"
        )

    # Validate EVERY declared backend, not just the active one — script
    # routing (fallback_backend, by_script) can send requests to any of
    # them at runtime, and we'd rather fail loudly at load time than
    # surface a missing-field error mid-batch.
    for name, be in cfg.ocr_backends.items():
        for field in ("base_url", "prompt"):
            if not be.get(field):
                raise ValueError(f"ocr_backends.{name}.{field} is required")

        # Model declaration: accept EITHER `model: <str>` OR `models: [...]`.
        # The list form is for llama-swap-style multi-replica routing —
        # same URL, different model aliases per replica, rotated by the
        # client. Both may be present; `models` wins (see ocr_client.py).
        single = be.get("model")
        many = be.get("models")
        if not single and not (many and len(many) > 0):
            raise ValueError(
                f"ocr_backends.{name}: declare a model — either "
                f"`model: <name>` or `models: [<name1>, <name2>, ...]`"
            )
        if many is not None and not isinstance(many, list):
            raise ValueError(
                f"ocr_backends.{name}.models must be a list of strings"
            )

    for k in ("document_workers", "page_workers", "ocr_concurrency"):
        if int(cfg.parallelism.get(k, 0)) < 1:
            raise ValueError(f"parallelism.{k} must be >= 1")

    # Alias-hop must be >= 1 (1 = disabled). We guard against 0 / negative
    # because the hop loop would never fire and the call would raise.
    if int(cfg.ocr.get("alias_hop_attempts") or 2) < 1:
        raise ValueError("ocr.alias_hop_attempts must be >= 1")

    # Health-check sanity.
    hc = cfg.ocr.get("health_check") or {}
    if hc:
        if float(hc.get("probe_timeout_s") or 10) <= 0:
            raise ValueError("ocr.health_check.probe_timeout_s must be > 0")
        if int(hc.get("min_healthy") or 1) < 1:
            raise ValueError("ocr.health_check.min_healthy must be >= 1")

    # Script routing: every referenced backend must exist in ocr_backends.
    routing = cfg.ocr.get("script_routing") or {}
    if routing.get("enabled"):
        fallback = routing.get("fallback_backend")
        if fallback and fallback not in cfg.ocr_backends:
            raise ValueError(
                f"ocr.script_routing.fallback_backend='{fallback}' not in ocr_backends"
            )
        by_script = routing.get("by_script") or {}
        for script_name, be_name in by_script.items():
            if be_name not in cfg.ocr_backends:
                raise ValueError(
                    f"ocr.script_routing.by_script.{script_name}='{be_name}' "
                    f"not in ocr_backends"
                )
        mode = (routing.get("detection_mode") or "both").lower()
        if mode not in ("surrounding_text", "probe_ocr", "both"):
            raise ValueError(
                f"ocr.script_routing.detection_mode='{mode}' must be one of "
                f"surrounding_text | probe_ocr | both"
            )
        if int(routing.get("probe_max_tokens") or 512) < 32:
            raise ValueError("ocr.script_routing.probe_max_tokens must be >= 32")

    # Elasticsearch section is optional — only validated when enabled.
    _validate_elasticsearch(cfg)


def _validate_elasticsearch(cfg: Config) -> None:
    """Validate the `elasticsearch` section when elasticsearch.enabled is true.

    The section drives es_ingest.py and is irrelevant for plain
    file-based pipeline runs, so it's optional by default.
    """
    es = cfg.get("elasticsearch")
    if not es or not es.get("enabled"):
        return

    required_str = (
        "username", "password", "index",
        "input_dir", "output_dir",
        "system_path_field", "text_field", "status_field", "error_field",
    )
    for f in required_str:
        if not es.get(f):
            raise ValueError(
                f"elasticsearch.{f} is required when elasticsearch.enabled is true"
            )

    hosts = es.get("hosts")
    if not isinstance(hosts, list) or not hosts:
        raise ValueError("elasticsearch.hosts must be a non-empty list")

    if int(es.get("batch_size", 100)) < 1:
        raise ValueError("elasticsearch.batch_size must be >= 1")

    # path_strip_prefix may be an empty string (= strip nothing), but
    # the key must be explicitly present so the contract is clear.
    if "path_strip_prefix" not in es:
        raise ValueError(
            "elasticsearch.path_strip_prefix is required (use \"\" to disable)"
        )


def active_backend(cfg: Config) -> Config:
    """Return the config sub-tree for the currently selected OCR backend."""
    return cfg.ocr_backends[cfg.ocr.active_backend]
