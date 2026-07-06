"""Top-level orchestrator.

Responsibilities
----------------
1. Pick the right extractor for each input file.
2. Run the extractor (synchronous — CPU-bound, runs in a thread so we
   don't block the event loop).
3. Walk the resulting ExtractedDoc, gather every image block, and fan
   them out through the OCR client — up to
     * `parallelism.page_workers` async tasks per document, AND
     * `parallelism.ocr_concurrency` simultaneous OCR HTTP requests
       across everything (enforced inside OCRClient via a Semaphore,
       letting vLLM/SGLang batch on the server).
4. Splice the OCR results back into the block list in-place.
5. Render Markdown + JSON sidecar and write them to disk.

Multi-document ingestion is handled by `process_many`, which runs up to
`parallelism.document_workers` documents concurrently.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import httpx

from .config import Config
from .extractors import ExtractedDoc, get_extractor, supported_extensions
from .extractors.base import Block, Page
from .formatter import render_markdown, render_sidecar
from .ocr_client import OCRClient
from .scripts import detect_script
from .utils import setup_logging

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.output.directory)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._doc_semaphore = asyncio.Semaphore(int(cfg.parallelism.document_workers))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_one(self, path: str | Path,
                          ocr: OCRClient) -> Tuple[Path, Path]:
        """Extract + OCR + write outputs for a single file."""
        path = Path(path)
        stem = path.stem
        md_path = self.out_dir / self.cfg.output.markdown_filename.format(stem=stem)
        json_path = self.out_dir / self.cfg.output.sidecar_filename.format(stem=stem)

        per_doc_log = None
        if self.cfg.logging.get("per_document_log", False):
            per_doc_log = self.out_dir / f"{stem}.log"
            setup_logging(self.cfg.logging.level,
                          self.cfg.logging.format,
                          per_doc_log)

        t0 = time.perf_counter()
        log.info("Extracting %s", path)

        # 1. Native extraction off-thread (CPU bound / blocking).
        extractor = get_extractor(path, self.cfg)
        doc = await asyncio.to_thread(extractor.extract, path)

        t_extract = time.perf_counter() - t0

        # 2. OCR every image block in parallel, respecting page_workers.
        t_ocr_start = time.perf_counter()
        await self._run_ocr(doc, ocr)
        t_ocr = time.perf_counter() - t_ocr_start

        # 3. Render & write output.
        md = render_markdown(doc, bool(self.cfg.output.include_page_headings))
        md_path.write_text(md, encoding="utf-8")

        timings = {
            "extract_seconds": round(t_extract, 3),
            "ocr_seconds": round(t_ocr, 3),
            "total_seconds": round(time.perf_counter() - t0, 3),
            "pages": len(doc.pages),
            "ocr_blocks": self._count_ocr_blocks(doc),
        }
        json_path.write_text(
            render_sidecar(doc, str(md_path), timings),
            encoding="utf-8",
        )

        log.info("Done %s -> %s (%.2fs, %d pages, %d OCR blocks)",
                 path.name, md_path.name,
                 timings["total_seconds"], timings["pages"],
                 timings["ocr_blocks"])
        return md_path, json_path

    async def process_many(self, paths: Iterable[str | Path]) -> List[Tuple[Path, Path]]:
        paths = [Path(p) for p in paths]
        async with OCRClient(self.cfg) as ocr:
            tasks = [self._wrap_doc(p, ocr) for p in paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        successes: List[Tuple[Path, Path]] = []
        for p, r in zip(paths, results):
            if isinstance(r, Exception):
                log.error("FAILED %s: %s", p, r)
            else:
                successes.append(r)
        return successes

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _wrap_doc(self, path: Path, ocr: OCRClient):
        async with self._doc_semaphore:
            return await self.process_one(path, ocr)

    async def _run_ocr(self, doc: ExtractedDoc, ocr: OCRClient) -> None:
        """Fan out OCR calls for image blocks.

        Concurrency layers:
          - page_workers: how many pages are worked on concurrently.
          - ocr_concurrency (inside OCRClient): global cap across all HTTP
            requests so the OCR servers aren't overloaded. vLLM / SGLang
            batch these server-side on the GPUs.

        Per-image backend selection honors ocr.script_routing.detection_mode:
          - surrounding_text : use block.script_hint (set by the extractor
                               from native text near the image).
          - probe_ocr        : run a cheap short OCR pass on the fallback
                               backend, detect script from ITS output, then
                               route accordingly. Re-uses probe output
                               when the detected script maps back to the
                               same fallback backend (no double call).
          - both             : prefer surrounding_text; fall back to
                               probe_ocr when no hint is available.
        """
        page_sema = asyncio.Semaphore(int(self.cfg.parallelism.page_workers))

        async def ocr_page(page: Page) -> None:
            async with page_sema:
                img_blocks = [b for b in page.blocks if b.type == "image"]
                if not img_blocks:
                    return
                tasks = [asyncio.create_task(self._ocr_block(b, ocr))
                         for b in img_blocks]
                await asyncio.gather(*tasks, return_exceptions=False)

        await asyncio.gather(*(ocr_page(p) for p in doc.pages))

    async def _ocr_block(self, block: Block, ocr: OCRClient) -> None:
        """Run OCR on a single image block with script-aware routing.

        Mutates `block` in place: type becomes 'ocr', text is filled,
        image is freed, and meta is annotated with the chosen backend
        and detected script for the sidecar.
        """
        if block.type != "image" or block.image is None:
            return

        mode = ocr.detection_mode if ocr.routing_enabled else "surrounding_text"
        backend_used: str
        script_detected: Optional[str] = block.script_hint
        text: str = ""

        try:
            if mode == "surrounding_text" or not ocr.routing_enabled:
                backend_used = ocr.resolve_backend_for_script(block.script_hint)
                text = await ocr.ocr_image(block.image, backend=backend_used)

            elif mode == "probe_ocr":
                text, backend_used, script_detected = \
                    await self._probe_then_route(block.image, ocr)

            else:  # "both"
                if block.script_hint is not None:
                    backend_used = ocr.resolve_backend_for_script(block.script_hint)
                    text = await ocr.ocr_image(block.image, backend=backend_used)
                else:
                    text, backend_used, script_detected = \
                        await self._probe_then_route(block.image, ocr)

        except Exception as e:
            log.error("OCR failed on %s: %s", block.source, e)
            block.type = "ocr"
            block.text = f"<!-- OCR failed: {e} -->"
            block.image = None
            block.meta["ocr_error"] = str(e)
            return

        block.type = "ocr"
        block.text = (text or "").strip()
        block.image = None   # free memory
        block.meta["backend_used"] = backend_used
        if script_detected:
            block.meta["script_detected"] = script_detected

    async def _probe_then_route(self, image, ocr: OCRClient):
        """Run a cheap probe OCR on the fallback backend, detect script
        from its output, then re-OCR on the routed backend if different.
        Returns (final_text, backend_used, script_detected).

        If the routed backend rejects the image (4xx — typically the
        stricter-limit backend refusing a large full-page render), we
        keep the probe text as the final output rather than failing the
        whole block. This closes the regression where script routing
        could send a page render to dots.ocr and lose the result.
        """
        fallback = ocr.fallback_backend
        probe_text = await ocr.ocr_image(
            image, backend=fallback, max_tokens=ocr.probe_max_tokens
        )
        script = detect_script(probe_text or "")
        routed = ocr.resolve_backend_for_script(script)

        if routed == fallback:
            # The fallback backend is already the best choice — but the
            # probe was capped at probe_max_tokens, which may have
            # truncated the content. Do a full-token pass only if the
            # probe hit its cap; otherwise reuse the probe output.
            if len(probe_text) < ocr.probe_max_tokens * 3:  # rough char/token ratio
                return probe_text, fallback, script
            full_text = await ocr.ocr_image(image, backend=fallback)
            return full_text, fallback, script

        # Different backend is better for this script — re-OCR there.
        # If it rejects the image (e.g. still too large for the stricter
        # backend despite the per-backend cap), fall back to probe text.
        try:
            final_text = await ocr.ocr_image(image, backend=routed)
            return final_text, routed, script
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                log.warning(
                    "routed backend %s rejected image (HTTP %d); "
                    "falling back to probe text from %s",
                    routed, e.response.status_code, fallback,
                )
                return probe_text, fallback, script
            raise

    @staticmethod
    def _count_ocr_blocks(doc: ExtractedDoc) -> int:
        return sum(1 for p in doc.pages for b in p.blocks if b.type == "ocr")


# ----------------------------------------------------------------------
# Directory walking helpers (used by CLI)
# ----------------------------------------------------------------------

def collect_paths(inputs: Iterable[str | Path]) -> List[Path]:
    """Expand a mixed list of files and directories into concrete files
    whose extension has a registered extractor."""
    exts = set(supported_extensions())
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower().lstrip(".") in exts:
                    out.append(child)
        elif p.is_file():
            if p.suffix.lower().lstrip(".") in exts:
                out.append(p)
            else:
                log.warning("Skipping unsupported file: %s", p)
        else:
            log.warning("Path does not exist: %s", p)
    return out
