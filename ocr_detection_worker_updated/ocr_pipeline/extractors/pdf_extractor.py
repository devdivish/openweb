"""PDF extractor (PyMuPDF / fitz).

Resilient hybrid strategy per page. Each page travels through up to six
checks; only the last one (visual blankness) is allowed to declare a
page truly empty:

  0. **Doc-level fingerprint** — if page 1 already revealed a font
     pathology (failed trust check, or `>max_embedded_images_per_page`
     embedded images suggesting a Type 3 bitmap font), the rest of the
     PDF is rendered+OCR'd unconditionally without per-page checks.
     PDFs use consistent fonts throughout; once we see the pattern on
     page 1 it's almost guaranteed on every page.
  1. Pull native text via PyMuPDF structured dict (preserves blocks).
  2. **Trust check** — if native text count is high but the text fails a
     vowel/bigram heuristic, the fonts probably have a broken /ToUnicode
     CMap; treat the text as garbage and route the page to whole-page OCR.
  3. **Low-text branch** — if native text is below `min_native_chars_per_page`
     AND a low-DPI render shows visible ink on the page, render at full
     `render_dpi` and OCR. (The visual-ink probe replaces the older
     "has embedded raster image" heuristic, which silently dropped pages
     whose content was vector-only or wrapped in Form XObjects.)
  4. **Embedded-image flood** — if more than `max_embedded_images_per_page`
     embedded images would otherwise be OCR'd individually, treat that
     as a Type 3 bitmap-font pathology and switch to whole-page render.
     One full-page OCR call is dramatically faster (and produces better
     output) than N per-glyph fragment calls.
  5. **Hybrid path** — emit native text blocks plus, if enabled, embedded
     images > `min_image_side_px`.
  6. **Safety net** — before emitting an empty placeholder for any page,
     re-run the visual-ink probe. If the page actually has visible
     content, force-render it (logging a warning) so downstream content
     is never silently lost. This is the last-resort guarantee that no
     visually non-blank page is dropped, regardless of the failure mode
     in steps 1–5.

Block order follows the top-left -> bottom-right reading order returned
by PyMuPDF; image blocks are inserted at their y-position so the final
reconstructed Markdown mirrors the original flow.

Every page records a `decision` block in its meta sidecar describing
which branch fired and the signals that drove it (native_char_count,
embedded_images, visual_density, trust_scores) — so post-mortem audits
on production runs don't require re-extraction.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from .base import BaseExtractor, Block, ExtractedDoc, Page
from ..scripts import detect_script

log = logging.getLogger(__name__)


class PdfExtractor(BaseExtractor):
    extensions = ["pdf"]
    format_name = "pdf"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = str(path)
        doc_meta = ExtractedDoc(source_path=path, format="pdf")
        min_native = int(self.cfg.ocr.min_native_chars_per_page)
        min_side = int(self.cfg.ocr.min_image_side_px)
        render_dpi = int(self.cfg.ocr.render_dpi)
        ocr_embedded = bool(self.cfg.extractors.pdf.get("ocr_embedded_images", True))
        # When enabled, run a cheap heuristic on the extracted native text
        # to detect pages whose fonts lack a usable /ToUnicode CMap. Such
        # PDFs return plausible-length but unreadable output (effectively
        # a shift-cipher of the real text); without this check the page
        # sails past `min_native_chars_per_page` and OCR is never invoked.
        trust_native_check = bool(
            self.cfg.extractors.pdf.get("trust_native_text_check", True)
        )
        # Visual-ink probe — used as ground truth for "does this page have
        # visible content?" both for the low-text branch and the safety
        # net. Tunables (with sensible defaults):
        #   * visual_blank_threshold (default 0.005 = 0.5% non-white pixels)
        #     — pages above this are treated as visually non-blank.
        #     Lower values trigger over-eagerly on anti-alias noise; higher
        #     values risk missing pages with sparse/light-coloured content.
        #   * visual_density_dpi (default 50) — render resolution for the
        #     probe. 50 DPI is enough for an A4/Letter page in 3–8 ms;
        #     raising it slightly improves detection of very faint content
        #     at proportional cost.
        visual_blank_threshold = float(
            self.cfg.extractors.pdf.get("visual_blank_threshold", 0.005)
        )
        visual_density_dpi = int(
            self.cfg.extractors.pdf.get("visual_density_dpi", 50)
        )
        # Embedded-image flood guard: pages whose `_embedded_images()`
        # returns more than this go to whole-page render instead of
        # per-fragment OCR. Catches Type 3 / bitmap-glyph fonts where
        # each character is stored as its own tiny image — without this,
        # a 25-page doc can produce hundreds of OCR calls (one per
        # glyph) and run for many minutes. Default 3: typical docs have
        # 0–3 figures per page; anything above 3 is suspicious.
        max_embedded_per_page = int(
            self.cfg.extractors.pdf.get("max_embedded_images_per_page", 3)
        )

        # Doc-level fingerprint: when page 1 reveals font pathology,
        # this flag is set and every subsequent page goes straight to
        # whole-page render+OCR without per-page evaluation. Faster
        # (no probe / trust-check overhead) and more consistent
        # (uniform treatment for a uniformly-broken document).
        doc_force_whole_render = False

        with fitz.open(path) as pdf:
            for i, page in enumerate(pdf):
                page_obj = Page(index=i, label=f"Page {i + 1}")

                # --- 0. Doc-level shortcut ---
                # If a previous page (in practice always page 1) already
                # diagnosed the document as font-pathological, don't
                # bother running checks again — render and OCR.
                if doc_force_whole_render:
                    page_img = self._render_page(page, render_dpi)
                    page_obj.rendered_whole = True
                    page_obj.blocks.append(
                        Block(
                            type="image", image=page_img,
                            source=f"page_{i + 1}_full_render",
                            meta={
                                "w": page_img.width,
                                "h": page_img.height,
                                "reason": "doc_force_whole_render",
                                "decision": {
                                    "branch": "doc_force_whole_render",
                                    "trigger": "first_page_font_pathology",
                                },
                            },
                        )
                    )
                    doc_meta.pages.append(page_obj)
                    continue

                # --- 1. Collect native text blocks with bbox ---
                text_blocks = self._native_text_blocks(page)
                native_char_count = sum(len(b[1]) for b in text_blocks)

                # Detect the dominant script from all native text on this
                # page. Used to annotate image blocks so the orchestrator
                # can route them to the correct OCR backend.
                joined_text = "\n".join(t for _, t in text_blocks)
                page_script = detect_script(joined_text)

                # --- 2. Trust check on native text ---
                # Only meaningful when there's enough text to evaluate AND
                # the low-text branch below wouldn't already trigger OCR.
                # Garbled-encoding PDFs produce HIGH char counts of
                # nonsense, so this is exactly the case the existing
                # threshold misses.
                native_untrusted = False
                trust_scores: Dict[str, Any] = {}
                if trust_native_check and native_char_count >= min_native:
                    is_trusted, trust_scores = self._check_text_trust(joined_text)
                    if not is_trusted:
                        native_untrusted = True
                        log.warning(
                            "page %d: native text failed trust check (%s); "
                            "falling back to whole-page OCR — likely a PDF "
                            "with missing or broken ToUnicode CMap",
                            i + 1, trust_scores,
                        )

                # --- 3. Visual-ink probe (computed lazily) ---
                # Cached at function scope so each page is probed at most
                # once even when both the low-text branch and the safety
                # net consult it. Pages whose text is rich and trusted
                # never compute the probe at all (zero overhead on the
                # common path).
                visual_density: Optional[float] = None

                def density() -> float:
                    nonlocal visual_density
                    if visual_density is None:
                        visual_density = self._page_visual_density(
                            page, dpi=visual_density_dpi
                        )
                    return visual_density

                images = self._embedded_images(pdf, page, min_side)

                # --- 4. Embedded-image flood check ---
                # Type 3 / bitmap-glyph fonts make `get_images()` return
                # one tiny image per character. Per-fragment OCR would
                # generate dozens of calls per page with poor quality
                # (no surrounding-text context). Whole-page render once
                # is faster AND produces better output.
                embedded_image_flood = (
                    ocr_embedded and len(images) > max_embedded_per_page
                )
                has_trusted_native_text = (
                    native_char_count >= min_native and not native_untrusted
                )

                # --- 5. Decide whole-page OCR vs hybrid ---
                # Trigger whole-page render+OCR when ANY of:
                #   a) native text is present but garbled (trust check),
                #   b) native text is below threshold AND the visual-ink
                #      probe confirms the page has visible content,
                #   c) embedded-image flood (Type 3 bitmap font) when the
                #      native text layer is absent or untrusted.
                needs_whole_render = native_untrusted
                if not needs_whole_render and native_char_count < min_native:
                    if density() >= visual_blank_threshold:
                        needs_whole_render = True
                if (
                    not needs_whole_render
                    and embedded_image_flood
                    and not has_trusted_native_text
                ):
                    needs_whole_render = True
                    log.warning(
                        "page %d: %d embedded images detected (cap=%d) — "
                        "likely a Type 3 bitmap font where each glyph is "
                        "a separate image; rendering whole page once "
                        "instead of OCRing %d fragments",
                        i + 1, len(images), max_embedded_per_page,
                        len(images),
                    )

                # --- 6. Doc-level fingerprint (page 1 only) ---
                # Set the flag if page 1 shows pathology that's almost
                # certainly consistent across the whole document. We
                # trigger on `native_untrusted` (broken /ToUnicode) and
                # `embedded_image_flood` (Type 3 bitmap font), but NOT
                # on plain low-text — page 1 might just be a sparse
                # cover/title page in an otherwise normal doc.
                if i == 0 and not doc_force_whole_render:
                    if native_untrusted or (
                        embedded_image_flood and not has_trusted_native_text
                    ):
                        doc_force_whole_render = True
                        trigger = (
                            "trust check failure (broken /ToUnicode)"
                            if native_untrusted else
                            f"{len(images)} embedded images on first page "
                            f"(suggests Type 3 bitmap font)"
                        )
                        log.warning(
                            "page 1 indicates document-wide font pathology "
                            "(%s); subsequent pages will be rendered+OCR'd "
                            "whole without per-page checks",
                            trigger,
                        )

                if needs_whole_render:
                    page_img = self._render_page(page, render_dpi)
                    page_obj.rendered_whole = True
                    if native_untrusted:
                        reason = "untrusted_native_text"
                    elif embedded_image_flood:
                        reason = "embedded_image_flood"
                    elif native_char_count == 0:
                        reason = "no_native_text"
                    else:
                        reason = "low_native_text"

                    decision: Dict[str, Any] = {
                        "branch": reason,
                        "native_char_count": native_char_count,
                        "embedded_images": len(images),
                        "visual_density": (
                            round(visual_density, 4)
                            if visual_density is not None else None
                        ),
                    }
                    if native_untrusted and trust_scores:
                        decision["trust_scores"] = trust_scores

                    block_meta: Dict[str, Any] = {
                        "w": page_img.width, "h": page_img.height,
                        "reason": reason,
                        "decision": decision,
                    }
                    if native_untrusted and trust_scores:
                        block_meta["trust_scores"] = trust_scores
                    page_obj.blocks.append(
                        Block(type="image", image=page_img,
                              source=f"page_{i + 1}_full_render",
                              meta=block_meta,
                              script_hint=page_script)
                    )
                    doc_meta.pages.append(page_obj)
                    continue

                # --- 5. Hybrid: native text + (optionally) embedded images ---
                # Interleave by y-coordinate so reading order is preserved.
                ordered: List[Tuple[float, Block]] = []

                for bbox, text in text_blocks:
                    if text.strip():
                        ordered.append((bbox[1],
                                        Block(type="text", text=text.strip(),
                                              source=f"page_{i + 1}_native")))

                if ocr_embedded:
                    if embedded_image_flood and has_trusted_native_text:
                        log.warning(
                            "page %d: %d embedded images detected (cap=%d), "
                            "but native text is trusted; keeping native text "
                            "and skipping embedded-image OCR",
                            i + 1, len(images), max_embedded_per_page,
                        )
                    else:
                        for bbox, img in images:
                            ordered.append((bbox[1],
                                            Block(type="image", image=img,
                                                  source=f"page_{i + 1}_embedded",
                                                  meta={"bbox": list(bbox),
                                                        "w": img.width,
                                                        "h": img.height},
                                                  script_hint=page_script)))

                ordered.sort(key=lambda t: t[0])
                page_obj.blocks = [b for _, b in ordered]

                # --- 6. Safety net ---
                # If the hybrid path produced no blocks, the page is
                # either (a) genuinely blank or (b) something the upstream
                # checks didn't catch. The visual probe is the deciding
                # vote: ANY visible ink → render+OCR, no exceptions.
                # This is the last-line guarantee that no non-blank page
                # is silently dropped, irrespective of the failure mode
                # (custom CIDFonts, Type 3 fonts, deeply nested Form
                # XObjects, vector-only glyphs, etc.).
                if not page_obj.blocks:
                    if density() >= visual_blank_threshold:
                        log.warning(
                            "page %d: extraction produced no blocks but "
                            "page is visually non-blank (density=%.4f); "
                            "force-rendering as safety net — investigate "
                            "the source PDF for unusual font/content "
                            "encoding",
                            i + 1, visual_density,
                        )
                        page_img = self._render_page(page, render_dpi)
                        page_obj.rendered_whole = True
                        page_obj.blocks.append(
                            Block(
                                type="image", image=page_img,
                                source=f"page_{i + 1}_full_render",
                                meta={
                                    "w": page_img.width,
                                    "h": page_img.height,
                                    "reason": "safety_net_force_render",
                                    "decision": {
                                        "branch": "safety_net_force_render",
                                        "native_char_count": native_char_count,
                                        "embedded_images": len(images),
                                        "visual_density": round(
                                            visual_density, 4
                                        ),
                                    },
                                },
                                script_hint=page_script,
                            )
                        )
                    else:
                        # Genuinely blank page — record the rationale so
                        # the sidecar shows we VERIFIED blankness rather
                        # than fell through to a placeholder by accident.
                        page_obj.blocks.append(
                            Block(
                                type="text", text="",
                                source=f"page_{i + 1}_empty",
                                meta={
                                    "decision": {
                                        "branch": "empty_blank_verified",
                                        "native_char_count": native_char_count,
                                        "embedded_images": len(images),
                                        "visual_density": round(
                                            visual_density, 4
                                        ),
                                    },
                                },
                            )
                        )

                doc_meta.pages.append(page_obj)

        return doc_meta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _native_text_blocks(page: "fitz.Page") -> List[Tuple[Tuple[float, float, float, float], str]]:
        """Return (bbox, text) tuples for native text blocks."""
        data = page.get_text("dict")
        out: List[Tuple[Tuple[float, float, float, float], str]] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            lines = block.get("lines", [])
            text_parts: List[str] = []
            for line in lines:
                spans = line.get("spans", [])
                text_parts.append("".join(span.get("text", "") for span in spans))
            text = "\n".join(text_parts)
            if text:
                out.append((bbox, text))
        return out

    @staticmethod
    def _embedded_images(pdf: "fitz.Document", page: "fitz.Page",
                         min_side: int) -> List[Tuple[Tuple[float, float, float, float], Image.Image]]:
        """Return (bbox, PIL image) for each embedded raster image >= min_side."""
        out: List[Tuple[Tuple[float, float, float, float], Image.Image]] = []
        for info in page.get_images(full=True):
            xref = info[0]
            try:
                pix = fitz.Pixmap(pdf, xref)
                if pix.n - pix.alpha >= 4:       # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                pix = None
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                log.warning("Failed to extract image xref=%s: %s", xref, e)
                continue

            if img.width < min_side or img.height < min_side:
                continue

            # Find bounding box on page (first rect returned).
            rects = page.get_image_rects(xref)
            bbox = tuple(rects[0]) if rects else (0.0, 0.0, 0.0, 0.0)
            out.append((bbox, img))
        return out

    @staticmethod
    def _page_visual_density(
        page: "fitz.Page", dpi: int = 50, white_threshold: int = 250
    ) -> float:
        """Render the page at low DPI and return the fraction of
        non-white pixels.

        Used as ground truth for "does this page have visible content?".
        Beats `page.get_images()` / `page.get_drawings()` heuristics
        because it answers the question pixels-down rather than relying
        on PyMuPDF's content classifier — pages whose content is stored
        as Type 3 fonts, deeply nested Form XObjects, or vector-drawn
        glyphs still render correctly, so the probe sees them even when
        text/image extraction returns nothing.

        We render directly to grayscale and use PIL's C-implemented
        histogram to count pixels below `white_threshold` (default 250
        out of 255 — tolerant of anti-alias halos on white backgrounds).

        Cost: ~3–8 ms per page at 50 DPI for an A4/Letter page. Trivial
        compared to the milliseconds of `page.get_text("dict")` already
        happening, and only invoked on the small minority of pages
        where the cheap text/trust checks couldn't decide.

        Returns a float in [0.0, 1.0]. Empty/all-white page -> ~0.0;
        a dense block of text -> ~0.05–0.20; a fully inked page -> ~1.0.
        """
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)
        n = pix.width * pix.height
        if n == 0:
            return 0.0
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        # `histogram()` returns a 256-entry list; sum the non-white tail.
        hist = img.histogram()
        non_white = sum(hist[:white_threshold])
        return non_white / n

    @staticmethod
    def _render_page(page: "fitz.Page", dpi: int) -> Image.Image:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    @staticmethod
    def _check_text_trust(
        text: str, min_letters: int = 100
    ) -> Tuple[bool, Dict[str, Any]]:
        """Heuristic: does `text` look like real natural-language prose?

        Catches the common failure mode of PDFs whose embedded fonts have
        a missing or broken `/ToUnicode` CMap. PyMuPDF returns the raw
        glyph codes as if they were Unicode, producing output that has
        plausible length but is unreadable (effectively a Caesar-shifted
        ASCII cipher of the original text).

        Two cheap, orthogonal signals:
          1. **Vowel ratio** — English averages ~38%; European Latin
             languages 35–50%. Ciphered output usually lands well below
             20% or far above 60%.
          2. **Common-bigram coverage** — `th`, `er`, `in`, `de`, `le`,
             etc. cover ~25% of letter-bigrams in real prose. Garbled
             output scores near 0%.

        Returns `(is_trustworthy, scores)`. Scores are stamped into the
        per-page meta when the check fires, so .meta.json sidecars
        explain *why* OCR was triggered.

        Pages with fewer than `min_letters` alphabetic characters default
        to "trusted" — there's not enough signal to reject confidently,
        and it's cheaper to keep the fast path than to needlessly re-OCR
        short pages.
        """
        # The vowel / bigram heuristic is Latin-specific. Non-Latin
        # letters such as Han, Indic, Arabic, etc. pass `isalpha()` but
        # should not be scored against English vowel and bigram ratios.
        letters = [
            c for c in text.lower()
            if ("a" <= c <= "z") or (0x00C0 <= ord(c) <= 0x024F)
        ]
        if len(letters) < min_letters:
            return True, {}

        vowels = sum(1 for c in letters if c in "aeiou")
        vowel_ratio = vowels / len(letters)

        text_letters = "".join(letters)
        bigrams = [text_letters[i:i + 2]
                   for i in range(len(text_letters) - 1)]
        common_bigrams = {
            # Top-30 English bigrams.
            "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
            "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
            "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
            # Plus a few high-frequency bigrams from French / Spanish /
            # German / Italian — avoids false positives on European
            # Latin-script text. Some overlap with English is expected.
            "de", "la", "ch", "ei", "ie", "ge", "un", "ne", "me", "el",
        }
        common_count = sum(1 for bg in bigrams if bg in common_bigrams)
        bigram_ratio = common_count / len(bigrams) if bigrams else 0.0

        scores: Dict[str, Any] = {
            "vowel_ratio": round(vowel_ratio, 3),
            "bigram_ratio": round(bigram_ratio, 3),
            "letter_count": len(letters),
        }

        # Both signals must agree. Thresholds tuned conservatively: real
        # text scores ~0.30–0.45 vowel and ~0.20–0.35 bigram; ciphered
        # text scores near zero on both. The asymmetric vowel band also
        # catches "all consonants" and "all vowels" cipher mappings.
        if not (0.25 <= vowel_ratio <= 0.60):
            return False, scores
        if bigram_ratio < 0.08:
            return False, scores
        return True, scores
