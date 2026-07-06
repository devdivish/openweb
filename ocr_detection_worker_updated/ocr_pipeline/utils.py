"""Small utilities shared across the pipeline."""
from __future__ import annotations

import base64
import io
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

from PIL import Image


def setup_logging(level: str, fmt: str,
                  file_path: Optional[Path] = None) -> logging.Logger:
    """Configure root logger; optionally tee to a file."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Reset handlers so repeated setup doesn't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter = logging.Formatter(fmt)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(file_path, encoding="utf-8")
        file_h.setFormatter(formatter)
        root.addHandler(file_h)

    return root


def encode_image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a `data:` URL suitable for OpenAI-compatible
    multimodal endpoints."""
    buf = io.BytesIO()
    # Keep alpha only if format supports it — otherwise flatten to RGB.
    if fmt.upper() in ("JPEG", "JPG"):
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
    img.save(buf, format=fmt, optimize=True)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fit_image_for_ocr(
    img: Image.Image,
    max_pixels: Optional[int] = None,
    max_long_side_px: Optional[int] = None,
) -> Image.Image:
    """Downscale `img` so it fits the VLM server's input constraints.

    vLLM / Qwen-VL / dots.ocr deployments commonly reject images that
    exceed `max_pixels` (total area) or an absolute long-side cap with
    HTTP 400. Full-page PDF renders at 200+ DPI can easily trip either
    limit; embedded image crops almost never do.

    Returns the original image unchanged when both constraints are None
    or already satisfied. Uses LANCZOS resampling to preserve legibility
    of small text — critical for OCR quality.
    """
    if not max_pixels and not max_long_side_px:
        return img

    w, h = img.size
    scale = 1.0

    if max_long_side_px:
        long_side = max(w, h)
        if long_side > max_long_side_px:
            scale = min(scale, max_long_side_px / float(long_side))

    if max_pixels:
        area = w * h
        if area > max_pixels:
            scale = min(scale, (max_pixels / float(area)) ** 0.5)

    if scale >= 1.0:
        return img

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def small_image(img: Image.Image, min_side: int) -> bool:
    """Return True if either image dimension is below the threshold."""
    w, h = img.size
    return w < min_side or h < min_side


def path_stem(p: str | Path) -> str:
    return Path(p).stem


# ----------------------------------------------------------------------
# Run-log capture (logging + plain print)
# ----------------------------------------------------------------------

class _TeeStream:
    """Duplicate writes from a stdout/stderr stream into a file handle.

    Leaves all other attributes (fileno, isatty, etc.) delegated to the
    original stream so libraries that probe for TTY behavior still get
    the right answer from the terminal.
    """

    def __init__(self, original: TextIO, file_handle: TextIO):
        self._original = original
        self._file = file_handle

    def write(self, s):
        self._original.write(s)
        try:
            self._file.write(s)
        except Exception:
            pass
        return len(s) if s is not None else 0

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


def _resolve_run_log_path(template: str, out_dir: Optional[Path] = None) -> Path:
    """Expand {timestamp} and resolve relative paths against out_dir."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = Path(template.replace("{timestamp}", ts))
    if not p.is_absolute() and out_dir is not None:
        p = out_dir / p
    return p


def install_run_log(
    log_file: str | Path,
    level: str,
    fmt: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """Tee the full terminal session (logging + print) into `log_file`.

    Order matters: the tee is installed on `sys.stdout` / `sys.stderr`
    BEFORE the logging StreamHandler is created, so the StreamHandler
    picks up the teed `sys.stderr` and every log record also reaches
    the file. Plain `print()` calls are captured via the stdout tee.

    {timestamp} in `log_file` is replaced with YYYYMMDD_HHMMSS so each
    invocation writes its own file.
    """
    log_path = _resolve_run_log_path(str(log_file), out_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered so tail -f works in real time.
    fh = open(log_path, "a", encoding="utf-8", buffering=1)

    # Replace BEFORE reconfiguring logging so new StreamHandlers use the
    # teed streams.
    sys.stdout = _TeeStream(sys.stdout, fh)
    sys.stderr = _TeeStream(sys.stderr, fh)

    # (Re)configure logging so it emits to the (now-teed) stderr.
    setup_logging(level, fmt, None)

    root = logging.getLogger()
    root.info("run log -> %s", log_path)
    return log_path
