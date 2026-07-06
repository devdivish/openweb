"""Shared LibreOffice conversion helper.

All `.doc`, `.ppt`, and (fallback) `.rtf` extraction paths share the same
two failure modes, and both are fixable from one place:

1. **Concurrent profile lock.**
   `soffice --headless` shares `~/.config/libreoffice/4/user` across all
   invocations by default. Run two at once and the second silently
   loses the race (often exits 0 with no output file), which is the
   origin of the "LibreOffice did not produce ..." error.

   Fix: each `convert()` gets its own temp user profile via
   `-env:UserInstallation=file:///tmp/lo_profile_*`. No shared lock →
   full parallelism is safe.

2. **Output filename sanitization.**
   LibreOffice rewrites weird characters in the output filename, so
   `tmp / (path.stem + ".docx")` doesn't always match. We glob the
   output dir for any file with the target extension instead.

Also:
    * Tries `soffice` first, then `libreoffice` (distros vary).
    * Single automatic retry on failure — transient LO hiccups are real.
    * Reports stdout + stderr on failure, not just stderr.
    * Configurable timeout (default 300s for big decks).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


class LibreOfficeError(RuntimeError):
    """Raised when LibreOffice conversion fails after retries."""


def _resolve_binary() -> Optional[str]:
    """Return a usable LibreOffice binary path, or None if not found."""
    env_path = os.getenv("LIBREOFFICE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for name in ("soffice", "libreoffice"):
        p = shutil.which(name)
        if p:
            return p
    return None


def convert(
    src: str | Path,
    target_ext: str,
    out_dir: Optional[str | Path] = None,
    timeout_s: float = 300.0,
    retries: int = 1,
) -> Path:
    """Convert `src` to `target_ext` via LibreOffice.

    Parameters
    ----------
    src         Path to the source file (e.g. legacy .doc/.ppt/.xls/.rtf).
    target_ext  Target extension WITHOUT leading dot (e.g. "docx", "pptx",
                "txt"). Passed to `soffice --convert-to`.
    out_dir     Output directory. Created if not supplied.
    timeout_s   Max seconds per attempt.
    retries     Additional attempts after the first failure.

    Returns
    -------
    Path to the converted file. Caller owns it and is responsible for
    cleanup (typically by cleaning up `out_dir`).

    Raises
    ------
    LibreOfficeError if conversion fails after all retries.
    """
    src = Path(src)
    if not src.exists():
        raise LibreOfficeError(f"source file does not exist: {src}")

    binary = _resolve_binary()
    if binary is None:
        raise LibreOfficeError(
            "LibreOffice not found on PATH (tried 'soffice' and "
            "'libreoffice'). Install it or disable the legacy_* flag."
        )

    out_dir_path = Path(out_dir) if out_dir else Path(
        tempfile.mkdtemp(prefix=f"lo_{target_ext}_")
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    last_err = ""
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        # Unique user profile per invocation -> no lock clash across
        # concurrent callers. Scoped to a temp dir we clean up after.
        profile_dir = Path(tempfile.mkdtemp(prefix="lo_profile_"))
        user_installation = "file://" + profile_dir.as_posix()
        cmd = [
            binary,
            f"-env:UserInstallation={user_installation}",
            "--headless",
            "--norestore",
            "--nolockcheck",
            "--nodefault",
            "--nofirststartwizard",
            "--convert-to", target_ext,
            "--outdir", str(out_dir_path),
            str(src),
        ]

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            rc = result.returncode
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout_s}s"
            log.warning(
                "LibreOffice convert attempt %d/%d TIMEOUT for %s",
                attempt, attempts, src,
            )
            _rmtree_quiet(profile_dir)
            continue
        except FileNotFoundError as e:
            raise LibreOfficeError(f"LibreOffice binary not runnable: {e}")

        _rmtree_quiet(profile_dir)
        elapsed = time.perf_counter() - t0

        # Find any file in out_dir with the requested extension.
        produced = _find_output(out_dir_path, target_ext)

        if produced is not None:
            if attempt > 1:
                log.info(
                    "LibreOffice convert recovered on attempt %d (%.1fs) -> %s",
                    attempt, elapsed, produced.name,
                )
            else:
                log.debug(
                    "LibreOffice convert ok (%.1fs) -> %s", elapsed, produced.name,
                )
            return produced

        # No output file. Build a rich error message.
        last_err = (
            f"rc={rc} no output file in {out_dir_path} "
            f"(stdout={stdout!r}, stderr={stderr!r})"
        )
        log.warning(
            "LibreOffice convert attempt %d/%d FAILED for %s: %s",
            attempt, attempts, src, last_err,
        )

    raise LibreOfficeError(
        f"LibreOffice failed to produce .{target_ext} for {src} "
        f"after {attempts} attempt(s): {last_err}"
    )


def _find_output(out_dir: Path, target_ext: str) -> Optional[Path]:
    target_ext = target_ext.lower().lstrip(".")
    # Most recent matching file wins — LibreOffice writes exactly one
    # per invocation, but the temp dir may be reused across retries.
    candidates: List[Path] = [
        p for p in out_dir.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") == target_ext
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _rmtree_quiet(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
