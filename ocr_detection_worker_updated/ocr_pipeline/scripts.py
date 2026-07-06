"""Lightweight Unicode-range script detector.

No external dependencies — uses hand-rolled ranges from the Unicode
standard. Good enough for routing decisions (Arabic vs Devanagari vs
Latin vs CJK...). Returns the *dominant* script in a text sample, i.e.
the script with the most non-whitespace, non-digit characters.

Script names follow Unicode's ISO 15924 "Script_Name" convention so they
line up 1:1 with the keys users write in config.yaml.

Note: Urdu, Persian, Pashto, and Arabic all share the `Arabic` script,
so one rule in config covers all of them — which is exactly what the
user wants for routing to dots.ocr.
"""
from __future__ import annotations

from typing import Optional, Tuple


# (start, end, script_name). Ranges are inclusive.
# Covers the majority of writing systems likely to appear in documents.
_RANGES: Tuple[Tuple[int, int, str], ...] = (
    # ---- Arabic family (covers Urdu, Persian, Pashto, Sindhi, Uyghur) ----
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF, "Arabic"),   # Arabic Presentation Forms-B

    # ---- Hebrew ----
    (0x0590, 0x05FF, "Hebrew"),
    (0xFB1D, 0xFB4F, "Hebrew"),

    # ---- Indic scripts ----
    (0x0900, 0x097F, "Devanagari"),   # Hindi, Sanskrit, Marathi, Nepali
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),     # Punjabi
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),

    # ---- Southeast Asian ----
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x1780, 0x17FF, "Khmer"),

    # ---- CJK ----
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x31F0, 0x31FF, "Katakana"),
    (0x3400, 0x4DBF, "Han"),          # CJK Ext A
    (0x4E00, 0x9FFF, "Han"),          # CJK Unified
    (0xF900, 0xFAFF, "Han"),          # CJK Compat
    (0x20000, 0x2A6DF, "Han"),        # CJK Ext B
    (0xAC00, 0xD7AF, "Hangul"),       # Korean syllables
    (0x1100, 0x11FF, "Hangul"),       # Hangul Jamo

    # ---- European ----
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic"),
    (0x0370, 0x03FF, "Greek"),
    (0x0530, 0x058F, "Armenian"),
    (0x10A0, 0x10FF, "Georgian"),

    # ---- Latin (keep LAST so more specific scripts win ties) ----
    (0x0041, 0x005A, "Latin"),        # A-Z
    (0x0061, 0x007A, "Latin"),        # a-z
    (0x00C0, 0x024F, "Latin"),        # Latin-1 supplement + Extended
    (0x1E00, 0x1EFF, "Latin"),        # Latin Extended Additional
)


def _classify_char(cp: int) -> Optional[str]:
    for start, end, name in _RANGES:
        if start <= cp <= end:
            return name
    return None


def detect_script(text: str, min_chars: int = 8) -> Optional[str]:
    """Return the dominant script in `text`, or None if inconclusive.

    - Ignores whitespace, digits, and common punctuation (they exist in
      every script and add noise).
    - Requires at least `min_chars` classified characters before returning
      a result — short snippets (e.g. a 3-letter page number) give None.
    """
    if not text:
        return None
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        # Skip ASCII punctuation — it's script-neutral.
        if 0x20 <= ord(ch) <= 0x40 or 0x5B <= ord(ch) <= 0x60 or 0x7B <= ord(ch) <= 0x7E:
            continue
        name = _classify_char(ord(ch))
        if name is None:
            continue
        counts[name] = counts.get(name, 0) + 1
        total += 1

    if total < min_chars or not counts:
        return None
    # Return the script with the highest count.
    return max(counts.items(), key=lambda kv: kv[1])[0]


def detect_script_from_samples(*samples: Optional[str]) -> Optional[str]:
    """Concatenate several text fragments and detect dominant script."""
    joined = "\n".join(s for s in samples if s)
    return detect_script(joined)
