import re
import html
from bs4 import BeautifulSoup


def _normalize_unicode_text(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\r\n": "\n",
        "\r": "\n",
        "\t": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _remove_control_characters(text: str) -> str:
    return "".join(
        ch for ch in text
        if ch == "\n" or ord(ch) >= 32
    )


def _normalize_spaces(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ ]{2,}", " ", line)
        cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)


def _normalize_newlines(text: str) -> str:
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_cell_text(cell) -> str:
    text = cell.get_text(separator=" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("|", "\\|")
    return text.strip()


def _html_table_to_markdown(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")

    if table is None:
        return table_html

    rows = []
    rowspan_map = {}

    for tr in table.find_all("tr"):
        row = []
        col_idx = 0

        cells = tr.find_all(["th", "td"])

        for cell in cells:
            while col_idx in rowspan_map:
                value, remaining = rowspan_map[col_idx]
                row.append(value)

                remaining -= 1
                if remaining <= 0:
                    del rowspan_map[col_idx]
                else:
                    rowspan_map[col_idx] = (value, remaining)

                col_idx += 1

            text = _clean_cell_text(cell)

            try:
                colspan = int(cell.get("colspan", 1))
            except Exception:
                colspan = 1

            try:
                rowspan = int(cell.get("rowspan", 1))
            except Exception:
                rowspan = 1

            row.append(text)

            for _ in range(colspan - 1):
                row.append("")

            if rowspan > 1:
                rowspan_map[col_idx] = (text, rowspan - 1)

            col_idx += colspan

        while col_idx in rowspan_map:
            value, remaining = rowspan_map[col_idx]
            row.append(value)

            remaining -= 1
            if remaining <= 0:
                del rowspan_map[col_idx]
            else:
                rowspan_map[col_idx] = (value, remaining)

            col_idx += 1

        if row:
            rows.append(row)

    if not rows:
        return table_html

    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    header = rows[0]
    body = rows[1:]

    markdown_lines = []
    markdown_lines.append("| " + " | ".join(header) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    for row in body:
        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


def _convert_html_tables_to_markdown(text: str) -> str:
    table_pattern = re.compile(
        r"<table[\s\S]*?</table>",
        re.IGNORECASE
    )

    def replace_table(match):
        table_html = match.group(0)

        try:
            markdown_table = _html_table_to_markdown(table_html)
            return f"\n\n{markdown_table}\n\n"
        except Exception:
            return table_html

    return table_pattern.sub(replace_table, text)


def _convert_basic_html_to_markdown_text(text: str) -> str:
    """
    Converts leftover simple HTML tags into readable markdown/text.
    Tables should already be converted before this.
    """

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"</div\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<div[^>]*>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"<strong[^>]*>(.*?)</strong>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<b[^>]*>(.*?)</b>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"<em[^>]*>(.*?)</em>", r"*\1*", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<i[^>]*>(.*?)</i>", r"*\1*", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text


def _fix_common_ocr_artifacts(text: str) -> str:
    # Too many dots
    text = re.sub(r"\.{5,}", "...", text)

    # Long separator-like garbage
    text = re.sub(r"={5,}", "---", text)
    text = re.sub(r"_{5,}", "---", text)
    text = re.sub(r"-{5,}", "---", text)

    # Bad spaces around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Bracket spacing
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)

    return text


def _clean_markdown_table_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            stripped = "| " + " | ".join(parts) + " |"

        cleaned.append(stripped)

    return "\n".join(cleaned)


def _remove_empty_table_rows(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]

            # Keep markdown separator row
            if all(re.fullmatch(r"-+", c.replace(" ", "")) for c in cells):
                cleaned.append(line)
                continue

            # Remove completely empty rows
            if all(c == "" for c in cells):
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


def _fix_heading_spacing(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", line)
        cleaned.append(line)

    return "\n".join(cleaned)


def clean_ocr_markdown_text(markdown_text: str) -> str:
    """
    Pass raw OCR markdown/html text directly.
    Returns cleaned markdown text.

    Example:
        cleaned_md = clean_ocr_markdown_text(raw_ocr_md)
    """

    text = markdown_text or ""

    # 1. Basic cleanup
    text = _normalize_unicode_text(text)
    text = _remove_control_characters(text)
    text = html.unescape(text)

    # 2. Convert HTML tables to markdown tables
    text = _convert_html_tables_to_markdown(text)

    # 3. Convert/remove remaining HTML
    text = _convert_basic_html_to_markdown_text(text)

    # 4. OCR cleanup
    text = _fix_common_ocr_artifacts(text)

    # 5. Markdown cleanup
    text = _clean_markdown_table_lines(text)
    text = _remove_empty_table_rows(text)
    text = _fix_heading_spacing(text)

    # 6. Final whitespace cleanup
    text = _normalize_spaces(text)
    text = _normalize_newlines(text)

    return text + "\n"