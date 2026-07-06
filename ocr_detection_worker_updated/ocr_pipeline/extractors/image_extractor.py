"""Plain image extractor — any single image file becomes one image block."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import BaseExtractor, Block, ExtractedDoc, Page


class ImageExtractor(BaseExtractor):
    extensions = ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"]
    format_name = "image"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        img = Image.open(str(path)).convert("RGB")
        page = Page(index=0, label="Image", rendered_whole=True)
        page.blocks.append(
            Block(type="image", image=img, source="standalone_image",
                  meta={"w": img.width, "h": img.height})
        )
        return ExtractedDoc(source_path=str(path), format="image", pages=[page])
