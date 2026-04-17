"""Multimodal ingestion for exercise2.

Supports: txt/md/pdf/image
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image

from exercise2_nexdr.core.models import DocumentChunk

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _chunk_text(text: str, source_file: str, page_no: int | None = None, chunk_size: int = 1200) -> list[DocumentChunk]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    idx = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        part = normalized[start:end]
        chunks.append(
            DocumentChunk(
                content=part,
                source_file=source_file,
                chunk_id=f"{Path(source_file).name}#C{idx}",
                page_no=page_no,
            )
        )
        idx += 1
        start = end
    return chunks


def _read_text_file(path: Path) -> list[DocumentChunk]:
    content = path.read_text(encoding="utf-8", errors="replace")
    return _chunk_text(content, str(path))


def _extract_pdf_page_text(page: fitz.Page) -> str:
    return page.get_text("text") or ""


def _extract_image_text_with_tesseract(image: Image.Image) -> str:
    if pytesseract is None:
        return ""
    try:
        return pytesseract.image_to_string(image, lang="eng+chi_sim")
    except Exception:
        try:
            return pytesseract.image_to_string(image)
        except Exception:
            return ""


def _can_run_tesseract() -> bool:
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=False)
        return True
    except Exception:
        return False


def _extract_image_text_with_llm_vision(image: Image.Image) -> str:
    model = os.getenv("MULTI_MODAL_LLM_MODEL")
    base_url = os.getenv("MULTI_MODAL_LLM_BASE_URL")
    api_key = os.getenv("MULTI_MODAL_LLM_API_KEY")
    if not (model and base_url and api_key):
        return ""

    from openai import OpenAI

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Extract all readable text from the image. Keep original language.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract readable text from this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _extract_image_text(path: Path) -> str:
    with Image.open(path) as image:
        if _can_run_tesseract():
            text = _extract_image_text_with_tesseract(image)
            if text.strip():
                return text

        vision_text = _extract_image_text_with_llm_vision(image)
        if vision_text.strip():
            return vision_text

    return ""


def _read_image_file(path: Path) -> list[DocumentChunk]:
    text = _extract_image_text(path)
    if not text.strip():
        text = (
            "[Image ingestion fallback] No OCR/vision text extracted. "
            "Set pytesseract+tesseract or MULTI_MODAL_LLM_* env vars for better extraction."
        )
    return _chunk_text(text, str(path), page_no=1)


def _read_pdf_file(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc = fitz.open(path)
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = _extract_pdf_page_text(page)
            if not page_text.strip():
                # Fallback: render page image then OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.open(BytesIO(pix.tobytes("png")))
                page_text = _extract_image_text_with_tesseract(image)
                if not page_text.strip():
                    page_text = _extract_image_text_with_llm_vision(image)
                if not page_text.strip():
                    page_text = "[PDF page contains no extractable text]"

            chunks.extend(
                _chunk_text(
                    text=page_text,
                    source_file=str(path),
                    page_no=page_idx + 1,
                )
            )
    finally:
        doc.close()

    return chunks


def ingest_path(path: str) -> list[DocumentChunk]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return _read_text_file(file_path)
    if ext == ".pdf":
        return _read_pdf_file(file_path)
    if ext in IMAGE_EXTENSIONS:
        return _read_image_file(file_path)

    raise ValueError(
        f"Unsupported input type: {ext}. Supported: text/markdown/pdf/image"
    )


def ingest_many(paths: Iterable[str]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in paths:
        chunks.extend(ingest_path(path))
    return chunks
