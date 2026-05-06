from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file type is not supported."""


class EmptyTextError(Exception):
    """Raised when no text can be extracted from the file."""


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract text from a PDF, DOCX or TXT file."""

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {extension}. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        text = _extract_text_from_pdf(content)
    elif extension == ".docx":
        text = _extract_text_from_docx(content)
    else:
        text = _extract_text_from_txt(content)

    text = _normalize_text(text)

    if not text:
        raise EmptyTextError("Could not extract text from the file.")

    return text


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from all PDF pages."""
    reader = PdfReader(BytesIO(content))

    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n".join(pages_text)


def _extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX paragraphs."""
    document = Document(BytesIO(content))

    paragraphs = []
    for paragraph in document.paragraphs:
        if paragraph.text:
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def _extract_text_from_txt(content: bytes) -> str:
    """Decode TXT content using common encodings."""
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    return content.decode("utf-8", errors="ignore")


def _normalize_text(text: str) -> str:
    """Remove empty lines and extra spaces."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)
