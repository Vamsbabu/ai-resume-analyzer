# =============================================================
# resume_parser.py - PDF Text Extraction Module
# Extracts raw text from PDF resumes using pdfminer.six
# =============================================================

import re
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.

    Args:
        pdf_path (str): The file path to the PDF resume.

    Returns:
        str: Cleaned text extracted from the PDF.
    """

    # ── Use pdfminer to extract text with layout analysis parameters ──
    # LAParams controls how text blocks and lines are detected
    laparams = LAParams(
        line_margin=0.5,      # Margin between lines in same paragraph
        word_margin=0.1,      # Margin between words
        char_margin=2.0,      # Margin between characters
        boxes_flow=0.5,       # How much to consider column layout
    )

    raw_text = extract_text(pdf_path, laparams=laparams)

    if not raw_text:
        return ""

    # ── Clean and normalize the extracted text ──
    cleaned_text = clean_text(raw_text)
    return cleaned_text


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text extracted from a PDF.

    - Removes excessive whitespace and blank lines
    - Normalizes Unicode characters
    - Removes non-printable characters

    Args:
        text (str): Raw text from PDF.

    Returns:
        str: Cleaned text string.
    """

    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove non-printable characters (keep standard ASCII + common Unicode)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def get_resume_preview(text: str, max_chars: int = 500) -> str:
    """
    Returns a short preview of the extracted resume text.
    Useful for debugging or displaying to the user.

    Args:
        text (str): Full resume text.
        max_chars (int): Maximum number of characters to return.

    Returns:
        str: Truncated preview of the resume.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
