
import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF.
    Preserve page numbers for source citations.
    """

    extracted_pages = []

    try:
        # Open PDF from uploaded file bytes
        pdf_document = fitz.open(
            stream=pdf_file,
            filetype="pdf"
        )

        if pdf_document.page_count == 0:
            raise ValueError("This PDF contains no pages.")

        for page_number, page in enumerate(pdf_document, start=1):

            # Extract text from current page
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page_number": page_number,
                    "text": text
                })

        pdf_document.close()

        if not extracted_pages:
            raise ValueError(
                "No readable text found. "
                "This may be a scanned PDF requiring OCR."
            )

        return extracted_pages

    except Exception as error:
        raise ValueError(
            f"PDF processing failed: {error}"
        ) from error


def get_pdf_summary(extracted_pages):
    """
    Generate basic PDF information.
    """

    total_pages = len(extracted_pages)

    total_words = sum(
        len(page["text"].split())
        for page in extracted_pages
    )

    return {
        "total_pages": total_pages,
        "total_words": total_words
    }
