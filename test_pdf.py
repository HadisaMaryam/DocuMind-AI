
from pathlib import Path
from pdf_processor import extract_text_from_pdf, get_pdf_summary

# Ask for a PDF path
pdf_path = input("Enter your PDF file path: ").strip().strip('"')

# Check file
if not Path(pdf_path).is_file():
    print("Error: PDF file not found.")
    raise SystemExit(1)

# Read PDF
with open(pdf_path, "rb") as file:
    pdf_bytes = file.read()

try:
    pages = extract_text_from_pdf(pdf_bytes)

    summary = get_pdf_summary(pages)

    print("\nPDF PROCESSED SUCCESSFULLY!")
    print("Total pages:", summary["total_pages"])
    print("Total words:", summary["total_words"])

    print("\nEXTRACTED TEXT PREVIEW:\n")

    for page in pages[:2]:
        print(f"--- Page {page['page_number']} ---")
        print(page["text"][:500])
        print()

except ValueError as error:
    print("PDF Error:", error)
