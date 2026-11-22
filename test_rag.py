
from pathlib import Path

from pdf_processor import extract_text_from_pdf
from vector_store import (
    create_chunks,
    create_embeddings,
    create_vector_database
)
from rag_engine import answer_question

# Find the PDF in the project folder
pdf_path = Path(__file__).parent / "sample.pdf"

print("Loading PDF...")

if not pdf_path.exists():
    raise FileNotFoundError(
        "sample.pdf not found. Place it in the DocuMind-AI folder."
    )

# Read PDF
with open(pdf_path, "rb") as file:
    pdf_bytes = file.read()

# Extract PDF text
pages = extract_text_from_pdf(pdf_bytes)
print("PDF text extracted successfully!")

# Create chunks
chunks = create_chunks(pages)
print("Chunks created:", len(chunks))

# Generate embeddings
embeddings = create_embeddings(chunks)
print("Embeddings generated!")

# Create FAISS database
index = create_vector_database(chunks, embeddings)
print("FAISS database created!")

print("\nDocuMind AI is ready!")

# Ask questions continuously
while True:
    question = input(
        "\nAsk a question about your PDF (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        print("Thank you for using DocuMind AI!")
        break

    result = answer_question(question, index, chunks)

    print("\n" + "=" * 50)
    print("DOCUMIND AI ANSWER")
    print("=" * 50)
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"Page {source['page_number']} "
            f"| Similarity: "
            f"{source['similarity_score']:.3f}"
        )
    print("=" * 50)
