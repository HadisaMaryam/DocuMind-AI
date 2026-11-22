
from pathlib import Path

from pdf_processor import extract_text_from_pdf
from vector_store import (
    create_chunks,
    create_embeddings,
    create_vector_database,
    search_relevant_chunks
)


# STEP 1: Find PDF automatically
pdf_path = Path(__file__).parent / "sample.pdf"

print("Looking for PDF...")

if not pdf_path.exists():
    print("ERROR: sample.pdf not found!")
    print("Please place sample.pdf inside your DocuMind-AI folder.")
    raise SystemExit(1)


# STEP 2: Read PDF
print("\nReading PDF...")

try:
    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()

    pages = extract_text_from_pdf(pdf_bytes)

    print("PDF text extracted successfully!")


    # STEP 3: Create text chunks
    print("\nCreating text chunks...")

    chunks = create_chunks(pages)

    if not chunks:
        print("ERROR: No text chunks were created.")
        raise SystemExit(1)

    print("Total chunks created:", len(chunks))


    # STEP 4: Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = create_embeddings(chunks)

    print("Embeddings generated successfully!")
    print("Embedding shape:", embeddings.shape)


    # STEP 5: Create FAISS database
    print("\nCreating FAISS database...")

    index = create_vector_database(
        chunks,
        embeddings
    )

    print("FAISS database created successfully!")
    print("Total vectors stored:", index.ntotal)


    # STEP 6: Ask a question
    print("\n" + "=" * 40)

    question = input(
        "Ask a question about your PDF: "
    ).strip()

    if not question:
        print("Please enter a question.")
        raise SystemExit(1)


    # STEP 7: Retrieve relevant information
    print("\nSearching for relevant information...")

    results = search_relevant_chunks(
        question,
        index,
        chunks,
        top_k=3
    )


    # STEP 8: Display results
    print("\n" + "=" * 40)
    print("RETRIEVED INFORMATION")
    print("=" * 40)

    if not results:
        print("No relevant information found.")

    else:
        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")

            print(
                "Page:",
                result["page_number"]
            )

            print(
                "Similarity:",
                round(result["similarity_score"], 3)
            )

            print(
                "Text:",
                result["text"]
            )

    print("\n" + "=" * 40)
    print("RAG RETRIEVAL TEST COMPLETED!")
    print("=" * 40)


except Exception as error:

    print("\nAN ERROR OCCURRED:")
    print(error)
