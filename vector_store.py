
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load the embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Split extracted PDF text into chunks
def create_chunks(extracted_pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in extracted_pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page_number": page["page_number"]
            })

    return chunks


# 3. Generate embeddings
def create_embeddings(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.astype("float32")


# 4. Create FAISS vector database
def create_vector_database(chunks, embeddings):

    if len(chunks) == 0:
        raise ValueError("No text chunks available.")

    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    # Add vectors
    index.add(embeddings)

    return index


# 5. Search relevant chunks
def search_relevant_chunks(
    question,
    index,
    chunks,
    top_k=3
):

    if index.ntotal == 0:
        return []

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search FAISS database
    scores, indices = index.search(
        question_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page_number": chunks[idx]["page_number"],
            "similarity_score": float(score)
        })

    return results
