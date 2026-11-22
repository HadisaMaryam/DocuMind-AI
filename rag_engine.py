
import os
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from vector_store import search_relevant_chunks

# Load API key from the project's .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

# Connect to Groq
client = Groq(api_key=api_key)


def answer_question(question, index, chunks):
    """
    Retrieve relevant PDF information and generate
    an answer using Groq AI.
    """

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    # 1. Retrieve relevant chunks from FAISS
    retrieved_chunks = search_relevant_chunks(
        question,
        index,
        chunks,
        top_k=3
    )

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in this PDF.",
            "sources": []
        }

    # 2. Prepare PDF context
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page_number']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    # 3. Send question and context to Groq
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are DocuMind AI, a helpful PDF study assistant. "
                        "Answer the user's question using only the provided "
                        "PDF context. Do not invent facts or use outside "
                        "knowledge. If the answer is not present in the "
                        "context, clearly say that the PDF does not provide "
                        "enough information. Keep answers clear and easy "
                        "to understand. Mention page numbers when useful."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"PDF CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ],
            max_completion_tokens=1000
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }

    except Exception as error:
        return {
            "answer": f"Groq API error: {error}",
            "sources": retrieved_chunks
        }
