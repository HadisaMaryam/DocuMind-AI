
import streamlit as st

from pdf_processor import extract_text_from_pdf
from vector_store import (
    create_chunks,
    create_embeddings,
    create_vector_database
)
from rag_engine import answer_question


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

# --------------------------------
# CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #8b9cff;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #b0b8c8;
    font-size: 17px;
    margin-bottom: 30px;
}

.stChatMessage {
    border-radius: 12px;
}

div[data-testid="stFileUploader"] {
    background-color: #1b2230;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">📚 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent AI assistant for PDF documents'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------
# SESSION STATE
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("⚙️ Document Settings")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()

        # Identify a changed PDF
        import hashlib

        signature = hashlib.md5(pdf_bytes).hexdigest()

        if signature != st.session_state.pdf_signature:

            with st.spinner("Processing your PDF..."):

                try:
                    pages = extract_text_from_pdf(pdf_bytes)

                    chunks = create_chunks(pages)

                    if not chunks:
                        raise ValueError(
                            "No text chunks found in this PDF."
                        )

                    embeddings = create_embeddings(chunks)

                    index = create_vector_database(
                        chunks,
                        embeddings
                    )

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_signature = signature
                    st.session_state.messages = []

                    st.success("PDF processed successfully!")

                except Exception as error:
                    st.error(f"PDF processing failed: {error}")

    if st.session_state.pdf_name:

        st.divider()

        st.success("Document ready")

        st.write("**File:**", st.session_state.pdf_name)

        st.write(
            "**Total chunks:**",
            len(st.session_state.chunks)
        )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Groq AI + FAISS")


# --------------------------------
# CHAT AREA
# --------------------------------

st.subheader("💬 Ask Your Document")

if st.session_state.index is None:

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )

else:

    st.success(
        f"Currently chatting with: {st.session_state.pdf_name}"
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("📖 View PDF Sources"):

                    for source in message["sources"]:

                        st.markdown(
                            f"**Page {source['page_number']}** "
                            f"| Similarity: "
                            f"{source['similarity_score']:.3f}"
                        )

    # User input
    question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("DocuMind AI is thinking..."):

                result = answer_question(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )

                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)

            if sources:

                with st.expander("📖 View PDF Sources"):

                    for source in sources:

                        st.markdown(
                            f"**Page {source['page_number']}** "
                            f"| Similarity: "
                            f"{source['similarity_score']:.3f}"
                        )

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.markdown(
    "<center>Built with ❤️ using Python, Streamlit, "
    "FAISS, Sentence Transformers and Groq AI</center>",
    unsafe_allow_html=True
)
