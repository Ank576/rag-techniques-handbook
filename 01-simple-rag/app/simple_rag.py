import os
import sys
import importlib
from typing import List

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ----- Environment & dependency sanity checks -----

st.write("Python version:", sys.version)
st.write("Has fitz module:", importlib.util.find_spec("fitz") is not None)

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    st.error(
        "PyMuPDF (fitz) is not installed in this environment.\n\n"
        "Make sure `PyMuPDF` is present in `requirements.txt` for this app path."
    )
    st.stop()

# ----- Configuration -----

load_dotenv()
OPENAI_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "Missing OPENAI_API_KEY.\n\n"
        "Set it in Streamlit secrets or a local .env file before using the app."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(
    page_title="Simple RAG – Financial Q&A",
    page_icon="📚",
)

st.title("📚 Simple RAG – Financial Document Q&A")
st.markdown("Upload a PDF and ask questions about its content. Created by Ankit for educational purpose only")

# ----- Helper functions -----


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from an uploaded PDF file-like object."""
    try:
        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            return ""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []
        for page in doc:
            all_text.append(page.get_text("text"))
        return "\n".join(all_text)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
    return chunks


def create_embeddings(inputs: List[str]):
    if not inputs:
        return None
    try:
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=inputs,
        )
    except Exception as e:
        import streamlit as st
        st.error(f"Embedding error: {e}")
        raise


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def semantic_search(
    query: str,
    text_chunks: List[str],
    embeddings,
    k: int = 3,
) -> List[str]:
    """Find top-k most relevant chunks for a query."""
    if not text_chunks or embeddings is None:
        return []

    query_embedding = create_embeddings([query]).data[0].embedding
    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding.embedding)
        similarity_scores.append((i, score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarity_scores[:k]]
    return [text_chunks[idx] for idx in top_indices]


def generate_response(query: str, context_chunks: List[str]) -> str:
    """Generate an answer using GPT based strictly on provided context."""
    if not context_chunks:
        return "I don't have enough information to answer that."

    context = "\n\n".join(
        [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    system_prompt = (
        "You are a helpful financial assistant. "
        "Answer questions based strictly on the provided context. "
        "If the answer is not in the context, say "
        "\"I don't have enough information to answer that.\""
    )

    user_prompt = f"{context}\n\nQuestion: {query}"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


# ----- Main UI -----

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text, chunk_size=1000, overlap=200)

        if not chunks:
            st.error("No text could be extracted from the PDF.")
        else:
            # Create embeddings for all chunks once
            embeddings_response = create_embeddings(chunks)

            # Store in session state
            st.session_state["chunks"] = chunks
            st.session_state["embeddings"] = embeddings_response.data
            st.session_state["text_length"] = len(text)

            st.success(
                f"✅ Document processed! {len(chunks)} chunks created from "
                f"{len(text)} characters."
            )

# Query interface
if "chunks" in st.session_state and st.session_state["chunks"]:
    st.markdown("---")
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Searching and generating answer..."):
            relevant_chunks = semantic_search(
                query,
                st.session_state["chunks"],
                st.session_state["embeddings"],
                k=3,
            )
            answer = generate_response(query, relevant_chunks)

        st.markdown("### 💡 Answer")
        st.write(answer)

        with st.expander("📄 View retrieved context"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Context {i+1}:**")
                if len(chunk) > 500:
                    st.text(chunk[:500] + "...")
                else:
                    st.text(chunk)
                st.markdown("---")
else:
    st.info("Upload a PDF to start asking questions about it.")

# Sidebar info
st.sidebar.markdown("## About Simple RAG")
st.sidebar.markdown(
    """
This app demonstrates a basic Retrieval-Augmented Generation (RAG) workflow:

1. **Document loading** – Extract text from a PDF.
2. **Chunking** – Split into 1000‑character chunks with 200‑character overlap.
3. **Embedding** – Convert chunks to vectors using OpenAI embeddings.
4. **Retrieval** – Find the top 3 most similar chunks by cosine similarity.
5. **Generation** – Use GPT to answer based on the retrieved context.

**Use case**: Financial document Q&A for loan agreements, policies, and guidelines.
"""
)
