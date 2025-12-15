import sys
import importlib
import streamlit as st

st.write("Python version:", sys.version)

spec = importlib.util.find_spec("fitz")
st.write("Has fitz module:", spec is not None)


st.write("Python:", sys.version)
installed = {d.project_name.lower(): d.version for d in pkg_resources.working_set}
st.write("Has PyMuPDF:", "pymupdf" in installed or "pymupdf" in installed or "pymupdf" in installed)

import fitz # PyMuPDF
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv




# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📚 Simple RAG - Financial Document Q&A")
st.markdown("Upload a PDF and ask questions about its content")

# Helper Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    mypdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_text = ""
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text
    return all_text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_embeddings(text):
    """Create embeddings using OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=3):
    """Find top k most relevant chunks"""
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []
    
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(
            np.array(query_embedding), 
            np.array(chunk_embedding.embedding)
        )
        similarity_scores.append((i, similarity_score))
    
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]

def generate_response(query, context_chunks):
    """Generate response using GPT"""
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    system_prompt = "You are a helpful financial assistant. Answer questions based strictly on the provided context. If the answer isn't in the context, say 'I don't have enough information to answer that.'"
    
    user_prompt = f"{context}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Extract and chunk text
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        # Create embeddings
        embeddings_response = create_embeddings(chunks)
        
        # Store in session state
        st.session_state['chunks'] = chunks
        st.session_state['embeddings'] = embeddings_response.data
        st.session_state['text_length'] = len(text)
        
    st.success(f"✅ Document processed! {len(chunks)} chunks created from {len(text)} characters")

# Query interface
if 'chunks' in st.session_state:
    st.markdown("---")
    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Searching and generating answer..."):
            # Retrieve relevant chunks
            relevant_chunks = semantic_search(
                query, 
                st.session_state['chunks'], 
                st.session_state['embeddings'],
                k=3
            )
            
            # Generate response
            answer = generate_response(query, relevant_chunks)
            
            # Display results
            st.markdown("### 💡 Answer")
            st.write(answer)
            
            # Show retrieved context
            with st.expander("📄 View Retrieved Context"):
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**Context {i+1}:**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.markdown("---")

# Sidebar info
st.sidebar.markdown("## About Simple RAG")
st.sidebar.markdown("""
This app demonstrates the basic RAG workflow:
1. **Document Loading**: Extract text from PDF
2. **Chunking**: Split into 1000-char chunks with 200-char overlap
3. **Embedding**: Convert chunks to vectors using OpenAI
4. **Retrieval**: Find top 3 most similar chunks using cosine similarity
5. **Generation**: Use GPT to answer based on retrieved context

**Use Case**: Financial document Q&A for loan agreements, policies, and guidelines
""")
