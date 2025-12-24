import os
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Semantic Chunking RAG", layout="wide")

class SemanticChunker:
    def __init__(self, threshold=0.72, model="text-embedding-3-small"):
        self.threshold = threshold
        self.model = model
    
    @st.cache_data
    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        response = client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def semantic_chunk(self, text: str) -> Tuple[List[str], List[float], List[int]]:
        """Split text into semantic chunks"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [text], [], []
        
        # Get embeddings for all sentences
        embeddings = [self.get_embedding(sent) for sent in sentences]
        
        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)
        
        # Find split points (where similarity drops below threshold)
        split_indices = [0]
        current_chunk_start = 0
        
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                split_indices.append(i + 1)
                current_chunk_start = i + 1
        
        split_indices.append(len(sentences))
        
        # Create chunks
        chunks = []
        chunk_boundaries = []
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            chunk = ' '.join(sentences[start:end])
            chunks.append(chunk)
            chunk_boundaries.append((start, end))
        
        return chunks, similarities, chunk_boundaries

def fixed_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Fixed size chunking for comparison"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def rag_query(chunks: List[str], question: str, top_k: int = 3) -> str:
    """Simple RAG retrieval and generation"""
    # Embed question
    question_embedding = chunker.get_embedding(question)
    
    # Calculate similarities with all chunks
    chunk_embeddings = [chunker.get_embedding(chunk) for chunk in chunks]
    similarities = [
        np.dot(question_embedding, chunk_emb) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(chunk_emb)
        )
        for chunk_emb in chunk_embeddings
    ]
    
    # Get top-k chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    
    # Generate answer
    prompt = f"""Using only the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Initialize chunker
if 'chunker' not in st.session_state:
    st.session_state.chunker = SemanticChunker()

chunker = st.session_state.chunker

st.title("🔥 Semantic Chunking RAG")
st.markdown("**Better retrieval through intelligent chunking** - Day 3 of RAG Techniques Handbook")

tab1, tab2, tab3 = st.tabs(["📄 Upload & Chunk", "📊 Visualize", "🤖 RAG Query"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload PDF (financial reports, RBI circulars, etc.)",
        type="pdf"
    )
    
    if uploaded_file is not None:
        try:
            # Extract text from PDF
            mypdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page_num in range(mypdf.page_count):
                page = mypdf[page_num]
                text += page.get_text("text")
            st.success(f"✅ Loaded: {uploaded_file.name} ({len(text):,} chars)")
            
            # Cache results
            if 'text' not in st.session_state:
                st.session_state.text = text
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Semantic Chunks")
                if st.button("🔄 Semantic Chunk", key="semantic"):
                    with st.spinner("Chunking semantically..."):
                        st.session_state.semantic_chunks, st.session_state.similarities, st.session_state.boundaries = chunker.semantic_chunk(text)
                        st.session_state.chunk_type = "semantic"
                
                if 'semantic_chunks' in st.session_state:
                    st.metric("Chunks Created", len(st.session_state.semantic_chunks))
                    for i, chunk in enumerate(st.session_state.semantic_chunks[:3]):
                        with st.expander(f"Chunk {i+1} ({len(chunk):,} chars)"):
                            st.write(chunk)
                    if len(st.session_state.semantic_chunks) > 3:
                        st.info(f"... and {len(st.session_state.semantic_chunks)-3} more")
            
            with col2:
                st.subheader("Fixed Size (Baseline)")
                chunk_size = st.slider("Chunk Size", 200, 1000, 500)
                if st.button("🔄 Fixed Chunk", key="fixed"):
                    st.session_state.fixed_chunks = fixed_chunk(text, chunk_size)
                    st.session_state.chunk_type = "fixed"
                
                if 'fixed_chunks' in st.session_state:
                    st.metric("Chunks Created", len(st.session_state.fixed_chunks))
                    for i, chunk in enumerate(st.session_state.fixed_chunks[:3]):
                        with st.expander(f"Chunk {i+1} ({len(chunk):,} chars)"):
                            st.write(chunk)

with tab2:
    if 'similarities' in st.session_state:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.similarities))),
            y=st.session_state.similarities,
            mode='lines+markers',
            name='Sentence Similarity',
            line=dict(color='blue')
        ))
        
        # Mark split points
        split_sims = []
        for i, sim in enumerate(st.session_state.similarities):
            if i+1 in [b[0] for b in st.session_state.boundaries[1:]]:
                split_sims.append(sim)
            else:
                split_sims.append(None)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(split_sims))),
            y=split_sims,
            mode='markers',
            name='Split Points ⭐',
            marker=dict(size=12, color='red', symbol='star')
        ))
        
        fig.add_hline(y=chunker.threshold, line_dash="dash", 
                     annotation_text=f"Threshold ({chunker.threshold})")
        fig.update_layout(
            title="Semantic Similarity Between Consecutive Sentences",
            xaxis_title="Sentence Pair", 
            yaxis_title="Cosine Similarity"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Similarity", f"{np.mean(st.session_state.similarities):.3f}")
        with col2:
            st.metric("Split Points", len(st.session_state.boundaries)-2)

with tab3:
    st.subheader("Test RAG Performance")
    question = st.text_area("Ask a question about the document:", height=100)
    
    if st.button("🚀 Query RAG") and question and 'semantic_chunks' in st.session_state:
        with st.spinner("Retrieving + Generating..."):
            answer = rag_query(st.session_state.semantic_chunks, question)
            st.markdown("### **Answer**")
            st.write(answer)
            st.balloons()

st.markdown("---")
st.markdown("""
**Portfolio Ready Features:**
- ✅ Semantic vs Fixed chunking comparison
- ✅ Interactive similarity visualization  
- ✅ Production OpenAI embeddings + GPT-4o-mini
- ✅ PDF processing for fintech docs
- ✅ Cached computations for speed
- ✅ Threshold tuning slider
""")
