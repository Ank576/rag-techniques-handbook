# Day 2: Semantic Chunking for RAG

## Overview
Semantic chunking improves upon basic text splitting by dividing documents based on meaning rather than arbitrary character or token limits. This technique ensures that semantically related content stays together, leading to more accurate retrieval and better context for LLM responses.

## What is Semantic Chunking?

Semantic chunking analyzes the semantic relationships between sentences or paragraphs and groups text based on conceptual similarity. Unlike fixed-size chunking (Day 1), this approach:
- Preserves contextual integrity by keeping related ideas together
- Creates variable-length chunks based on topic boundaries
- Improves retrieval accuracy by maintaining semantic coherence
- Reduces information fragmentation across chunks

## Key Concepts

### Sentence Embeddings
Embeddings are numerical vector representations of text that capture semantic meaning. Sentences with similar meanings have similar embeddings (measured by cosine similarity).

### Semantic Similarity
Measures how closely related two pieces of text are in meaning, typically using:
- **Cosine similarity**: Measures angle between embedding vectors (0 to 1)
- **Euclidean distance**: Measures geometric distance between vectors

### Breakpoint Detection
Identifies natural boundaries in text where topics shift by:
1. Computing embeddings for each sentence
2. Calculating similarity scores between consecutive sentences
3. Detecting significant drops in similarity (breakpoints)
4. Creating chunks at these semantic boundaries

## Implementation Approaches

### 1. **Percentile-Based Chunking**
Sets a threshold based on the distribution of similarity scores:
```python
threshold = np.percentile(similarities, 75)  # Top 25% dissimilarity
# Split where similarity falls below threshold
```

### 2. **Standard Deviation Method**
Uses statistical measures to identify outliers:
```python
threshold = mean_similarity - (std_dev * factor)
# Splits at points significantly below average
```

### 3. **Gradient-Based Detection**
Identifies sudden changes in similarity:
```python
gradients = np.diff(similarities)
# Large negative gradients indicate topic shifts
```

## Advantages Over Simple Chunking

- **Context Preservation**: Related information stays together
- **Improved Retrieval**: More relevant chunks returned for queries
- **Better LLM Responses**: Coherent context leads to accurate answers
- **Flexible Chunk Sizes**: Adapts to natural content structure
- **Reduced Noise**: Less fragmented information across boundaries

## Common Use Cases

- **Technical Documentation**: Keeps code examples with explanations
- **Legal Documents**: Maintains clause integrity
- **Research Papers**: Preserves argument flow and methodology
- **Financial Reports**: Groups related metrics and analysis
- **Knowledge Bases**: Maintains topical coherence

## Libraries and Tools

### Popular Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2`: Fast, lightweight (384 dimensions)
- `text-embedding-3-small` (OpenAI): High quality, 1536 dimensions
- `text-embedding-3-large` (OpenAI): Best performance, 3072 dimensions

### Implementation Libraries
- **LangChain**: `SemanticChunker` with configurable breakpoints
- **LlamaIndex**: `SemanticSplitterNodeParser`
- **sentence-transformers**: For generating embeddings
- **scikit-learn**: For similarity calculations

## Implementation Example Structure

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize semantic chunker
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=75
)

# Create semantic chunks
chunks = text_splitter.create_documents([document_text])
```

## Configuration Parameters

- **breakpoint_threshold_type**: Method for detecting splits (percentile, std_deviation, gradient)
- **breakpoint_threshold_amount**: Sensitivity of split detection
- **embedding_model**: Model used for semantic representation
- **buffer_size**: Number of sentences to consider for context

## Trade-offs and Considerations

### Advantages
- Superior context preservation
- More accurate retrieval results
- Better LLM response quality
- Adaptive to content structure

### Disadvantages
- Higher computational cost (embedding generation)
- Variable chunk sizes may complicate some workflows
- Requires tuning threshold parameters
- Longer processing time compared to simple splitting

## Best Practices

1. **Choose appropriate embedding models** based on domain and performance needs
2. **Tune threshold parameters** using sample documents from your corpus
3. **Monitor chunk size distribution** to avoid extremely large or small chunks
4. **Consider hybrid approaches** combining semantic and size-based constraints
5. **Test retrieval quality** with representative queries before production

## Next Steps

Day 3 will explore advanced chunking techniques including:
- Hierarchical chunking with parent-child relationships
- Context-enriched chunking with surrounding sentences
- Multi-level indexing strategies
- Chunk optimization for specific domains

## Resources

- [LangChain Semantic Chunking Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker)
- [Sentence Transformers Library](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Chunking Best Practices](https://www.pinecone.io/learn/chunking-strategies/)

---

**Repository**: [RAG Techniques Handbook](https://github.com/Ank576/rag-techniques-handbook)
**Day**: 2 of 30
**Focus**: Semantic Chunking and Embeddings-Based Text Splitting