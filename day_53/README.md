# Day 53: Vector Database for Semantic Memory

A vector database implementation using ChromaDB for semantic memory storage and retrieval.

## Features

- **Database Connection**: ChromaDB client with in-memory and persistent storage options
- **Collection Management**: Create, list, delete collections with metadata
- **Vector Storage**: Store documents with embeddings and metadata
- **Document Operations**: CRUD operations (Create, Read, Update, Delete)
- **Search Capabilities**: Similarity search with metadata filtering
- **Metadata Management**: Rich metadata support with filtering
- **Persistent Storage**: Optional file-based persistence
- **Error Handling**: Comprehensive error handling with custom exceptions

## Components

### VectorStore
Main class providing vector database functionality:
- Connection management (in-memory or persistent)
- Collection operations
- Document CRUD operations
- Search and retrieval
- Metadata management

### VectorDocument
Data class representing a document with vector embedding:
- Document ID and content
- Optional vector embedding
- Metadata dictionary

### SearchResult
Data class for search results:
- Document with metadata
- Similarity score and distance

### EmbeddingProvider
Abstract interface for embedding providers:
- Default provider uses ChromaDB's built-in embeddings
- Extensible for custom embedding models

## Usage

### Basic Operations

```python
from day_53.vector_store import VectorStore, VectorDocument

# Initialize vector store
store = VectorStore()

# Create collection
store.create_collection("documents", metadata={"description": "My documents"})

# Add document with pre-computed embedding
doc = VectorDocument(
    id="doc1",
    content="Python is a programming language",
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    metadata={"category": "programming"}
)
store.add_document("documents", doc)

# Retrieve document
retrieved = store.get_document("documents", "doc1")
print(retrieved.content)

# Search similar documents
results = store.search("documents", "programming language", n_results=5)
for result in results:
    print(f"Score: {result.score}, Content: {result.document.content}")

# Clean up
store.close()
```

### Persistent Storage

```python
# Create persistent store
store = VectorStore(persist_directory="./vector_data")

# Operations persist across sessions
store.create_collection("persistent_docs")
# ... add documents ...
store.close()

# Later session - data is still there
new_store = VectorStore(persist_directory="./vector_data")
docs = new_store.peek_collection("persistent_docs")
```

### Metadata Filtering

```python
# Add documents with metadata
docs = [
    VectorDocument(
        id="doc1",
        content="Python tutorial",
        metadata={"category": "programming", "level": "beginner"}
    ),
    VectorDocument(
        id="doc2", 
        content="Advanced Python",
        metadata={"category": "programming", "level": "advanced"}
    )
]
store.add_documents("documents", docs)

# Search with metadata filter
results = store.search(
    "documents",
    "Python",
    where={"category": "programming", "level": "beginner"}
)
```

## Error Handling

The implementation includes comprehensive error handling:

- `VectorStoreError`: Base exception for vector store operations
- `CollectionNotFoundError`: Collection doesn't exist
- `DocumentNotFoundError`: Document doesn't exist

```python
try:
    doc = store.get_document("nonexistent", "doc1")
except DocumentNotFoundError:
    print("Document not found")
except VectorStoreError as e:
    print(f"Vector store error: {e}")
```

## Testing

Run the simple test to verify functionality:

```bash
cd day_53
PYTHONPATH=.. python simple_test.py
```

## Dependencies

- `chromadb`: Vector database backend
- `typing`: Type hints
- `dataclasses`: Data structures
- `uuid`: ID generation
- `os`: File operations

## Notes

- ChromaDB requires non-empty metadata for collections and documents
- The implementation provides default metadata when none is specified
- Embeddings can be pre-computed or generated automatically by ChromaDB
- Network connectivity may be required for ChromaDB's default embedding models
- Use pre-computed embeddings to avoid network dependencies in testing