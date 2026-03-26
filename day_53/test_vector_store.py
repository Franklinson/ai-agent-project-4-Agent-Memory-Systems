"""
Tests for vector_store.py
"""

import pytest
import tempfile
import shutil
from typing import List
from day_53.vector_store import (
    VectorStore, VectorDocument, SearchResult, EmbeddingProvider,
    VectorStoreError, CollectionNotFoundError, DocumentNotFoundError
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def embed_text(self, text: str) -> List[float]:
        # Simple hash-based embedding for testing
        return [float(hash(text) % 100) / 100.0] * 5
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


class TestVectorStore:
    """Test cases for VectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore()
        self.persistent_store = VectorStore(persist_directory=self.temp_dir)
        
        # Clean up any existing collections
        try:
            for collection_name in self.store.list_collections():
                self.store.delete_collection(collection_name)
        except:
            pass
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.store.close()
        self.persistent_store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_collection(self):
        """Test collection creation."""
        self.store.create_collection("test_collection")
        collections = self.store.list_collections()
        assert "test_collection" in collections
    
    def test_create_collection_with_metadata(self):
        """Test collection creation with metadata."""
        metadata = {"description": "Test collection", "version": "1.0"}
        self.store.create_collection("test_collection", metadata=metadata)
        
        collection_metadata = self.store.get_collection_metadata("test_collection")
        assert collection_metadata["description"] == "Test collection"
        assert collection_metadata["version"] == "1.0"
    
    def test_get_nonexistent_collection(self):
        """Test getting a non-existent collection."""
        with pytest.raises(CollectionNotFoundError):
            self.store.get_collection("nonexistent")
    
    def test_delete_collection(self):
        """Test collection deletion."""
        self.store.create_collection("test_collection")
        assert "test_collection" in self.store.list_collections()
        
        self.store.delete_collection("test_collection")
        assert "test_collection" not in self.store.list_collections()
    
    def test_add_document(self):
        """Test adding a single document."""
        self.store.create_collection("test_collection")
        
        doc = VectorDocument(
            id="doc1",
            content="This is a test document",
            metadata={"type": "test", "category": "example"}
        )
        
        doc_id = self.store.add_document("test_collection", doc)
        assert doc_id == "doc1"
        
        # Verify document was added
        assert self.store.count_documents("test_collection") == 1
    
    def test_add_document_without_id(self):
        """Test adding a document without specifying ID."""
        self.store.create_collection("test_collection")
        
        doc = VectorDocument(
            content="This is a test document",
            metadata={"type": "test"}
        )
        
        doc_id = self.store.add_document("test_collection", doc)
        assert doc_id is not None
        assert len(doc_id) > 0
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        self.store.create_collection("test_collection")
        
        docs = [
            VectorDocument(id="doc1", content="First document"),
            VectorDocument(id="doc2", content="Second document"),
            VectorDocument(id="doc3", content="Third document")
        ]
        
        doc_ids = self.store.add_documents("test_collection", docs)
        assert len(doc_ids) == 3
        assert self.store.count_documents("test_collection") == 3
    
    def test_get_document(self):
        """Test retrieving a document."""
        self.store.create_collection("test_collection")
        
        original_doc = VectorDocument(
            id="doc1",
            content="This is a test document",
            metadata={"type": "test"}
        )
        
        self.store.add_document("test_collection", original_doc)
        retrieved_doc = self.store.get_document("test_collection", "doc1")
        
        assert retrieved_doc.id == "doc1"
        assert retrieved_doc.content == "This is a test document"
        assert retrieved_doc.metadata["type"] == "test"
    
    def test_get_nonexistent_document(self):
        """Test getting a non-existent document."""
        self.store.create_collection("test_collection")
        
        with pytest.raises(DocumentNotFoundError):
            self.store.get_document("test_collection", "nonexistent")
    
    def test_update_document(self):
        """Test updating a document."""
        self.store.create_collection("test_collection")
        
        # Add original document
        original_doc = VectorDocument(
            id="doc1",
            content="Original content",
            metadata={"version": "1.0"}
        )
        self.store.add_document("test_collection", original_doc)
        
        # Update document
        updated_doc = VectorDocument(
            id="doc1",
            content="Updated content",
            metadata={"version": "2.0"}
        )
        self.store.update_document("test_collection", updated_doc)
        
        # Verify update
        retrieved_doc = self.store.get_document("test_collection", "doc1")
        assert retrieved_doc.content == "Updated content"
        assert retrieved_doc.metadata["version"] == "2.0"
    
    def test_delete_document(self):
        """Test deleting a document."""
        self.store.create_collection("test_collection")
        
        doc = VectorDocument(id="doc1", content="Test document")
        self.store.add_document("test_collection", doc)
        assert self.store.count_documents("test_collection") == 1
        
        self.store.delete_document("test_collection", "doc1")
        assert self.store.count_documents("test_collection") == 0
    
    def test_search_documents(self):
        """Test searching for documents."""
        self.store.create_collection("test_collection")
        
        # Add test documents
        docs = [
            VectorDocument(id="doc1", content="Python programming language"),
            VectorDocument(id="doc2", content="Java programming tutorial"),
            VectorDocument(id="doc3", content="Machine learning with Python"),
            VectorDocument(id="doc4", content="Web development basics")
        ]
        self.store.add_documents("test_collection", docs)
        
        # Search for Python-related documents
        results = self.store.search("test_collection", "Python programming", n_results=3)
        
        assert len(results) <= 3
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(result.score >= 0 for result in results)
        
        # Results should be sorted by relevance (highest score first)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_metadata_filter(self):
        """Test searching with metadata filters."""
        self.store.create_collection("test_collection")
        
        # Add documents with different categories
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
            ),
            VectorDocument(
                id="doc3",
                content="Cooking recipes",
                metadata={"category": "cooking", "level": "beginner"}
            )
        ]
        self.store.add_documents("test_collection", docs)
        
        # Search with metadata filter
        results = self.store.search(
            "test_collection",
            "Python",
            where={"category": "programming"}
        )
        
        assert len(results) == 2
        for result in results:
            assert result.document.metadata["category"] == "programming"
    
    def test_search_by_vector(self):
        """Test searching using pre-computed embeddings."""
        provider = MockEmbeddingProvider()
        store = VectorStore(embedding_provider=provider)
        store.create_collection("test_collection")
        
        # Add documents with embeddings
        docs = [
            VectorDocument(
                id="doc1",
                content="Python programming",
                embedding=provider.embed_text("Python programming")
            ),
            VectorDocument(
                id="doc2",
                content="Java development",
                embedding=provider.embed_text("Java development")
            )
        ]
        store.add_documents("test_collection", docs)
        
        # Search using vector
        query_embedding = provider.embed_text("Python coding")
        results = store.search_by_vector("test_collection", query_embedding)
        
        assert len(results) > 0
        assert all(isinstance(result, SearchResult) for result in results)
        
        store.close()
    
    def test_peek_collection(self):
        """Test peeking at collection contents."""
        self.store.create_collection("test_collection")
        
        # Add test documents
        docs = [
            VectorDocument(id=f"doc{i}", content=f"Document {i}")
            for i in range(5)
        ]
        self.store.add_documents("test_collection", docs)
        
        # Peek at first 3 documents
        peeked_docs = self.store.peek_collection("test_collection", limit=3)
        
        assert len(peeked_docs) == 3
        assert all(isinstance(doc, VectorDocument) for doc in peeked_docs)
    
    def test_persistent_storage(self):
        """Test persistent storage functionality."""
        # Add document to persistent store
        self.persistent_store.create_collection("persistent_collection")
        doc = VectorDocument(id="doc1", content="Persistent document")
        self.persistent_store.add_document("persistent_collection", doc)
        
        # Close and recreate store
        self.persistent_store.close()
        new_store = VectorStore(persist_directory=self.temp_dir)
        
        # Verify document persisted
        retrieved_doc = new_store.get_document("persistent_collection", "doc1")
        assert retrieved_doc.content == "Persistent document"
        
        new_store.close()
    
    def test_count_documents(self):
        """Test document counting."""
        self.store.create_collection("test_collection")
        assert self.store.count_documents("test_collection") == 0
        
        # Add documents
        docs = [
            VectorDocument(id=f"doc{i}", content=f"Document {i}")
            for i in range(3)
        ]
        self.store.add_documents("test_collection", docs)
        
        assert self.store.count_documents("test_collection") == 3
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test adding document to non-existent collection
        doc = VectorDocument(id="doc1", content="Test")
        with pytest.raises(CollectionNotFoundError):
            self.store.add_document("nonexistent", doc)
        
        # Test searching non-existent collection
        with pytest.raises(CollectionNotFoundError):
            self.store.search("nonexistent", "query")
    
    def test_custom_embedding_provider(self):
        """Test using custom embedding provider."""
        provider = MockEmbeddingProvider()
        store = VectorStore(embedding_provider=provider)
        
        # Test embedding generation
        embedding = provider.embed_text("test text")
        assert len(embedding) == 5
        assert all(isinstance(x, float) for x in embedding)
        
        # Test batch embedding
        embeddings = provider.embed_batch(["text1", "text2"])
        assert len(embeddings) == 2
        assert all(len(emb) == 5 for emb in embeddings)
        
        store.close()


if __name__ == "__main__":
    pytest.main([__file__])