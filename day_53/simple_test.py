"""
Simple test for vector_store.py without network dependencies
"""

import pytest
from day_53.vector_store import VectorStore, VectorDocument, VectorStoreError

def test_basic_operations():
    """Test basic vector store operations without embeddings."""
    store = VectorStore()
    
    # Test collection creation
    store.create_collection("test_basic_ops")
    collections = store.list_collections()
    assert "test_basic_ops" in collections
    
    # Test document with pre-computed embedding
    doc = VectorDocument(
        id="doc1",
        content="Test document",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],  # Pre-computed embedding
        metadata={"type": "test"}
    )
    
    # Add document
    doc_id = store.add_document("test_basic_ops", doc)
    assert doc_id == "doc1"
    
    # Count documents
    count = store.count_documents("test_basic_ops")
    assert count == 1
    
    # Get document
    retrieved_doc = store.get_document("test_basic_ops", "doc1")
    assert retrieved_doc.content == "Test document"
    assert retrieved_doc.metadata["type"] == "test"
    
    # Update document
    updated_doc = VectorDocument(
        id="doc1",
        content="Updated test document",
        embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
        metadata={"type": "test", "updated": True}
    )
    store.update_document("test_basic_ops", updated_doc)
    
    # Verify update
    retrieved_doc = store.get_document("test_basic_ops", "doc1")
    assert retrieved_doc.content == "Updated test document"
    assert retrieved_doc.metadata["updated"] is True
    
    # Delete document
    store.delete_document("test_basic_ops", "doc1")
    assert store.count_documents("test_basic_ops") == 0
    
    # Clean up
    store.close()
    print("✅ Basic operations test passed!")

def test_collection_management():
    """Test collection management operations."""
    store = VectorStore()
    
    # Create collection with metadata
    metadata = {"description": "Test collection", "version": "1.0"}
    store.create_collection("test_collection_mgmt", metadata=metadata)
    
    # Get collection metadata
    collection_metadata = store.get_collection_metadata("test_collection_mgmt")
    assert collection_metadata["description"] == "Test collection"
    assert collection_metadata["version"] == "1.0"
    
    # List collections
    collections = store.list_collections()
    assert "test_collection_mgmt" in collections
    
    # Delete collection
    store.delete_collection("test_collection_mgmt")
    collections = store.list_collections()
    assert "test_collection_mgmt" not in collections
    
    store.close()
    print("✅ Collection management test passed!")

if __name__ == "__main__":
    test_basic_operations()
    test_collection_management()
    print("🎉 All tests passed!")