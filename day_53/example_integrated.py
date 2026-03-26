"""
Integrated example: Vector Store with Automatic Embedding Generation
"""

from day_53.vector_store import VectorStore, VectorDocument
from day_53.embedding_generator import create_embedding_generator
import tempfile
import shutil

def main():
    """Demonstrate integrated vector store with embedding generation."""
    print("=== Vector Store with Automatic Embedding Generation ===\n")
    
    # Create temporary directory for persistent storage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Create vector store with automatic embedding generation
        print("1. Creating vector store with local embeddings...")
        store = VectorStore(
            persist_directory=temp_dir,
            use_local_embeddings=True  # Uses sentence-transformers
        )
        
        # Create collection
        print("2. Creating collection...")
        store.create_collection("documents", metadata={"description": "Auto-embedded documents"})
        
        # 3. Add documents WITHOUT pre-computed embeddings
        print("3. Adding documents (embeddings will be generated automatically)...")
        docs = [
            VectorDocument(
                id="doc1",
                content="Python is a powerful programming language used for web development",
                metadata={"category": "programming", "topic": "python"}
            ),
            VectorDocument(
                id="doc2", 
                content="Machine learning algorithms can solve complex problems",
                metadata={"category": "ai", "topic": "machine_learning"}
            ),
            VectorDocument(
                id="doc3",
                content="Natural language processing enables computers to understand text",
                metadata={"category": "ai", "topic": "nlp"}
            ),
            VectorDocument(
                id="doc4",
                content="Web development frameworks make building websites easier",
                metadata={"category": "programming", "topic": "web"}
            ),
            VectorDocument(
                id="doc5",
                content="Data science combines statistics and programming",
                metadata={"category": "data", "topic": "statistics"}
            )
        ]
        
        doc_ids = store.add_documents("documents", docs)
        print(f"   Added {len(doc_ids)} documents with auto-generated embeddings")\n        \n        # 4. Verify documents were added\n        count = store.count_documents(\"documents\")\n        print(f\"   Total documents in collection: {count}\")\n        \n        # 5. Test similarity search\n        print(\"\\n4. Testing similarity search...\")\n        search_results = store.search(\n            \"documents\", \n            \"programming languages and coding\", \n            n_results=3\n        )\n        \n        print(f\"   Found {len(search_results)} similar documents:\")\n        for i, result in enumerate(search_results, 1):\n            print(f\"   {i}. Score: {result.score:.3f}\")\n            print(f\"      Content: {result.document.content}\")\n            print(f\"      Category: {result.document.metadata.get('category', 'N/A')}\")\n            print()\n        \n        # 6. Test metadata filtering\n        print(\"5. Testing search with metadata filtering...\")\n        ai_results = store.search(\n            \"documents\",\n            \"artificial intelligence\",\n            where={\"category\": \"ai\"},\n            n_results=5\n        )\n        \n        print(f\"   Found {len(ai_results)} AI-related documents:\")\n        for result in ai_results:\n            print(f\"   - {result.document.content} (score: {result.score:.3f})\")\n        \n        # 7. Add a single document and test caching\n        print(\"\\n6. Testing single document addition with caching...\")\n        new_doc = VectorDocument(\n            id=\"doc6\",\n            content=\"Deep learning neural networks process complex patterns\",\n            metadata={\"category\": \"ai\", \"topic\": \"deep_learning\"}\n        )\n        \n        doc_id = store.add_document(\"documents\", new_doc)\n        print(f\"   Added document: {doc_id}\")\n        \n        # 8. Update a document\n        print(\"\\n7. Testing document update...\")\n        updated_doc = VectorDocument(\n            id=\"doc1\",\n            content=\"Python is an excellent programming language for AI and web development\",\n            metadata={\"category\": \"programming\", \"topic\": \"python\", \"updated\": True}\n        )\n        \n        store.update_document(\"documents\", updated_doc)\n        print(\"   Document updated with new content and embedding\")\n        \n        # Verify update\n        retrieved = store.get_document(\"documents\", \"doc1\")\n        print(f\"   Updated content: {retrieved.content}\")\n        \n        # 9. Test search with updated content\n        print(\"\\n8. Testing search with updated content...\")\n        python_results = store.search(\n            \"documents\",\n            \"Python artificial intelligence\",\n            n_results=2\n        )\n        \n        print(\"   Top results for 'Python artificial intelligence':\")\n        for result in python_results:\n            print(f\"   - {result.document.content[:60]}... (score: {result.score:.3f})\")\n        \n        # 10. Test persistence\n        print(\"\\n9. Testing persistence...\")\n        store.close()\n        \n        # Create new store instance\n        new_store = VectorStore(\n            persist_directory=temp_dir,\n            use_local_embeddings=True\n        )\n        \n        # Verify data persisted\n        persisted_count = new_store.count_documents(\"documents\")\n        print(f\"   Documents persisted: {persisted_count}\")\n        \n        # Test search on persisted data\n        persisted_results = new_store.search(\"documents\", \"machine learning\", n_results=2)\n        print(f\"   Search on persisted data found {len(persisted_results)} results\")\n        \n        new_store.close()\n        \n        print(\"\\n✅ Integrated vector store example completed successfully!\")\n        print(\"\\n🎯 Key Features Demonstrated:\")\n        print(\"   • Automatic embedding generation using sentence-transformers\")\n        print(\"   • Batch processing for efficient embedding generation\")\n        print(\"   • Embedding caching for improved performance\")\n        print(\"   • Semantic similarity search\")\n        print(\"   • Metadata filtering\")\n        print(\"   • Document updates with re-embedding\")\n        print(\"   • Persistent storage with embeddings\")\n        \n    except Exception as e:\n        print(f\"❌ Error: {e}\")\n        import traceback\n        traceback.print_exc()\n    finally:\n        # Clean up temporary directory\n        shutil.rmtree(temp_dir, ignore_errors=True)\n\n\nif __name__ == \"__main__\":\n    main()