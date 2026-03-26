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
        print(f"   Added {len(doc_ids)} documents with auto-generated embeddings")
        
        # 4. Verify documents were added
        count = store.count_documents("documents")
        print(f"   Total documents in collection: {count}")
        
        # 5. Test similarity search
        print("\\n4. Testing similarity search...")
        search_results = store.search(
            "documents", 
            "programming languages and coding", 
            n_results=3
        )
        
        print(f"   Found {len(search_results)} similar documents:")
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. Score: {result.score:.3f}")
            print(f"      Content: {result.document.content}")
            print(f"      Category: {result.document.metadata.get('category', 'N/A')}")
            print()
        
        # 6. Test metadata filtering
        print("5. Testing search with metadata filtering...")
        ai_results = store.search(
            "documents",
            "artificial intelligence",
            where={"category": "ai"},
            n_results=5
        )
        
        print(f"   Found {len(ai_results)} AI-related documents:")
        for result in ai_results:
            print(f"   - {result.document.content} (score: {result.score:.3f})")
        
        print("\\n✅ Integrated vector store example completed successfully!")
        print("\\n🎯 Key Features Demonstrated:")
        print("   • Automatic embedding generation using sentence-transformers")
        print("   • Batch processing for efficient embedding generation")
        print("   • Embedding caching for improved performance")
        print("   • Semantic similarity search")
        print("   • Metadata filtering")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()