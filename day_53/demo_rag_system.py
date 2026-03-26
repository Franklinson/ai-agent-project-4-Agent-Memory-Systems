#!/usr/bin/env python3
"""
Simple demonstration of the complete RAG system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from day_53.rag_system import create_rag_system
from day_50.fact_store import FactStore, FactType
from day_50.knowledge_graph import KnowledgeGraph, NodeType, EdgeType


def main():
    """Demonstrate complete RAG system functionality."""
    print("=== RAG System Demo ===\n")
    
    try:
        # Create RAG system
        print("Creating RAG system...")
        rag_system = create_rag_system(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_path="./demo_rag_db",
            collection_name="demo_knowledge",
            response_generator_type="mock",
            max_retrieved_docs=5
        )
        print("✓ RAG system created")
        
        # Create and populate knowledge bases
        print("\nSetting up knowledge bases...")
        
        # Fact store
        fact_store = FactStore()
        facts_data = [
            ("Python", "is_a", "programming language", FactType.DEFINITION),
            ("Python", "created_by", "Guido van Rossum", FactType.ASSERTION),
            ("Python", "supports", "multiple paradigms", FactType.ATTRIBUTE),
            ("Django", "is_a", "web framework", FactType.DEFINITION),
            ("Django", "written_in", "Python", FactType.ASSERTION)
        ]
        
        for subject, predicate, obj, fact_type in facts_data:
            fact_store.store(subject, predicate, obj, fact_type=fact_type)
        
        # Knowledge graph
        kg = KnowledgeGraph()
        python_node = kg.add_node("Python", NodeType.ENTITY)
        django_node = kg.add_node("Django", NodeType.ENTITY)
        web_framework_node = kg.add_node("Web Framework", NodeType.CONCEPT)
        
        kg.add_edge(django_node.id, web_framework_node.id, EdgeType.IS_A)
        kg.add_edge(django_node.id, python_node.id, EdgeType.DEPENDS_ON)
        
        # Connect knowledge bases to RAG system
        rag_system.fact_store = fact_store
        rag_system.knowledge_graph = kg
        
        print("✓ Knowledge bases created")
        
        # Index knowledge base
        print("\nIndexing knowledge base...")
        indexed = rag_system.index_knowledge_base()
        print(f"✓ Indexed {indexed['facts']} facts and {indexed['nodes']} nodes")
        
        # Add some additional documents
        print("\nAdding additional documents...")
        documents = [
            {
                "id": "python_intro",
                "content": "Python is a high-level programming language known for its simplicity and readability",
                "metadata": {"type": "introduction"}
            },
            {
                "id": "python_uses",
                "content": "Python is widely used in web development, data science, artificial intelligence, and automation",
                "metadata": {"type": "applications"}
            },
            {
                "id": "django_intro",
                "content": "Django is a high-level Python web framework that encourages rapid development and clean design",
                "metadata": {"type": "introduction"}
            }
        ]
        
        for doc in documents:
            rag_system.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                source_type="document"
            )
        
        print(f"✓ Added {len(documents)} additional documents")
        
        # Execute demo queries
        print("\nExecuting demo queries:")
        queries = [
            "What is Python?",
            "Who created Python?",
            "What is Django?",
            "What is Python used for?",
            "How are Python and Django related?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            
            result = rag_system.query(query)
            
            if result.success:
                print(f"   ✓ Success ({result.total_time:.3f}s)")
                print(f"   Retrieved {len(result.context.retrieved_documents)} documents:")
                
                for j, doc in enumerate(result.context.retrieved_documents, 1):
                    print(f"     {j}. [{doc.source_type.upper()}] {doc.content[:60]}... (score: {doc.similarity_score:.3f})")
                
                print(f"   Response: {result.response[:120]}...")
            else:
                print(f"   ✗ Failed: {result.error_message}")
        
        # Show system statistics
        print("\n=== System Statistics ===")
        stats = rag_system.get_stats()
        
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful queries: {stats['successful_queries']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average retrieved docs: {stats['average_retrieved_docs']:.1f}")
        print(f"Average retrieval time: {stats['avg_retrieval_time']:.3f}s")
        print(f"Average generation time: {stats['avg_generation_time']:.3f}s")
        
        # Show embedding cache stats
        cache_stats = stats.get('embedding_cache_stats', {})
        if cache_stats:
            print(f"\nEmbedding cache:")
            print(f"  Hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            print(f"  Cache size: {cache_stats.get('memory_cache_size', 0)}")
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()