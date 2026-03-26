"""
Example usage of the complete RAG system for semantic memory.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from day_53.rag_system import create_rag_system, MockResponseGenerator, OpenAIResponseGenerator
from day_50.fact_store import FactStore, FactType
from day_50.knowledge_graph import KnowledgeGraph, NodeType, EdgeType


def example_basic_rag_system():
    """Example of basic RAG system setup and usage."""
    print("=== Basic RAG System Example ===\n")
    
    try:
        # Create RAG system with default settings
        rag_system = create_rag_system(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_path="./example_rag_db",
            collection_name="basic_knowledge",
            response_generator_type="mock"
        )
        
        print("✓ RAG system created successfully")
        
        # Index some documents
        documents = [
            {
                "id": "python_def",
                "content": "Python is a high-level, interpreted programming language with dynamic semantics",
                "metadata": {"type": "definition", "topic": "programming"}
            },
            {
                "id": "python_features",
                "content": "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming",
                "metadata": {"type": "features", "topic": "programming"}
            },
            {
                "id": "python_applications",
                "content": "Python is widely used for web development, data science, artificial intelligence, and automation",
                "metadata": {"type": "applications", "topic": "programming"}
            },
            {
                "id": "ml_definition",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed",
                "metadata": {"type": "definition", "topic": "ai"}
            }
        ]
        
        print("Indexing documents...")
        for doc in documents:
            rag_system.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                source_type="document"
            )
        
        print(f"✓ Indexed {len(documents)} documents")
        
        # Execute queries
        queries = [
            "What is Python?",
            "What can Python be used for?",
            "How does Python support different programming paradigms?",
            "What is machine learning?"
        ]
        
        print("\nExecuting queries:")
        for query in queries:
            print(f"\n--- Query: {query} ---")
            
            result = rag_system.query(query)
            
            if result.success:
                print(f"✓ Query successful")
                print(f"Retrieved {len(result.context.retrieved_documents)} documents")
                print(f"Response: {result.response[:200]}...")
                print(f"Total time: {result.total_time:.3f}s")
                
                # Show retrieved documents
                print("Retrieved documents:")
                for i, doc in enumerate(result.context.retrieved_documents, 1):
                    print(f"  {i}. [{doc.source_type.upper()}] {doc.content[:100]}... (score: {doc.similarity_score:.3f})")
            else:
                print(f"✗ Query failed: {result.error_message}")
        
        # Show statistics
        print("\n--- RAG System Statistics ---")
        stats = rag_system.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            elif isinstance(value, dict):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"✗ Basic RAG example failed: {e}")
        import traceback
        traceback.print_exc()


def example_knowledge_base_integration():
    """Example integrating RAG with fact store and knowledge graph."""
    print("\n=== Knowledge Base Integration Example ===\n")
    
    try:
        # Create knowledge bases
        fact_store = FactStore()
        knowledge_graph = KnowledgeGraph()
        
        # Populate fact store
        print("Populating fact store...")
        facts = [
            ("Python", "is_a", "programming language", FactType.DEFINITION, 0.95),
            ("Python", "created_by", "Guido van Rossum", FactType.ASSERTION, 0.90),
            ("Python", "first_released", "1991", FactType.ASSERTION, 0.95),
            ("Python", "supports", "object-oriented programming", FactType.ATTRIBUTE, 0.90),
            ("Django", "is_a", "web framework", FactType.DEFINITION, 0.95),
            ("Django", "written_in", "Python", FactType.ASSERTION, 0.95),
            ("NumPy", "is_a", "library", FactType.DEFINITION, 0.90),
            ("NumPy", "used_for", "numerical computing", FactType.ATTRIBUTE, 0.90)
        ]
        
        stored_facts = []
        for subject, predicate, obj, fact_type, confidence in facts:
            fact = fact_store.store(subject, predicate, obj, fact_type=fact_type, confidence=confidence)
            stored_facts.append(fact)
        
        print(f"✓ Stored {len(stored_facts)} facts")
        
        # Populate knowledge graph
        print("Populating knowledge graph...")
        
        # Add nodes
        python_node = knowledge_graph.add_node("Python", NodeType.ENTITY, properties={"type": "language", "paradigm": "multi"})
        guido_node = knowledge_graph.add_node("Guido van Rossum", NodeType.ENTITY, properties={"type": "person", "role": "creator"})
        django_node = knowledge_graph.add_node("Django", NodeType.ENTITY, properties={"type": "framework", "domain": "web"})
        numpy_node = knowledge_graph.add_node("NumPy", NodeType.ENTITY, properties={"type": "library", "domain": "numerical"})
        programming_node = knowledge_graph.add_node("Programming Language", NodeType.CONCEPT)
        web_framework_node = knowledge_graph.add_node("Web Framework", NodeType.CONCEPT)
        
        # Add edges
        knowledge_graph.add_edge(python_node.id, programming_node.id, EdgeType.IS_A)
        knowledge_graph.add_edge(python_node.id, guido_node.id, EdgeType.RELATED_TO, properties={"relationship": "created_by"})
        knowledge_graph.add_edge(django_node.id, web_framework_node.id, EdgeType.IS_A)
        knowledge_graph.add_edge(django_node.id, python_node.id, EdgeType.DEPENDS_ON)
        knowledge_graph.add_edge(numpy_node.id, python_node.id, EdgeType.DEPENDS_ON)
        
        print(f"✓ Created knowledge graph with {len(knowledge_graph.get_all_nodes())} nodes")
        
        # Create RAG system with knowledge bases
        rag_system = create_rag_system(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_path="./knowledge_rag_db",
            collection_name="semantic_knowledge",
            response_generator_type="mock",
            fact_store=fact_store,
            knowledge_graph=knowledge_graph,
            max_retrieved_docs=7,
            min_similarity_score=0.1
        )
        
        print("✓ RAG system created with knowledge bases")
        
        # Index the knowledge base
        print("Indexing knowledge base...")
        indexed = rag_system.index_knowledge_base()
        print(f"✓ Indexed {indexed['facts']} facts, {indexed['nodes']} nodes")
        
        # Add some additional contextual documents
        additional_docs = [
            {
                "id": "python_history",
                "content": "Python was conceived in the late 1980s and its implementation began in December 1989 by Guido van Rossum",
                "metadata": {"type": "history", "topic": "python"}
            },
            {
                "id": "python_philosophy",
                "content": "The Zen of Python emphasizes code readability and simplicity with the principle that there should be one obvious way to do it",
                "metadata": {"type": "philosophy", "topic": "python"}
            }
        ]
        
        for doc in additional_docs:
            rag_system.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                source_type="document"
            )
        
        print(f"✓ Added {len(additional_docs)} additional documents")
        
        # Execute complex queries
        complex_queries = [
            "Who created Python and when?",
            "What is the relationship between Python and Django?",
            "What libraries are available for Python?",
            "What is Python's design philosophy?",
            "How does Python support different programming paradigms?"
        ]
        
        print("\nExecuting complex queries:")
        for query in complex_queries:
            print(f"\n--- Query: {query} ---")
            
            result = rag_system.query(query)
            
            if result.success:
                print(f"✓ Retrieved {len(result.context.retrieved_documents)} relevant items")
                
                # Categorize retrieved documents by source type
                by_source = {}
                for doc in result.context.retrieved_documents:
                    source_type = doc.source_type
                    if source_type not in by_source:
                        by_source[source_type] = []
                    by_source[source_type].append(doc)
                
                for source_type, docs in by_source.items():
                    print(f"  {source_type.upper()}: {len(docs)} items")
                    for doc in docs[:2]:  # Show top 2 per source
                        print(f"    - {doc.content[:80]}... (score: {doc.similarity_score:.3f})")
                
                print(f"Response preview: {result.response[:150]}...")
            else:
                print(f"✗ Query failed: {result.error_message}")
        
        # Show final statistics
        print("\n--- Final Statistics ---")
        stats = rag_system.get_stats()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average retrieved docs: {stats['average_retrieved_docs']:.1f}")
        print(f"Average retrieval time: {stats['avg_retrieval_time']:.3f}s")
        
    except Exception as e:
        print(f"✗ Knowledge base integration example failed: {e}")
        import traceback
        traceback.print_exc()


def example_custom_response_generator():
    """Example with custom response generator and templates."""
    print("\n=== Custom Response Generator Example ===\n")
    
    try:
        # Create custom response generator with specific template
        custom_template = """Based on the retrieved knowledge:

{context}

Question: {query}

Answer: I can provide the following information based on the knowledge base:

"""
        
        custom_generator = MockResponseGenerator(response_template=custom_template + "This is a detailed response incorporating the retrieved context.")
        
        # Create RAG system with custom generator
        rag_system = create_rag_system(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_path="./custom_rag_db",
            collection_name="custom_knowledge",
            response_generator_type="mock"
        )
        
        # Replace with custom generator
        rag_system.response_generator = custom_generator
        
        # Custom context template
        rag_system.context_template = """Retrieved Knowledge:
{context}

User Question: {query}

Please provide a comprehensive answer based on the retrieved knowledge above."""
        
        print("✓ RAG system created with custom response generator")
        
        # Index some technical documents
        tech_docs = [
            {
                "id": "ai_overview",
                "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn",
                "metadata": {"type": "definition", "field": "ai"}
            },
            {
                "id": "ml_types",
                "content": "Machine learning includes supervised learning, unsupervised learning, and reinforcement learning approaches",
                "metadata": {"type": "classification", "field": "ml"}
            },
            {
                "id": "deep_learning",
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data",
                "metadata": {"type": "technique", "field": "dl"}
            }
        ]
        
        for doc in tech_docs:
            rag_system.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                source_type="document"
            )
        
        print(f"✓ Indexed {len(tech_docs)} technical documents")
        
        # Test queries with custom formatting
        queries = [
            "What is artificial intelligence?",
            "What are the types of machine learning?",
            "How does deep learning work?"
        ]
        
        print("\nTesting custom response formatting:")
        for query in queries:
            print(f"\n--- Query: {query} ---")
            
            result = rag_system.query(query)
            
            if result.success:
                print("Custom formatted response:")
                print(result.response)
                print(f"\nMetadata:")
                print(f"  - Retrieved docs: {len(result.context.retrieved_documents)}")
                print(f"  - Context tokens: {result.context.total_tokens}")
                print(f"  - Total time: {result.total_time:.3f}s")
            else:
                print(f"✗ Query failed: {result.error_message}")
        
    except Exception as e:
        print(f"✗ Custom response generator example failed: {e}")
        import traceback
        traceback.print_exc()


def example_openai_integration():
    """Example with OpenAI response generator (requires API key)."""
    print("\n=== OpenAI Integration Example ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Skipping OpenAI example - set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Create RAG system with OpenAI response generator
        rag_system = create_rag_system(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_path="./openai_rag_db",
            collection_name="openai_knowledge",
            response_generator_type="openai",
            openai_api_key=api_key,
            openai_model="gpt-3.5-turbo",
            max_tokens=300
        )
        
        print("✓ RAG system created with OpenAI response generator")
        
        # Index some documents
        documents = [
            {
                "id": "climate_change",
                "content": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities",
                "metadata": {"type": "definition", "topic": "environment"}
            },
            {
                "id": "renewable_energy",
                "content": "Renewable energy sources include solar, wind, hydroelectric, and geothermal power that can be replenished naturally",
                "metadata": {"type": "classification", "topic": "energy"}
            },
            {
                "id": "carbon_footprint",
                "content": "A carbon footprint measures the total greenhouse gas emissions caused by an individual, organization, or activity",
                "metadata": {"type": "definition", "topic": "environment"}
            }
        ]
        
        for doc in documents:
            rag_system.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                source_type="document"
            )
        
        print(f"✓ Indexed {len(documents)} documents")
        
        # Test with OpenAI generation
        query = "What is climate change and how can renewable energy help?"
        print(f"\n--- Query: {query} ---")
        
        result = rag_system.query(query)
        
        if result.success:
            print("✓ OpenAI response generated successfully")
            print(f"Response: {result.response}")
            print(f"\nMetadata:")
            print(f"  - Retrieved docs: {len(result.context.retrieved_documents)}")
            print(f"  - Generation time: {result.generation_time:.3f}s")
            print(f"  - Total time: {result.total_time:.3f}s")
        else:
            print(f"✗ Query failed: {result.error_message}")
        
    except Exception as e:
        print(f"✗ OpenAI integration example failed: {e}")
        import traceback
        traceback.print_exc()


def example_performance_analysis():
    """Example analyzing RAG system performance."""
    print("\n=== Performance Analysis Example ===\n")
    
    try:
        # Create RAG system
        rag_system = create_rag_system(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_path="./perf_rag_db",
            collection_name="performance_test",
            response_generator_type="mock",
            max_retrieved_docs=10
        )
        
        print("✓ RAG system created for performance testing")
        
        # Index a larger set of documents
        print("Indexing documents for performance testing...")
        
        topics = ["technology", "science", "history", "literature", "mathematics"]
        documents = []
        
        for i in range(50):  # Create 50 documents
            topic = topics[i % len(topics)]
            doc = {
                "id": f"doc_{i:03d}",
                "content": f"This is document {i} about {topic}. It contains detailed information and explanations about various aspects of {topic}.",
                "metadata": {"type": "article", "topic": topic, "doc_num": i}
            }
            documents.append(doc)
            
            rag_system.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                source_type="document"
            )
        
        print(f"✓ Indexed {len(documents)} documents")
        
        # Test queries with different patterns
        test_queries = [
            "What is technology?",
            "Tell me about science",
            "Explain history",
            "What is literature?",
            "How does mathematics work?",
            "What are the relationships between technology and science?",
            "Compare history and literature",
            "What topics are covered in the knowledge base?"
        ]
        
        print("\nExecuting performance test queries...")
        
        results = []
        for i, query in enumerate(test_queries):
            print(f"Query {i+1}/{len(test_queries)}: {query}")
            
            result = rag_system.query(query)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Success - {len(result.context.retrieved_documents)} docs, {result.total_time:.3f}s")
            else:
                print(f"  ✗ Failed - {result.error_message}")
        
        # Analyze performance
        print("\n--- Performance Analysis ---")
        
        successful_results = [r for r in results if r.success]
        if successful_results:
            avg_time = sum(r.total_time for r in successful_results) / len(successful_results)
            avg_docs = sum(len(r.context.retrieved_documents) for r in successful_results) / len(successful_results)
            avg_similarity = sum(
                sum(doc.similarity_score for doc in r.context.retrieved_documents) / len(r.context.retrieved_documents)
                for r in successful_results if r.context.retrieved_documents
            ) / len([r for r in successful_results if r.context.retrieved_documents])
            
            print(f"Successful queries: {len(successful_results)}/{len(results)}")
            print(f"Average response time: {avg_time:.3f}s")
            print(f"Average documents retrieved: {avg_docs:.1f}")
            print(f"Average similarity score: {avg_similarity:.3f}")
        
        # Show system statistics
        print("\n--- System Statistics ---")
        stats = rag_system.get_stats()
        for key, value in stats.items():
            if key.endswith('_time') and isinstance(value, float):
                print(f"{key}: {value:.3f}s")
            elif isinstance(value, float):
                print(f"{key}: {value:.3f}")
            elif not isinstance(value, dict):
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"✗ Performance analysis example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("RAG System Examples\n")
    print("This script demonstrates various RAG system configurations and use cases.")
    print("=" * 80)
    
    # Run examples
    example_basic_rag_system()
    example_knowledge_base_integration()
    example_custom_response_generator()
    example_openai_integration()
    example_performance_analysis()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("\nTo use the RAG system in your own code:")
    print("1. Install dependencies: pip install sentence-transformers")
    print("2. For OpenAI: pip install openai and set OPENAI_API_KEY")
    print("3. Import and use create_rag_system() function")
    print("4. Index your knowledge base and start querying!")