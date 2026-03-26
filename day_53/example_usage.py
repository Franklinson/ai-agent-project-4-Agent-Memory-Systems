"""
Example usage of the embedding generator for semantic memory.
"""

import os
from day_53.embedding_generator import (
    create_embedding_generator, EmbeddingGenerator, TextPreprocessor, EmbeddingCache,
    SentenceTransformerProvider, OpenAIProvider
)


def example_sentence_transformers():
    """Example using Sentence Transformers (local model)."""
    print("=== Sentence Transformers Example ===")
    
    try:
        # Create generator with sentence transformers
        generator = create_embedding_generator(
            provider_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",  # Small, fast model
            cache_dir="./embeddings_cache",
            lowercase=True,
            remove_special_chars=False,
            max_length=512
        )
        
        # Single text embedding
        print("Generating single embedding...")
        result = generator.generate("Python is a programming language")
        print(f"Text: {result.text}")
        print(f"Model: {result.model}")
        print(f"Dimension: {result.dimension}")
        print(f"Processing time: {result.processing_time:.4f}s")
        print(f"Cached: {result.cached}")
        print(f"Embedding preview: {result.embedding[:5]}...")
        
        # Generate same text again (should be cached)
        print("\nGenerating same text again (cached)...")
        result2 = generator.generate("Python is a programming language")
        print(f"Cached: {result2.cached}")
        print(f"Processing time: {result2.processing_time:.4f}s")
        
        # Batch processing
        print("\nBatch processing...")
        texts = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing handles text",
            "Computer vision processes images"
        ]
        
        batch_result = generator.generate_batch(texts, batch_size=2)
        print(f"Processed {len(batch_result.texts)} texts")
        print(f"Total processing time: {batch_result.total_processing_time:.4f}s")
        print(f"Cache hits: {batch_result.cache_hits}")
        print(f"Cache misses: {batch_result.cache_misses}")
        
        # Cache statistics
        print("\nCache statistics:")
        stats = generator.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install sentence-transformers")


def example_openai():
    """Example using OpenAI embeddings (requires API key)."""
    print("\n=== OpenAI Example ===")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI example - set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Create generator with OpenAI
        generator = create_embedding_generator(
            provider_type="openai",
            model_name="text-embedding-3-small",
            api_key=api_key,
            cache_dir="./embeddings_cache"
        )
        
        # Generate embedding
        print("Generating OpenAI embedding...")
        result = generator.generate("Artificial intelligence transforms industries")
        print(f"Text: {result.text}")
        print(f"Model: {result.model}")
        print(f"Dimension: {result.dimension}")
        print(f"Processing time: {result.processing_time:.4f}s")
        print(f"Embedding preview: {result.embedding[:5]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install openai")


def example_custom_configuration():
    """Example with custom preprocessing and caching configuration."""
    print("\n=== Custom Configuration Example ===")
    
    try:
        # Custom preprocessor
        preprocessor = TextPreprocessor(
            lowercase=False,  # Keep original case
            remove_special_chars=True,  # Remove special characters
            min_length=3,
            max_length=256
        )
        
        # Custom cache with smaller size
        cache = EmbeddingCache(
            cache_dir="./custom_cache",
            max_cache_size=50
        )
        
        # Create provider
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        
        # Create generator
        generator = EmbeddingGenerator(provider, preprocessor, cache)
        
        # Test preprocessing
        texts = [
            "  Hello World!  ",
            "PYTHON @#$% Programming",
            "AI & Machine Learning",
            "hi"  # Too short
        ]
        
        for text in texts:
            try:
                result = generator.generate(text)
                print(f"Original: '{text}' -> Processed: '{result.text}'")
            except Exception as e:
                print(f"Error processing '{text}': {e}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_semantic_memory_integration():
    """Example integrating with semantic memory components."""
    print("\n=== Semantic Memory Integration Example ===")
    
    try:
        # Create generator
        generator = create_embedding_generator(
            provider_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            cache_dir="./semantic_cache"
        )
        
        # Simulate semantic memory content
        facts = [
            "Python is a high-level programming language",
            "Machine learning algorithms learn from data",
            "Neural networks are inspired by biological neurons",
            "Natural language processing enables computers to understand text"
        ]
        
        knowledge_concepts = [
            "Programming Language",
            "Machine Learning",
            "Artificial Intelligence",
            "Data Science",
            "Neural Network"
        ]
        
        # Generate embeddings for facts
        print("Generating embeddings for facts...")
        fact_embeddings = []
        for fact in facts:
            result = generator.generate(fact)
            fact_embeddings.append({
                "text": fact,
                "embedding": result.embedding,
                "dimension": result.dimension
            })
        
        # Generate embeddings for concepts
        print("Generating embeddings for concepts...")
        concept_embeddings = []
        for concept in knowledge_concepts:
            result = generator.generate(concept)
            concept_embeddings.append({
                "text": concept,
                "embedding": result.embedding,
                "dimension": result.dimension
            })
        
        print(f"Generated {len(fact_embeddings)} fact embeddings")
        print(f"Generated {len(concept_embeddings)} concept embeddings")
        
        # Demonstrate similarity (simple cosine similarity)
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        # Find most similar concept for each fact
        print("\nFinding similar concepts for facts:")
        for fact_emb in fact_embeddings:
            similarities = []
            for concept_emb in concept_embeddings:
                sim = cosine_similarity(fact_emb["embedding"], concept_emb["embedding"])
                similarities.append((concept_emb["text"], sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_match = similarities[0]
            
            print(f"Fact: '{fact_emb['text'][:50]}...'")
            print(f"  Most similar concept: '{best_match[0]}' (similarity: {best_match[1]:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")


def example_performance_comparison():
    """Example comparing different models and configurations."""
    print("\n=== Performance Comparison Example ===")
    
    try:
        import time
        
        # Test different models
        models = [
            ("all-MiniLM-L6-v2", "Small, fast model"),
            ("all-mpnet-base-v2", "Better quality, slower")
        ]
        
        test_texts = [
            "Artificial intelligence and machine learning",
            "Natural language processing and text analysis",
            "Computer vision and image recognition",
            "Deep learning and neural networks"
        ]
        
        for model_name, description in models:
            try:
                print(f"\nTesting {model_name} ({description}):")
                
                generator = create_embedding_generator(
                    provider_type="sentence-transformers",
                    model_name=model_name,
                    cache_dir=f"./cache_{model_name.replace('-', '_')}"
                )
                
                # Time single generation
                start_time = time.time()
                result = generator.generate(test_texts[0])
                single_time = time.time() - start_time
                
                print(f"  Single embedding: {single_time:.4f}s")
                print(f"  Dimension: {result.dimension}")
                
                # Time batch generation
                start_time = time.time()
                batch_result = generator.generate_batch(test_texts)
                batch_time = time.time() - start_time
                
                print(f"  Batch embedding: {batch_time:.4f}s")
                print(f"  Average per text: {batch_time/len(test_texts):.4f}s")
                
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run examples
    example_sentence_transformers()
    example_openai()
    example_custom_configuration()
    example_semantic_memory_integration()
    example_performance_comparison()
    
    print("\n=== Examples Complete ===")
    print("To use with your own data:")
    print("1. Install dependencies: pip install sentence-transformers")
    print("2. For OpenAI: pip install openai and set OPENAI_API_KEY")
    print("3. Import and use create_embedding_generator() in your code")