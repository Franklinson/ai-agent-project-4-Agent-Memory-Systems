"""
Usage example for embedding_generator.py
"""

import tempfile
import shutil
from day_53.embedding_generator import (
    create_embedding_generator, TextPreprocessor, EmbeddingCache,
    SentenceTransformerProvider, EmbeddingGenerator
)


def main():
    """Demonstrate embedding generator usage."""
    print("=== Embedding Generator Usage Example ===\n")
    
    # Create temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Create embedding generator with default settings
        print("1. Creating embedding generator...")
        generator = create_embedding_generator(
            provider_type="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            cache_dir=temp_dir
        )
        print(f"   Model: {generator.provider.get_model_name()}")
        print(f"   Dimension: {generator.provider.get_dimension()}")
        
        # 2. Generate single embedding
        print("\n2. Generating single embedding...")
        text = "Python is a powerful programming language"
        result = generator.generate(text)
        
        print(f"   Original text: '{text}'")
        print(f"   Processed text: '{result.text}'")
        print(f"   Embedding dimension: {result.dimension}")
        print(f"   Processing time: {result.processing_time:.4f}s")
        print(f"   Cached: {result.cached}")
        print(f"   First 5 values: {result.embedding[:5]}")
        
        # 3. Generate same embedding again (should be cached)
        print("\n3. Generating same embedding again...")
        result2 = generator.generate(text)
        print(f"   Processing time: {result2.processing_time:.4f}s")
        print(f"   Cached: {result2.cached}")
        
        # 4. Generate batch embeddings
        print("\n4. Generating batch embeddings...")
        texts = [
            "Machine learning is fascinating",
            "Natural language processing",
            "Deep learning with neural networks",
            "Computer vision applications",
            "Artificial intelligence research"
        ]
        
        batch_result = generator.generate_batch(texts, batch_size=3)
        print(f"   Processed {len(batch_result.texts)} texts")
        print(f"   Total processing time: {batch_result.total_processing_time:.4f}s")
        print(f"   Cache hits: {batch_result.cache_hits}")
        print(f"   Cache misses: {batch_result.cache_misses}")
        print(f"   Model: {batch_result.model}")
        
        # 5. Test text preprocessing
        print("\n5. Testing text preprocessing...")
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_extra_whitespace=True,
            remove_special_chars=True,
            max_length=50
        )
        
        messy_text = "  Hello @#$% WORLD!!!   This is a VERY long text that should be truncated  "
        clean_text = preprocessor.preprocess(messy_text)
        print(f"   Original: '{messy_text}'")
        print(f"   Cleaned: '{clean_text}'")
        
        # 6. Custom embedding generator
        print("\n6. Creating custom embedding generator...")
        custom_preprocessor = TextPreprocessor(
            lowercase=False,
            remove_special_chars=False,
            max_length=100
        )
        custom_cache = EmbeddingCache(cache_dir=None, max_cache_size=1000)
        
        custom_generator = EmbeddingGenerator(
            provider=SentenceTransformerProvider("all-MiniLM-L6-v2"),
            preprocessor=custom_preprocessor,
            cache=custom_cache,
            batch_size=16
        )
        
        custom_result = custom_generator.generate("Hello World!")
        print(f"   Custom processed text: '{custom_result.text}'")
        print(f"   Maintains case: {custom_result.text != custom_result.text.lower()}")
        
        # 7. Cache statistics
        print("\n7. Cache statistics...")
        stats = generator.get_cache_stats()
        print(f"   Memory cache size: {stats['memory_cache_size']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Hit rate: {stats['hit_rate']:.2%}")
        print(f"   Cache directory: {stats['cache_dir']}")
        
        # 8. Similarity demonstration
        print("\n8. Demonstrating semantic similarity...")
        similar_texts = [
            "Python programming language",
            "Python is a programming language",
            "Cooking recipes with ingredients",
            "Machine learning algorithms"
        ]
        
        similar_results = generator.generate_batch(similar_texts)
        
        # Calculate simple cosine similarity between first two (should be high)
        emb1 = similar_results.embeddings[0]
        emb2 = similar_results.embeddings[1]
        emb3 = similar_results.embeddings[2]
        
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)
        
        print(f"   Similarity between '{similar_texts[0]}' and '{similar_texts[1]}': {sim_1_2:.3f}")
        print(f"   Similarity between '{similar_texts[0]}' and '{similar_texts[2]}': {sim_1_3:.3f}")
        print(f"   Related texts are more similar: {sim_1_2 > sim_1_3}")
        
        print("\n✅ Embedding generator example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()