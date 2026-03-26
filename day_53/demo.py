#!/usr/bin/env python3
"""
Simple demonstration of the embedding generator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from day_53.embedding_generator import create_embedding_generator, EmbeddingError


def main():
    """Demonstrate basic embedding generation functionality."""
    print("=== Embedding Generator Demo ===\n")
    
    try:
        # Create generator with sentence transformers (fallback to mock if not available)
        print("Creating embedding generator...")
        try:
            generator = create_embedding_generator(
                provider_type="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
                cache_dir="./demo_cache"
            )
            print("✓ Using Sentence Transformers provider")
        except Exception as e:
            print(f"⚠ Sentence Transformers not available: {e}")
            print("Using mock provider for demonstration...")
            
            # Create a simple mock provider for demo
            from day_53.embedding_generator import EmbeddingGenerator, TextPreprocessor, EmbeddingCache
            
            class DemoProvider:
                def __init__(self):
                    self.model_name = "demo-model"
                    self.dimension = 384
                
                def get_embedding(self, text):
                    # Simple hash-based embedding for demo
                    import hashlib
                    hash_obj = hashlib.md5(text.encode())
                    hash_hex = hash_obj.hexdigest()
                    # Convert to float values
                    return [int(hash_hex[i:i+2], 16) / 255.0 for i in range(0, min(len(hash_hex), self.dimension*2), 2)][:self.dimension]
                
                def get_embeddings(self, texts):
                    return [self.get_embedding(text) for text in texts]
                
                def get_dimension(self):
                    return self.dimension
                
                def get_model_name(self):
                    return self.model_name
            
            generator = EmbeddingGenerator(
                DemoProvider(),
                TextPreprocessor(),
                EmbeddingCache(cache_dir="./demo_cache")
            )
        
        # Test single embedding
        print("\n1. Single Text Embedding:")
        text = "Artificial intelligence is transforming the world"
        result = generator.generate(text)
        
        print(f"   Original: '{text}'")
        print(f"   Processed: '{result.text}'")
        print(f"   Model: {result.model}")
        print(f"   Dimension: {result.dimension}")
        print(f"   Processing time: {result.processing_time:.4f}s")
        print(f"   Cached: {result.cached}")
        print(f"   Embedding preview: {result.embedding[:5]}...")
        
        # Test caching
        print("\n2. Testing Cache (same text):")
        result2 = generator.generate(text)
        print(f"   Processing time: {result2.processing_time:.4f}s")
        print(f"   Cached: {result2.cached}")
        
        # Test batch processing
        print("\n3. Batch Processing:")
        texts = [
            "Machine learning algorithms",
            "Natural language processing",
            "Computer vision systems",
            "Deep neural networks"
        ]
        
        batch_result = generator.generate_batch(texts, batch_size=2)
        print(f"   Processed {len(batch_result.texts)} texts")
        print(f"   Total time: {batch_result.total_processing_time:.4f}s")
        print(f"   Cache hits: {batch_result.cache_hits}")
        print(f"   Cache misses: {batch_result.cache_misses}")
        print(f"   Average time per text: {batch_result.total_processing_time/len(texts):.4f}s")
        
        # Test preprocessing
        print("\n4. Text Preprocessing:")
        messy_text = "  HELLO @#$% World!!!  \n\t  "
        result3 = generator.generate(messy_text)
        print(f"   Original: {repr(messy_text)}")
        print(f"   Processed: {repr(result3.text)}")
        
        # Cache statistics
        print("\n5. Cache Statistics:")
        stats = generator.get_cache_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()