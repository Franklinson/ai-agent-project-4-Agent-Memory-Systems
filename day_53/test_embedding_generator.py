"""
Tests for embedding_generator.py
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from day_53.embedding_generator import (
    EmbeddingGenerator, TextPreprocessor, EmbeddingCache, EmbeddingResult,
    BatchEmbeddingResult, SentenceTransformerProvider, OpenAIProvider, create_embedding_generator,
    EmbeddingError, PreprocessingError, ModelNotAvailableError
)


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def __init__(self, model_name: str = "mock-model", dimension: int = 5):
        self.model_name = model_name
        self.dimension = dimension
    
    def get_embedding(self, text: str) -> list:
        # Simple hash-based embedding for testing
        hash_val = hash(text)
        return [(hash_val >> i) % 100 / 100.0 for i in range(self.dimension)]
    
    def get_embeddings(self, texts: list) -> list:
        return [self.get_embedding(text) for text in texts]
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def get_model_name(self) -> str:
        return self.model_name


class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""
    
    def test_default_preprocessing(self):
        """Test default preprocessing settings."""
        preprocessor = TextPreprocessor()
        
        text = "  Hello WORLD!  \n\t  "
        result = preprocessor.preprocess(text)
        
        assert result == "hello world!"
    
    def test_lowercase_disabled(self):
        """Test preprocessing with lowercase disabled."""
        preprocessor = TextPreprocessor(lowercase=False)
        
        text = "Hello WORLD!"
        result = preprocessor.preprocess(text)
        
        assert result == "Hello WORLD!"
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        preprocessor = TextPreprocessor(remove_special_chars=True)
        
        text = "Hello @#$% World! 123"
        result = preprocessor.preprocess(text)
        
        # Special chars are removed but extra spaces are normalized to single spaces
        assert result == "hello world! 123"
    
    def test_length_limits(self):
        """Test minimum and maximum length limits."""
        preprocessor = TextPreprocessor(min_length=5, max_length=10)
        
        # Test minimum length violation
        with pytest.raises(PreprocessingError):
            preprocessor.preprocess("hi")
        
        # Test maximum length truncation
        result = preprocessor.preprocess("this is a very long text")
        assert len(result) <= 10
    
    def test_control_character_removal(self):
        """Test removal of control characters."""
        preprocessor = TextPreprocessor()
        
        text = "Hello\x00\x08World\x7f"
        result = preprocessor.preprocess(text)
        
        assert result == "helloworld"
    
    def test_batch_preprocessing(self):
        """Test batch text preprocessing."""
        preprocessor = TextPreprocessor()
        
        texts = ["  Hello  ", "WORLD!", "  Test  "]
        results = preprocessor.preprocess_batch(texts)
        
        expected = ["hello", "world!", "test"]
        assert results == expected
    
    def test_non_string_input(self):
        """Test preprocessing of non-string input."""
        preprocessor = TextPreprocessor()
        
        result = preprocessor.preprocess(123)
        assert result == "123"


class TestEmbeddingCache:
    """Test cases for EmbeddingCache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EmbeddingCache(cache_dir=self.temp_dir, max_cache_size=3)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_cache(self):
        """Test in-memory caching."""
        cache = EmbeddingCache(cache_dir=None)  # Memory only
        
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
            processing_time=0.1
        )
        
        # Cache miss
        assert cache.get("test", "test-model") is None
        
        # Cache put
        cache.put("test", "test-model", result)
        
        # Cache hit
        cached = cache.get("test", "test-model")
        assert cached is not None
        assert cached.text == "test"
        assert cached.cached is True
    
    def test_disk_cache(self):
        """Test disk-based caching."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
            processing_time=0.1
        )
        
        # Cache put
        self.cache.put("test", "test-model", result)
        
        # Create new cache instance (simulates restart)
        new_cache = EmbeddingCache(cache_dir=self.temp_dir)
        
        # Should load from disk
        cached = new_cache.get("test", "test-model")
        assert cached is not None
        assert cached.text == "test"
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache = EmbeddingCache(cache_dir=None, max_cache_size=2)
        
        # Add items up to limit
        for i in range(3):
            result = EmbeddingResult(
                text=f"test{i}",
                embedding=[0.1 * i],
                model="test-model",
                dimension=1,
                processing_time=0.1
            )
            cache.put(f"test{i}", "test-model", result)
        
        # First item should be evicted
        assert cache.get("test0", "test-model") is None
        assert cache.get("test1", "test-model") is not None
        assert cache.get("test2", "test-model") is not None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
            processing_time=0.1
        )
        
        # Initial stats
        stats = self.cache.stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Cache miss
        self.cache.get("test", "test-model")
        
        # Cache put and hit
        self.cache.put("test", "test-model", result)
        self.cache.get("test", "test-model")
        
        stats = self.cache.stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_cache_clear(self):
        """Test cache clearing."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
            processing_time=0.1
        )
        
        self.cache.put("test", "test-model", result)
        assert self.cache.get("test", "test-model") is not None
        
        self.cache.clear()
        assert self.cache.get("test", "test-model") is None


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = MockEmbeddingProvider()
        self.preprocessor = TextPreprocessor()
        self.cache = EmbeddingCache(cache_dir=None)
        self.generator = EmbeddingGenerator(
            self.provider, self.preprocessor, self.cache
        )
    
    def test_single_embedding_generation(self):
        """Test single text embedding generation."""
        result = self.generator.generate("Hello World!")
        
        assert isinstance(result, EmbeddingResult)
        assert result.text == "hello world!"
        assert len(result.embedding) == 5
        assert result.model == "mock-model"
        assert result.dimension == 5
        assert result.processing_time > 0
        assert result.cached is False
    
    def test_cached_embedding_generation(self):
        """Test cached embedding generation."""
        # First generation
        result1 = self.generator.generate("Hello World!")
        assert result1.cached is False
        
        # Second generation (should be cached)
        result2 = self.generator.generate("Hello World!")
        assert result2.cached is True
        assert result2.embedding == result1.embedding
    
    def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        texts = ["Hello", "World", "Test"]
        result = self.generator.generate_batch(texts)
        
        assert isinstance(result, BatchEmbeddingResult)
        assert len(result.texts) == 3
        assert len(result.embeddings) == 3
        assert result.model == "mock-model"
        assert result.dimension == 5
        assert result.total_processing_time > 0
        assert result.cache_misses == 3
        assert result.cache_hits == 0
    
    def test_batch_with_cache(self):
        """Test batch generation with some cached results."""
        texts = ["Hello", "World", "Hello"]  # "Hello" appears twice
        
        # First batch - within the batch, duplicate "Hello" should be cached
        result1 = self.generator.generate_batch(texts)
        assert result1.cache_misses == 3  # All are cache misses initially
        assert result1.cache_hits == 0
        
        # Second batch (all cache hits)
        result2 = self.generator.generate_batch(texts)
        assert result2.cache_misses == 0
        assert result2.cache_hits == 3
    
    def test_batch_size_limit(self):
        """Test batch processing with size limits."""
        texts = [f"text{i}" for i in range(5)]
        result = self.generator.generate_batch(texts, batch_size=2)
        
        assert len(result.embeddings) == 5
        assert result.cache_misses == 5
    
    def test_preprocessing_integration(self):
        """Test integration with text preprocessing."""
        result = self.generator.generate("  HELLO WORLD!  ")
        assert result.text == "hello world!"
    
    def test_provider_error_handling(self):
        """Test handling of provider errors."""
        # Mock provider that raises an error
        error_provider = Mock()
        error_provider.get_embedding.side_effect = Exception("Provider error")
        error_provider.get_model_name.return_value = "error-model"
        
        generator = EmbeddingGenerator(error_provider, self.preprocessor, self.cache)
        
        with pytest.raises(Exception):
            generator.generate("test")
    
    def test_cache_stats(self):
        """Test cache statistics retrieval."""
        # Generate some embeddings
        self.generator.generate("test1")
        self.generator.generate("test2")
        self.generator.generate("test1")  # Cache hit
        
        stats = self.generator.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == 1/3
    
    def test_clear_cache(self):
        """Test cache clearing."""
        self.generator.generate("test")
        initial_stats = self.generator.get_cache_stats()
        assert initial_stats["cache_misses"] == 1
        
        self.generator.clear_cache()
        
        # Should be cache miss again (stats don't reset, just cache content)
        self.generator.generate("test")
        stats = self.generator.get_cache_stats()
        # Cache was cleared, so this is another miss
        assert stats["cache_misses"] == 2


class TestSentenceTransformerProvider:
    """Test cases for SentenceTransformerProvider."""
    
    def test_sentence_transformer_provider(self):
        """Test SentenceTransformer provider."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            # Mock the model
            mock_model = Mock()
            # Return a list directly (not numpy array)
            mock_model.encode.return_value = [0.1, 0.2, 0.3]
            mock_st.return_value = mock_model
            
            provider = SentenceTransformerProvider("test-model")
            
            # Test single embedding
            embedding = provider.get_embedding("test")
            assert embedding == [0.1, 0.2, 0.3]
            
            # Test model name
            assert provider.get_model_name() == "test-model"
            
            # Test dimension
            assert provider.get_dimension() == 3
    
    def test_sentence_transformer_not_installed(self):
        """Test error when sentence-transformers not installed."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            provider = SentenceTransformerProvider()
            
            with pytest.raises(ModelNotAvailableError):
                provider.get_embedding("test")


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""
    
    def test_openai_provider(self):
        """Test OpenAI provider."""
        with patch('openai.OpenAI') as mock_openai:
            # Mock the client and response
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            provider = OpenAIProvider("text-embedding-3-small", "test-key")
            
            # Test single embedding
            embedding = provider.get_embedding("test")
            assert embedding == [0.1, 0.2, 0.3]
            
            # Test model name
            assert provider.get_model_name() == "text-embedding-3-small"
            
            # Test dimension
            assert provider.get_dimension() == 1536
    
    def test_openai_not_installed(self):
        """Test error when openai not installed."""
        with patch.dict('sys.modules', {'openai': None}):
            provider = OpenAIProvider()
            
            with pytest.raises(ModelNotAvailableError):
                provider.get_embedding("test")


class TestCreateEmbeddingGenerator:
    """Test cases for factory function."""
    
    @patch('day_53.embedding_generator.SentenceTransformerProvider')
    def test_create_sentence_transformer_generator(self, mock_provider_class):
        """Test creating generator with sentence-transformers."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        generator = create_embedding_generator(
            provider_type="sentence-transformers",
            model_name="custom-model",
            cache_dir="/tmp/cache"
        )
        
        assert isinstance(generator, EmbeddingGenerator)
        mock_provider_class.assert_called_once_with("custom-model")
    
    @patch('day_53.embedding_generator.OpenAIProvider')
    def test_create_openai_generator(self, mock_provider_class):
        """Test creating generator with OpenAI."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        generator = create_embedding_generator(
            provider_type="openai",
            model_name="text-embedding-3-large",
            api_key="test-key"
        )
        
        assert isinstance(generator, EmbeddingGenerator)
        mock_provider_class.assert_called_once_with("text-embedding-3-large", "test-key")
    
    def test_create_unknown_provider(self):
        """Test error with unknown provider type."""
        with pytest.raises(ValueError):
            create_embedding_generator(provider_type="unknown")
    
    def test_create_with_preprocessing_options(self):
        """Test creating generator with preprocessing options."""
        with patch('day_53.embedding_generator.SentenceTransformerProvider'):
            generator = create_embedding_generator(
                provider_type="sentence-transformers",
                lowercase=False,
                remove_special_chars=True,
                min_length=5,
                max_length=1000
            )
            
            assert isinstance(generator, EmbeddingGenerator)
            assert generator.preprocessor.lowercase is False
            assert generator.preprocessor.remove_special_chars is True
            assert generator.preprocessor.min_length == 5
            assert generator.preprocessor.max_length == 1000


if __name__ == "__main__":
    pytest.main([__file__])