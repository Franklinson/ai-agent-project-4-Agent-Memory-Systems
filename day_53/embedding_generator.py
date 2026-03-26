"""
Embedding generation for semantic memory with preprocessing, caching, and batch processing.
"""

import re
import time
import json
import hashlib
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from collections import OrderedDict


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class PreprocessingError(EmbeddingError):
    """Exception raised during text preprocessing."""
    pass


class ModelNotAvailableError(EmbeddingError):
    """Exception raised when embedding model is not available."""
    pass


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a single text."""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    processing_time: float
    cached: bool = False


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""
    texts: List[str]
    embeddings: List[List[float]]
    model: str
    dimension: int
    total_processing_time: float
    cache_hits: int
    cache_misses: int


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get dimension by encoding a test string
                test_embedding = self._model.encode("test")
                self._dimension = len(test_embedding)
            except ImportError:
                raise ModelNotAvailableError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            except Exception as e:
                raise ModelNotAvailableError(f"Failed to load model {self.model_name}: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._load_model()
        embedding = self._model.encode(text)
        # Handle both numpy arrays and lists
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        embeddings = self._model.encode(texts)
        # Handle both numpy arrays and lists
        if hasattr(embeddings[0], 'tolist'):
            return [emb.tolist() for emb in embeddings]
        return [list(emb) for emb in embeddings]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None
        self._dimension = None
    
    def _load_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                # Set dimension based on model
                if "text-embedding-3-small" in self.model_name:
                    self._dimension = 1536
                elif "text-embedding-3-large" in self.model_name:
                    self._dimension = 3072
                else:
                    self._dimension = 1536  # default
            except ImportError:
                raise ModelNotAvailableError(
                    "openai not installed. Run: pip install openai"
                )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._load_client()
        try:
            response = self._client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI API error: {e}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_client()
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI API error: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self._load_client()
        return self._dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class TextPreprocessor:
    """Text preprocessing for embedding generation."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = False,
        min_length: int = 1,
        max_length: int = 8192
    ):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
        self.max_length = max_length
    
    def preprocess(self, text: Union[str, Any]) -> str:
        """Preprocess a single text."""
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Strip whitespace
        text = text.strip()
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s!?.,;:\'"()-]', '', text)
        
        # Normalize whitespace (after special char removal to handle extra spaces)
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Check length constraints
        if len(text) < self.min_length:
            raise PreprocessingError(f"Text too short: {len(text)} < {self.min_length}")
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def preprocess_batch(self, texts: List[Union[str, Any]]) -> List[str]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]


class EmbeddingCache:
    """Cache for embedding results with memory and disk storage."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 1000
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_size = max_cache_size
        self._memory_cache = OrderedDict()
        self._stats = {"cache_hits": 0, "cache_misses": 0}
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        content = f"{text}:{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_disk_path(self, cache_key: str) -> Path:
        """Get disk path for cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, text: str, model: str) -> Optional[EmbeddingResult]:
        """Get cached embedding result."""
        cache_key = self._get_cache_key(text, model)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            # Move to end (LRU)
            result = self._memory_cache.pop(cache_key)
            self._memory_cache[cache_key] = result
            self._stats["cache_hits"] += 1
            result.cached = True
            return result
        
        # Check disk cache
        if self.cache_dir:
            disk_path = self._get_disk_path(cache_key)
            if disk_path.exists():
                try:
                    with open(disk_path, 'r') as f:
                        data = json.load(f)
                    result = EmbeddingResult(**data)
                    result.cached = True
                    
                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, result)
                    self._stats["cache_hits"] += 1
                    return result
                except Exception:
                    # Remove corrupted cache file
                    disk_path.unlink(missing_ok=True)
        
        self._stats["cache_misses"] += 1
        return None
    
    def put(self, text: str, model: str, result: EmbeddingResult):
        """Cache embedding result."""
        cache_key = self._get_cache_key(text, model)
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, result)
        
        # Save to disk cache
        if self.cache_dir:
            disk_path = self._get_disk_path(cache_key)
            try:
                with open(disk_path, 'w') as f:
                    json.dump(asdict(result), f)
            except Exception:
                pass  # Ignore disk cache errors
    
    def _add_to_memory_cache(self, cache_key: str, result: EmbeddingResult):
        """Add result to memory cache with LRU eviction."""
        # Remove if already exists
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Add to end
        self._memory_cache[cache_key] = result
        
        # Evict oldest if over limit
        while len(self._memory_cache) > self.max_cache_size:
            self._memory_cache.popitem(last=False)
    
    def clear(self):
        """Clear all caches."""
        self._memory_cache.clear()
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink(missing_ok=True)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = self._stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "max_cache_size": self.max_cache_size
        }


class EmbeddingGenerator:
    """Main embedding generator with preprocessing, caching, and batch processing."""
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        preprocessor: Optional[TextPreprocessor] = None,
        cache: Optional[EmbeddingCache] = None
    ):
        self.provider = provider
        self.preprocessor = preprocessor or TextPreprocessor()
        self.cache = cache or EmbeddingCache()
    
    def generate(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        start_time = time.time()
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Check cache
        cached_result = self.cache.get(processed_text, self.provider.get_model_name())
        if cached_result:
            return cached_result
        
        # Generate embedding
        embedding = self.provider.get_embedding(processed_text)
        processing_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            text=processed_text,
            embedding=embedding,
            model=self.provider.get_model_name(),
            dimension=self.provider.get_dimension(),
            processing_time=processing_time,
            cached=False
        )
        
        # Cache result
        self.cache.put(processed_text, self.provider.get_model_name(), result)
        
        return result
    
    def generate_batch(self, texts: List[str], batch_size: int = 100) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts with batching."""
        start_time = time.time()
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        results = []
        cache_hits = 0
        cache_misses = 0
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text in batch
            for j, text in enumerate(batch_texts):
                cached_result = self.cache.get(text, self.provider.get_model_name())
                if cached_result:
                    batch_results.append(cached_result.embedding)
                    cache_hits += 1
                else:
                    batch_results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
                    cache_misses += 1
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                uncached_embeddings = self.provider.get_embeddings(uncached_texts)
                
                # Fill in results and cache
                for k, (text, embedding) in enumerate(zip(uncached_texts, uncached_embeddings)):
                    batch_idx = uncached_indices[k]
                    batch_results[batch_idx] = embedding
                    
                    # Cache result
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model=self.provider.get_model_name(),
                        dimension=self.provider.get_dimension(),
                        processing_time=0.0,  # Individual timing not available in batch
                        cached=False
                    )
                    self.cache.put(text, self.provider.get_model_name(), result)
            
            results.extend(batch_results)
        
        total_processing_time = time.time() - start_time
        
        return BatchEmbeddingResult(
            texts=processed_texts,
            embeddings=results,
            model=self.provider.get_model_name(),
            dimension=self.provider.get_dimension(),
            total_processing_time=total_processing_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()


def create_embedding_generator(
    provider_type: str = "sentence-transformers",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> EmbeddingGenerator:
    """Factory function to create embedding generator with common configurations."""
    
    # Create provider
    if provider_type == "sentence-transformers":
        model_name = model_name or "all-MiniLM-L6-v2"
        provider = SentenceTransformerProvider(model_name)
    elif provider_type == "openai":
        model_name = model_name or "text-embedding-3-small"
        provider = OpenAIProvider(model_name, api_key)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    # Create preprocessor
    preprocessor = TextPreprocessor(**kwargs)
    
    # Create cache
    cache = EmbeddingCache(cache_dir=cache_dir)
    
    return EmbeddingGenerator(provider, preprocessor, cache)