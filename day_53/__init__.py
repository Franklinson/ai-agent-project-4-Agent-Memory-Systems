"""
Day 53: Embedding Generation for Semantic Memory

This module provides embedding generation capabilities with text preprocessing,
caching, batch processing, and support for multiple embedding providers.
"""

from .embedding_generator import (
    # Main classes
    EmbeddingGenerator,
    TextPreprocessor,
    EmbeddingCache,
    
    # Providers
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    
    # Data classes
    EmbeddingResult,
    BatchEmbeddingResult,
    
    # Factory function
    create_embedding_generator,
    
    # Exceptions
    EmbeddingError,
    PreprocessingError,
    ModelNotAvailableError,
)

from .rag_system import (
    # Main RAG system
    RAGSystem,
    create_rag_system,
    
    # Response generators
    ResponseGenerator,
    MockResponseGenerator,
    OpenAIResponseGenerator,
    
    # Data classes
    RetrievedDocument,
    RAGContext,
    RAGResponse,
    
    # Exceptions
    RAGError,
    QueryProcessingError,
    RetrievalError,
    ContextBuildingError,
    GenerationError,
)

__all__ = [
    # Main classes
    "EmbeddingGenerator",
    "TextPreprocessor", 
    "EmbeddingCache",
    
    # Providers
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIProvider",
    
    # Data classes
    "EmbeddingResult",
    "BatchEmbeddingResult",
    
    # Factory function
    "create_embedding_generator",
    
    # Exceptions
    "EmbeddingError",
    "PreprocessingError",
    "ModelNotAvailableError",
    
    # Main RAG system
    "RAGSystem",
    "create_rag_system",
    
    # Response generators
    "ResponseGenerator",
    "MockResponseGenerator", 
    "OpenAIResponseGenerator",
    
    # Data classes
    "RetrievedDocument",
    "RAGContext",
    "RAGResponse",
    
    # RAG exceptions
    "RAGError",
    "QueryProcessingError",
    "RetrievalError",
    "ContextBuildingError",
    "GenerationError",
]