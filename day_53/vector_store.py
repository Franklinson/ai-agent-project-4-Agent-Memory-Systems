"""
Vector database implementation for semantic memory using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid
import os
from day_53.embedding_generator import EmbeddingGenerator, create_embedding_generator


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when a collection is not found."""
    pass


class DocumentNotFoundError(VectorStoreError):
    """Raised when a document is not found."""
    pass


@dataclass
class VectorDocument:
    """Represents a document with vector embedding."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Represents a search result."""
    document: VectorDocument
    score: float
    distance: float


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class DefaultEmbeddingProvider(EmbeddingProvider):
    """Default embedding provider using ChromaDB's built-in embeddings."""
    
    def embed_text(self, text: str) -> List[float]:
        # ChromaDB handles embeddings automatically
        return []
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # ChromaDB handles embeddings automatically
        return []


class VectorStore:
    """Vector database implementation using ChromaDB."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        use_local_embeddings: bool = True
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist data (None for in-memory)
            embedding_provider: Custom embedding provider (deprecated, use embedding_generator)
            embedding_generator: Embedding generator for automatic embedding generation
            use_local_embeddings: Whether to use local embeddings (sentence-transformers)
        """
        # Handle embedding generation
        if embedding_generator:
            self.embedding_generator = embedding_generator
        elif use_local_embeddings:
            try:
                self.embedding_generator = create_embedding_generator(
                    provider_type="sentence_transformers",
                    model_name="all-MiniLM-L6-v2",
                    cache_dir=persist_directory + "_embeddings" if persist_directory else None
                )
            except Exception as e:
                print(f"Warning: Could not create local embedding generator: {e}")
                self.embedding_generator = None
        else:
            self.embedding_generator = None
        
        # Legacy embedding provider support
        self.embedding_provider = embedding_provider or DefaultEmbeddingProvider()
        
        # Configure ChromaDB
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        
        self._collections = {}
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None
    ) -> None:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Collection metadata
            embedding_function: Custom embedding function
        """
        try:
            # ChromaDB requires non-empty metadata
            collection_metadata = metadata or {"created_by": "vector_store"}
            collection = self.client.create_collection(
                name=name,
                metadata=collection_metadata,
                embedding_function=embedding_function
            )
            self._collections[name] = collection
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection '{name}': {e}")
    
    def get_collection(self, name: str):
        """Get existing collection."""
        if name in self._collections:
            return self._collections[name]
        
        try:
            collection = self.client.get_collection(name)
            self._collections[name] = collection
            return collection
        except Exception as e:
            raise CollectionNotFoundError(f"Collection '{name}' not found: {e}")
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            raise VectorStoreError(f"Failed to list collections: {e}")
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            if name in self._collections:
                del self._collections[name]
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection '{name}': {e}")
    
    def add_document(
        self,
        collection_name: str,
        document: VectorDocument
    ) -> str:
        """
        Add a document to a collection.
        
        Args:
            collection_name: Name of the collection
            document: Document to add
            
        Returns:
            Document ID
        """
        collection = self.get_collection(collection_name)
        
        doc_id = document.id or str(uuid.uuid4())
        
        # Generate embedding if not provided
        embedding = document.embedding
        if not embedding and self.embedding_generator:
            try:
                result = self.embedding_generator.generate(document.content)
                embedding = result.embedding
            except Exception as e:
                print(f"Warning: Could not generate embedding: {e}")
        
        try:
            collection.add(
                ids=[doc_id],
                documents=[document.content],
                metadatas=[document.metadata or {"source": "vector_store"}],
                embeddings=[embedding] if embedding else None
            )
            return doc_id
        except Exception as e:
            raise VectorStoreError(f"Failed to add document: {e}")
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[VectorDocument]
    ) -> List[str]:
        """Add multiple documents to a collection."""
        collection = self.get_collection(collection_name)
        
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        
        # Generate embeddings for documents that don't have them
        embeddings = []
        texts_to_embed = []
        embed_indices = []
        
        for i, doc in enumerate(documents):
            if doc.embedding:
                embeddings.append(doc.embedding)
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(doc.content)
                embed_indices.append(i)
        
        # Generate missing embeddings in batch
        if texts_to_embed and self.embedding_generator:
            try:
                batch_result = self.embedding_generator.generate_batch(texts_to_embed)
                for idx, embedding in zip(embed_indices, batch_result.embeddings):
                    embeddings[idx] = embedding
            except Exception as e:
                print(f"Warning: Could not generate batch embeddings: {e}")
        
        # Filter out None embeddings for ChromaDB
        final_embeddings = [emb for emb in embeddings if emb is not None]
        
        try:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=[meta or {"source": "vector_store"} for meta in metadatas],
                embeddings=final_embeddings if final_embeddings else None
            )
            return ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    def get_document(
        self,
        collection_name: str,
        document_id: str
    ) -> VectorDocument:
        """Get a document by ID."""
        collection = self.get_collection(collection_name)
        
        try:
            result = collection.get(ids=[document_id])
            if not result['ids']:
                raise DocumentNotFoundError(f"Document '{document_id}' not found")
            
            return VectorDocument(
                id=result['ids'][0],
                content=result['documents'][0],
                metadata=result['metadatas'][0] if result['metadatas'] else None,
                embedding=result['embeddings'][0] if result['embeddings'] else None
            )
        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to get document: {e}")
    
    def update_document(
        self,
        collection_name: str,
        document: VectorDocument
    ) -> None:
        """Update a document."""
        collection = self.get_collection(collection_name)
        
        # Generate embedding if not provided
        embedding = document.embedding
        if not embedding and self.embedding_generator:
            try:
                result = self.embedding_generator.generate(document.content)
                embedding = result.embedding
            except Exception as e:
                print(f"Warning: Could not generate embedding for update: {e}")
        
        try:
            collection.update(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.metadata or {"source": "vector_store"}],
                embeddings=[embedding] if embedding else None
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to update document: {e}")
    
    def delete_document(
        self,
        collection_name: str,
        document_id: str
    ) -> None:
        """Delete a document."""
        collection = self.get_collection(collection_name)
        
        try:
            collection.delete(ids=[document_id])
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document: {e}")
    
    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            collection_name: Name of the collection
            query: Query text
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            List of search results
        """
        collection = self.get_collection(collection_name)
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                doc = VectorDocument(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else None,
                    embedding=results['embeddings'][0][i] if results['embeddings'] else None
                )
                
                search_results.append(SearchResult(
                    document=doc,
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    distance=results['distances'][0][i]
                ))
            
            return search_results
        except Exception as e:
            raise VectorStoreError(f"Failed to search: {e}")
    
    def search_by_vector(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search using a pre-computed embedding vector."""
        collection = self.get_collection(collection_name)
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                doc = VectorDocument(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else None,
                    embedding=results['embeddings'][0][i] if results['embeddings'] else None
                )
                
                search_results.append(SearchResult(
                    document=doc,
                    score=1.0 - results['distances'][0][i],
                    distance=results['distances'][0][i]
                ))
            
            return search_results
        except Exception as e:
            raise VectorStoreError(f"Failed to search by vector: {e}")
    
    def count_documents(self, collection_name: str) -> int:
        """Count documents in a collection."""
        collection = self.get_collection(collection_name)
        try:
            return collection.count()
        except Exception as e:
            raise VectorStoreError(f"Failed to count documents: {e}")
    
    def get_collection_metadata(self, collection_name: str) -> Dict[str, Any]:
        """Get collection metadata."""
        collection = self.get_collection(collection_name)
        try:
            return collection.metadata or {}
        except Exception as e:
            raise VectorStoreError(f"Failed to get collection metadata: {e}")
    
    def peek_collection(
        self,
        collection_name: str,
        limit: int = 10
    ) -> List[VectorDocument]:
        """Peek at documents in a collection."""
        collection = self.get_collection(collection_name)
        
        try:
            result = collection.peek(limit=limit)
            documents = []
            
            for i in range(len(result['ids'])):
                doc = VectorDocument(
                    id=result['ids'][i],
                    content=result['documents'][i],
                    metadata=result['metadatas'][i] if result['metadatas'] else None,
                    embedding=result['embeddings'][i] if result['embeddings'] else None
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            raise VectorStoreError(f"Failed to peek collection: {e}")
    
    def close(self) -> None:
        """Close the vector store connection."""
        # ChromaDB handles cleanup automatically
        self._collections.clear()