"""
Complete RAG (Retrieval-Augmented Generation) system for semantic memory.

This module provides a comprehensive RAG implementation that integrates:
- Query processing with embedding generation
- Knowledge retrieval from vector databases
- Context building and prompt augmentation
- Response generation coordination
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from day_53.embedding_generator import EmbeddingGenerator, create_embedding_generator
from day_53.vector_store import VectorStore
from day_50.fact_store import FactStore, Fact
from day_50.knowledge_graph import KnowledgeGraph, Node


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass


class QueryProcessingError(RAGError):
    """Exception raised during query processing."""
    pass


class RetrievalError(RAGError):
    """Exception raised during knowledge retrieval."""
    pass


class ContextBuildingError(RAGError):
    """Exception raised during context building."""
    pass


class GenerationError(RAGError):
    """Exception raised during response generation."""
    pass


@dataclass
class RetrievedDocument:
    """A document retrieved from the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_type: str  # 'fact', 'node', 'document'
    source_object: Optional[Any] = None  # Original Fact, Node, or document


@dataclass
class RAGContext:
    """Context built from retrieved documents."""
    query: str
    retrieved_documents: List[RetrievedDocument]
    formatted_context: str
    augmented_prompt: str
    retrieval_time: float
    context_building_time: float
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete RAG response with metadata."""
    query: str
    response: str
    context: RAGContext
    generation_time: float
    total_time: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseGenerator(ABC):
    """Abstract base class for response generators."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from augmented prompt."""
        pass


class MockResponseGenerator(ResponseGenerator):
    """Mock response generator for testing and demonstration."""
    
    def __init__(self, response_template: str = None):
        self.response_template = response_template or "Based on the context: {context}\n\nQuery: {query}\n\nResponse: This is a mock response based on the retrieved knowledge."
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        # Extract context and query from prompt for template
        lines = prompt.split('\n')
        context = ""
        query = ""
        
        for i, line in enumerate(lines):
            if "Context:" in line:
                # Find context section
                context_start = i + 1
                for j in range(context_start, len(lines)):
                    if lines[j].startswith("Query:") or lines[j].startswith("Question:"):
                        break
                    context += lines[j] + "\n"
            elif "Query:" in line or "Question:" in line:
                query = line.split(":", 1)[1].strip()
        
        try:
            return self.response_template.format(
                context=context.strip(),
                query=query,
                **kwargs
            )
        except Exception as e:
            # Fallback response if template formatting fails
            return f"Mock response for query: {query}. Context available: {bool(context.strip())}"


class OpenAIResponseGenerator(ResponseGenerator):
    """OpenAI-based response generator."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise GenerationError("OpenAI library not installed. Run: pip install openai")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            client = self._get_client()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise GenerationError(f"OpenAI generation failed: {e}")


class RAGSystem:
    """Complete RAG system integrating all components."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        response_generator: ResponseGenerator,
        fact_store: Optional[FactStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        collection_name: str = "rag_knowledge",
        max_retrieved_docs: int = 5,
        min_similarity_score: float = 0.0,
        context_template: str = None
    ):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.response_generator = response_generator
        self.fact_store = fact_store
        self.knowledge_graph = knowledge_graph
        self.collection_name = collection_name
        self.max_retrieved_docs = max_retrieved_docs
        self.min_similarity_score = min_similarity_score
        
        # Create collection if it doesn't exist
        try:
            self.vector_store.get_collection(self.collection_name)
        except:
            self.vector_store.create_collection(self.collection_name)
        
        # Default context template
        self.context_template = context_template or """Context:
{context}

Query: {query}

Please provide a comprehensive answer based on the context above."""
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
            "average_retrieved_docs": 0.0
        }
    
    def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: str = "document",
        source_object: Any = None
    ) -> str:
        """Index a document in the vector store."""
        try:
            # Generate embedding
            embedding_result = self.embedding_generator.generate(content)
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "source_type": source_type,
                "content": content,
                "indexed_at": time.time()
            })
            
            # Create VectorDocument
            from day_53.vector_store import VectorDocument
            vector_doc = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding_result.embedding,
                metadata=doc_metadata
            )
            
            # Store in vector database
            return self.vector_store.add_document(self.collection_name, vector_doc)
        
        except Exception as e:
            raise RetrievalError(f"Failed to index document {doc_id}: {e}")
    
    def index_fact(self, fact: Fact) -> str:
        """Index a fact from the fact store."""
        content = f"{fact.subject} {fact.predicate} {fact.object}"
        metadata = {
            "fact_id": fact.id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "fact_type": fact.fact_type.value if fact.fact_type else None,
            "confidence": fact.confidence
        }
        
        return self.index_document(
            doc_id=f"fact_{fact.id}",
            content=content,
            metadata=metadata,
            source_type="fact",
            source_object=fact
        )
    
    def index_node(self, node: Node) -> str:
        """Index a knowledge graph node."""
        content = f"{node.label} ({node.node_type.value})"
        if node.properties:
            content += f" - {', '.join(f'{k}: {v}' for k, v in node.properties.items())}"
        
        # Flatten properties for ChromaDB compatibility
        metadata = {
            "node_id": node.id,
            "label": node.label,
            "node_type": node.node_type.value
        }
        
        # Add properties as individual metadata fields
        if node.properties:
            for key, value in node.properties.items():
                # Ensure values are ChromaDB-compatible types
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"prop_{key}"] = value
                else:
                    metadata[f"prop_{key}"] = str(value)
        
        return self.index_document(
            doc_id=f"node_{node.id}",
            content=content,
            metadata=metadata,
            source_type="node",
            source_object=node
        )
    
    def index_knowledge_base(self) -> Dict[str, int]:
        """Index all facts and nodes from connected knowledge bases."""
        indexed = {"facts": 0, "nodes": 0, "errors": 0}
        
        # Index facts
        if self.fact_store:
            try:
                # Get all facts using the list method
                facts = list(self.fact_store._facts.values())
                for fact in facts:
                    try:
                        self.index_fact(fact)
                        indexed["facts"] += 1
                    except Exception as e:
                        logger.error(f"Failed to index fact {fact.id}: {e}")
                        indexed["errors"] += 1
            except Exception as e:
                logger.error(f"Failed to retrieve facts: {e}")
        
        # Index nodes
        if self.knowledge_graph:
            try:
                # Get all nodes using the nodes dictionary
                nodes = list(self.knowledge_graph._nodes.values())
                for node in nodes:
                    try:
                        self.index_node(node)
                        indexed["nodes"] += 1
                    except Exception as e:
                        logger.error(f"Failed to index node {node.id}: {e}")
                        indexed["errors"] += 1
            except Exception as e:
                logger.error(f"Failed to retrieve nodes: {e}")
        
        logger.info(f"Indexed {indexed['facts']} facts, {indexed['nodes']} nodes with {indexed['errors']} errors")
        return indexed
    
    def process_query(self, query: str) -> List[float]:
        """Process query and generate embedding."""
        try:
            if not query or not query.strip():
                raise QueryProcessingError("Query cannot be empty")
            
            # Generate query embedding
            embedding_result = self.embedding_generator.generate(query.strip())
            return embedding_result.embedding
        
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query: {e}")
    
    def retrieve_knowledge(self, query_embedding: List[float], query: str) -> List[RetrievedDocument]:
        """Retrieve relevant knowledge from vector store."""
        try:
            start_time = time.time()
            
            # Search vector store
            search_results = self.vector_store.search_by_vector(
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                n_results=self.max_retrieved_docs
            )
            
            retrieval_time = time.time() - start_time
            
            # Convert to RetrievedDocument objects and filter by similarity
            retrieved_docs = []
            for result in search_results:
                if result.score >= self.min_similarity_score:
                    doc = RetrievedDocument(
                        id=result.document.id,
                        content=result.document.content,
                        metadata=result.document.metadata or {},
                        similarity_score=result.score,
                        source_type=result.document.metadata.get("source_type", "document") if result.document.metadata else "document",
                        source_object=None  # Could be populated if needed
                    )
                    retrieved_docs.append(doc)
            
            # Update statistics
            self.stats["total_retrieval_time"] += retrieval_time
            
            return retrieved_docs
        
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve knowledge: {e}")
    
    def build_context(self, query: str, retrieved_docs: List[RetrievedDocument]) -> RAGContext:
        """Build context from retrieved documents."""
        try:
            start_time = time.time()
            
            if not retrieved_docs:
                formatted_context = "No relevant information found."
            else:
                # Format retrieved documents
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    source_info = f"[{doc.source_type.upper()}]"
                    if doc.source_type == "fact":
                        source_info += f" (confidence: {doc.metadata.get('confidence', 'N/A')})"
                    
                    context_parts.append(
                        f"{i}. {source_info} {doc.content} (similarity: {doc.similarity_score:.3f})"
                    )
                
                formatted_context = "\n".join(context_parts)
            
            # Build augmented prompt
            augmented_prompt = self.context_template.format(
                context=formatted_context,
                query=query
            )
            
            context_building_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            total_tokens = len(augmented_prompt.split())
            
            return RAGContext(
                query=query,
                retrieved_documents=retrieved_docs,
                formatted_context=formatted_context,
                augmented_prompt=augmented_prompt,
                retrieval_time=0.0,  # Set by caller
                context_building_time=context_building_time,
                total_tokens=total_tokens,
                metadata={
                    "num_retrieved_docs": len(retrieved_docs),
                    "avg_similarity_score": sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
                }
            )
        
        except Exception as e:
            raise ContextBuildingError(f"Failed to build context: {e}")
    
    def generate_response(self, context: RAGContext) -> str:
        """Generate response using the response generator."""
        try:
            start_time = time.time()
            
            response = self.response_generator.generate(context.augmented_prompt)
            
            generation_time = time.time() - start_time
            self.stats["total_generation_time"] += generation_time
            
            return response
        
        except Exception as e:
            raise GenerationError(f"Failed to generate response: {e}")
    
    def query(self, query: str, **kwargs) -> RAGResponse:
        """Complete RAG query processing."""
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            # Step 1: Process query
            logger.info(f"Processing query: {query}")
            query_embedding = self.process_query(query)
            
            # Step 2: Retrieve knowledge
            logger.info("Retrieving relevant knowledge...")
            retrieval_start = time.time()
            retrieved_docs = self.retrieve_knowledge(query_embedding, query)
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 3: Build context
            logger.info("Building context...")
            context = self.build_context(query, retrieved_docs)
            context.retrieval_time = retrieval_time
            
            # Step 4: Generate response
            logger.info("Generating response...")
            response = self.generate_response(context)
            
            total_time = time.time() - start_time
            
            # Update statistics
            self.stats["successful_queries"] += 1
            if retrieved_docs:
                current_avg = self.stats["average_retrieved_docs"]
                total_successful = self.stats["successful_queries"]
                self.stats["average_retrieved_docs"] = (
                    (current_avg * (total_successful - 1) + len(retrieved_docs)) / total_successful
                )
            
            return RAGResponse(
                query=query,
                response=response,
                context=context,
                generation_time=self.stats["total_generation_time"],
                total_time=total_time,
                success=True,
                metadata={
                    "embedding_model": self.embedding_generator.provider.get_model_name(),
                    "embedding_dimension": self.embedding_generator.provider.get_dimension(),
                    "vector_store_collection": self.collection_name
                }
            )
        
        except Exception as e:
            total_time = time.time() - start_time
            self.stats["failed_queries"] += 1
            
            logger.error(f"RAG query failed: {e}")
            
            return RAGResponse(
                query=query,
                response="",
                context=RAGContext(
                    query=query,
                    retrieved_documents=[],
                    formatted_context="",
                    augmented_prompt="",
                    retrieval_time=0.0,
                    context_building_time=0.0
                ),
                generation_time=0.0,
                total_time=total_time,
                success=False,
                error_message=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
            stats["failure_rate"] = stats["failed_queries"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        if stats["successful_queries"] > 0:
            stats["avg_retrieval_time"] = stats["total_retrieval_time"] / stats["successful_queries"]
            stats["avg_generation_time"] = stats["total_generation_time"] / stats["successful_queries"]
        else:
            stats["avg_retrieval_time"] = 0.0
            stats["avg_generation_time"] = 0.0
        
        # Add component stats
        stats["embedding_cache_stats"] = self.embedding_generator.get_cache_stats()
        stats["vector_store_stats"] = {
            "collection_name": self.collection_name,
            "document_count": self.vector_store.count_documents(self.collection_name) if hasattr(self.vector_store, 'count_documents') else 0
        }
        
        return stats
    
    def clear_stats(self):
        """Clear all statistics."""
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
            "average_retrieved_docs": 0.0
        }


def create_rag_system(
    embedding_provider: str = "sentence-transformers",
    embedding_model: str = "all-MiniLM-L6-v2",
    vector_store_path: str = "./rag_vector_db",
    collection_name: str = "rag_knowledge",
    response_generator_type: str = "mock",
    **kwargs
) -> RAGSystem:
    """Factory function to create a complete RAG system."""
    
    # Create embedding generator
    embedding_generator = create_embedding_generator(
        provider_type=embedding_provider,
        model_name=embedding_model,
        cache_dir=kwargs.get("embedding_cache_dir", "./rag_embeddings_cache"),
        **{k: v for k, v in kwargs.items() if k.startswith("embedding_")}
    )
    
    # Create vector store
    vector_store = VectorStore(
        persist_directory=vector_store_path,
        use_local_embeddings=False  # We'll use our own embedding generator
    )
    
    # Create response generator
    if response_generator_type == "mock":
        response_generator = MockResponseGenerator(
            kwargs.get("response_template", None)
        )
    elif response_generator_type == "openai":
        response_generator = OpenAIResponseGenerator(
            api_key=kwargs.get("openai_api_key"),
            model=kwargs.get("openai_model", "gpt-3.5-turbo"),
            max_tokens=kwargs.get("max_tokens", 500)
        )
    else:
        raise ValueError(f"Unknown response generator type: {response_generator_type}")
    
    # Create optional components
    fact_store = kwargs.get("fact_store")
    knowledge_graph = kwargs.get("knowledge_graph")
    
    return RAGSystem(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        response_generator=response_generator,
        fact_store=fact_store,
        knowledge_graph=knowledge_graph,
        collection_name=collection_name,
        max_retrieved_docs=kwargs.get("max_retrieved_docs", 5),
        min_similarity_score=kwargs.get("min_similarity_score", 0.0),
        context_template=kwargs.get("context_template")
    )