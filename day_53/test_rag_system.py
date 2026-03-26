"""
Tests for rag_system.py
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from day_53.rag_system import (
    RAGSystem, MockResponseGenerator, OpenAIResponseGenerator,
    RetrievedDocument, RAGContext, RAGResponse,
    create_rag_system, RAGError, QueryProcessingError, RetrievalError,
    ContextBuildingError, GenerationError
)
from day_53.embedding_generator import EmbeddingGenerator, TextPreprocessor, EmbeddingCache
from day_53.vector_store import VectorStore
from day_50.fact_store import FactStore, FactType
from day_50.knowledge_graph import KnowledgeGraph, NodeType


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def __init__(self, model_name: str = "mock-model", dimension: int = 384):
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


class TestMockResponseGenerator:
    """Test cases for MockResponseGenerator."""
    
    def test_default_response_generation(self):
        """Test default response generation."""
        generator = MockResponseGenerator()
        
        prompt = """Context:
Python is a programming language
Machine learning uses algorithms

Query: What is Python?"""
        
        response = generator.generate(prompt)
        assert "mock response" in response.lower()
        assert "retrieved knowledge" in response.lower()
    
    def test_custom_template(self):
        """Test custom response template."""
        template = "Custom response for query: {query}"
        generator = MockResponseGenerator(response_template=template)
        
        prompt = "Query: What is Python?"
        response = generator.generate(prompt)
        
        assert "Custom response for query: What is Python?" == response
    
    def test_context_extraction(self):
        """Test context and query extraction from prompt."""
        generator = MockResponseGenerator(
            response_template="Context: {context} | Query: {query}"
        )
        
        prompt = """Context:
Line 1
Line 2

Query: Test query"""
        
        response = generator.generate(prompt)
        assert "Line 1\nLine 2" in response
        assert "Test query" in response


class TestOpenAIResponseGenerator:
    """Test cases for OpenAIResponseGenerator."""
    
    def test_initialization(self):
        """Test OpenAI generator initialization."""
        generator = OpenAIResponseGenerator(
            api_key="test-key",
            model="gpt-4",
            max_tokens=1000
        )
        
        assert generator.api_key == "test-key"
        assert generator.model == "gpt-4"
        assert generator.max_tokens == 1000
    
    @patch('openai.OpenAI')
    def test_successful_generation(self, mock_openai):
        """Test successful response generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        generator = OpenAIResponseGenerator("test-key")
        response = generator.generate("Test prompt")
        
        assert response == "Generated response"
        mock_client.chat.completions.create.assert_called_once()
    
    def test_openai_not_installed(self):
        """Test error when OpenAI not installed."""
        with patch.dict('sys.modules', {'openai': None}):
            generator = OpenAIResponseGenerator("test-key")
            
            with pytest.raises(GenerationError):
                generator.generate("Test prompt")


class TestRAGSystem:
    """Test cases for RAGSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock components
        self.embedding_provider = MockEmbeddingProvider()
        self.embedding_generator = EmbeddingGenerator(
            self.embedding_provider,
            TextPreprocessor(),
            EmbeddingCache(cache_dir=None)
        )
        
        self.vector_store = VectorStore(
            persist_directory=str(Path(self.temp_dir) / "test_db"),
            use_local_embeddings=False
        )
        
        self.response_generator = MockResponseGenerator()
        
        self.rag_system = RAGSystem(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            response_generator=self.response_generator,
            collection_name="test_collection",
            max_retrieved_docs=3,
            min_similarity_score=0.1
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test RAG system initialization."""
        assert self.rag_system.embedding_generator is not None
        assert self.rag_system.vector_store is not None
        assert self.rag_system.response_generator is not None
        assert self.rag_system.max_retrieved_docs == 3
        assert self.rag_system.min_similarity_score == 0.1
    
    def test_index_document(self):
        """Test document indexing."""
        doc_id = self.rag_system.index_document(
            doc_id="test_doc",
            content="Python is a programming language",
            metadata={"type": "definition"},
            source_type="document"
        )
        
        assert doc_id == "test_doc"
        
        # Verify document was indexed
        count = self.vector_store.count_documents("test_collection")
        assert count == 1
    
    def test_index_fact(self):
        """Test fact indexing."""
        # Create a fact store and fact
        fact_store = FactStore()
        fact = fact_store.store(
            "Python", "is_a", "programming language",
            fact_type=FactType.DEFINITION,
            confidence=0.95
        )
        
        # Index the fact
        doc_id = self.rag_system.index_fact(fact)
        
        assert doc_id == f"fact_{fact.id}"
        
        # Verify document was indexed
        count = self.vector_store.count_documents("test_collection")
        assert count == 1
    
    def test_index_node(self):
        """Test knowledge graph node indexing."""
        # Create a knowledge graph and node
        kg = KnowledgeGraph()
        node = kg.add_node("Python", NodeType.ENTITY, properties={"type": "language"})
        
        # Index the node
        doc_id = self.rag_system.index_node(node)
        
        assert doc_id == f"node_{node.id}"
        
        # Verify document was indexed
        count = self.vector_store.count_documents("test_collection")
        assert count == 1
    
    def test_index_knowledge_base(self):
        """Test indexing entire knowledge base."""
        # Create fact store and knowledge graph
        fact_store = FactStore()
        kg = KnowledgeGraph()
        
        # Add some facts and nodes
        fact1 = fact_store.store("Python", "is_a", "programming language")
        fact2 = fact_store.store("Java", "is_a", "programming language")
        node1 = kg.add_node("Python", NodeType.ENTITY)
        node2 = kg.add_node("Java", NodeType.ENTITY)
        
        # Set up RAG system with knowledge bases
        self.rag_system.fact_store = fact_store
        self.rag_system.knowledge_graph = kg
        
        # Index knowledge base
        result = self.rag_system.index_knowledge_base()
        
        assert result["facts"] == 2
        assert result["nodes"] == 2
        assert result["errors"] == 0
        
        # Verify documents were indexed
        count = self.vector_store.count_documents("test_collection")
        assert count == 4
    
    def test_process_query(self):
        """Test query processing."""
        query = "What is Python?"
        embedding = self.rag_system.process_query(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Mock provider dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_process_empty_query(self):
        """Test processing empty query."""
        with pytest.raises(QueryProcessingError):
            self.rag_system.process_query("")
        
        with pytest.raises(QueryProcessingError):
            self.rag_system.process_query("   ")
    
    def test_retrieve_knowledge(self):
        """Test knowledge retrieval."""
        # Index some documents first
        self.rag_system.index_document(
            "doc1", "Python is a programming language",
            {"type": "definition"}, "document"
        )
        self.rag_system.index_document(
            "doc2", "Java is also a programming language",
            {"type": "definition"}, "document"
        )
        
        # Generate query embedding
        query_embedding = self.rag_system.process_query("What is Python?")
        
        # Retrieve knowledge
        docs = self.rag_system.retrieve_knowledge(query_embedding, "What is Python?")
        
        assert isinstance(docs, list)
        assert len(docs) <= self.rag_system.max_retrieved_docs
        
        for doc in docs:
            assert isinstance(doc, RetrievedDocument)
            assert doc.similarity_score >= self.rag_system.min_similarity_score
    
    def test_build_context(self):
        """Test context building."""
        # Create mock retrieved documents
        docs = [
            RetrievedDocument(
                id="doc1",
                content="Python is a programming language",
                metadata={"type": "definition"},
                similarity_score=0.9,
                source_type="document"
            ),
            RetrievedDocument(
                id="doc2",
                content="Python was created by Guido van Rossum",
                metadata={"type": "fact"},
                similarity_score=0.8,
                source_type="fact"
            )
        ]
        
        query = "What is Python?"
        context = self.rag_system.build_context(query, docs)
        
        assert isinstance(context, RAGContext)
        assert context.query == query
        assert len(context.retrieved_documents) == 2
        assert "Python is a programming language" in context.formatted_context
        assert "Python was created by Guido van Rossum" in context.formatted_context
        assert query in context.augmented_prompt
        assert context.total_tokens > 0
    
    def test_build_context_no_docs(self):
        """Test context building with no retrieved documents."""
        query = "What is Python?"
        context = self.rag_system.build_context(query, [])
        
        assert isinstance(context, RAGContext)
        assert context.query == query
        assert len(context.retrieved_documents) == 0
        assert "No relevant information found" in context.formatted_context
    
    def test_generate_response(self):
        """Test response generation."""
        context = RAGContext(
            query="What is Python?",
            retrieved_documents=[],
            formatted_context="Python is a programming language",
            augmented_prompt="Context: Python is a programming language\nQuery: What is Python?",
            retrieval_time=0.1,
            context_building_time=0.05
        )
        
        response = self.rag_system.generate_response(context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_complete_query_flow(self):
        """Test complete RAG query flow."""
        # Index some documents
        self.rag_system.index_document(
            "doc1", "Python is a high-level programming language",
            {"type": "definition"}, "document"
        )
        self.rag_system.index_document(
            "doc2", "Python supports multiple programming paradigms",
            {"type": "feature"}, "document"
        )
        
        # Execute query
        result = self.rag_system.query("What is Python?")
        
        assert isinstance(result, RAGResponse)
        assert result.success is True
        assert result.query == "What is Python?"
        assert len(result.response) > 0
        assert result.context is not None
        assert result.total_time > 0
        assert result.error_message is None
    
    def test_query_with_no_results(self):
        """Test query with no matching documents."""
        # Don't index any documents
        result = self.rag_system.query("What is quantum computing?")
        
        assert isinstance(result, RAGResponse)
        assert result.success is True  # Should still succeed with no results
        assert len(result.context.retrieved_documents) == 0
        assert "No relevant information found" in result.context.formatted_context
    
    def test_query_error_handling(self):
        """Test query error handling."""
        # Mock embedding generator to raise an error
        self.rag_system.embedding_generator = Mock()
        self.rag_system.embedding_generator.generate.side_effect = Exception("Embedding failed")
        
        result = self.rag_system.query("What is Python?")
        
        assert isinstance(result, RAGResponse)
        assert result.success is False
        assert result.error_message is not None
        assert "Embedding failed" in result.error_message
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        # Initial stats
        stats = self.rag_system.get_stats()
        assert stats["total_queries"] == 0
        assert stats["successful_queries"] == 0
        assert stats["failed_queries"] == 0
        
        # Index a document and make a successful query
        self.rag_system.index_document(
            "doc1", "Python is a programming language",
            {"type": "definition"}, "document"
        )
        
        result = self.rag_system.query("What is Python?")
        assert result.success is True
        
        # Check updated stats
        stats = self.rag_system.get_stats()
        assert stats["total_queries"] == 1
        assert stats["successful_queries"] == 1
        assert stats["failed_queries"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["average_retrieved_docs"] >= 0
    
    def test_clear_statistics(self):
        """Test clearing statistics."""
        # Make a query to generate stats
        self.rag_system.query("test query")
        
        stats = self.rag_system.get_stats()
        assert stats["total_queries"] > 0
        
        # Clear stats
        self.rag_system.clear_stats()
        
        stats = self.rag_system.get_stats()
        assert stats["total_queries"] == 0
        assert stats["successful_queries"] == 0
        assert stats["failed_queries"] == 0


class TestCreateRAGSystem:
    """Test cases for create_rag_system factory function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('day_53.rag_system.create_embedding_generator')
    @patch('day_53.rag_system.VectorStore')
    def test_create_with_defaults(self, mock_vector_store, mock_create_embedding):
        """Test creating RAG system with default parameters."""
        # Mock the dependencies
        mock_embedding_generator = Mock()
        mock_create_embedding.return_value = mock_embedding_generator
        
        mock_vs = Mock()
        mock_vector_store.return_value = mock_vs
        
        rag_system = create_rag_system()
        
        assert isinstance(rag_system, RAGSystem)
        assert rag_system.embedding_generator == mock_embedding_generator
        assert rag_system.vector_store == mock_vs
        assert isinstance(rag_system.response_generator, MockResponseGenerator)
    
    @patch('day_53.rag_system.create_embedding_generator')
    @patch('day_53.rag_system.VectorStore')
    def test_create_with_openai_generator(self, mock_vector_store, mock_create_embedding):
        """Test creating RAG system with OpenAI response generator."""
        # Mock the dependencies
        mock_embedding_generator = Mock()
        mock_create_embedding.return_value = mock_embedding_generator
        
        mock_vs = Mock()
        mock_vector_store.return_value = mock_vs
        
        rag_system = create_rag_system(
            response_generator_type="openai",
            openai_api_key="test-key",
            openai_model="gpt-4"
        )
        
        assert isinstance(rag_system, RAGSystem)
        assert isinstance(rag_system.response_generator, OpenAIResponseGenerator)
        assert rag_system.response_generator.api_key == "test-key"
        assert rag_system.response_generator.model == "gpt-4"
    
    def test_create_with_invalid_generator_type(self):
        """Test creating RAG system with invalid response generator type."""
        with pytest.raises(ValueError):
            create_rag_system(response_generator_type="invalid")
    
    @patch('day_53.rag_system.create_embedding_generator')
    @patch('day_53.rag_system.VectorStore')
    def test_create_with_custom_parameters(self, mock_vector_store, mock_create_embedding):
        """Test creating RAG system with custom parameters."""
        # Mock the dependencies
        mock_embedding_generator = Mock()
        mock_create_embedding.return_value = mock_embedding_generator
        
        mock_vs = Mock()
        mock_vector_store.return_value = mock_vs
        
        fact_store = FactStore()
        kg = KnowledgeGraph()
        
        rag_system = create_rag_system(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            vector_store_path=str(Path(self.temp_dir) / "custom_db"),
            collection_name="custom_collection",
            max_retrieved_docs=10,
            min_similarity_score=0.5,
            fact_store=fact_store,
            knowledge_graph=kg
        )
        
        assert isinstance(rag_system, RAGSystem)
        assert rag_system.max_retrieved_docs == 10
        assert rag_system.min_similarity_score == 0.5
        assert rag_system.fact_store == fact_store
        assert rag_system.knowledge_graph == kg


class TestIntegration:
    """Integration tests for RAG system with real components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end RAG workflow."""
        # Create RAG system with real components
        rag_system = create_rag_system(
            vector_store_path=str(Path(self.temp_dir) / "test_db"),
            collection_name="test_collection",
            response_generator_type="mock"
        )
        
        # Create knowledge bases
        fact_store = FactStore()
        kg = KnowledgeGraph()
        
        # Add some knowledge
        fact1 = fact_store.store("Python", "is_a", "programming language", fact_type=FactType.DEFINITION)
        fact2 = fact_store.store("Python", "created_by", "Guido van Rossum", fact_type=FactType.ASSERTION)
        node1 = kg.add_node("Python", NodeType.ENTITY, properties={"type": "language"})
        node2 = kg.add_node("Programming", NodeType.CONCEPT)
        
        # Set up RAG system with knowledge bases
        rag_system.fact_store = fact_store
        rag_system.knowledge_graph = kg
        
        # Index knowledge base
        indexed = rag_system.index_knowledge_base()
        assert indexed["facts"] == 2
        assert indexed["nodes"] == 2
        
        # Add some additional documents
        rag_system.index_document(
            "doc1", "Python is widely used for data science and machine learning",
            {"type": "application"}, "document"
        )
        
        # Execute queries
        result1 = rag_system.query("What is Python?")
        assert result1.success is True
        assert len(result1.context.retrieved_documents) > 0
        
        result2 = rag_system.query("Who created Python?")
        assert result2.success is True
        
        result3 = rag_system.query("What is Python used for?")
        assert result3.success is True
        
        # Check statistics
        stats = rag_system.get_stats()
        assert stats["total_queries"] == 3
        assert stats["successful_queries"] == 3
        assert stats["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])