"""Tests for unified memory system."""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from unified_memory import UnifiedMemory, MemoryLayer, UnifiedResult, UnifiedMemoryError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_47'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_51'))

from context_manager import Priority
from event_store import EventType
from experience_tracker import Outcome
from fact_store import FactType
from knowledge_graph import NodeType, EdgeType
from cross_type_queries import QueryPattern


class TestUnifiedMemory:
    """Test unified memory system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory = UnifiedMemory(
            total_budget=4000,
            response_reserve=500,
            max_messages=10
        )
    
    # ------------------------------------------------------------------
    # Store interface tests
    # ------------------------------------------------------------------
    
    def test_store_message(self):
        """Test storing conversation messages."""
        item = self.memory.store("Hello!", role="user")
        
        assert item.id == "msg_1"
        assert "user: Hello!" in item.content
        assert item.metadata["layer"] == MemoryLayer.WORKING
        assert self.memory.conversation.get_message_count() == 1
    
    def test_store_event(self):
        """Test storing events."""
        item = self.memory.store(
            "User logged in",
            memory_type="event",
            event_type=EventType.ACTION,
            participants=["alice"]
        )
        
        assert item.memory_type.value == "episodic"
        assert item.source.value == "event_store"
        assert "action" in item.content.lower()
        assert self.memory.events.count == 1
    
    def test_store_experience(self):
        """Test storing experiences."""
        item = self.memory.store(
            "Search completed successfully",
            memory_type="experience",
            action="search",
            outcome=Outcome.SUCCESS,
            score=0.9
        )
        
        assert item.memory_type.value == "episodic"
        assert item.source.value == "experience_tracker"
        assert "search: success" in item.content.lower()
        assert self.memory.experiences.count == 1
    
    def test_store_fact(self):
        """Test storing facts."""
        item = self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a",
            fact_type=FactType.DEFINITION
        )
        
        assert item.memory_type.value == "semantic"
        assert item.source.value == "fact_store"
        assert "Python is_a programming language" in item.content
        assert self.memory.facts.count == 1
    
    def test_store_node(self):
        """Test storing knowledge graph nodes."""
        item = self.memory.store(
            "Python",
            memory_type="node",
            node_type=NodeType.ENTITY
        )
        
        assert item.memory_type.value == "semantic"
        assert item.source.value == "knowledge_graph"
        assert "Python" in item.content and "entity" in item.content.lower()
        assert self.memory.graph.node_count == 1
    
    def test_store_auto_detection(self):
        """Test automatic type detection."""
        # Should default to fact
        item = self.memory.store("Some content")
        assert item.memory_type.value == "semantic"
        assert item.source.value == "fact_store"
    
    # ------------------------------------------------------------------
    # Retrieval tests
    # ------------------------------------------------------------------
    
    def test_get_message(self):
        """Test retrieving messages."""
        self.memory.store("Hello!", role="user")
        item = self.memory.get("msg_1", layer=MemoryLayer.WORKING)
        
        assert "user: Hello!" in item.content
        assert item.metadata["layer"] == MemoryLayer.WORKING
    
    def test_get_hybrid_item(self):
        """Test retrieving from hybrid memory."""
        stored = self.memory.store(
            "Test event",
            memory_type="event",
            event_type=EventType.CUSTOM
        )
        
        retrieved = self.memory.get(stored.id)
        assert retrieved.id == stored.id
        assert retrieved.content == stored.content
    
    def test_get_nonexistent(self):
        """Test retrieving nonexistent item."""
        with pytest.raises(Exception):
            self.memory.get("nonexistent")
    
    # ------------------------------------------------------------------
    # Query interface tests
    # ------------------------------------------------------------------
    
    def test_query_working_memory(self):
        """Test querying working memory."""
        self.memory.store("Hello world!", role="user")
        self.memory.store("Hi there!", role="assistant")
        
        result = self.memory.query("hello", layer=MemoryLayer.WORKING)
        
        assert isinstance(result, UnifiedResult)
        assert result.layer == MemoryLayer.WORKING
        assert result.count == 1
        assert "Hello world!" in result.items[0].content
    
    def test_query_short_term_memory(self):
        """Test querying short-term memory."""
        self.memory.store(
            "Login event",
            memory_type="event",
            event_type=EventType.ACTION
        )
        self.memory.store(
            "Search worked well",
            memory_type="experience",
            action="search",
            outcome=Outcome.SUCCESS
        )
        
        result = self.memory.query("login", layer=MemoryLayer.SHORT_TERM)
        
        assert result.layer == MemoryLayer.SHORT_TERM
        assert result.count >= 1
        assert any("login" in item.content.lower() for item in result.items)
    
    def test_query_long_term_memory(self):
        """Test querying long-term memory."""
        self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a"
        )
        self.memory.store("Django", memory_type="node", node_type=NodeType.ENTITY)
        
        result = self.memory.query("Python", layer=MemoryLayer.LONG_TERM)
        
        assert result.layer == MemoryLayer.LONG_TERM
        assert result.count >= 1
        assert any("Python" in item.content for item in result.items)
    
    def test_query_intelligent_routing(self):
        """Test intelligent query routing."""
        # Store different types of content
        self.memory.store(
            "Login happened",
            memory_type="event",
            event_type=EventType.ACTION
        )
        self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a"
        )
        
        # Temporal query should route to episodic
        result = self.memory.query("when did login happen")
        assert result.layer == MemoryLayer.HYBRID
        
        # Factual query should find semantic content
        result = self.memory.query("what is Python")
        assert result.layer == MemoryLayer.HYBRID
    
    def test_query_cross_type_combined(self):
        """Test combined cross-type queries."""
        self.memory.store(
            "Python event",
            memory_type="event",
            event_type=EventType.CUSTOM
        )
        self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a"
        )
        
        result = self.memory.query(
            "Python",
            pattern=QueryPattern.COMBINED,
            limit=10
        )
        
        assert result.pattern == QueryPattern.COMBINED
        assert result.count >= 2
    
    def test_query_cross_type_sequential(self):
        """Test sequential cross-type queries."""
        self.memory.store(
            "Used Python",
            memory_type="experience",
            action="coding",
            outcome=Outcome.SUCCESS
        )
        self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a"
        )
        
        result = self.memory.query(
            "coding",
            pattern=QueryPattern.SEQUENTIAL,
            primary_type="episodic"
        )
        
        assert result.pattern == QueryPattern.SEQUENTIAL
        assert "metadata" in result.metadata
    
    # ------------------------------------------------------------------
    # Context management tests
    # ------------------------------------------------------------------
    
    def test_add_to_context(self):
        """Test adding content to context."""
        self.memory.add_to_context(
            "system_prompt",
            "You are helpful",
            Priority.CRITICAL
        )
        
        assert len(self.memory.context.components) == 1
        assert self.memory.context.components[0].priority == Priority.CRITICAL
    
    def test_build_context(self):
        """Test building context."""
        self.memory.add_to_context("test", "Test content", Priority.HIGH)
        context = self.memory.build_context()
        
        assert "test" in context
        assert "Test content" in context
    
    def test_build_context_with_memory(self):
        """Test building context with memory retrieval."""
        self.memory.store(
            "Important fact",
            memory_type="fact",
            subject="test",
            predicate="contains"
        )
        
        context = self.memory.build_context(
            include_memory=True,
            query="test"
        )
        
        assert "retrieved_memory" in context
        assert "Important fact" in context
    
    def test_build_messages(self):
        """Test building message list."""
        self.memory.add_to_context("system", "You are helpful", Priority.CRITICAL)
        self.memory.store("Hello!", role="user")
        
        messages = self.memory.build_messages()
        
        assert len(messages) >= 1
        assert any(msg["role"] == "system" for msg in messages)
    
    # ------------------------------------------------------------------
    # Memory management tests
    # ------------------------------------------------------------------
    
    def test_clear_working_memory(self):
        """Test clearing working memory."""
        self.memory.store("Hello!", role="user")
        self.memory.add_to_context("test", "content", Priority.MEDIUM)
        
        self.memory.clear_working_memory()
        
        assert self.memory.conversation.get_message_count() == 0
        assert len(self.memory.context.components) == 0
    
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        self.memory.store("Hello!", role="user")
        self.memory.store(
            "Test event",
            memory_type="event",
            event_type=EventType.CUSTOM
        )
        
        stats = self.memory.get_memory_stats()
        
        assert "working" in stats
        assert "hybrid" in stats
        assert "routing" in stats
        assert stats["working"]["messages"] == 1
        assert stats["hybrid"]["events"] == 1
    
    def test_link_memories(self):
        """Test linking memories."""
        event_item = self.memory.store(
            "Python used",
            memory_type="event",
            event_type=EventType.ACTION
        )
        fact_item = self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a"
        )
        
        self.memory.link_memories(
            event_item.id,
            fact_item.id,
            "relates_to"
        )
        
        related = self.memory.get_related_memories(event_item.id)
        assert len(related) == 1
        assert related[0].id == fact_item.id
    
    def test_optimize_memory(self):
        """Test memory optimization."""
        # Add lots of context to trigger optimization
        for i in range(10):
            self.memory.add_to_context(
                f"component_{i}",
                "x" * 1000,  # Large content
                Priority.LOW
            )
        
        result = self.memory.optimize_memory()
        
        assert "context_optimized" in result
        assert "overflow_detected" in result
        assert "memory_stats" in result
    
    def test_export_memory(self):
        """Test memory export."""
        self.memory.store("Hello!", role="user")
        self.memory.store(
            "Test fact",
            memory_type="fact",
            subject="test",
            predicate="is"
        )
        
        export_data = self.memory.export_memory()
        
        assert "timestamp" in export_data
        assert "stats" in export_data
        assert "working" in export_data
        assert "items" in export_data
        assert len(export_data["working"]["messages"]) == 1
        assert len(export_data["items"]) >= 1
    
    def test_export_memory_by_layer(self):
        """Test exporting specific memory layers."""
        self.memory.store("Hello!", role="user")
        
        working_export = self.memory.export_memory(layer=MemoryLayer.WORKING)
        
        assert "working" in working_export
        assert "items" not in working_export
        assert len(working_export["working"]["messages"]) == 1


class TestUnifiedMemoryIntegration:
    """Integration tests for unified memory system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory = UnifiedMemory()
    
    def test_full_workflow(self):
        """Test complete memory workflow."""
        # 1. Store conversation
        self.memory.store("I want to learn Python", role="user")
        self.memory.store("I can help with that!", role="assistant")
        
        # 2. Store knowledge
        python_fact = self.memory.store(
            "programming language",
            memory_type="fact",
            subject="Python",
            predicate="is_a",
            fact_type=FactType.DEFINITION
        )
        
        # 3. Store experience
        learning_exp = self.memory.store(
            "Python tutorial completed",
            memory_type="experience",
            action="learn_python",
            outcome=Outcome.SUCCESS,
            score=0.8
        )
        
        # 4. Link memories
        self.memory.link_memories(
            learning_exp.id,
            python_fact.id,
            "learned_about"
        )
        
        # 5. Query across types
        result = self.memory.query("Python")
        assert result.count >= 2
        
        # 6. Build context with memory
        context = self.memory.build_context(
            include_memory=True,
            query="Python learning"
        )
        assert "Python" in context
        
        # 7. Get comprehensive stats
        stats = self.memory.get_memory_stats()
        assert stats["working"]["messages"] == 2
        assert stats["hybrid"]["facts"] == 1
        assert stats["hybrid"]["experiences"] == 1
    
    def test_memory_layers_interaction(self):
        """Test interaction between memory layers."""
        # Working memory
        self.memory.store("Learning about AI", role="user")
        
        # Short-term memory
        self.memory.store(
            "Started AI course",
            memory_type="event",
            event_type=EventType.ACTION
        )
        
        # Long-term memory
        self.memory.store(
            "machine learning technique",
            memory_type="fact",
            subject="AI",
            predicate="includes"
        )
        
        # Query each layer
        working_result = self.memory.query("AI", layer=MemoryLayer.WORKING)
        short_result = self.memory.query("AI", layer=MemoryLayer.SHORT_TERM)
        long_result = self.memory.query("AI", layer=MemoryLayer.LONG_TERM)
        
        assert working_result.count >= 1
        assert short_result.count >= 1
        assert long_result.count >= 1
        
        # Query all layers
        all_result = self.memory.query("AI")
        assert all_result.count >= 3
    
    def test_context_with_memory_retrieval(self):
        """Test context building with automatic memory retrieval."""
        # Store relevant memories
        self.memory.store(
            "Django framework",
            memory_type="fact",
            subject="Django",
            predicate="is_a"
        )
        self.memory.store(
            "Built web app",
            memory_type="experience",
            action="web_development",
            outcome=Outcome.SUCCESS
        )
        
        # Add system prompt
        self.memory.add_to_context(
            "system",
            "You are a Python expert",
            Priority.CRITICAL
        )
        
        # Build context with memory retrieval
        context = self.memory.build_context(
            include_memory=True,
            query="Django web development"
        )
        
        assert "Python expert" in context
        assert "Django" in context
        assert "web app" in context or "web_development" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])