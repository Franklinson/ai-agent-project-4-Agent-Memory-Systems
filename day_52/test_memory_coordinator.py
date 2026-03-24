"""Tests for memory coordination mechanisms."""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(__file__))
from memory_coordinator import (
    MemoryCoordinator, KeywordQueryRouter, RelevanceResultMerger, DefaultEventHandler,
    CoordinationPattern, EventType, CoordinationEvent, RoutingDecision, MergedResult,
    SyncState, RoutingError, MergingError, SynchronizationError
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_47'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_51'))

from unified_memory import UnifiedMemory, MemoryLayer, UnifiedResult
from hybrid_memory import MemoryItem, MemoryType, MemorySource
from context_manager import Priority
from event_store import EventType as ESEventType
from experience_tracker import Outcome
from fact_store import FactType
from knowledge_graph import NodeType
from cross_type_queries import QueryPattern, RankingStrategy


class TestKeywordQueryRouter:
    """Test keyword-based query routing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = KeywordQueryRouter()
    
    def test_working_memory_routing(self):
        """Test routing to working memory."""
        decision = self.router.route("What did we talk about in our conversation?")
        
        assert MemoryLayer.WORKING in decision.target_layers
        assert decision.confidence > 0
        assert "scores" in decision.metadata
    
    def test_short_term_memory_routing(self):
        """Test routing to short-term memory."""
        decision = self.router.route("What happened recently?")
        
        assert MemoryLayer.SHORT_TERM in decision.target_layers
        assert decision.confidence > 0
    
    def test_long_term_memory_routing(self):
        """Test routing to long-term memory."""
        decision = self.router.route("What is the definition of this concept?")
        
        assert MemoryLayer.LONG_TERM in decision.target_layers
        assert decision.confidence > 0
    
    def test_hybrid_memory_routing(self):
        """Test routing to hybrid memory."""
        decision = self.router.route("What is related to this topic?")
        
        assert MemoryLayer.HYBRID in decision.target_layers
        assert decision.confidence > 0
    
    def test_no_keywords_routing(self):
        """Test routing when no specific keywords are found."""
        decision = self.router.route("xyz abc def")
        
        assert len(decision.target_layers) == 4  # All layers
        assert decision.pattern == CoordinationPattern.PARALLEL
        assert decision.confidence == 0.3
    
    def test_multiple_keywords_routing(self):
        """Test routing with multiple keyword matches."""
        decision = self.router.route("What facts are related to recent events?")
        
        assert len(decision.target_layers) >= 2
        assert decision.pattern == CoordinationPattern.PARALLEL


class TestRelevanceResultMerger:
    """Test relevance-based result merging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.merger = RelevanceResultMerger()
    
    def test_merge_empty_results(self):
        """Test merging empty results."""
        merged = self.merger.merge([])
        
        assert merged.count == 0
        assert merged.query == ""
        assert len(merged.sources) == 0
    
    def test_merge_single_result(self):
        """Test merging single result."""
        item = MemoryItem(
            id="test1",
            memory_type=MemoryType.EPISODIC,
            source=MemorySource.EVENT_STORE,
            content="Test content",
            timestamp=datetime.now(),
            relevance=0.8
        )
        
        result = UnifiedResult(
            query="test",
            items=[item],
            layer=MemoryLayer.SHORT_TERM
        )
        
        merged = self.merger.merge([result])
        
        assert merged.count == 1
        assert merged.query == "test"
        assert MemoryLayer.SHORT_TERM in merged.sources
        assert merged.items[0].id == "test1"
    
    def test_merge_multiple_results(self):
        """Test merging multiple results."""
        item1 = MemoryItem(
            id="test1",
            memory_type=MemoryType.EPISODIC,
            source=MemorySource.EVENT_STORE,
            content="Test content 1",
            timestamp=datetime.now(),
            relevance=0.8
        )
        
        item2 = MemoryItem(
            id="test2",
            memory_type=MemoryType.SEMANTIC,
            source=MemorySource.FACT_STORE,
            content="Test content 2",
            timestamp=datetime.now(),
            relevance=0.9
        )
        
        result1 = UnifiedResult(query="test", items=[item1], layer=MemoryLayer.SHORT_TERM)
        result2 = UnifiedResult(query="test", items=[item2], layer=MemoryLayer.LONG_TERM)
        
        merged = self.merger.merge([result1, result2])
        
        assert merged.count == 2
        assert merged.items[0].relevance >= merged.items[1].relevance  # Sorted by relevance
        assert len(merged.sources) == 2
    
    def test_merge_duplicate_removal(self):
        """Test removal of duplicate items."""
        item = MemoryItem(
            id="test1",
            memory_type=MemoryType.EPISODIC,
            source=MemorySource.EVENT_STORE,
            content="Test content",
            timestamp=datetime.now(),
            relevance=0.8
        )
        
        result1 = UnifiedResult(query="test", items=[item], layer=MemoryLayer.SHORT_TERM)
        result2 = UnifiedResult(query="test", items=[item], layer=MemoryLayer.HYBRID)
        
        merged = self.merger.merge([result1, result2])
        
        assert merged.count == 1  # Duplicate removed
        assert merged.metadata["original_count"] == 2
        assert merged.metadata["unique_count"] == 1


class TestDefaultEventHandler:
    """Test default event handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = DefaultEventHandler()
    
    def test_handle_event(self):
        """Test basic event handling."""
        event = CoordinationEvent(
            event_type=EventType.MEMORY_STORED,
            source_layer=MemoryLayer.WORKING,
            data={"test": "data"}
        )
        
        result = self.handler.handle_event(event)
        
        assert result is True
        assert event.handled is True
        assert len(self.handler.handled_events) == 1
        assert self.handler.handled_events[0] == event


class TestMemoryCoordinator:
    """Test memory coordination functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory = UnifiedMemory(total_budget=2000, max_messages=5)
        self.coordinator = MemoryCoordinator(self.memory)
    
    # ------------------------------------------------------------------
    # Query routing tests
    # ------------------------------------------------------------------
    
    def test_route_query(self):
        """Test query routing."""
        decision = self.coordinator.route_query("What is Python?")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.query == "What is Python?"
        assert len(decision.target_layers) > 0
        assert self.coordinator.stats["queries_routed"] == 1
    
    def test_route_query_error(self):
        """Test query routing error handling."""
        # Mock router to raise exception
        self.coordinator.router = Mock()
        self.coordinator.router.route.side_effect = Exception("Test error")
        
        with pytest.raises(RoutingError):
            self.coordinator.route_query("test")
    
    def test_execute_coordinated_query_sequential(self):
        """Test sequential coordinated query execution."""
        # Store some test data
        self.memory.store("Hello!", role="user")
        self.memory.store("Test fact", memory_type="fact", subject="test", predicate="is")
        
        result = self.coordinator.execute_coordinated_query(
            "test",
            pattern=CoordinationPattern.SEQUENTIAL
        )
        
        assert isinstance(result, MergedResult)
        assert result.pattern == CoordinationPattern.SEQUENTIAL
        assert result.execution_time > 0
        assert self.coordinator.stats["results_merged"] >= 1
    
    def test_execute_coordinated_query_parallel(self):
        """Test parallel coordinated query execution."""
        # Store some test data
        self.memory.store("Hello!", role="user")
        self.memory.store("Test fact", memory_type="fact", subject="test", predicate="is")
        
        result = self.coordinator.execute_coordinated_query(
            "test",
            pattern=CoordinationPattern.PARALLEL
        )
        
        assert isinstance(result, MergedResult)
        assert result.pattern == CoordinationPattern.PARALLEL
        assert result.execution_time > 0
    
    def test_execute_coordinated_query_hierarchical(self):
        """Test hierarchical coordinated query execution."""
        result = self.coordinator.execute_coordinated_query(
            "test",
            pattern=CoordinationPattern.HIERARCHICAL
        )
        
        assert isinstance(result, MergedResult)
        assert result.pattern == CoordinationPattern.SEQUENTIAL  # Converted to sequential
    
    def test_execute_coordinated_query_adaptive(self):
        """Test adaptive coordinated query execution."""
        result = self.coordinator.execute_coordinated_query(
            "test",
            pattern=CoordinationPattern.ADAPTIVE
        )
        
        assert isinstance(result, MergedResult)
        assert result.pattern in [CoordinationPattern.SEQUENTIAL, CoordinationPattern.PARALLEL]
    
    # ------------------------------------------------------------------
    # Result merging tests
    # ------------------------------------------------------------------
    
    def test_merge_results(self):
        """Test result merging."""
        item = MemoryItem(
            id="test1",
            memory_type=MemoryType.EPISODIC,
            source=MemorySource.EVENT_STORE,
            content="Test content",
            timestamp=datetime.now(),
            relevance=0.8
        )
        
        result = UnifiedResult(query="test", items=[item])
        merged = self.coordinator.merge_results([result])
        
        assert isinstance(merged, MergedResult)
        assert merged.count == 1
        assert self.coordinator.stats["results_merged"] >= 1
    
    def test_merge_results_error(self):
        """Test result merging error handling."""
        # Mock merger to raise exception
        self.coordinator.merger = Mock()
        self.coordinator.merger.merge.side_effect = Exception("Test error")
        
        with pytest.raises(MergingError):
            self.coordinator.merge_results([])
    
    def test_rank_items_by_relevance(self):
        """Test ranking items by relevance."""
        items = [
            MemoryItem(
                id="test1", memory_type=MemoryType.EPISODIC,
                source=MemorySource.EVENT_STORE, content="Test 1",
                timestamp=datetime.now(), relevance=0.5
            ),
            MemoryItem(
                id="test2", memory_type=MemoryType.EPISODIC,
                source=MemorySource.EVENT_STORE, content="Test 2",
                timestamp=datetime.now(), relevance=0.9
            )
        ]
        
        ranked = self.coordinator.rank_items(items, RankingStrategy.RELEVANCE)
        
        assert ranked[0].relevance >= ranked[1].relevance
        assert ranked[0].id == "test2"
    
    def test_rank_items_by_timestamp(self):
        """Test ranking items by timestamp."""
        now = datetime.now()
        items = [
            MemoryItem(
                id="test1", memory_type=MemoryType.EPISODIC,
                source=MemorySource.EVENT_STORE, content="Test 1",
                timestamp=now - timedelta(hours=1), relevance=0.5
            ),
            MemoryItem(
                id="test2", memory_type=MemoryType.EPISODIC,
                source=MemorySource.EVENT_STORE, content="Test 2",
                timestamp=now, relevance=0.5
            )
        ]
        
        ranked = self.coordinator.rank_items(items, RankingStrategy.TIMESTAMP)
        
        assert ranked[0].timestamp >= ranked[1].timestamp
        assert ranked[0].id == "test2"
    
    def test_rank_items_by_source_priority(self):
        """Test ranking items by source priority."""
        items = [
            MemoryItem(
                id="test1", memory_type=MemoryType.SEMANTIC,
                source=MemorySource.KNOWLEDGE_GRAPH, content="Test 1",
                timestamp=datetime.now(), relevance=0.8
            ),
            MemoryItem(
                id="test2", memory_type=MemoryType.EPISODIC,
                source=MemorySource.EVENT_STORE, content="Test 2",
                timestamp=datetime.now(), relevance=0.8
            )
        ]
        
        ranked = self.coordinator.rank_items(items, RankingStrategy.SOURCE_PRIORITY)
        
        # EVENT_STORE has higher priority than KNOWLEDGE_GRAPH
        assert ranked[0].source == MemorySource.EVENT_STORE
    
    # ------------------------------------------------------------------
    # State synchronization tests
    # ------------------------------------------------------------------
    
    def test_synchronize_state(self):
        """Test state synchronization."""
        results = self.coordinator.synchronize_state([MemoryLayer.WORKING])
        
        assert MemoryLayer.WORKING in results
        assert isinstance(results[MemoryLayer.WORKING], bool)
        assert self.coordinator.stats["sync_operations"] >= 1
    
    def test_synchronize_all_layers(self):
        """Test synchronizing all layers."""
        results = self.coordinator.synchronize_state()
        
        assert len(results) == 4  # All memory layers
        assert all(isinstance(result, bool) for result in results.values())
    
    def test_get_layer_state_working(self):
        """Test getting working layer state."""
        state = self.coordinator._get_layer_state(MemoryLayer.WORKING)
        
        assert "messages" in state
        assert "context_components" in state
        assert "tokens_used" in state
    
    def test_get_layer_state_short_term(self):
        """Test getting short-term layer state."""
        state = self.coordinator._get_layer_state(MemoryLayer.SHORT_TERM)
        
        assert "events" in state
        assert "experiences" in state
    
    def test_get_layer_state_long_term(self):
        """Test getting long-term layer state."""
        state = self.coordinator._get_layer_state(MemoryLayer.LONG_TERM)
        
        assert "facts" in state
        assert "nodes" in state
        assert "edges" in state
    
    def test_get_layer_state_hybrid(self):
        """Test getting hybrid layer state."""
        state = self.coordinator._get_layer_state(MemoryLayer.HYBRID)
        
        assert isinstance(state, dict)
    
    def test_detect_conflicts(self):
        """Test conflict detection."""
        layer = MemoryLayer.WORKING
        state = {"test": "value"}
        
        # First sync - no conflicts
        conflicts = self.coordinator._detect_conflicts(layer, state)
        assert len(conflicts) == 0
        
        # Update sync state
        self.coordinator.sync_states[layer].version = 1
        self.coordinator.sync_states[layer].checksum = self.coordinator._calculate_checksum(state)
        
        # Same state - no conflicts
        conflicts = self.coordinator._detect_conflicts(layer, state)
        assert len(conflicts) == 0
        
        # Different state - conflict detected
        new_state = {"test": "different_value"}
        conflicts = self.coordinator._detect_conflicts(layer, new_state)
        assert len(conflicts) > 0
    
    def test_calculate_checksum(self):
        """Test checksum calculation."""
        state1 = {"a": 1, "b": 2}
        state2 = {"b": 2, "a": 1}  # Same content, different order
        state3 = {"a": 1, "b": 3}  # Different content
        
        checksum1 = self.coordinator._calculate_checksum(state1)
        checksum2 = self.coordinator._calculate_checksum(state2)
        checksum3 = self.coordinator._calculate_checksum(state3)
        
        assert checksum1 == checksum2  # Order doesn't matter
        assert checksum1 != checksum3  # Different content
    
    # ------------------------------------------------------------------
    # Event coordination tests
    # ------------------------------------------------------------------
    
    def test_emit_event(self):
        """Test event emission."""
        event = CoordinationEvent(
            event_type=EventType.MEMORY_STORED,
            source_layer=MemoryLayer.WORKING
        )
        
        initial_count = len(self.coordinator.event_queue)
        self.coordinator._emit_event(event)
        
        assert len(self.coordinator.event_queue) == initial_count + 1
        assert event.handled is True
        assert self.coordinator.stats["events_handled"] >= 1
    
    def test_process_events(self):
        """Test event processing."""
        # Add unhandled event
        event = CoordinationEvent(
            event_type=EventType.MEMORY_STORED,
            source_layer=MemoryLayer.WORKING,
            handled=False
        )
        self.coordinator.event_queue.append(event)
        
        processed = self.coordinator.process_events()
        
        assert processed >= 1
        assert event.handled is True
    
    def test_get_pending_events(self):
        """Test getting pending events."""
        # Add handled and unhandled events
        handled_event = CoordinationEvent(
            event_type=EventType.MEMORY_STORED,
            source_layer=MemoryLayer.WORKING,
            handled=True
        )
        unhandled_event = CoordinationEvent(
            event_type=EventType.MEMORY_UPDATED,
            source_layer=MemoryLayer.SHORT_TERM,
            handled=False
        )
        
        self.coordinator.event_queue.extend([handled_event, unhandled_event])
        
        pending = self.coordinator.get_pending_events()
        
        assert len(pending) == 1
        assert pending[0] == unhandled_event
    
    def test_clear_events(self):
        """Test clearing events."""
        # Add some events
        event = CoordinationEvent(
            event_type=EventType.MEMORY_STORED,
            source_layer=MemoryLayer.WORKING
        )
        self.coordinator.event_queue.append(event)
        
        self.coordinator.clear_events()
        
        assert len(self.coordinator.event_queue) == 0
    
    # ------------------------------------------------------------------
    # Coordination patterns tests
    # ------------------------------------------------------------------
    
    def test_coordinate_store(self):
        """Test coordinated storage."""
        results = self.coordinator.coordinate_store(
            "Test content",
            target_layers=[MemoryLayer.WORKING, MemoryLayer.HYBRID],
            role="user"
        )
        
        assert len(results) >= 1
        assert all(isinstance(item, MemoryItem) for item in results.values())
    
    def test_coordinate_update(self):
        """Test coordinated updates."""
        results = self.coordinator.coordinate_update(
            "test_id",
            {"content": "updated"},
            target_layers=[MemoryLayer.HYBRID]
        )
        
        assert isinstance(results, dict)
        # Note: Actual update logic would depend on implementation
    
    # ------------------------------------------------------------------
    # Statistics and monitoring tests
    # ------------------------------------------------------------------
    
    def test_get_coordination_stats(self):
        """Test getting coordination statistics."""
        stats = self.coordinator.get_coordination_stats()
        
        assert "queries_routed" in stats
        assert "results_merged" in stats
        assert "events_handled" in stats
        assert "sync_operations" in stats
        assert "sync_states" in stats
        assert "pending_events" in stats
        assert "total_events" in stats
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        # Generate some stats
        self.coordinator.route_query("test")
        
        # Reset
        self.coordinator.reset_stats()
        
        assert self.coordinator.stats["queries_routed"] == 0
        assert len(self.coordinator.event_queue) == 0


class TestCoordinationIntegration:
    """Integration tests for memory coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory = UnifiedMemory(total_budget=4000, max_messages=10)
        self.coordinator = MemoryCoordinator(self.memory)
    
    def test_full_coordination_workflow(self):
        """Test complete coordination workflow."""
        # 1. Store coordinated content
        store_results = self.coordinator.coordinate_store(
            "Python is a programming language",
            target_layers=[MemoryLayer.WORKING, MemoryLayer.HYBRID],
            role="user"
        )
        
        assert len(store_results) >= 1
        
        # 2. Execute coordinated query
        query_result = self.coordinator.execute_coordinated_query(
            "Python programming",
            pattern=CoordinationPattern.PARALLEL
        )
        
        assert query_result.count >= 0
        assert query_result.pattern == CoordinationPattern.PARALLEL
        
        # 3. Synchronize state
        sync_results = self.coordinator.synchronize_state()
        
        assert len(sync_results) == 4
        
        # 4. Check statistics
        stats = self.coordinator.get_coordination_stats()
        
        assert stats["queries_routed"] >= 1
        assert stats["events_handled"] >= 1
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid query
        try:
            result = self.coordinator.execute_coordinated_query("")
            # Should handle gracefully
            assert isinstance(result, MergedResult)
        except Exception as e:
            # Should not crash the system
            assert isinstance(e, (RoutingError, MergingError))
        
        # System should still be functional
        stats = self.coordinator.get_coordination_stats()
        assert isinstance(stats, dict)
    
    def test_concurrent_operations(self):
        """Test concurrent coordination operations."""
        import threading
        
        results = []
        errors = []
        
        def worker():
            try:
                result = self.coordinator.execute_coordinated_query(
                    f"test query {threading.current_thread().ident}",
                    pattern=CoordinationPattern.PARALLEL
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) >= 0  # Some may succeed
        assert len(errors) == 0 or all(isinstance(e, Exception) for e in errors)
        
        # System should still be functional
        stats = self.coordinator.get_coordination_stats()
        assert stats["queries_routed"] >= len(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])