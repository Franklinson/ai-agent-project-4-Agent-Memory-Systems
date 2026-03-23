"""Tests for the hybrid memory system."""

import sys
import os
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))
sys.path.insert(0, os.path.dirname(__file__))

from hybrid_memory import (
    HybridMemory, HybridMemoryError, MemoryNotFoundError,
    MemoryType, MemorySource, MemoryItem, QueryResult, CrossLink,
)
from event_store import EventType
from experience_tracker import Outcome
from fact_store import FactType
from knowledge_graph import NodeType, EdgeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hm():
    return HybridMemory()


@pytest.fixture
def populated(hm):
    """HybridMemory pre-loaded with sample data across all subsystems."""
    hm.store_event(EventType.ACTION, data={"action": "login"}, participants=["alice"])
    hm.store_event(EventType.OBSERVATION, data={"saw": "dashboard"}, participants=["alice"])
    hm.store_event(EventType.ERROR, data={"error": "timeout"}, participants=["bob"])

    hm.store_experience("search", Outcome.SUCCESS, score=0.9, tags=["web"])
    hm.store_experience("search", Outcome.FAILURE, score=0.3, feedback="timeout")
    hm.store_experience("summarize", Outcome.SUCCESS, score=0.8, tags=["nlp"])

    hm.store_fact("Python", "is_a", "programming language", fact_type=FactType.DEFINITION)
    hm.store_fact("Python", "created_by", "Guido van Rossum")
    hm.store_fact("Django", "is_a", "web framework", fact_type=FactType.DEFINITION, confidence=0.95)

    hm.store_node("Python", NodeType.ENTITY)
    hm.store_node("Programming Language", NodeType.CONCEPT)
    hm.store_node("Django", NodeType.ENTITY)

    return hm


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_subsystems_created(self):
        hm = HybridMemory()
        assert hm.events is not None
        assert hm.experiences is not None
        assert hm.facts is not None
        assert hm.graph is not None

    def test_custom_subsystems_accepted(self):
        from event_store import EventStore
        from experience_tracker import ExperienceTracker
        from fact_store import FactStore
        from knowledge_graph import KnowledgeGraph

        es, et, fs, kg = EventStore(), ExperienceTracker(), FactStore(), KnowledgeGraph()
        hm = HybridMemory(event_store=es, experience_tracker=et,
                          fact_store=fs, knowledge_graph=kg)
        assert hm.events is es
        assert hm.experiences is et
        assert hm.facts is fs
        assert hm.graph is kg

    def test_empty_stats(self, hm):
        s = hm.stats()
        assert s == {"events": 0, "experiences": 0, "facts": 0,
                     "graph_nodes": 0, "graph_edges": 0, "cross_links": 0}


# ---------------------------------------------------------------------------
# Unified store
# ---------------------------------------------------------------------------

class TestUnifiedStore:
    def test_store_event(self, hm):
        item = hm.store_event(EventType.ACTION, data={"a": 1})
        assert item.memory_type == MemoryType.EPISODIC
        assert item.source == MemorySource.EVENT_STORE
        assert hm.events.count == 1

    def test_store_experience(self, hm):
        item = hm.store_experience("search", Outcome.SUCCESS, score=0.9)
        assert item.memory_type == MemoryType.EPISODIC
        assert item.source == MemorySource.EXPERIENCE_TRACKER
        assert hm.experiences.count == 1

    def test_store_fact(self, hm):
        item = hm.store_fact("Python", "is_a", "language")
        assert item.memory_type == MemoryType.SEMANTIC
        assert item.source == MemorySource.FACT_STORE
        assert hm.facts.count == 1

    def test_store_node(self, hm):
        item = hm.store_node("Python", NodeType.ENTITY)
        assert item.memory_type == MemoryType.SEMANTIC
        assert item.source == MemorySource.KNOWLEDGE_GRAPH
        assert hm.graph.node_count == 1

    def test_store_edge(self, hm):
        n1 = hm.store_node("A", NodeType.ENTITY)
        n2 = hm.store_node("B", NodeType.ENTITY)
        edge = hm.store_edge(n1.id, n2.id, EdgeType.RELATED_TO)
        assert hm.graph.edge_count == 1
        assert edge.source_id == n1.id

    def test_memory_item_fields(self, hm):
        item = hm.store_fact("X", "rel", "Y", confidence=0.7)
        assert item.id
        assert item.content == "X rel Y"
        assert isinstance(item.timestamp, datetime)
        assert item.data is not None
        assert item.metadata["confidence"] == 0.7
        assert item.relevance == 0.7


# ---------------------------------------------------------------------------
# Unified get
# ---------------------------------------------------------------------------

class TestUnifiedGet:
    def test_get_event(self, hm):
        stored = hm.store_event(EventType.ACTION, data={"x": 1})
        retrieved = hm.get(stored.id)
        assert retrieved.id == stored.id
        assert retrieved.source == MemorySource.EVENT_STORE

    def test_get_experience(self, hm):
        stored = hm.store_experience("act", Outcome.SUCCESS)
        retrieved = hm.get(stored.id)
        assert retrieved.id == stored.id
        assert retrieved.source == MemorySource.EXPERIENCE_TRACKER

    def test_get_fact(self, hm):
        stored = hm.store_fact("A", "b", "C")
        retrieved = hm.get(stored.id)
        assert retrieved.id == stored.id
        assert retrieved.source == MemorySource.FACT_STORE

    def test_get_node(self, hm):
        stored = hm.store_node("N", NodeType.CONCEPT)
        retrieved = hm.get(stored.id)
        assert retrieved.id == stored.id
        assert retrieved.source == MemorySource.KNOWLEDGE_GRAPH

    def test_get_with_source_hint(self, hm):
        stored = hm.store_fact("A", "b", "C")
        retrieved = hm.get(stored.id, source=MemorySource.FACT_STORE)
        assert retrieved.id == stored.id

    def test_get_not_found(self, hm):
        with pytest.raises(MemoryNotFoundError):
            hm.get("nonexistent")

    def test_get_wrong_source_hint(self, hm):
        stored = hm.store_fact("A", "b", "C")
        with pytest.raises(MemoryNotFoundError):
            hm.get(stored.id, source=MemorySource.EVENT_STORE)


# ---------------------------------------------------------------------------
# Unified delete
# ---------------------------------------------------------------------------

class TestUnifiedDelete:
    def test_delete_event(self, hm):
        item = hm.store_event(EventType.ACTION, data={"x": 1})
        assert hm.delete(item.id) is True
        assert hm.events.count == 0

    def test_delete_fact(self, hm):
        item = hm.store_fact("A", "b", "C")
        assert hm.delete(item.id) is True
        assert hm.facts.count == 0

    def test_delete_node(self, hm):
        item = hm.store_node("N", NodeType.ENTITY)
        assert hm.delete(item.id) is True
        assert hm.graph.node_count == 0

    def test_delete_not_found(self, hm):
        with pytest.raises(MemoryNotFoundError):
            hm.delete("nonexistent")

    def test_delete_cleans_cross_links(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        fact = hm.store_fact("A", "b", "C")
        hm.cross_link(ev.id, fact.id)
        assert len(hm.get_cross_links(ev.id)) == 1
        hm.delete(ev.id)
        assert len(hm.get_cross_links(ev.id)) == 0


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

class TestTypeDetection:
    def test_detect_episodic_event(self, hm):
        item = hm.store_event(EventType.ACTION, data={"x": 1})
        assert hm.detect_type(item.id) == MemoryType.EPISODIC

    def test_detect_episodic_experience(self, hm):
        item = hm.store_experience("act", Outcome.SUCCESS)
        assert hm.detect_type(item.id) == MemoryType.EPISODIC

    def test_detect_semantic_fact(self, hm):
        item = hm.store_fact("A", "b", "C")
        assert hm.detect_type(item.id) == MemoryType.SEMANTIC

    def test_detect_semantic_node(self, hm):
        item = hm.store_node("N", NodeType.ENTITY)
        assert hm.detect_type(item.id) == MemoryType.SEMANTIC

    def test_detect_unknown(self, hm):
        assert hm.detect_type("nonexistent") is None

    def test_detect_source_event(self, hm):
        item = hm.store_event(EventType.ACTION, data={"x": 1})
        assert hm.detect_source(item.id) == MemorySource.EVENT_STORE

    def test_detect_source_fact(self, hm):
        item = hm.store_fact("A", "b", "C")
        assert hm.detect_source(item.id) == MemorySource.FACT_STORE

    def test_detect_source_unknown(self, hm):
        assert hm.detect_source("nonexistent") is None


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

class TestQueryRouting:
    def test_query_both_types(self, populated):
        result = populated.query("")
        assert result.count > 0
        types = {i.memory_type for i in result.items}
        assert MemoryType.EPISODIC in types
        assert MemoryType.SEMANTIC in types

    def test_query_episodic_only(self, populated):
        result = populated.query("", memory_type=MemoryType.EPISODIC)
        assert all(i.memory_type == MemoryType.EPISODIC for i in result.items)
        assert MemorySource.EVENT_STORE in result.sources_queried
        assert MemorySource.FACT_STORE not in result.sources_queried

    def test_query_semantic_only(self, populated):
        result = populated.query("", memory_type=MemoryType.SEMANTIC)
        assert all(i.memory_type == MemoryType.SEMANTIC for i in result.items)
        assert MemorySource.FACT_STORE in result.sources_queried
        assert MemorySource.EVENT_STORE not in result.sources_queried

    def test_query_text_filter(self, populated):
        result = populated.query("Python")
        assert result.count > 0
        for item in result.items:
            assert "python" in item.content.lower()

    def test_query_by_event_type(self, populated):
        result = populated.query("", event_type=EventType.ERROR)
        events = result.by_source(MemorySource.EVENT_STORE)
        assert len(events) >= 1
        assert all("error" in e.metadata.get("event_type", "") for e in events)

    def test_query_by_participant(self, populated):
        result = populated.query("", participant="alice",
                                 memory_type=MemoryType.EPISODIC)
        events = result.by_source(MemorySource.EVENT_STORE)
        assert len(events) >= 1

    def test_query_by_action(self, populated):
        result = populated.query("", action="search",
                                 memory_type=MemoryType.EPISODIC)
        exps = result.by_source(MemorySource.EXPERIENCE_TRACKER)
        assert len(exps) == 2

    def test_query_by_outcome(self, populated):
        result = populated.query("", outcome=Outcome.SUCCESS,
                                 memory_type=MemoryType.EPISODIC)
        exps = result.by_source(MemorySource.EXPERIENCE_TRACKER)
        assert len(exps) >= 2

    def test_query_by_tag(self, populated):
        result = populated.query("", tag="web",
                                 memory_type=MemoryType.EPISODIC)
        exps = result.by_source(MemorySource.EXPERIENCE_TRACKER)
        assert len(exps) >= 1

    def test_query_by_subject(self, populated):
        result = populated.query("", subject="Python",
                                 memory_type=MemoryType.SEMANTIC)
        facts = result.by_source(MemorySource.FACT_STORE)
        assert len(facts) == 2

    def test_query_by_fact_type(self, populated):
        result = populated.query("", fact_type=FactType.DEFINITION,
                                 memory_type=MemoryType.SEMANTIC)
        facts = result.by_source(MemorySource.FACT_STORE)
        assert len(facts) == 2

    def test_query_by_min_confidence(self, populated):
        result = populated.query("", min_confidence=0.96,
                                 memory_type=MemoryType.SEMANTIC)
        facts = result.by_source(MemorySource.FACT_STORE)
        assert all(i.metadata["confidence"] >= 0.96 for i in facts)

    def test_query_by_node_type(self, populated):
        result = populated.query("", node_type=NodeType.CONCEPT,
                                 memory_type=MemoryType.SEMANTIC)
        nodes = result.by_source(MemorySource.KNOWLEDGE_GRAPH)
        assert len(nodes) == 1

    def test_query_by_node_label(self, populated):
        result = populated.query("", node_label="Django",
                                 memory_type=MemoryType.SEMANTIC)
        nodes = result.by_source(MemorySource.KNOWLEDGE_GRAPH)
        assert len(nodes) == 1

    def test_query_with_limit(self, populated):
        result = populated.query("", limit=3)
        assert result.count == 3

    def test_query_results_sorted_newest_first(self, hm):
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2024, 6, 1)
        hm.store_event(EventType.ACTION, data={"a": 1}, timestamp=t1)
        hm.store_event(EventType.ACTION, data={"a": 2}, timestamp=t2)
        result = hm.query("", memory_type=MemoryType.EPISODIC)
        timestamps = [i.timestamp for i in result.items
                      if i.source == MemorySource.EVENT_STORE]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_by_time_range(self, hm):
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2024, 6, 1)
        t3 = datetime(2024, 12, 1)
        hm.store_event(EventType.ACTION, data={"a": 1}, timestamp=t1)
        hm.store_event(EventType.ACTION, data={"a": 2}, timestamp=t2)
        hm.store_event(EventType.ACTION, data={"a": 3}, timestamp=t3)
        result = hm.query("", memory_type=MemoryType.EPISODIC,
                          time_range=(datetime(2024, 3, 1), datetime(2024, 9, 1)))
        events = result.by_source(MemorySource.EVENT_STORE)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Result merging (QueryResult helpers)
# ---------------------------------------------------------------------------

class TestQueryResult:
    def test_by_source(self, populated):
        result = populated.query("")
        events = result.by_source(MemorySource.EVENT_STORE)
        assert all(i.source == MemorySource.EVENT_STORE for i in events)

    def test_by_type(self, populated):
        result = populated.query("")
        episodic = result.by_type(MemoryType.EPISODIC)
        semantic = result.by_type(MemoryType.SEMANTIC)
        assert len(episodic) + len(semantic) == result.count


# ---------------------------------------------------------------------------
# Cross-memory linking
# ---------------------------------------------------------------------------

class TestCrossLinking:
    def test_create_cross_link(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        fact = hm.store_fact("A", "b", "C")
        link = hm.cross_link(ev.id, fact.id, relationship="caused_by")
        assert isinstance(link, CrossLink)
        assert link.episodic_id == ev.id
        assert link.semantic_id == fact.id
        assert link.relationship == "caused_by"

    def test_get_cross_links(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        f1 = hm.store_fact("A", "b", "C")
        f2 = hm.store_fact("D", "e", "F")
        hm.cross_link(ev.id, f1.id)
        hm.cross_link(ev.id, f2.id)
        links = hm.get_cross_links(ev.id)
        assert len(links) == 2

    def test_get_linked_items(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        fact = hm.store_fact("A", "b", "C")
        hm.cross_link(ev.id, fact.id)
        linked = hm.get_linked_items(ev.id)
        assert len(linked) == 1
        assert linked[0].id == fact.id
        # Reverse lookup
        linked_rev = hm.get_linked_items(fact.id)
        assert len(linked_rev) == 1
        assert linked_rev[0].id == ev.id

    def test_remove_cross_link(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        fact = hm.store_fact("A", "b", "C")
        hm.cross_link(ev.id, fact.id)
        assert hm.remove_cross_link(ev.id, fact.id) is True
        assert len(hm.get_cross_links(ev.id)) == 0

    def test_remove_nonexistent_cross_link(self, hm):
        assert hm.remove_cross_link("a", "b") is False

    def test_cross_link_wrong_types(self, hm):
        f1 = hm.store_fact("A", "b", "C")
        f2 = hm.store_fact("D", "e", "F")
        with pytest.raises(HybridMemoryError):
            hm.cross_link(f1.id, f2.id)

    def test_cross_link_reversed_types(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        fact = hm.store_fact("A", "b", "C")
        with pytest.raises(HybridMemoryError):
            hm.cross_link(fact.id, ev.id)

    def test_cross_link_not_found(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        with pytest.raises(MemoryNotFoundError):
            hm.cross_link(ev.id, "nonexistent")


# ---------------------------------------------------------------------------
# Stats & get_all
# ---------------------------------------------------------------------------

class TestStatsAndGetAll:
    def test_stats(self, populated):
        s = populated.stats()
        assert s["events"] == 3
        assert s["experiences"] == 3
        assert s["facts"] == 3
        assert s["graph_nodes"] == 3
        assert s["graph_edges"] == 0
        assert s["cross_links"] == 0

    def test_get_all_both(self, populated):
        items = populated.get_all()
        assert len(items) == 12  # 3+3+3+3

    def test_get_all_episodic(self, populated):
        items = populated.get_all(memory_type=MemoryType.EPISODIC)
        assert all(i.memory_type == MemoryType.EPISODIC for i in items)
        assert len(items) == 6

    def test_get_all_semantic(self, populated):
        items = populated.get_all(memory_type=MemoryType.SEMANTIC)
        assert all(i.memory_type == MemoryType.SEMANTIC for i in items)
        assert len(items) == 6

    def test_get_all_with_limit(self, populated):
        items = populated.get_all(limit=5)
        assert len(items) == 5

    def test_get_all_sorted_newest_first(self, hm):
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2024, 12, 1)
        hm.store_event(EventType.ACTION, data={"a": 1}, timestamp=t1)
        hm.store_event(EventType.ACTION, data={"a": 2}, timestamp=t2)
        items = hm.get_all()
        assert items[0].timestamp >= items[-1].timestamp


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query(self, hm):
        result = hm.query("")
        assert result.count == 0

    def test_query_no_match(self, populated):
        result = populated.query("zzzznonexistent")
        assert result.count == 0

    def test_store_and_retrieve_round_trip(self, hm):
        item = hm.store_fact("X", "y", "Z", confidence=0.5)
        got = hm.get(item.id)
        assert got.content == "X y Z"
        assert got.data.confidence == 0.5

    def test_delete_then_get_raises(self, hm):
        item = hm.store_event(EventType.ACTION, data={"x": 1})
        hm.delete(item.id)
        with pytest.raises(MemoryNotFoundError):
            hm.get(item.id)

    def test_cross_link_survives_semantic_delete(self, hm):
        ev = hm.store_event(EventType.ACTION, data={"x": 1})
        fact = hm.store_fact("A", "b", "C")
        hm.cross_link(ev.id, fact.id)
        hm.delete(fact.id)
        assert len(hm.get_cross_links(ev.id)) == 0

    def test_multiple_subsystem_independence(self, hm):
        hm.store_event(EventType.ACTION, data={"x": 1})
        hm.store_fact("A", "b", "C")
        hm.delete(list(hm.facts._facts.keys())[0])
        assert hm.events.count == 1
        assert hm.facts.count == 0
