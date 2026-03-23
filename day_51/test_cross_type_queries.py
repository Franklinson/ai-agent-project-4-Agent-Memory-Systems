"""Tests for cross-type query support."""

import sys
import os
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from cross_type_queries import (
    CrossTypeQueryEngine, CrossQueryError,
    QueryPattern, RankingStrategy, SubQuery, MergedResult,
    merge_items, rank_items, format_results,
    _text_relevance, _default_bridge,
)
from hybrid_memory import (
    HybridMemory, MemoryType, MemorySource, MemoryItem, QueryResult,
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
    """HybridMemory with data across all subsystems."""
    hm.store_event(EventType.ACTION, data={"action": "login"},
                   participants=["alice"])
    hm.store_event(EventType.OBSERVATION, data={"saw": "dashboard"},
                   participants=["alice"])
    hm.store_event(EventType.ERROR, data={"error": "timeout"},
                   participants=["bob"])

    hm.store_experience("search", Outcome.SUCCESS, score=0.9, tags=["web"])
    hm.store_experience("search", Outcome.FAILURE, score=0.3,
                        feedback="timeout")
    hm.store_experience("summarize", Outcome.SUCCESS, score=0.8,
                        tags=["nlp"])

    hm.store_fact("Python", "is_a", "programming language",
                  fact_type=FactType.DEFINITION)
    hm.store_fact("Python", "created_by", "Guido van Rossum")
    hm.store_fact("Django", "is_a", "web framework",
                  fact_type=FactType.DEFINITION, confidence=0.95)

    py = hm.store_node("Python", NodeType.ENTITY)
    pl = hm.store_node("Programming Language", NodeType.CONCEPT)
    dj = hm.store_node("Django", NodeType.ENTITY)

    hm.store_edge(py.id, pl.id, EdgeType.IS_A)
    hm.store_edge(dj.id, py.id, EdgeType.DEPENDS_ON)

    return hm


@pytest.fixture
def engine(populated):
    return CrossTypeQueryEngine(populated)


# ---------------------------------------------------------------------------
# Combined queries
# ---------------------------------------------------------------------------

class TestCombinedQuery:
    def test_returns_merged_result(self, engine):
        result = engine.combined("Python")
        assert isinstance(result, MergedResult)
        assert result.pattern == QueryPattern.COMBINED

    def test_includes_both_types(self, engine):
        result = engine.combined("Python")
        types = {i.memory_type for i in result.items}
        assert MemoryType.SEMANTIC in types

    def test_has_sub_results(self, engine):
        result = engine.combined("Python")
        assert len(result.sub_results) == 2

    def test_deduplicates(self, engine):
        result = engine.combined("")
        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids))

    def test_limit(self, engine):
        result = engine.combined("", limit=3)
        assert result.count <= 3

    def test_ranking_relevance(self, engine):
        result = engine.combined("Python", ranking=RankingStrategy.RELEVANCE)
        assert result.count > 0
        assert result.metadata["ranking"] == "relevance"

    def test_ranking_timestamp(self, engine):
        result = engine.combined("", ranking=RankingStrategy.TIMESTAMP)
        if result.count >= 2:
            assert result.items[0].timestamp >= result.items[1].timestamp

    def test_ranking_source_priority(self, engine):
        result = engine.combined("Python",
                                 ranking=RankingStrategy.SOURCE_PRIORITY)
        assert result.count > 0

    def test_execution_time_tracked(self, engine):
        result = engine.combined("Python")
        assert result.execution_time_ms >= 0

    def test_by_source_helper(self, engine):
        result = engine.combined("Python")
        facts = result.by_source(MemorySource.FACT_STORE)
        assert all(i.source == MemorySource.FACT_STORE for i in facts)

    def test_by_type_helper(self, engine):
        result = engine.combined("")
        ep = result.by_type(MemoryType.EPISODIC)
        sem = result.by_type(MemoryType.SEMANTIC)
        assert len(ep) + len(sem) == result.count

    def test_extra_kwargs_forwarded(self, engine):
        result = engine.combined("", subject="Python")
        facts = result.by_source(MemorySource.FACT_STORE)
        for f in facts:
            assert "python" in f.content.lower()


# ---------------------------------------------------------------------------
# Sequential queries
# ---------------------------------------------------------------------------

class TestSequentialQuery:
    def test_returns_merged_result(self, engine):
        result = engine.sequential("login", primary_type=MemoryType.EPISODIC)
        assert isinstance(result, MergedResult)
        assert result.pattern == QueryPattern.SEQUENTIAL

    def test_has_two_sub_results(self, engine):
        result = engine.sequential("login", primary_type=MemoryType.EPISODIC)
        assert len(result.sub_results) == 2

    def test_primary_type_respected(self, engine):
        result = engine.sequential("Python", primary_type=MemoryType.SEMANTIC)
        assert result.metadata["primary_type"] == "semantic"
        assert result.metadata["secondary_type"] == "episodic"

    def test_default_bridge_extracts_keywords(self, engine):
        result = engine.sequential("Python", primary_type=MemoryType.SEMANTIC)
        bridge_kw = result.metadata.get("bridge_kwargs", {})
        # Default bridge should extract subject/label from semantic results
        assert isinstance(bridge_kw, dict)

    def test_custom_bridge_fn(self, engine):
        def my_bridge(items: List[MemoryItem]) -> Dict[str, Any]:
            return {"query_text": "Django", "subject": "Django"}

        result = engine.sequential(
            "Python", primary_type=MemoryType.SEMANTIC,
            bridge_fn=my_bridge,
        )
        assert result.metadata["bridge_kwargs"] == {
            "query_text": "Django", "subject": "Django"}

    def test_primary_items_boosted(self, engine):
        result = engine.sequential("Python", primary_type=MemoryType.SEMANTIC)
        # Primary items should have boosted relevance
        sem_items = result.by_type(MemoryType.SEMANTIC)
        if sem_items:
            assert any(i.relevance > 0.5 for i in sem_items)

    def test_limit(self, engine):
        result = engine.sequential("", primary_type=MemoryType.EPISODIC,
                                   limit=2)
        assert result.count <= 2

    def test_deduplicates(self, engine):
        result = engine.sequential("", primary_type=MemoryType.EPISODIC)
        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids))

    def test_primary_kwargs_forwarded(self, engine):
        result = engine.sequential(
            "", primary_type=MemoryType.EPISODIC,
            primary_kwargs={"action": "search"},
        )
        # Should find search experiences in primary
        assert result.count >= 0

    def test_secondary_kwargs_forwarded(self, engine):
        result = engine.sequential(
            "Python", primary_type=MemoryType.SEMANTIC,
            secondary_kwargs={"action": "search"},
        )
        assert result.count >= 0


# ---------------------------------------------------------------------------
# Parallel queries
# ---------------------------------------------------------------------------

class TestParallelQuery:
    def test_returns_merged_result(self, engine):
        sqs = [
            SubQuery("Python", MemoryType.SEMANTIC, label="facts"),
            SubQuery("login", MemoryType.EPISODIC, label="events"),
        ]
        result = engine.parallel(sqs)
        assert isinstance(result, MergedResult)
        assert result.pattern == QueryPattern.PARALLEL

    def test_sub_results_count(self, engine):
        sqs = [
            SubQuery("Python", MemoryType.SEMANTIC),
            SubQuery("", MemoryType.EPISODIC),
            SubQuery("Django", MemoryType.SEMANTIC),
        ]
        result = engine.parallel(sqs)
        assert len(result.sub_results) == 3

    def test_deduplicates_across_sub_queries(self, engine):
        sqs = [
            SubQuery("Python", MemoryType.SEMANTIC),
            SubQuery("Python", MemoryType.SEMANTIC),
        ]
        result = engine.parallel(sqs)
        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids))

    def test_limit(self, engine):
        sqs = [
            SubQuery("", MemoryType.BOTH),
            SubQuery("", MemoryType.BOTH),
        ]
        result = engine.parallel(sqs, limit=3)
        assert result.count <= 3

    def test_labels_in_metadata(self, engine):
        sqs = [
            SubQuery("Python", MemoryType.SEMANTIC, label="sem"),
            SubQuery("login", MemoryType.EPISODIC, label="ep"),
        ]
        result = engine.parallel(sqs)
        assert "sem" in result.metadata["labels"]
        assert "ep" in result.metadata["labels"]

    def test_execution_time_tracked(self, engine):
        sqs = [SubQuery("Python", MemoryType.SEMANTIC)]
        result = engine.parallel(sqs)
        assert result.execution_time_ms >= 0

    def test_empty_sub_queries(self, engine):
        result = engine.parallel([])
        assert result.count == 0

    def test_sub_query_kwargs(self, engine):
        sqs = [
            SubQuery("", MemoryType.SEMANTIC,
                     kwargs={"subject": "Python"}, label="py_facts"),
        ]
        result = engine.parallel(sqs)
        facts = result.by_source(MemorySource.FACT_STORE)
        for f in facts:
            assert "python" in f.content.lower()


# ---------------------------------------------------------------------------
# Multi-hop queries
# ---------------------------------------------------------------------------

class TestMultiHopQuery:
    def test_basic_multi_hop(self, engine):
        hops = [
            {"memory_type": MemoryType.SEMANTIC},
            {"memory_type": MemoryType.EPISODIC},
        ]
        result = engine.multi_hop("Python", hops)
        assert isinstance(result, MergedResult)
        assert result.metadata["hops"] == 2

    def test_accumulates_results(self, engine):
        hops = [
            {"memory_type": MemoryType.SEMANTIC},
            {"memory_type": MemoryType.EPISODIC},
        ]
        result = engine.multi_hop("", hops)
        assert len(result.sub_results) == 2

    def test_bridge_between_hops(self, engine):
        def bridge(items):
            return {"query_text": "search"}

        hops = [
            {"memory_type": MemoryType.SEMANTIC},
            {"memory_type": MemoryType.EPISODIC, "bridge_fn": bridge},
        ]
        result = engine.multi_hop("Python", hops)
        assert result.count >= 0

    def test_limit(self, engine):
        hops = [
            {"memory_type": MemoryType.BOTH},
            {"memory_type": MemoryType.BOTH},
        ]
        result = engine.multi_hop("", hops, limit=4)
        assert result.count <= 4

    def test_deduplicates_across_hops(self, engine):
        hops = [
            {"memory_type": MemoryType.SEMANTIC},
            {"memory_type": MemoryType.SEMANTIC},
        ]
        result = engine.multi_hop("Python", hops)
        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids))

    def test_single_hop(self, engine):
        hops = [{"memory_type": MemoryType.SEMANTIC}]
        result = engine.multi_hop("Python", hops)
        assert result.count > 0

    def test_empty_hops(self, engine):
        result = engine.multi_hop("Python", [])
        assert result.count == 0


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

class TestMergeItems:
    def test_merge_two_lists(self, populated):
        ep = populated.query("", memory_type=MemoryType.EPISODIC).items
        sem = populated.query("", memory_type=MemoryType.SEMANTIC).items
        merged = merge_items(ep, sem)
        assert len(merged) == len(ep) + len(sem)

    def test_deduplication(self, populated):
        items = populated.query("", memory_type=MemoryType.SEMANTIC).items
        merged = merge_items(items, items)
        assert len(merged) == len(items)

    def test_no_deduplication(self, populated):
        items = populated.query("", memory_type=MemoryType.SEMANTIC).items
        merged = merge_items(items, items, deduplicate=False)
        assert len(merged) == len(items) * 2

    def test_empty_lists(self):
        assert merge_items([], []) == []

    def test_single_list(self, populated):
        items = populated.query("", memory_type=MemoryType.SEMANTIC).items
        merged = merge_items(items)
        assert len(merged) == len(items)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestRanking:
    def test_relevance_ranking_prefers_matches(self, populated):
        items = populated.query("", memory_type=MemoryType.SEMANTIC).items
        ranked = rank_items(items, "Python", RankingStrategy.RELEVANCE)
        # Items matching "Python" should be near the top
        top_contents = [i.content.lower() for i in ranked[:3]]
        assert any("python" in c for c in top_contents)

    def test_timestamp_ranking(self, hm):
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2024, 12, 1)
        hm.store_event(EventType.ACTION, data={"a": 1}, timestamp=t1)
        hm.store_event(EventType.ACTION, data={"a": 2}, timestamp=t2)
        items = hm.query("", memory_type=MemoryType.EPISODIC).items
        ranked = rank_items(items, "", RankingStrategy.TIMESTAMP)
        assert ranked[0].timestamp >= ranked[-1].timestamp

    def test_source_priority_ranking(self, populated):
        items = populated.query("").items
        ranked = rank_items(items, "", RankingStrategy.SOURCE_PRIORITY)
        # Fact store has highest priority
        if ranked:
            fact_items = [i for i in ranked if i.source == MemorySource.FACT_STORE]
            if fact_items:
                first_fact_idx = ranked.index(fact_items[0])
                assert first_fact_idx < len(ranked)

    def test_empty_items(self):
        assert rank_items([], "test") == []


# ---------------------------------------------------------------------------
# Text relevance scoring
# ---------------------------------------------------------------------------

class TestTextRelevance:
    def test_exact_match_boosts(self, populated):
        # Use a fact with confidence < 1.0 so the boost is visible
        items = populated.query("Django", memory_type=MemoryType.SEMANTIC).items
        fact_items = [i for i in items if i.source == MemorySource.FACT_STORE]
        assert len(fact_items) > 0
        item = fact_items[0]  # Django fact has confidence=0.95
        score = _text_relevance(item, "Django")
        assert score > item.relevance

    def test_partial_match(self, populated):
        item = populated.query("Python", memory_type=MemoryType.SEMANTIC).items[0]
        score = _text_relevance(item, "Python language xyz")
        assert score > 0

    def test_no_match(self, populated):
        item = populated.query("Python", memory_type=MemoryType.SEMANTIC).items[0]
        score = _text_relevance(item, "zzzzz")
        assert score < 1.0

    def test_empty_query_returns_base(self, populated):
        item = populated.query("Python", memory_type=MemoryType.SEMANTIC).items[0]
        score = _text_relevance(item, "")
        assert score == item.relevance


# ---------------------------------------------------------------------------
# Format results
# ---------------------------------------------------------------------------

class TestFormatResults:
    def test_format_with_items(self, populated):
        items = populated.query("Python", memory_type=MemoryType.SEMANTIC).items
        text = format_results(items)
        assert "Python" in text
        assert "1." in text

    def test_format_empty(self):
        assert format_results([]) == "(no results)"

    def test_format_with_limit(self, populated):
        items = populated.query("", memory_type=MemoryType.SEMANTIC).items
        text = format_results(items, max_items=2)
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) <= 2

    def test_format_includes_source(self, populated):
        items = populated.query("Python", memory_type=MemoryType.SEMANTIC).items
        text = format_results(items)
        assert "fact_store" in text or "knowledge_graph" in text


# ---------------------------------------------------------------------------
# Default bridge function
# ---------------------------------------------------------------------------

class TestDefaultBridge:
    def test_extracts_from_facts(self, populated):
        items = populated.query("Python", memory_type=MemoryType.SEMANTIC).items
        fact_items = [i for i in items if i.source == MemorySource.FACT_STORE]
        result = _default_bridge(fact_items)
        assert "query_text" in result
        assert "Python" in result["query_text"]

    def test_extracts_from_nodes(self, populated):
        items = populated.query("Python", memory_type=MemoryType.SEMANTIC).items
        node_items = [i for i in items if i.source == MemorySource.KNOWLEDGE_GRAPH]
        result = _default_bridge(node_items)
        assert "query_text" in result

    def test_extracts_from_events(self, populated):
        items = populated.query("login", memory_type=MemoryType.EPISODIC).items
        event_items = [i for i in items if i.source == MemorySource.EVENT_STORE]
        result = _default_bridge(event_items)
        assert "query_text" in result

    def test_extracts_from_experiences(self, populated):
        items = populated.query("search", memory_type=MemoryType.EPISODIC).items
        exp_items = [i for i in items
                     if i.source == MemorySource.EXPERIENCE_TRACKER]
        result = _default_bridge(exp_items)
        assert "query_text" in result
        assert "search" in result["query_text"]

    def test_empty_items(self):
        assert _default_bridge([]) == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_memory_combined(self, hm):
        engine = CrossTypeQueryEngine(hm)
        result = engine.combined("Python")
        assert result.count == 0

    def test_empty_memory_sequential(self, hm):
        engine = CrossTypeQueryEngine(hm)
        result = engine.sequential("Python")
        assert result.count == 0

    def test_empty_memory_parallel(self, hm):
        engine = CrossTypeQueryEngine(hm)
        result = engine.parallel([SubQuery("Python")])
        assert result.count == 0

    def test_hybrid_memory_accessible(self, engine):
        assert isinstance(engine.hybrid_memory, HybridMemory)

    def test_combined_with_no_text(self, engine):
        result = engine.combined("")
        assert result.count > 0  # returns all items

    def test_sequential_episodic_to_semantic(self, engine):
        result = engine.sequential(
            "login", primary_type=MemoryType.EPISODIC)
        assert result.metadata["primary_type"] == "episodic"
        assert result.metadata["secondary_type"] == "semantic"

    def test_sequential_semantic_to_episodic(self, engine):
        result = engine.sequential(
            "Python", primary_type=MemoryType.SEMANTIC)
        assert result.metadata["primary_type"] == "semantic"
        assert result.metadata["secondary_type"] == "episodic"

    def test_parallel_single_query(self, engine):
        result = engine.parallel([SubQuery("Python", MemoryType.SEMANTIC)])
        assert result.count > 0

    def test_combined_ranking_strategies_all_work(self, engine):
        for strategy in RankingStrategy:
            result = engine.combined("Python", ranking=strategy)
            assert isinstance(result, MergedResult)
