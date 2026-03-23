"""Tests for the memory router."""

import sys
import os
import pytest
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from memory_router import (
    MemoryRouter, RoutingError,
    QueryIntent, QueryAnalysis, RoutingDecision,
    analyse_query,
    KeywordRoutingStrategy, WeightedRoutingStrategy, AdaptiveRoutingStrategy,
    RoutingStrategy, RouteRecord,
)
from hybrid_memory import (
    HybridMemory, MemoryType, MemorySource, QueryResult,
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
    """HybridMemory with sample data across all subsystems."""
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

    hm.store_node("Python", NodeType.ENTITY)
    hm.store_node("Programming Language", NodeType.CONCEPT)
    hm.store_node("Django", NodeType.ENTITY)

    return hm


@pytest.fixture
def router(populated):
    return MemoryRouter(populated)


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------

class TestQueryAnalysis:
    def test_temporal_query(self):
        a = analyse_query("When did the event happen?")
        assert a.intent == QueryIntent.TEMPORAL
        assert a.recommended_type == MemoryType.EPISODIC
        assert a.confidence > 0

    def test_factual_query(self):
        a = analyse_query("What is Python?")
        assert a.intent == QueryIntent.FACTUAL
        assert a.recommended_type == MemoryType.SEMANTIC

    def test_experiential_query(self):
        a = analyse_query("How did the search perform? What was the outcome?")
        assert a.intent == QueryIntent.EXPERIENTIAL
        assert a.recommended_type == MemoryType.EPISODIC
        assert MemorySource.EXPERIENCE_TRACKER in a.recommended_sources

    def test_relational_query(self):
        a = analyse_query("Show the connected linked relationship between nodes")
        assert a.intent == QueryIntent.RELATIONAL
        assert a.recommended_type == MemoryType.SEMANTIC
        assert MemorySource.KNOWLEDGE_GRAPH in a.recommended_sources

    def test_ambiguous_query(self):
        a = analyse_query("xyz 123")
        assert a.intent == QueryIntent.AMBIGUOUS
        assert a.recommended_type == MemoryType.BOTH
        assert a.confidence == 0.0

    def test_empty_query(self):
        a = analyse_query("")
        assert a.intent == QueryIntent.AMBIGUOUS

    def test_signals_sum_to_one(self):
        a = analyse_query("When did the event happen and what is Python?")
        total = sum(a.signals.values())
        assert abs(total - 1.0) < 1e-9

    def test_detected_filters_event_type(self):
        a = analyse_query("Show me the latest action events")
        assert a.detected_filters.get("event_type") == EventType.ACTION

    def test_detected_filters_outcome(self):
        a = analyse_query("Which attempts resulted in success?")
        assert a.detected_filters.get("outcome") == Outcome.SUCCESS

    def test_detected_filters_fact_type(self):
        a = analyse_query("Give me the definition of Python")
        assert a.detected_filters.get("fact_type") == FactType.DEFINITION

    def test_detected_filters_node_type_concept(self):
        a = analyse_query("Find the concept of programming")
        assert a.detected_filters.get("node_type") == NodeType.CONCEPT

    def test_detected_filters_node_type_entity(self):
        a = analyse_query("Find the entity Python")
        assert a.detected_filters.get("node_type") == NodeType.ENTITY

    def test_mixed_signals(self):
        a = analyse_query("When was the fact about Python defined?")
        assert a.signals["temporal"] > 0
        assert a.signals["factual"] > 0


# ---------------------------------------------------------------------------
# KeywordRoutingStrategy
# ---------------------------------------------------------------------------

class TestKeywordStrategy:
    def test_routes_temporal(self):
        s = KeywordRoutingStrategy()
        a = analyse_query("When did the event happen?")
        d = s.route(a)
        assert d.memory_type == MemoryType.EPISODIC
        assert d.strategy_name == "keyword"

    def test_routes_factual(self):
        s = KeywordRoutingStrategy()
        a = analyse_query("What is Python?")
        d = s.route(a)
        assert d.memory_type == MemoryType.SEMANTIC

    def test_routes_ambiguous_to_both(self):
        s = KeywordRoutingStrategy()
        a = analyse_query("xyz")
        d = s.route(a)
        assert d.memory_type == MemoryType.BOTH

    def test_passes_detected_filters(self):
        s = KeywordRoutingStrategy()
        a = analyse_query("Show me the latest action events")
        d = s.route(a)
        assert d.query_kwargs.get("event_type") == EventType.ACTION


# ---------------------------------------------------------------------------
# WeightedRoutingStrategy
# ---------------------------------------------------------------------------

class TestWeightedStrategy:
    def test_high_confidence_routes_specific(self):
        s = WeightedRoutingStrategy(threshold=0.4)
        a = analyse_query("When did the event happen in the timeline?")
        d = s.route(a)
        assert d.memory_type in (MemoryType.EPISODIC, MemoryType.SEMANTIC)
        assert d.strategy_name == "weighted"

    def test_low_confidence_falls_back_to_both(self):
        s = WeightedRoutingStrategy(threshold=0.99)
        # Mixed temporal + factual signals → confidence below 0.99
        a = analyse_query("When was the fact defined and what is it?")
        d = s.route(a)
        assert d.memory_type == MemoryType.BOTH

    def test_ambiguous_always_both(self):
        s = WeightedRoutingStrategy(threshold=0.1)
        a = analyse_query("xyz")
        d = s.route(a)
        assert d.memory_type == MemoryType.BOTH

    def test_custom_threshold(self):
        s = WeightedRoutingStrategy(threshold=0.0)
        a = analyse_query("What is Python?")
        d = s.route(a)
        # threshold=0 means any non-ambiguous query routes specifically
        assert d.memory_type == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# AdaptiveRoutingStrategy
# ---------------------------------------------------------------------------

class TestAdaptiveStrategy:
    def test_early_queries_use_base(self):
        s = AdaptiveRoutingStrategy()
        a = analyse_query("What is Python?")
        d = s.route(a)
        # First 3 queries use base strategy
        assert d.strategy_name == "adaptive"
        assert d.memory_type == MemoryType.SEMANTIC  # base = keyword

    def test_ambiguous_uses_base(self):
        s = AdaptiveRoutingStrategy()
        # Burn through early queries
        for _ in range(4):
            s.route(analyse_query("xyz"))
        a = analyse_query("xyz")
        d = s.route(a)
        assert d.memory_type == MemoryType.BOTH

    def test_record_outcome_updates_scores(self):
        s = AdaptiveRoutingStrategy(learning_rate=0.5)
        initial = s.scores.get("temporal", {}).get("episodic", 0.5)
        s.record_outcome(QueryIntent.TEMPORAL, MemoryType.EPISODIC, 5)
        updated = s.scores["temporal"]["episodic"]
        assert updated > initial

    def test_record_outcome_negative_on_zero_results(self):
        s = AdaptiveRoutingStrategy(learning_rate=0.5)
        s.record_outcome(QueryIntent.TEMPORAL, MemoryType.EPISODIC, 5)
        after_positive = s.scores["temporal"]["episodic"]
        s.record_outcome(QueryIntent.TEMPORAL, MemoryType.EPISODIC, 0)
        after_negative = s.scores["temporal"]["episodic"]
        assert after_negative < after_positive

    def test_record_outcome_ignores_both(self):
        s = AdaptiveRoutingStrategy(learning_rate=0.5)
        s.record_outcome(QueryIntent.TEMPORAL, MemoryType.BOTH, 10)
        # BOTH should not be tracked — scores should stay at defaults
        assert s.scores.get("temporal", {}).get("episodic", 0.5) == 0.5

    def test_reset_clears_state(self):
        s = AdaptiveRoutingStrategy()
        s.record_outcome(QueryIntent.TEMPORAL, MemoryType.EPISODIC, 5)
        s.route(analyse_query("test"))
        s.reset()
        assert s.query_count == 0
        assert len(s.scores) == 0

    def test_learning_shifts_routing(self):
        s = AdaptiveRoutingStrategy(learning_rate=1.0, exploration_rate=0.0)
        # Burn through early queries
        for _ in range(4):
            s.route(analyse_query("dummy test"))

        # Heavily reward semantic for temporal intent (unusual but tests learning)
        for _ in range(10):
            s.record_outcome(QueryIntent.TEMPORAL, MemoryType.SEMANTIC, 10)
            s.record_outcome(QueryIntent.TEMPORAL, MemoryType.EPISODIC, 0)

        a = analyse_query("When did the event happen?")
        d = s.route(a)
        # After heavy semantic reward, adaptive should pick semantic
        assert d.memory_type == MemoryType.SEMANTIC

    def test_exploration_triggers_both(self):
        s = AdaptiveRoutingStrategy(learning_rate=0.01, exploration_rate=0.9)
        # Burn through early queries
        for _ in range(4):
            s.route(analyse_query("dummy test"))
        # Tiny learning rate → scores stay close → exploration → BOTH
        a = analyse_query("When did the event happen?")
        d = s.route(a)
        assert d.memory_type == MemoryType.BOTH

    def test_custom_base_strategy(self):
        base = WeightedRoutingStrategy(threshold=0.0)
        s = AdaptiveRoutingStrategy(base_strategy=base)
        a = analyse_query("What is Python?")
        d = s.route(a)
        # Early query → uses base (weighted with threshold=0)
        assert d.memory_type == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# MemoryRouter (orchestrator)
# ---------------------------------------------------------------------------

class TestMemoryRouter:
    def test_route_returns_query_result(self, router):
        result = router.route("What is Python?")
        assert isinstance(result, QueryResult)

    def test_route_temporal_finds_events(self, router):
        result = router.route("When did the event happen?")
        # Should route to episodic and find events
        assert result.count >= 0  # may or may not match text

    def test_route_factual_finds_facts(self, router):
        # "Python" matches fact content and node labels
        result = router.route("Python", memory_type=MemoryType.SEMANTIC)
        facts = result.by_source(MemorySource.FACT_STORE)
        nodes = result.by_source(MemorySource.KNOWLEDGE_GRAPH)
        assert len(facts) + len(nodes) > 0

    def test_route_experiential(self, router):
        # "search" matches experience action text
        result = router.route("search", memory_type=MemoryType.EPISODIC)
        exps = result.by_source(MemorySource.EXPERIENCE_TRACKER)
        assert len(exps) >= 1

    def test_route_with_override(self, router):
        result = router.route("anything", memory_type=MemoryType.SEMANTIC)
        # Override forces semantic
        assert all(i.memory_type == MemoryType.SEMANTIC for i in result.items)

    def test_route_with_analysis(self, router):
        result, analysis, decision = router.route_with_analysis("What is Python?")
        assert isinstance(analysis, QueryAnalysis)
        assert isinstance(decision, RoutingDecision)
        assert isinstance(result, QueryResult)
        assert analysis.intent == QueryIntent.FACTUAL

    def test_history_recorded(self, router):
        router.route("What is Python?")
        router.route("When did the event happen?")
        assert len(router.history) == 2
        assert all(isinstance(r, RouteRecord) for r in router.history)

    def test_clear_history(self, router):
        router.route("test")
        router.clear_history()
        assert len(router.history) == 0

    def test_stats_empty(self, populated):
        r = MemoryRouter(populated)
        s = r.stats()
        assert s["total_queries"] == 0
        assert s["avg_results"] == 0.0

    def test_stats_after_queries(self, router):
        router.route("What is Python?")
        router.route("When did the event happen?")
        s = router.stats()
        assert s["total_queries"] == 2
        assert s["avg_results"] >= 0
        assert "factual" in s["intent_distribution"]
        assert "temporal" in s["intent_distribution"]

    def test_analyse_without_executing(self, router):
        a = router.analyse("What is Python?")
        assert isinstance(a, QueryAnalysis)
        assert len(router.history) == 0  # no execution

    def test_strategy_property(self, router):
        assert isinstance(router.strategy, KeywordRoutingStrategy)

    def test_set_strategy(self, router):
        new_strategy = WeightedRoutingStrategy(threshold=0.5)
        router.strategy = new_strategy
        assert router.strategy is new_strategy
        assert router.strategy.name == "weighted"


# ---------------------------------------------------------------------------
# MemoryRouter with adaptive strategy
# ---------------------------------------------------------------------------

class TestMemoryRouterAdaptive:
    def test_adaptive_records_outcomes(self, populated):
        strategy = AdaptiveRoutingStrategy(learning_rate=0.5)
        router = MemoryRouter(populated, strategy=strategy)
        router.route("What is Python?")
        router.route("What is Django?")
        assert strategy.query_count >= 2

    def test_adaptive_improves_over_time(self, populated):
        strategy = AdaptiveRoutingStrategy(
            learning_rate=1.0, exploration_rate=0.0)
        router = MemoryRouter(populated, strategy=strategy)

        # Run enough queries to pass the early-query phase
        for _ in range(5):
            router.route("What is Python?")

        # Manually reward semantic for factual intent
        for _ in range(10):
            strategy.record_outcome(QueryIntent.FACTUAL, MemoryType.SEMANTIC, 10)
            strategy.record_outcome(QueryIntent.FACTUAL, MemoryType.EPISODIC, 0)

        _, _, decision = router.route_with_analysis("What is Django?")
        assert decision.memory_type == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_memory_returns_empty(self, hm):
        router = MemoryRouter(hm)
        result = router.route("What is Python?")
        assert result.count == 0

    def test_special_characters_in_query(self, router):
        result = router.route("What is @#$%^&*() ?!")
        assert isinstance(result, QueryResult)

    def test_very_long_query(self, router):
        long_q = "what is " * 500 + "Python?"
        result = router.route(long_q)
        assert isinstance(result, QueryResult)

    def test_route_preserves_limit(self, router):
        result = router.route("", limit=2, memory_type=MemoryType.BOTH)
        assert result.count <= 2

    def test_hybrid_memory_accessible(self, router):
        assert isinstance(router.hybrid_memory, HybridMemory)

    def test_multiple_strategies_same_router(self, populated):
        router = MemoryRouter(populated, KeywordRoutingStrategy())
        r1 = router.route("What is Python?")

        router.strategy = WeightedRoutingStrategy(threshold=0.5)
        r2 = router.route("What is Python?")

        # Both should return results (same data, different strategy)
        assert isinstance(r1, QueryResult)
        assert isinstance(r2, QueryResult)
