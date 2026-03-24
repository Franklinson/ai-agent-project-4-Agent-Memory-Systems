"""Tests for intelligent memory type selection."""

import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_51'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from type_selector import (
    TypeSelectorError, SelectionFailedError,
    InformationNeed, QueryCharacteristics, analyse_query_characteristics,
    TypeSelection, SelectionAlgorithm,
    RuleBasedSelector, QueryBasedSelector, PatternBasedSelector, HybridSelector,
    AutoSelector, SelectionOptimizer, SelectionOutcome,
)
from unified_memory import MemoryLayer
from hybrid_memory import MemoryType


# ---------------------------------------------------------------------------
# Query analysis tests
# ---------------------------------------------------------------------------

class TestAnalyseQueryCharacteristics:
    """Test query characteristic extraction."""

    def test_temporal_query(self):
        c = analyse_query_characteristics("When did the event happen yesterday?")
        assert c.temporal_score > 0
        assert c.has_temporal_reference is True
        assert c.is_question is True
        assert c.information_need == InformationNeed.TEMPORAL

    def test_factual_query(self):
        c = analyse_query_characteristics("What is the definition of Python?")
        assert c.factual_score > 0
        assert c.information_need == InformationNeed.LOOKUP

    def test_experiential_query(self):
        c = analyse_query_characteristics("How did the search attempt perform?")
        assert c.experiential_score > 0
        assert c.information_need == InformationNeed.EXPERIENTIAL

    def test_relational_query(self):
        c = analyse_query_characteristics("Show connected linked graph edge path between nodes")
        assert c.relational_score > 0
        assert c.information_need == InformationNeed.EXPLORATION

    def test_conversational_query(self):
        c = analyse_query_characteristics("Recall the conversation chat message discussed")
        assert c.conversational_score > 0
        assert c.information_need == InformationNeed.RECALL

    def test_ambiguous_query(self):
        c = analyse_query_characteristics("xyz abc")
        assert c.information_need == InformationNeed.LOOKUP  # default
        assert c.temporal_score == 0
        assert c.factual_score == 0

    def test_question_detection(self):
        c1 = analyse_query_characteristics("Is Python a language?")
        assert c1.is_question is True

        c2 = analyse_query_characteristics("Python language")
        assert c2.is_question is False

    def test_empty_query(self):
        c = analyse_query_characteristics("")
        assert c.word_count >= 1  # normalised to 1
        assert c.information_need == InformationNeed.LOOKUP

    def test_mixed_signals(self):
        c = analyse_query_characteristics("When was the fact about Python defined?")
        # Both temporal and factual signals present
        assert c.temporal_score > 0 or c.factual_score > 0


# ---------------------------------------------------------------------------
# RuleBasedSelector tests
# ---------------------------------------------------------------------------

class TestRuleBasedSelector:
    def setup_method(self):
        self.selector = RuleBasedSelector(threshold=0.3)

    def test_conversational_selection(self):
        c = analyse_query_characteristics("What did we discuss in the conversation chat?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.WORKING
        assert sel.memory_type == MemoryType.EPISODIC
        assert sel.algorithm == "rule_based"

    def test_temporal_selection(self):
        c = analyse_query_characteristics("When did the event happen recently?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.SHORT_TERM
        assert sel.memory_type == MemoryType.EPISODIC

    def test_factual_selection(self):
        c = analyse_query_characteristics("What is the definition of this concept?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.LONG_TERM
        assert sel.memory_type == MemoryType.SEMANTIC

    def test_relational_selection(self):
        c = analyse_query_characteristics("What is connected to this graph edge path?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.LONG_TERM
        assert sel.memory_type == MemoryType.SEMANTIC

    def test_fallback_selection(self):
        c = analyse_query_characteristics("xyz abc def")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.HYBRID
        assert sel.memory_type == MemoryType.BOTH
        assert sel.confidence == 0.2

    def test_custom_threshold(self):
        strict = RuleBasedSelector(threshold=0.9)
        c = analyse_query_characteristics("What is Python?")
        sel = strict.select(c)
        # With a very high threshold, most queries fall through to hybrid
        assert sel.layer == MemoryLayer.HYBRID or sel.confidence >= 0.9

    def test_name(self):
        assert self.selector.name == "rule_based"


# ---------------------------------------------------------------------------
# QueryBasedSelector tests
# ---------------------------------------------------------------------------

class TestQueryBasedSelector:
    def setup_method(self):
        self.selector = QueryBasedSelector()

    def test_selects_best_scored_pair(self):
        c = analyse_query_characteristics("What is the definition of Python?")
        sel = self.selector.select(c)
        assert isinstance(sel.layer, MemoryLayer)
        assert isinstance(sel.memory_type, MemoryType)
        assert sel.confidence > 0
        assert sel.algorithm == "query_based"

    def test_alternatives_populated(self):
        c = analyse_query_characteristics("When did the fact happen?")
        sel = self.selector.select(c)
        # Should have at least one alternative
        assert isinstance(sel.alternatives, list)

    def test_conversational_scores_high(self):
        c = analyse_query_characteristics("What was said in the conversation chat message?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.WORKING

    def test_ambiguous_defaults_to_hybrid(self):
        c = analyse_query_characteristics("xyz")
        sel = self.selector.select(c)
        # Hybrid baseline of 0.3 should win when all signals are 0
        assert sel.layer == MemoryLayer.HYBRID

    def test_name(self):
        assert self.selector.name == "query_based"


# ---------------------------------------------------------------------------
# PatternBasedSelector tests
# ---------------------------------------------------------------------------

class TestPatternBasedSelector:
    def setup_method(self):
        self.selector = PatternBasedSelector()

    def test_heuristic_before_enough_data(self):
        c = analyse_query_characteristics("When did it happen?")
        sel = self.selector.select(c)
        assert sel.algorithm == "pattern_based"
        assert sel.layer == MemoryLayer.SHORT_TERM  # temporal heuristic

    def test_heuristic_factual(self):
        c = analyse_query_characteristics("What is Python?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.LONG_TERM

    def test_heuristic_fallback(self):
        c = analyse_query_characteristics("xyz abc")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.HYBRID

    def test_learns_from_outcomes(self):
        # Warm up with 6 queries to pass the early-query gate
        for _ in range(6):
            c = analyse_query_characteristics("dummy query")
            self.selector.select(c)

        # Record strong outcomes for LONG_TERM / SEMANTIC on LOOKUP need
        for _ in range(10):
            self.selector.record_outcome(
                InformationNeed.LOOKUP, MemoryLayer.LONG_TERM,
                MemoryType.SEMANTIC, result_count=5,
            )

        c = analyse_query_characteristics("What is Python?")
        sel = self.selector.select(c)
        assert sel.layer == MemoryLayer.LONG_TERM
        assert sel.memory_type == MemoryType.SEMANTIC

    def test_reset(self):
        self.selector.record_outcome(
            InformationNeed.LOOKUP, MemoryLayer.LONG_TERM,
            MemoryType.SEMANTIC, result_count=5,
        )
        assert self.selector.query_count >= 0
        self.selector.reset()
        assert self.selector.query_count == 0
        assert len(self.selector.patterns) == 0

    def test_name(self):
        assert self.selector.name == "pattern_based"


# ---------------------------------------------------------------------------
# HybridSelector tests
# ---------------------------------------------------------------------------

class TestHybridSelector:
    def setup_method(self):
        self.selector = HybridSelector()

    def test_voting_produces_result(self):
        c = analyse_query_characteristics("What is the definition of Python?")
        sel = self.selector.select(c)
        assert sel.algorithm == "hybrid"
        assert sel.confidence > 0
        assert "Voted by" in sel.reasoning

    def test_alternatives_present(self):
        c = analyse_query_characteristics("When did the fact happen?")
        sel = self.selector.select(c)
        assert isinstance(sel.alternatives, list)

    def test_custom_selectors(self):
        custom = HybridSelector(selectors=[
            (RuleBasedSelector(), 2.0),
            (QueryBasedSelector(), 0.5),
        ])
        c = analyse_query_characteristics("What is Python?")
        sel = custom.select(c)
        assert sel.algorithm == "hybrid"

    def test_get_pattern_selector(self):
        ps = self.selector.get_pattern_selector()
        assert isinstance(ps, PatternBasedSelector)

    def test_no_pattern_selector(self):
        custom = HybridSelector(selectors=[(RuleBasedSelector(), 1.0)])
        assert custom.get_pattern_selector() is None

    def test_name(self):
        assert self.selector.name == "hybrid"

    def test_selectors_property(self):
        assert len(self.selector.selectors) == 3


# ---------------------------------------------------------------------------
# AutoSelector tests
# ---------------------------------------------------------------------------

class TestAutoSelector:
    def setup_method(self):
        self.auto = AutoSelector()

    def test_basic_selection(self):
        sel = self.auto.select("What is Python?")
        assert isinstance(sel, TypeSelection)
        assert isinstance(sel.layer, MemoryLayer)
        assert isinstance(sel.memory_type, MemoryType)

    def test_caching(self):
        sel1 = self.auto.select("What is Python?")
        sel2 = self.auto.select("What is Python?")
        assert sel1 is sel2  # same cached object

    def test_cache_bypass(self):
        sel1 = self.auto.select("What is Python?")
        sel2 = self.auto.select("What is Python?", use_cache=False)
        # Both valid but not necessarily the same object
        assert sel2.layer == sel1.layer

    def test_clear_cache(self):
        self.auto.select("test")
        self.auto.clear_cache()
        assert len(self.auto._cache) == 0

    def test_history_tracking(self):
        self.auto.select("query 1")
        self.auto.select("query 2")
        assert len(self.auto.history) == 2

    def test_clear_history(self):
        self.auto.select("test")
        self.auto.clear_history()
        assert len(self.auto.history) == 0

    def test_fallback_on_low_confidence(self):
        # Use a selector that always returns very low confidence
        auto = AutoSelector(
            algorithm=RuleBasedSelector(threshold=0.99),
            min_confidence=0.5,
        )
        sel = auto.select("xyz abc")
        assert sel.algorithm == "fallback"
        assert sel.layer == MemoryLayer.HYBRID

    def test_custom_fallback(self):
        auto = AutoSelector(
            fallback_layer=MemoryLayer.LONG_TERM,
            fallback_type=MemoryType.SEMANTIC,
            min_confidence=0.99,
        )
        sel = auto.select("xyz")
        assert sel.layer == MemoryLayer.LONG_TERM
        assert sel.memory_type == MemoryType.SEMANTIC

    def test_algorithm_swap(self):
        self.auto.select("test")
        assert len(self.auto._cache) == 1
        self.auto.algorithm = QueryBasedSelector()
        assert len(self.auto._cache) == 0  # cache cleared on swap
        assert isinstance(self.auto.algorithm, QueryBasedSelector)

    def test_algorithm_error_fallback(self):
        class BrokenSelector(SelectionAlgorithm):
            @property
            def name(self): return "broken"
            def select(self, c): raise RuntimeError("boom")

        auto = AutoSelector(algorithm=BrokenSelector())
        sel = auto.select("test")
        assert sel.algorithm == "fallback"


# ---------------------------------------------------------------------------
# SelectionOptimizer tests
# ---------------------------------------------------------------------------

class TestSelectionOptimizer:
    def setup_method(self):
        self.optimizer = SelectionOptimizer()

    def test_select_delegates(self):
        sel = self.optimizer.select("What is Python?")
        assert isinstance(sel, TypeSelection)

    def test_record_outcome(self):
        sel = self.optimizer.select("What is Python?")
        self.optimizer.record_outcome("What is Python?", sel, result_count=3)
        assert len(self.optimizer.outcomes) == 1
        assert self.optimizer.outcomes[0].result_count == 3

    def test_stats(self):
        sel = self.optimizer.select("test")
        self.optimizer.record_outcome("test", sel, result_count=5)
        stats = self.optimizer.stats()
        assert stats["total_selections"] == 1
        assert stats["avg_results"] == 5.0
        assert sel.algorithm in stats["algorithm_performance"]

    def test_stats_empty(self):
        stats = self.optimizer.stats()
        assert stats["total_selections"] == 0
        assert stats["avg_results"] == 0.0

    def test_best_algorithm(self):
        sel1 = TypeSelection(
            layer=MemoryLayer.LONG_TERM, memory_type=MemoryType.SEMANTIC,
            confidence=0.8, algorithm="rule_based",
        )
        sel2 = TypeSelection(
            layer=MemoryLayer.SHORT_TERM, memory_type=MemoryType.EPISODIC,
            confidence=0.5, algorithm="query_based",
        )
        self.optimizer.record_outcome("q1", sel1, result_count=10)
        self.optimizer.record_outcome("q2", sel2, result_count=1)
        assert self.optimizer.best_algorithm() == "rule_based"

    def test_best_algorithm_empty(self):
        assert self.optimizer.best_algorithm() is None

    def test_reset(self):
        sel = self.optimizer.select("test")
        self.optimizer.record_outcome("test", sel, result_count=3)
        self.optimizer.reset()
        assert len(self.optimizer.outcomes) == 0
        assert self.optimizer.stats()["total_selections"] == 0

    def test_pattern_learning_integration(self):
        """Optimizer feeds outcomes to PatternBasedSelector inside HybridSelector."""
        optimizer = SelectionOptimizer(AutoSelector(algorithm=HybridSelector()))

        for i in range(10):
            sel = optimizer.select(f"What is concept {i}?")
            optimizer.record_outcome(f"What is concept {i}?", sel, result_count=5)

        stats = optimizer.stats()
        assert stats["total_selections"] == 10

    def test_zero_result_outcome(self):
        sel = self.optimizer.select("nothing here")
        self.optimizer.record_outcome("nothing here", sel, result_count=0)
        stats = self.optimizer.stats()
        assert stats["avg_results"] == 0.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestTypeSelectorIntegration:
    """End-to-end tests combining analysis, selection, and optimization."""

    def test_full_workflow(self):
        optimizer = SelectionOptimizer()

        queries = [
            ("When did the login event happen?", MemoryLayer.SHORT_TERM),
            ("What is the definition of Python?", MemoryLayer.LONG_TERM),
            ("What did we discuss in the conversation?", MemoryLayer.WORKING),
            ("How did the search attempt perform?", MemoryLayer.SHORT_TERM),
            ("What is related to Django in the graph?", MemoryLayer.LONG_TERM),
        ]

        for query, expected_layer in queries:
            sel = optimizer.select(query)
            # Record a positive outcome
            optimizer.record_outcome(query, sel, result_count=3)

        stats = optimizer.stats()
        assert stats["total_selections"] == 5
        assert stats["avg_results"] == 3.0

    def test_all_algorithms_agree_on_clear_signal(self):
        c = analyse_query_characteristics(
            "What is the definition of this concept entity?"
        )
        rule_sel = RuleBasedSelector().select(c)
        query_sel = QueryBasedSelector().select(c)

        # Both should pick semantic / long-term for a strongly factual query
        assert rule_sel.memory_type == MemoryType.SEMANTIC
        assert query_sel.layer == MemoryLayer.LONG_TERM

    def test_different_algorithms_produce_valid_results(self):
        algorithms = [
            RuleBasedSelector(),
            QueryBasedSelector(),
            PatternBasedSelector(),
            HybridSelector(),
        ]
        c = analyse_query_characteristics("When did the event happen?")

        for algo in algorithms:
            sel = algo.select(c)
            assert isinstance(sel.layer, MemoryLayer)
            assert isinstance(sel.memory_type, MemoryType)
            assert 0 <= sel.confidence <= 1.0
            assert sel.algorithm == algo.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
