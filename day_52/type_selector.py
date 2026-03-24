"""Intelligent memory type selection for unified memory queries.

Analyses query characteristics (temporal signals, factual signals, information
needs, patterns) and selects the optimal MemoryLayer and MemoryType using one
of several pluggable algorithms:

- RuleBasedSelector:    deterministic keyword/signal threshold rules
- QueryBasedSelector:   scores each type against the query and picks the best
- PatternBasedSelector: learns from past query→result patterns
- HybridSelector:       combines rule, query, and pattern selectors via voting

AutoSelector wraps any algorithm with fallback and caching.
SelectionOptimizer tracks outcomes and tunes selector weights over time.
"""

import re
import sys
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_51'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from unified_memory import MemoryLayer
from hybrid_memory import MemoryType, MemorySource


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TypeSelectorError(Exception):
    """Base exception for type selection operations."""


class SelectionFailedError(TypeSelectorError):
    """Raised when no algorithm can produce a selection."""


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------

class InformationNeed(str, Enum):
    RECALL = "recall"            # retrieve a specific past item
    LOOKUP = "lookup"            # find a fact / definition
    EXPLORATION = "exploration"  # browse related items
    TEMPORAL = "temporal"        # time-based retrieval
    EXPERIENTIAL = "experiential"  # outcome / performance data


_TEMPORAL_WORDS = frozenset([
    "when", "time", "timeline", "history", "happened", "occurred",
    "before", "after", "during", "recent", "latest", "yesterday",
    "today", "last", "ago", "sequence", "event", "events",
])

_FACTUAL_WORDS = frozenset([
    "what", "define", "definition", "is", "are", "means", "meaning",
    "fact", "facts", "describe", "explain", "concept", "entity",
    "knowledge", "who", "which",
])

_EXPERIENTIAL_WORDS = frozenset([
    "how", "outcome", "result", "success", "failure", "score",
    "performance", "experience", "tried", "attempt", "worked",
    "failed", "improve", "lesson", "feedback", "rate",
])

_RELATIONAL_WORDS = frozenset([
    "related", "relationship", "connected", "link", "linked",
    "between", "connection", "depends", "causes", "part",
    "neighbour", "neighbor", "graph", "edge", "path",
])

_CONVERSATION_WORDS = frozenset([
    "conversation", "message", "chat", "talk", "said", "told",
    "asked", "replied", "discussed",
])


@dataclass
class QueryCharacteristics:
    """Extracted characteristics of a query."""
    query: str
    word_count: int = 0
    temporal_score: float = 0.0
    factual_score: float = 0.0
    experiential_score: float = 0.0
    relational_score: float = 0.0
    conversational_score: float = 0.0
    information_need: InformationNeed = InformationNeed.LOOKUP
    is_question: bool = False
    has_temporal_reference: bool = False


def analyse_query_characteristics(query: str) -> QueryCharacteristics:
    """Analyse a query and return its characteristics."""
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))
    word_count = len(words) or 1

    scores = {
        "temporal": len(words & _TEMPORAL_WORDS),
        "factual": len(words & _FACTUAL_WORDS),
        "experiential": len(words & _EXPERIENTIAL_WORDS),
        "relational": len(words & _RELATIONAL_WORDS),
        "conversational": len(words & _CONVERSATION_WORDS),
    }
    total = sum(scores.values()) or 1.0

    # Determine information need from dominant signal
    best = max(scores, key=scores.get)
    need_map = {
        "temporal": InformationNeed.TEMPORAL,
        "factual": InformationNeed.LOOKUP,
        "experiential": InformationNeed.EXPERIENTIAL,
        "relational": InformationNeed.EXPLORATION,
        "conversational": InformationNeed.RECALL,
    }
    info_need = need_map.get(best, InformationNeed.LOOKUP) if scores[best] > 0 else InformationNeed.LOOKUP

    return QueryCharacteristics(
        query=query,
        word_count=word_count,
        temporal_score=scores["temporal"] / total,
        factual_score=scores["factual"] / total,
        experiential_score=scores["experiential"] / total,
        relational_score=scores["relational"] / total,
        conversational_score=scores["conversational"] / total,
        information_need=info_need,
        is_question=q.strip().endswith("?") or any(w in words for w in {"what", "when", "how", "who", "which", "where"}),
        has_temporal_reference=bool(words & _TEMPORAL_WORDS),
    )


# ---------------------------------------------------------------------------
# Selection result
# ---------------------------------------------------------------------------

@dataclass
class TypeSelection:
    """Result of a type selection decision."""
    layer: MemoryLayer
    memory_type: MemoryType
    confidence: float = 0.0
    algorithm: str = ""
    reasoning: str = ""
    alternatives: List[Tuple[MemoryLayer, MemoryType, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract selector
# ---------------------------------------------------------------------------

class SelectionAlgorithm(ABC):
    """Abstract base for type selection algorithms."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def select(self, characteristics: QueryCharacteristics) -> TypeSelection: ...


# ---------------------------------------------------------------------------
# Rule-based selector
# ---------------------------------------------------------------------------

class RuleBasedSelector(SelectionAlgorithm):
    """Deterministic rules mapping signal thresholds to memory types."""

    def __init__(self, threshold: float = 0.3):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "rule_based"

    def select(self, c: QueryCharacteristics) -> TypeSelection:
        # Conversational → working memory
        if c.conversational_score >= self._threshold:
            return TypeSelection(
                layer=MemoryLayer.WORKING, memory_type=MemoryType.EPISODIC,
                confidence=c.conversational_score, algorithm=self.name,
                reasoning="Conversational keywords detected",
            )

        # Temporal / experiential → short-term (episodic)
        if c.temporal_score >= self._threshold or c.experiential_score >= self._threshold:
            conf = max(c.temporal_score, c.experiential_score)
            return TypeSelection(
                layer=MemoryLayer.SHORT_TERM, memory_type=MemoryType.EPISODIC,
                confidence=conf, algorithm=self.name,
                reasoning="Temporal or experiential keywords detected",
            )

        # Factual / relational → long-term (semantic)
        if c.factual_score >= self._threshold or c.relational_score >= self._threshold:
            conf = max(c.factual_score, c.relational_score)
            return TypeSelection(
                layer=MemoryLayer.LONG_TERM, memory_type=MemoryType.SEMANTIC,
                confidence=conf, algorithm=self.name,
                reasoning="Factual or relational keywords detected",
            )

        # Fallback → hybrid / both
        return TypeSelection(
            layer=MemoryLayer.HYBRID, memory_type=MemoryType.BOTH,
            confidence=0.2, algorithm=self.name,
            reasoning="No dominant signal — fallback to hybrid",
        )


# ---------------------------------------------------------------------------
# Query-based selector
# ---------------------------------------------------------------------------

class QueryBasedSelector(SelectionAlgorithm):
    """Scores each (layer, type) pair against the query and picks the best."""

    @property
    def name(self) -> str:
        return "query_based"

    def select(self, c: QueryCharacteristics) -> TypeSelection:
        candidates = [
            (MemoryLayer.WORKING, MemoryType.EPISODIC,
             c.conversational_score * 1.2),
            (MemoryLayer.SHORT_TERM, MemoryType.EPISODIC,
             c.temporal_score + c.experiential_score),
            (MemoryLayer.LONG_TERM, MemoryType.SEMANTIC,
             c.factual_score + c.relational_score),
            (MemoryLayer.HYBRID, MemoryType.BOTH, 0.3),  # baseline
        ]

        candidates.sort(key=lambda x: x[2], reverse=True)
        best_layer, best_type, best_score = candidates[0]

        alternatives = [
            (layer, mtype, score)
            for layer, mtype, score in candidates[1:]
            if score > 0
        ]

        return TypeSelection(
            layer=best_layer, memory_type=best_type,
            confidence=min(best_score, 1.0), algorithm=self.name,
            reasoning=f"Highest scored pair: {best_layer.value}/{best_type.value}",
            alternatives=alternatives,
        )


# ---------------------------------------------------------------------------
# Pattern-based selector
# ---------------------------------------------------------------------------

class PatternBasedSelector(SelectionAlgorithm):
    """Learns from past (information_need → layer/type) result counts."""

    def __init__(self):
        # need → {(layer, type) → cumulative result count}
        self._patterns: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(
            lambda: {
                (MemoryLayer.WORKING.value, MemoryType.EPISODIC.value): 0.0,
                (MemoryLayer.SHORT_TERM.value, MemoryType.EPISODIC.value): 0.0,
                (MemoryLayer.LONG_TERM.value, MemoryType.SEMANTIC.value): 0.0,
                (MemoryLayer.HYBRID.value, MemoryType.BOTH.value): 0.0,
            }
        )
        self._query_count = 0

    @property
    def name(self) -> str:
        return "pattern_based"

    @property
    def patterns(self) -> Dict[str, Dict[Tuple[str, str], float]]:
        return dict(self._patterns)

    @property
    def query_count(self) -> int:
        return self._query_count

    def select(self, c: QueryCharacteristics) -> TypeSelection:
        self._query_count += 1
        need_key = c.information_need.value
        scores = self._patterns[need_key]

        # Not enough data → use simple heuristic
        if self._query_count <= 5 or max(scores.values()) == 0:
            return self._heuristic_select(c)

        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]
        total = sum(scores.values()) or 1.0

        return TypeSelection(
            layer=MemoryLayer(best_key[0]),
            memory_type=MemoryType(best_key[1]),
            confidence=min(best_score / total, 1.0),
            algorithm=self.name,
            reasoning=f"Learned pattern for {need_key}: {best_key}",
        )

    def record_outcome(
        self, need: InformationNeed, layer: MemoryLayer,
        memory_type: MemoryType, result_count: int,
    ) -> None:
        """Record how many results a selection produced."""
        key = (layer.value, memory_type.value)
        self._patterns[need.value][key] += max(result_count, 0)

    def reset(self) -> None:
        self._patterns.clear()
        self._query_count = 0

    @staticmethod
    def _heuristic_select(c: QueryCharacteristics) -> TypeSelection:
        if c.has_temporal_reference:
            return TypeSelection(
                layer=MemoryLayer.SHORT_TERM, memory_type=MemoryType.EPISODIC,
                confidence=0.4, algorithm="pattern_based",
                reasoning="Heuristic: temporal reference detected",
            )
        if c.factual_score > 0:
            return TypeSelection(
                layer=MemoryLayer.LONG_TERM, memory_type=MemoryType.SEMANTIC,
                confidence=0.4, algorithm="pattern_based",
                reasoning="Heuristic: factual signal detected",
            )
        return TypeSelection(
            layer=MemoryLayer.HYBRID, memory_type=MemoryType.BOTH,
            confidence=0.2, algorithm="pattern_based",
            reasoning="Heuristic: no strong signal",
        )


# ---------------------------------------------------------------------------
# Hybrid selector (voting)
# ---------------------------------------------------------------------------

class HybridSelector(SelectionAlgorithm):
    """Combines multiple selectors via weighted voting."""

    def __init__(
        self,
        selectors: Optional[List[Tuple[SelectionAlgorithm, float]]] = None,
    ):
        self._selectors: List[Tuple[SelectionAlgorithm, float]] = selectors or [
            (RuleBasedSelector(), 1.0),
            (QueryBasedSelector(), 1.0),
            (PatternBasedSelector(), 0.8),
        ]

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def selectors(self) -> List[Tuple[SelectionAlgorithm, float]]:
        return list(self._selectors)

    def select(self, c: QueryCharacteristics) -> TypeSelection:
        # Collect votes: (layer, type) → weighted confidence sum
        votes: Dict[Tuple[str, str], float] = defaultdict(float)
        sub_selections: List[TypeSelection] = []

        for selector, weight in self._selectors:
            sel = selector.select(c)
            sub_selections.append(sel)
            key = (sel.layer.value, sel.memory_type.value)
            votes[key] += sel.confidence * weight

        best_key = max(votes, key=votes.get)
        total_weight = sum(w for _, w in self._selectors) or 1.0
        confidence = votes[best_key] / total_weight

        alternatives = [
            (MemoryLayer(k[0]), MemoryType(k[1]), v / total_weight)
            for k, v in votes.items()
            if k != best_key and v > 0
        ]

        voters = ", ".join(s.algorithm for s in sub_selections)
        return TypeSelection(
            layer=MemoryLayer(best_key[0]),
            memory_type=MemoryType(best_key[1]),
            confidence=min(confidence, 1.0),
            algorithm=self.name,
            reasoning=f"Voted by: {voters}",
            alternatives=alternatives,
        )

    def get_pattern_selector(self) -> Optional[PatternBasedSelector]:
        """Return the PatternBasedSelector if present (for recording outcomes)."""
        for sel, _ in self._selectors:
            if isinstance(sel, PatternBasedSelector):
                return sel
        return None


# ---------------------------------------------------------------------------
# AutoSelector — automatic selection with fallback
# ---------------------------------------------------------------------------

class AutoSelector:
    """Wraps a SelectionAlgorithm with automatic fallback and caching."""

    def __init__(
        self,
        algorithm: Optional[SelectionAlgorithm] = None,
        fallback_layer: MemoryLayer = MemoryLayer.HYBRID,
        fallback_type: MemoryType = MemoryType.BOTH,
        min_confidence: float = 0.1,
    ):
        self._algorithm = algorithm or HybridSelector()
        self._fallback_layer = fallback_layer
        self._fallback_type = fallback_type
        self._min_confidence = min_confidence
        self._cache: Dict[str, TypeSelection] = {}
        self._history: List[TypeSelection] = []

    @property
    def algorithm(self) -> SelectionAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: SelectionAlgorithm) -> None:
        self._algorithm = value
        self._cache.clear()

    @property
    def history(self) -> List[TypeSelection]:
        return list(self._history)

    def select(self, query: str, use_cache: bool = True) -> TypeSelection:
        """Analyse *query* and select the optimal memory type."""
        if use_cache and query in self._cache:
            return self._cache[query]

        characteristics = analyse_query_characteristics(query)

        try:
            selection = self._algorithm.select(characteristics)
        except Exception:
            selection = self._make_fallback("Algorithm error")

        # Apply minimum confidence gate
        if selection.confidence < self._min_confidence:
            selection = self._make_fallback("Below confidence threshold")

        self._cache[query] = selection
        self._history.append(selection)
        return selection

    def clear_cache(self) -> None:
        self._cache.clear()

    def clear_history(self) -> None:
        self._history.clear()

    def _make_fallback(self, reason: str) -> TypeSelection:
        return TypeSelection(
            layer=self._fallback_layer,
            memory_type=self._fallback_type,
            confidence=0.1,
            algorithm="fallback",
            reasoning=reason,
        )


# ---------------------------------------------------------------------------
# SelectionOptimizer — learns from outcomes
# ---------------------------------------------------------------------------

@dataclass
class SelectionOutcome:
    """Recorded outcome of a type selection."""
    query: str
    selection: TypeSelection
    result_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class SelectionOptimizer:
    """Tracks selection outcomes and optimizes selector weights."""

    def __init__(self, auto_selector: Optional[AutoSelector] = None):
        self._selector = auto_selector or AutoSelector()
        self._outcomes: List[SelectionOutcome] = []
        self._performance: Dict[str, List[float]] = defaultdict(list)

    @property
    def selector(self) -> AutoSelector:
        return self._selector

    @property
    def outcomes(self) -> List[SelectionOutcome]:
        return list(self._outcomes)

    def select(self, query: str) -> TypeSelection:
        """Delegate to the underlying AutoSelector."""
        return self._selector.select(query)

    def record_outcome(self, query: str, selection: TypeSelection, result_count: int) -> None:
        """Record the outcome of a selection for learning."""
        outcome = SelectionOutcome(query=query, selection=selection, result_count=result_count)
        self._outcomes.append(outcome)

        # Track per-algorithm performance (normalised result count)
        score = min(result_count / 5.0, 1.0) if result_count > 0 else 0.0
        self._performance[selection.algorithm].append(score)

        # Feed pattern-based selectors if available
        algo = self._selector.algorithm
        pattern_sel = None
        if isinstance(algo, PatternBasedSelector):
            pattern_sel = algo
        elif isinstance(algo, HybridSelector):
            pattern_sel = algo.get_pattern_selector()

        if pattern_sel is not None:
            chars = analyse_query_characteristics(query)
            pattern_sel.record_outcome(
                chars.information_need, selection.layer,
                selection.memory_type, result_count,
            )

    def stats(self) -> Dict[str, Any]:
        """Return optimisation statistics."""
        total = len(self._outcomes)
        if total == 0:
            return {"total_selections": 0, "avg_results": 0.0, "algorithm_performance": {}}

        avg_results = sum(o.result_count for o in self._outcomes) / total
        algo_perf = {
            algo: sum(scores) / len(scores) if scores else 0.0
            for algo, scores in self._performance.items()
        }
        return {
            "total_selections": total,
            "avg_results": avg_results,
            "algorithm_performance": algo_perf,
        }

    def best_algorithm(self) -> Optional[str]:
        """Return the name of the best-performing algorithm so far."""
        if not self._performance:
            return None
        return max(
            self._performance,
            key=lambda k: sum(self._performance[k]) / len(self._performance[k])
            if self._performance[k] else 0.0,
        )

    def reset(self) -> None:
        self._outcomes.clear()
        self._performance.clear()
        self._selector.clear_cache()
        self._selector.clear_history()
