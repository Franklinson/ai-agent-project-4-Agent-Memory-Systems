"""Memory type routing logic for hybrid memory queries.

Analyses incoming queries, selects the appropriate memory type(s), and routes
them through HybridMemory.  Supports multiple pluggable routing strategies
including an adaptive strategy that learns from past query outcomes.
"""

import sys
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from hybrid_memory import (
    HybridMemory, MemoryType, MemorySource, MemoryItem, QueryResult,
)
from event_store import EventType
from experience_tracker import Outcome
from fact_store import FactType
from knowledge_graph import NodeType


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RoutingError(Exception):
    """Base exception for routing operations."""


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    TEMPORAL = "temporal"        # when / timeline / history
    FACTUAL = "factual"          # what is / definition / fact
    EXPERIENTIAL = "experiential"  # how did / outcome / performance
    RELATIONAL = "relational"    # related / connected / linked
    AMBIGUOUS = "ambiguous"      # unclear intent


# Keyword sets used by the keyword-based analyser
_TEMPORAL_KEYWORDS = frozenset([
    "when", "time", "timeline", "history", "happened", "occurred",
    "before", "after", "during", "recent", "latest", "yesterday",
    "today", "last", "ago", "sequence", "order", "event", "events",
])

_FACTUAL_KEYWORDS = frozenset([
    "what", "define", "definition", "is", "are", "means", "meaning",
    "fact", "facts", "describe", "explain", "concept", "entity",
    "knowledge", "who", "which",
])

_EXPERIENTIAL_KEYWORDS = frozenset([
    "how", "outcome", "result", "success", "failure", "score",
    "performance", "experience", "tried", "attempt", "worked",
    "failed", "improve", "improving", "lesson", "feedback", "rate",
])

_RELATIONAL_KEYWORDS = frozenset([
    "related", "relationship", "connected", "link", "linked",
    "between", "connection", "depends", "causes", "part",
    "neighbour", "neighbor", "graph", "edge", "path",
])


@dataclass
class QueryAnalysis:
    """Result of analysing a query string."""
    query: str
    intent: QueryIntent
    signals: Dict[str, float] = field(default_factory=dict)
    detected_filters: Dict[str, Any] = field(default_factory=dict)
    recommended_type: MemoryType = MemoryType.BOTH
    recommended_sources: List[MemorySource] = field(default_factory=list)
    confidence: float = 0.0


def analyse_query(query: str) -> QueryAnalysis:
    """Analyse a query string and return intent signals and recommendations.

    Uses keyword matching to produce a signal score for each intent category,
    then picks the strongest signal.  Also extracts structured filter hints
    (e.g. detected event types, outcome keywords).
    """
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))

    # Score each intent category
    scores: Dict[str, float] = {
        "temporal": len(words & _TEMPORAL_KEYWORDS),
        "factual": len(words & _FACTUAL_KEYWORDS),
        "experiential": len(words & _EXPERIENTIAL_KEYWORDS),
        "relational": len(words & _RELATIONAL_KEYWORDS),
    }

    total = sum(scores.values()) or 1.0
    signals = {k: v / total for k, v in scores.items()}

    # Pick dominant intent
    best_key = max(scores, key=scores.get)
    best_val = scores[best_key]
    if best_val == 0:
        intent = QueryIntent.AMBIGUOUS
        confidence = 0.0
    else:
        intent = QueryIntent(best_key)
        confidence = signals[best_key]

    # Map intent → recommended memory type & sources
    rec_type, rec_sources = _intent_to_targets(intent)

    # Extract structured filter hints
    detected_filters = _extract_filters(q, words)

    return QueryAnalysis(
        query=query, intent=intent, signals=signals,
        detected_filters=detected_filters,
        recommended_type=rec_type, recommended_sources=rec_sources,
        confidence=confidence,
    )


def _intent_to_targets(
    intent: QueryIntent,
) -> Tuple[MemoryType, List[MemorySource]]:
    """Map a QueryIntent to a MemoryType and preferred sources."""
    if intent == QueryIntent.TEMPORAL:
        return MemoryType.EPISODIC, [MemorySource.EVENT_STORE]
    if intent == QueryIntent.FACTUAL:
        return MemoryType.SEMANTIC, [
            MemorySource.FACT_STORE, MemorySource.KNOWLEDGE_GRAPH]
    if intent == QueryIntent.EXPERIENTIAL:
        return MemoryType.EPISODIC, [MemorySource.EXPERIENCE_TRACKER]
    if intent == QueryIntent.RELATIONAL:
        return MemoryType.SEMANTIC, [MemorySource.KNOWLEDGE_GRAPH]
    # AMBIGUOUS → search everything
    return MemoryType.BOTH, list(MemorySource)


def _extract_filters(q: str, words: Set[str]) -> Dict[str, Any]:
    """Extract structured filter hints from the query text."""
    filters: Dict[str, Any] = {}

    # Event type hints
    for et in EventType:
        if et.value in q:
            filters["event_type"] = et

    # Outcome hints
    for oc in Outcome:
        if oc.value in words:
            filters["outcome"] = oc

    # Fact type hints
    if "definition" in words or "define" in words:
        filters["fact_type"] = FactType.DEFINITION
    elif "rule" in words:
        filters["fact_type"] = FactType.RULE

    # Node type hints
    if "concept" in words:
        filters["node_type"] = NodeType.CONCEPT
    elif "entity" in words:
        filters["node_type"] = NodeType.ENTITY

    return filters


# ---------------------------------------------------------------------------
# Routing strategies (pluggable)
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """The output of a routing strategy."""
    memory_type: MemoryType
    sources: List[MemorySource]
    query_kwargs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    strategy_name: str = ""


class RoutingStrategy(ABC):
    """Abstract base for routing strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def route(self, analysis: QueryAnalysis) -> RoutingDecision: ...


class KeywordRoutingStrategy(RoutingStrategy):
    """Routes based purely on the dominant keyword-intent signal."""

    @property
    def name(self) -> str:
        return "keyword"

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        kwargs = dict(analysis.detected_filters)
        return RoutingDecision(
            memory_type=analysis.recommended_type,
            sources=list(analysis.recommended_sources),
            query_kwargs=kwargs,
            confidence=analysis.confidence,
            strategy_name=self.name,
        )


class WeightedRoutingStrategy(RoutingStrategy):
    """Routes using configurable signal-weight thresholds.

    If the top signal exceeds *threshold* the query is routed to a single
    memory type; otherwise it falls back to BOTH.
    """

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "weighted"

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        kwargs = dict(analysis.detected_filters)

        if analysis.confidence >= self._threshold:
            return RoutingDecision(
                memory_type=analysis.recommended_type,
                sources=list(analysis.recommended_sources),
                query_kwargs=kwargs,
                confidence=analysis.confidence,
                strategy_name=self.name,
            )

        # Below threshold → query both
        return RoutingDecision(
            memory_type=MemoryType.BOTH,
            sources=list(MemorySource),
            query_kwargs=kwargs,
            confidence=analysis.confidence,
            strategy_name=self.name,
        )


class AdaptiveRoutingStrategy(RoutingStrategy):
    """Learns from past query outcomes to improve routing over time.

    Tracks how many results each (intent → memory_type) pairing produced
    and biases future decisions toward pairings that yielded results.
    """

    def __init__(self, base_strategy: Optional[RoutingStrategy] = None,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.2):
        self._base = base_strategy or KeywordRoutingStrategy()
        self._lr = learning_rate
        self._exploration = exploration_rate
        # intent → {memory_type_value → cumulative score}
        self._scores: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {mt.value: 0.5 for mt in MemoryType if mt != MemoryType.BOTH}
        )
        self._query_count: int = 0

    @property
    def name(self) -> str:
        return "adaptive"

    @property
    def scores(self) -> Dict[str, Dict[str, float]]:
        return dict(self._scores)

    @property
    def query_count(self) -> int:
        return self._query_count

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        self._query_count += 1
        intent_key = analysis.intent.value
        kwargs = dict(analysis.detected_filters)

        # Early queries or ambiguous intent → use base strategy
        if (analysis.intent == QueryIntent.AMBIGUOUS
                or self._query_count <= 3):
            base_decision = self._base.route(analysis)
            base_decision.strategy_name = self.name
            return base_decision

        # Pick the memory type with the highest learned score for this intent
        intent_scores = self._scores[intent_key]
        best_type_val = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_type_val]

        # Exploration: if scores are close, fall back to BOTH
        other_scores = [v for k, v in intent_scores.items() if k != best_type_val]
        spread = best_score - (max(other_scores) if other_scores else 0)
        if spread < self._exploration:
            return RoutingDecision(
                memory_type=MemoryType.BOTH,
                sources=list(MemorySource),
                query_kwargs=kwargs,
                confidence=analysis.confidence,
                strategy_name=self.name,
            )

        chosen_type = MemoryType(best_type_val)
        _, sources = _intent_to_targets(analysis.intent)
        # Override sources if the learned type differs from the keyword hint
        if chosen_type != analysis.recommended_type:
            _, sources = _intent_to_targets(
                QueryIntent.TEMPORAL if chosen_type == MemoryType.EPISODIC
                else QueryIntent.FACTUAL
            )

        return RoutingDecision(
            memory_type=chosen_type,
            sources=sources,
            query_kwargs=kwargs,
            confidence=analysis.confidence,
            strategy_name=self.name,
        )

    def record_outcome(self, intent: QueryIntent, memory_type: MemoryType,
                       result_count: int) -> None:
        """Feed back the number of results a routing decision produced.

        Positive reward when results > 0, negative when 0.
        """
        if memory_type == MemoryType.BOTH:
            return  # don't update for BOTH — it's the fallback
        intent_key = intent.value
        mt_key = memory_type.value
        reward = min(result_count / 5.0, 1.0) if result_count > 0 else -0.2
        self._scores[intent_key][mt_key] += self._lr * reward

    def reset(self) -> None:
        """Reset learned scores."""
        self._scores.clear()
        self._query_count = 0


# ---------------------------------------------------------------------------
# MemoryRouter — main entry point
# ---------------------------------------------------------------------------

@dataclass
class RouteRecord:
    """Audit log entry for a routed query."""
    query: str
    analysis: QueryAnalysis
    decision: RoutingDecision
    result_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryRouter:
    """Routes queries to the appropriate memory subsystems via HybridMemory.

    Wraps HybridMemory.query with automatic query analysis, strategy-based
    routing, and optional outcome tracking for adaptive strategies.
    """

    def __init__(self, hybrid_memory: HybridMemory,
                 strategy: Optional[RoutingStrategy] = None):
        self._hm = hybrid_memory
        self._strategy = strategy or KeywordRoutingStrategy()
        self._history: List[RouteRecord] = []

    @property
    def hybrid_memory(self) -> HybridMemory:
        return self._hm

    @property
    def strategy(self) -> RoutingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: RoutingStrategy) -> None:
        self._strategy = value

    @property
    def history(self) -> List[RouteRecord]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route(self, query: str, **override_kwargs) -> QueryResult:
        """Analyse *query*, pick a routing strategy, execute, and return results.

        Any *override_kwargs* are merged into the query kwargs produced by the
        strategy (overrides win).
        """
        analysis = analyse_query(query)
        decision = self._strategy.route(analysis)

        # Build final kwargs for HybridMemory.query
        kwargs: Dict[str, Any] = {
            "memory_type": decision.memory_type,
        }
        kwargs.update(decision.query_kwargs)
        kwargs.update(override_kwargs)

        result = self._hm.query(query, **kwargs)

        # Record for audit / adaptive learning
        record = RouteRecord(
            query=query, analysis=analysis, decision=decision,
            result_count=result.count,
        )
        self._history.append(record)

        # Feed back to adaptive strategy if applicable
        if isinstance(self._strategy, AdaptiveRoutingStrategy):
            self._strategy.record_outcome(
                analysis.intent, decision.memory_type, result.count)

        return result

    def route_with_analysis(self, query: str,
                            **override_kwargs) -> Tuple[QueryResult, QueryAnalysis, RoutingDecision]:
        """Like route() but also returns the analysis and decision for inspection."""
        analysis = analyse_query(query)
        decision = self._strategy.route(analysis)

        kwargs: Dict[str, Any] = {"memory_type": decision.memory_type}
        kwargs.update(decision.query_kwargs)
        kwargs.update(override_kwargs)

        result = self._hm.query(query, **kwargs)

        record = RouteRecord(
            query=query, analysis=analysis, decision=decision,
            result_count=result.count,
        )
        self._history.append(record)

        if isinstance(self._strategy, AdaptiveRoutingStrategy):
            self._strategy.record_outcome(
                analysis.intent, decision.memory_type, result.count)

        return result, analysis, decision

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def analyse(self, query: str) -> QueryAnalysis:
        """Analyse a query without executing it."""
        return analyse_query(query)

    def stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        total = len(self._history)
        if total == 0:
            return {"total_queries": 0, "avg_results": 0.0,
                    "strategy": self._strategy.name,
                    "intent_distribution": {}}

        intent_counts: Dict[str, int] = defaultdict(int)
        total_results = 0
        for r in self._history:
            intent_counts[r.analysis.intent.value] += 1
            total_results += r.result_count

        return {
            "total_queries": total,
            "avg_results": total_results / total,
            "strategy": self._strategy.name,
            "intent_distribution": dict(intent_counts),
        }

    def clear_history(self) -> None:
        """Clear the routing audit log."""
        self._history.clear()
