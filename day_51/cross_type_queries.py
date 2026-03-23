"""Cross-type query support for hybrid memory.

Provides combined, sequential, and parallel query patterns across episodic
and semantic memory, with result merging, deduplication, and relevance ranking.
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from hybrid_memory import (
    HybridMemory, MemoryType, MemorySource, MemoryItem, QueryResult,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CrossQueryError(Exception):
    """Base exception for cross-type query operations."""


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class QueryPattern(str, Enum):
    COMBINED = "combined"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RankingStrategy(str, Enum):
    RELEVANCE = "relevance"
    TIMESTAMP = "timestamp"
    SOURCE_PRIORITY = "source_priority"


@dataclass
class SubQuery:
    """A single sub-query to execute against HybridMemory."""
    query_text: str
    memory_type: MemoryType = MemoryType.BOTH
    kwargs: Dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass
class MergedResult:
    """Unified result from a cross-type query execution."""
    pattern: QueryPattern
    items: List[MemoryItem] = field(default_factory=list)
    sub_results: List[QueryResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.items)

    def by_source(self, source: MemorySource) -> List[MemoryItem]:
        return [i for i in self.items if i.source == source]

    def by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        return [i for i in self.items if i.memory_type == memory_type]


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def _text_relevance(item: MemoryItem, query: str) -> float:
    """Score 0.0–1.0 based on how well item content matches the query."""
    if not query:
        return item.relevance
    q = query.lower()
    content = item.content.lower()
    if q in content:
        return min(1.0, item.relevance + 0.3)
    words = q.split()
    if not words:
        return item.relevance
    matches = sum(1 for w in words if w in content)
    return item.relevance * 0.5 + 0.5 * (matches / len(words))


_SOURCE_PRIORITY: Dict[MemorySource, float] = {
    MemorySource.FACT_STORE: 0.05,
    MemorySource.KNOWLEDGE_GRAPH: 0.04,
    MemorySource.EVENT_STORE: 0.03,
    MemorySource.EXPERIENCE_TRACKER: 0.02,
}


def rank_items(
    items: List[MemoryItem],
    query: str = "",
    strategy: RankingStrategy = RankingStrategy.RELEVANCE,
) -> List[MemoryItem]:
    """Return a new list of items sorted by the chosen ranking strategy."""
    if strategy == RankingStrategy.TIMESTAMP:
        return sorted(items, key=lambda i: i.timestamp, reverse=True)

    if strategy == RankingStrategy.SOURCE_PRIORITY:
        return sorted(
            items,
            key=lambda i: (-_SOURCE_PRIORITY.get(i.source, 0), -i.relevance),
        )

    # RELEVANCE (default): text match + base relevance + source tiebreak
    def score(item: MemoryItem) -> float:
        return _text_relevance(item, query) + _SOURCE_PRIORITY.get(item.source, 0)

    return sorted(items, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_items(
    *item_lists: List[MemoryItem],
    deduplicate: bool = True,
) -> List[MemoryItem]:
    """Flatten multiple item lists, optionally removing duplicates by ID."""
    merged: List[MemoryItem] = []
    seen: set = set()
    for lst in item_lists:
        for item in lst:
            if deduplicate and item.id in seen:
                continue
            seen.add(item.id)
            merged.append(item)
    return merged


def format_results(items: List[MemoryItem], max_items: Optional[int] = None) -> str:
    """Format merged items into a readable string."""
    if max_items is not None:
        items = items[:max_items]
    lines: List[str] = []
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. [{item.source.value}] {item.content} "
            f"(relevance={item.relevance:.2f})"
        )
    return "\n".join(lines) if lines else "(no results)"


# ---------------------------------------------------------------------------
# CrossTypeQueryEngine
# ---------------------------------------------------------------------------

class CrossTypeQueryEngine:
    """Executes combined, sequential, and parallel queries across memory types."""

    def __init__(self, hybrid_memory: HybridMemory, max_workers: int = 4):
        self._hm = hybrid_memory
        self._max_workers = max_workers

    @property
    def hybrid_memory(self) -> HybridMemory:
        return self._hm

    # ------------------------------------------------------------------
    # Combined query
    # ------------------------------------------------------------------

    def combined(
        self,
        query_text: str,
        ranking: RankingStrategy = RankingStrategy.RELEVANCE,
        limit: Optional[int] = None,
        **kwargs,
    ) -> MergedResult:
        """Query both episodic and semantic memory, merge and rank results."""
        start = datetime.now()

        ep_result = self._hm.query(
            query_text, memory_type=MemoryType.EPISODIC, **kwargs)
        sem_result = self._hm.query(
            query_text, memory_type=MemoryType.SEMANTIC, **kwargs)

        items = merge_items(ep_result.items, sem_result.items)
        items = rank_items(items, query_text, ranking)
        if limit is not None:
            items = items[:limit]

        elapsed = (datetime.now() - start).total_seconds() * 1000
        return MergedResult(
            pattern=QueryPattern.COMBINED,
            items=items,
            sub_results=[ep_result, sem_result],
            execution_time_ms=elapsed,
            metadata={"query": query_text, "ranking": ranking.value},
        )

    # ------------------------------------------------------------------
    # Sequential query
    # ------------------------------------------------------------------

    def sequential(
        self,
        primary_query: str,
        primary_type: MemoryType = MemoryType.EPISODIC,
        secondary_type: Optional[MemoryType] = None,
        bridge_fn: Optional[Callable[[List[MemoryItem]], Dict[str, Any]]] = None,
        ranking: RankingStrategy = RankingStrategy.RELEVANCE,
        limit: Optional[int] = None,
        primary_kwargs: Optional[Dict[str, Any]] = None,
        secondary_kwargs: Optional[Dict[str, Any]] = None,
    ) -> MergedResult:
        """Query one type first, then use results to inform a second query.

        *bridge_fn* receives the primary results and returns kwargs to inject
        into the secondary query.  If omitted, a default bridge extracts
        subject/label hints from the primary results.
        """
        start = datetime.now()
        sec_type = secondary_type or (
            MemoryType.SEMANTIC if primary_type == MemoryType.EPISODIC
            else MemoryType.EPISODIC
        )

        # Phase 1: primary query
        p_kwargs = dict(primary_kwargs or {})
        primary_result = self._hm.query(
            primary_query, memory_type=primary_type, **p_kwargs)

        # Phase 2: derive secondary kwargs via bridge
        bridge = bridge_fn or _default_bridge
        derived = bridge(primary_result.items)

        s_kwargs = dict(secondary_kwargs or {})
        s_kwargs.update(derived)
        secondary_query = s_kwargs.pop("query_text", primary_query)
        secondary_result = self._hm.query(
            secondary_query, memory_type=sec_type, **s_kwargs)

        # Boost primary items slightly so they rank above secondary ties
        for item in primary_result.items:
            item.relevance = min(1.0, item.relevance + 0.1)

        items = merge_items(primary_result.items, secondary_result.items)
        items = rank_items(items, primary_query, ranking)
        if limit is not None:
            items = items[:limit]

        elapsed = (datetime.now() - start).total_seconds() * 1000
        return MergedResult(
            pattern=QueryPattern.SEQUENTIAL,
            items=items,
            sub_results=[primary_result, secondary_result],
            execution_time_ms=elapsed,
            metadata={
                "primary_type": primary_type.value,
                "secondary_type": sec_type.value,
                "bridge_kwargs": derived,
            },
        )

    # ------------------------------------------------------------------
    # Parallel query
    # ------------------------------------------------------------------

    def parallel(
        self,
        sub_queries: List[SubQuery],
        ranking: RankingStrategy = RankingStrategy.RELEVANCE,
        limit: Optional[int] = None,
    ) -> MergedResult:
        """Execute multiple sub-queries concurrently and merge results."""
        start = datetime.now()
        sub_results: List[QueryResult] = []
        all_items: List[List[MemoryItem]] = []

        def _run(sq: SubQuery) -> QueryResult:
            return self._hm.query(
                sq.query_text, memory_type=sq.memory_type, **sq.kwargs)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_run, sq): sq for sq in sub_queries}
            for future in as_completed(futures):
                result = future.result()
                sub_results.append(result)
                all_items.append(result.items)

        items = merge_items(*all_items)
        # Use the first sub-query's text for ranking context
        rank_query = sub_queries[0].query_text if sub_queries else ""
        items = rank_items(items, rank_query, ranking)
        if limit is not None:
            items = items[:limit]

        elapsed = (datetime.now() - start).total_seconds() * 1000
        return MergedResult(
            pattern=QueryPattern.PARALLEL,
            items=items,
            sub_results=sub_results,
            execution_time_ms=elapsed,
            metadata={
                "sub_query_count": len(sub_queries),
                "labels": [sq.label for sq in sub_queries],
            },
        )

    # ------------------------------------------------------------------
    # Multi-hop query (convenience sequential chain)
    # ------------------------------------------------------------------

    def multi_hop(
        self,
        query_text: str,
        hops: List[Dict[str, Any]],
        ranking: RankingStrategy = RankingStrategy.RELEVANCE,
        limit: Optional[int] = None,
    ) -> MergedResult:
        """Chain multiple sequential queries, each hop feeding the next.

        Each hop dict may contain: memory_type, kwargs, bridge_fn.
        """
        start = datetime.now()
        all_sub_results: List[QueryResult] = []
        accumulated: List[MemoryItem] = []
        current_query = query_text

        for hop in hops:
            mt = hop.get("memory_type", MemoryType.BOTH)
            kw = dict(hop.get("kwargs", {}))
            bridge = hop.get("bridge_fn")

            # Apply bridge from previous results
            if bridge and accumulated:
                derived = bridge(accumulated)
                kw.update(derived)
                current_query = kw.pop("query_text", current_query)

            result = self._hm.query(current_query, memory_type=mt, **kw)
            all_sub_results.append(result)
            accumulated = merge_items(accumulated, result.items)

        items = rank_items(accumulated, query_text, ranking)
        if limit is not None:
            items = items[:limit]

        elapsed = (datetime.now() - start).total_seconds() * 1000
        return MergedResult(
            pattern=QueryPattern.SEQUENTIAL,
            items=items,
            sub_results=all_sub_results,
            execution_time_ms=elapsed,
            metadata={"hops": len(hops)},
        )


# ---------------------------------------------------------------------------
# Default bridge function
# ---------------------------------------------------------------------------

def _default_bridge(items: List[MemoryItem]) -> Dict[str, Any]:
    """Extract query hints from primary results for the secondary query.

    Looks for subject/label keywords in the primary items and passes them
    as secondary query text.
    """
    keywords: List[str] = []
    for item in items:
        if item.source == MemorySource.FACT_STORE and item.data:
            keywords.append(item.data.subject)
        elif item.source == MemorySource.KNOWLEDGE_GRAPH and item.data:
            keywords.append(item.data.label)
        elif item.source == MemorySource.EVENT_STORE and item.data:
            for v in item.data.data.values():
                if isinstance(v, str):
                    keywords.append(v)
        elif item.source == MemorySource.EXPERIENCE_TRACKER and item.data:
            keywords.append(item.data.action)

    if keywords:
        return {"query_text": " ".join(dict.fromkeys(keywords))}
    return {}
