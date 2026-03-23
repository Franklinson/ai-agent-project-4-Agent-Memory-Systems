"""Hybrid memory system combining episodic and semantic memory.

Provides a unified interface over:
- Episodic memory: EventStore (temporal events) and ExperienceTracker (learning)
- Semantic memory: FactStore (triples) and KnowledgeGraph (nodes/edges)

Supports type detection, query routing, result merging, and cross-memory linking.
"""

import sys
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))

from event_store import EventStore, EventType, Event, EventNotFoundError
from experience_tracker import ExperienceTracker, Outcome, Experience
from fact_store import FactStore, FactType, Fact, FactNotFoundError
from knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType, Node, Edge,
    NodeNotFoundError,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class HybridMemoryError(Exception):
    """Base exception for hybrid memory operations."""


class MemoryNotFoundError(HybridMemoryError):
    """Raised when a memory item is not found."""


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    BOTH = "both"


class MemorySource(str, Enum):
    EVENT_STORE = "event_store"
    EXPERIENCE_TRACKER = "experience_tracker"
    FACT_STORE = "fact_store"
    KNOWLEDGE_GRAPH = "knowledge_graph"


@dataclass
class MemoryItem:
    """Unified wrapper around any memory object."""
    id: str
    memory_type: MemoryType
    source: MemorySource
    content: str
    timestamp: datetime
    data: Any = None  # original object
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance: float = 1.0


@dataclass
class QueryResult:
    """Merged results from one or more memory subsystems."""
    query: str
    items: List[MemoryItem] = field(default_factory=list)
    sources_queried: List[MemorySource] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def by_source(self, source: MemorySource) -> List[MemoryItem]:
        return [i for i in self.items if i.source == source]

    def by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        return [i for i in self.items if i.memory_type == memory_type]


@dataclass
class CrossLink:
    """A link between an episodic and a semantic memory item."""
    episodic_id: str
    episodic_source: MemorySource
    semantic_id: str
    semantic_source: MemorySource
    relationship: str = ""
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Helpers — wrap subsystem objects into MemoryItem
# ---------------------------------------------------------------------------

def _wrap_event(event: Event) -> MemoryItem:
    content = f"[{event.event_type.value}] {event.data}"
    return MemoryItem(
        id=event.id, memory_type=MemoryType.EPISODIC,
        source=MemorySource.EVENT_STORE, content=content,
        timestamp=event.timestamp, data=event,
        metadata={"event_type": event.event_type.value,
                  "participants": event.participants},
    )


def _wrap_experience(exp: Experience) -> MemoryItem:
    content = f"{exp.action}: {exp.outcome.value}"
    if exp.feedback:
        content += f" — {exp.feedback}"
    return MemoryItem(
        id=exp.id, memory_type=MemoryType.EPISODIC,
        source=MemorySource.EXPERIENCE_TRACKER, content=content,
        timestamp=exp.timestamp, data=exp,
        metadata={"action": exp.action, "outcome": exp.outcome.value,
                  "score": exp.score, "tags": exp.tags},
    )


def _wrap_fact(fact: Fact) -> MemoryItem:
    content = f"{fact.subject} {fact.predicate} {fact.object}"
    return MemoryItem(
        id=fact.id, memory_type=MemoryType.SEMANTIC,
        source=MemorySource.FACT_STORE, content=content,
        timestamp=fact.created_at, data=fact,
        metadata={"fact_type": fact.fact_type.value,
                  "confidence": fact.confidence},
        relevance=fact.confidence,
    )


def _wrap_node(node: Node) -> MemoryItem:
    content = f"{node.label} ({node.node_type.value})"
    return MemoryItem(
        id=node.id, memory_type=MemoryType.SEMANTIC,
        source=MemorySource.KNOWLEDGE_GRAPH, content=content,
        timestamp=node.created_at, data=node,
        metadata={"node_type": node.node_type.value,
                  "properties": node.properties},
    )


# ---------------------------------------------------------------------------
# HybridMemory
# ---------------------------------------------------------------------------

class HybridMemory:
    """Unified interface over episodic and semantic memory subsystems."""

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        experience_tracker: Optional[ExperienceTracker] = None,
        fact_store: Optional[FactStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        self.events = event_store or EventStore()
        self.experiences = experience_tracker or ExperienceTracker()
        self.facts = fact_store or FactStore()
        self.graph = knowledge_graph or KnowledgeGraph()
        self._cross_links: Dict[str, CrossLink] = {}  # key = "ep_id:sem_id"

    # ------------------------------------------------------------------
    # Unified store helpers
    # ------------------------------------------------------------------

    def store_event(self, event_type: EventType, **kwargs) -> MemoryItem:
        """Store an episodic event and return a unified MemoryItem."""
        event = self.events.store(event_type, **kwargs)
        return _wrap_event(event)

    def store_experience(self, action: str, outcome: Outcome, **kwargs) -> MemoryItem:
        """Log an experience and return a unified MemoryItem."""
        exp = self.experiences.log(action, outcome, **kwargs)
        return _wrap_experience(exp)

    def store_fact(self, subject: str, predicate: str, object_: str, **kwargs) -> MemoryItem:
        """Store a semantic fact and return a unified MemoryItem."""
        fact = self.facts.store(subject, predicate, object_, **kwargs)
        return _wrap_fact(fact)

    def store_node(self, label: str, node_type: NodeType = NodeType.ENTITY, **kwargs) -> MemoryItem:
        """Add a knowledge-graph node and return a unified MemoryItem."""
        node = self.graph.add_node(label, node_type, **kwargs)
        return _wrap_node(node)

    def store_edge(self, source_id: str, target_id: str,
                   edge_type: EdgeType = EdgeType.RELATED_TO, **kwargs) -> Edge:
        """Add a knowledge-graph edge (thin pass-through)."""
        return self.graph.add_edge(source_id, target_id, edge_type, **kwargs)

    # ------------------------------------------------------------------
    # Unified get
    # ------------------------------------------------------------------

    def get(self, item_id: str, source: Optional[MemorySource] = None) -> MemoryItem:
        """Retrieve a memory item by ID, optionally scoped to a source.

        Without *source*, tries each subsystem in turn.
        """
        sources = [source] if source else list(MemorySource)
        for src in sources:
            try:
                if src == MemorySource.EVENT_STORE:
                    return _wrap_event(self.events.get(item_id))
                if src == MemorySource.EXPERIENCE_TRACKER:
                    return _wrap_experience(self.experiences.get(item_id))
                if src == MemorySource.FACT_STORE:
                    return _wrap_fact(self.facts.get(item_id))
                if src == MemorySource.KNOWLEDGE_GRAPH:
                    return _wrap_node(self.graph.get_node(item_id))
            except Exception:
                continue
        raise MemoryNotFoundError(f"Memory item '{item_id}' not found")

    # ------------------------------------------------------------------
    # Unified delete
    # ------------------------------------------------------------------

    def delete(self, item_id: str, source: Optional[MemorySource] = None) -> bool:
        """Delete a memory item. Cleans up any cross-links."""
        sources = [source] if source else list(MemorySource)
        for src in sources:
            try:
                if src == MemorySource.EVENT_STORE:
                    self.events.delete(item_id)
                elif src == MemorySource.FACT_STORE:
                    self.facts.delete(item_id)
                elif src == MemorySource.KNOWLEDGE_GRAPH:
                    self.graph.remove_node(item_id)
                else:
                    continue
                self._remove_cross_links_for(item_id)
                return True
            except Exception:
                continue
        raise MemoryNotFoundError(f"Memory item '{item_id}' not found")

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    def detect_type(self, item_id: str) -> Optional[MemoryType]:
        """Detect whether an ID belongs to episodic or semantic memory."""
        for src in (MemorySource.EVENT_STORE, MemorySource.EXPERIENCE_TRACKER):
            try:
                self.get(item_id, source=src)
                return MemoryType.EPISODIC
            except (MemoryNotFoundError, Exception):
                continue
        for src in (MemorySource.FACT_STORE, MemorySource.KNOWLEDGE_GRAPH):
            try:
                self.get(item_id, source=src)
                return MemoryType.SEMANTIC
            except (MemoryNotFoundError, Exception):
                continue
        return None

    def detect_source(self, item_id: str) -> Optional[MemorySource]:
        """Detect which subsystem owns an ID."""
        for src in MemorySource:
            try:
                self.get(item_id, source=src)
                return src
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        memory_type: MemoryType = MemoryType.BOTH,
        *,
        event_type: Optional[EventType] = None,
        participant: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[Outcome] = None,
        tag: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        fact_type: Optional[FactType] = None,
        min_confidence: Optional[float] = None,
        node_type: Optional[NodeType] = None,
        node_label: Optional[str] = None,
        time_range: Optional[tuple] = None,
        limit: Optional[int] = None,
    ) -> QueryResult:
        """Route a query to the appropriate subsystems and merge results."""
        items: List[MemoryItem] = []
        sources_queried: List[MemorySource] = []

        if memory_type in (MemoryType.EPISODIC, MemoryType.BOTH):
            items.extend(self._query_episodic(
                query_text, event_type=event_type, participant=participant,
                action=action, outcome=outcome, tag=tag,
                time_range=time_range,
            ))
            sources_queried.extend([
                MemorySource.EVENT_STORE, MemorySource.EXPERIENCE_TRACKER])

        if memory_type in (MemoryType.SEMANTIC, MemoryType.BOTH):
            items.extend(self._query_semantic(
                query_text, subject=subject, predicate=predicate,
                object_=object_, fact_type=fact_type,
                min_confidence=min_confidence, node_type=node_type,
                node_label=node_label,
            ))
            sources_queried.extend([
                MemorySource.FACT_STORE, MemorySource.KNOWLEDGE_GRAPH])

        # Sort by timestamp descending (most recent first)
        items.sort(key=lambda i: i.timestamp, reverse=True)
        if limit is not None:
            items = items[:limit]

        return QueryResult(query=query_text, items=items,
                           sources_queried=sources_queried)

    # ------------------------------------------------------------------
    # Cross-memory linking
    # ------------------------------------------------------------------

    def cross_link(self, episodic_id: str, semantic_id: str,
                   relationship: str = "") -> CrossLink:
        """Create a link between an episodic and a semantic memory item."""
        ep_item = self.get(episodic_id)
        sem_item = self.get(semantic_id)
        if ep_item.memory_type != MemoryType.EPISODIC:
            raise HybridMemoryError(
                f"'{episodic_id}' is not an episodic memory item")
        if sem_item.memory_type != MemoryType.SEMANTIC:
            raise HybridMemoryError(
                f"'{semantic_id}' is not a semantic memory item")

        key = f"{episodic_id}:{semantic_id}"
        link = CrossLink(
            episodic_id=episodic_id, episodic_source=ep_item.source,
            semantic_id=semantic_id, semantic_source=sem_item.source,
            relationship=relationship,
        )
        self._cross_links[key] = link
        return link

    def get_cross_links(self, item_id: str) -> List[CrossLink]:
        """Get all cross-links involving the given ID."""
        return [
            cl for cl in self._cross_links.values()
            if cl.episodic_id == item_id or cl.semantic_id == item_id
        ]

    def get_linked_items(self, item_id: str) -> List[MemoryItem]:
        """Get all memory items linked to the given ID via cross-links."""
        links = self.get_cross_links(item_id)
        items: List[MemoryItem] = []
        for cl in links:
            target_id = cl.semantic_id if cl.episodic_id == item_id else cl.episodic_id
            try:
                items.append(self.get(target_id))
            except MemoryNotFoundError:
                continue
        return items

    def remove_cross_link(self, episodic_id: str, semantic_id: str) -> bool:
        """Remove a specific cross-link."""
        key = f"{episodic_id}:{semantic_id}"
        if key not in self._cross_links:
            return False
        del self._cross_links[key]
        return True

    # ------------------------------------------------------------------
    # Summaries / stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return counts from each subsystem."""
        return {
            "events": self.events.count,
            "experiences": self.experiences.count,
            "facts": self.facts.count,
            "graph_nodes": self.graph.node_count,
            "graph_edges": self.graph.edge_count,
            "cross_links": len(self._cross_links),
        }

    def get_all(self, memory_type: MemoryType = MemoryType.BOTH,
                limit: Optional[int] = None) -> List[MemoryItem]:
        """Return all memory items of the given type, newest first."""
        items: List[MemoryItem] = []
        if memory_type in (MemoryType.EPISODIC, MemoryType.BOTH):
            items.extend(_wrap_event(e) for e in self.events.timeline())
            for eid in list(self.experiences._timeline):
                items.append(_wrap_experience(self.experiences._experiences[eid]))
        if memory_type in (MemoryType.SEMANTIC, MemoryType.BOTH):
            for f in self.facts.query():
                items.append(_wrap_fact(f))
            for n in self.graph.find_nodes():
                items.append(_wrap_node(n))
        items.sort(key=lambda i: i.timestamp, reverse=True)
        if limit is not None:
            items = items[:limit]
        return items

    # ------------------------------------------------------------------
    # Internal query helpers
    # ------------------------------------------------------------------

    def _query_episodic(
        self, query_text: str, *,
        event_type: Optional[EventType] = None,
        participant: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[Outcome] = None,
        tag: Optional[str] = None,
        time_range: Optional[tuple] = None,
    ) -> List[MemoryItem]:
        items: List[MemoryItem] = []

        # --- EventStore ---
        events: Optional[List[Event]] = None
        if event_type:
            events = self.events.by_type(event_type)
        elif participant:
            events = self.events.by_participant(participant)
        elif time_range:
            events = self.events.by_time_range(time_range[0], time_range[1])
        else:
            events = self.events.timeline()

        if time_range and event_type:
            events = [e for e in events
                      if time_range[0] <= e.timestamp <= time_range[1]]
        if participant and event_type:
            events = [e for e in events if participant in e.participants]

        q = query_text.lower()
        for ev in (events or []):
            if q and q not in str(ev.data).lower() and q not in ev.event_type.value:
                continue
            items.append(_wrap_event(ev))

        # --- ExperienceTracker ---
        experiences: List[Experience]
        if action:
            experiences = self.experiences.by_action(action)
        elif outcome:
            experiences = self.experiences.by_outcome(outcome)
        elif tag:
            experiences = self.experiences.by_tag(tag)
        elif time_range:
            experiences = self.experiences.by_time_range(
                time_range[0], time_range[1])
        else:
            experiences = [
                self.experiences._experiences[eid]
                for eid in self.experiences._timeline
            ]

        for exp in experiences:
            if q and q not in exp.action.lower() and q not in (exp.feedback or "").lower():
                continue
            items.append(_wrap_experience(exp))

        return items

    def _query_semantic(
        self, query_text: str, *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        fact_type: Optional[FactType] = None,
        min_confidence: Optional[float] = None,
        node_type: Optional[NodeType] = None,
        node_label: Optional[str] = None,
    ) -> List[MemoryItem]:
        items: List[MemoryItem] = []

        # --- FactStore ---
        facts = self.facts.query(
            subject=subject, predicate=predicate, object_=object_,
            fact_type=fact_type, min_confidence=min_confidence,
        )
        q = query_text.lower()
        for f in facts:
            text = f"{f.subject} {f.predicate} {f.object}".lower()
            if q and q not in text:
                continue
            items.append(_wrap_fact(f))

        # --- KnowledgeGraph ---
        nodes = self.graph.find_nodes(node_type=node_type, label=node_label)
        for n in nodes:
            if q and q not in n.label.lower():
                continue
            items.append(_wrap_node(n))

        return items

    def _remove_cross_links_for(self, item_id: str) -> None:
        keys_to_remove = [
            k for k, cl in self._cross_links.items()
            if cl.episodic_id == item_id or cl.semantic_id == item_id
        ]
        for k in keys_to_remove:
            del self._cross_links[k]
