"""Memory coordination mechanisms for managing interactions between memory types.

Provides query routing, result merging, state synchronization, and event coordination
across working, short-term, long-term, and hybrid memory systems.
"""

import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths for all memory systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_47'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_51'))

from unified_memory import UnifiedMemory, MemoryLayer, UnifiedResult
from hybrid_memory import MemoryItem, MemoryType, MemorySource
from cross_type_queries import QueryPattern, RankingStrategy


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CoordinationError(Exception):
    """Base exception for coordination operations."""


class RoutingError(CoordinationError):
    """Raised when query routing fails."""


class MergingError(CoordinationError):
    """Raised when result merging fails."""


class SynchronizationError(CoordinationError):
    """Raised when state synchronization fails."""


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class CoordinationPattern(str, Enum):
    SEQUENTIAL = "sequential"      # Process one type at a time
    PARALLEL = "parallel"          # Process multiple types concurrently
    HIERARCHICAL = "hierarchical"  # Process in priority order
    ADAPTIVE = "adaptive"          # Adapt based on query characteristics


class EventType(str, Enum):
    MEMORY_STORED = "memory_stored"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    QUERY_EXECUTED = "query_executed"
    SYNC_REQUIRED = "sync_required"
    CONFLICT_DETECTED = "conflict_detected"


@dataclass
class CoordinationEvent:
    """Event for coordinating memory operations."""
    event_type: EventType
    source_layer: MemoryLayer
    target_layers: List[MemoryLayer] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    handled: bool = False


@dataclass
class RoutingDecision:
    """Decision about how to route a query."""
    query: str
    target_layers: List[MemoryLayer]
    pattern: CoordinationPattern
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergedResult:
    """Result from merging multiple memory queries."""
    query: str
    items: List[MemoryItem] = field(default_factory=list)
    sources: List[MemoryLayer] = field(default_factory=list)
    pattern: CoordinationPattern = CoordinationPattern.SEQUENTIAL
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        return len(self.items)


@dataclass
class SyncState:
    """State synchronization information."""
    layer: MemoryLayer
    last_sync: datetime = field(default_factory=datetime.now)
    version: int = 0
    checksum: str = ""
    conflicts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class QueryRouter(ABC):
    """Abstract interface for query routing strategies."""
    
    @abstractmethod
    def route(self, query: str, **kwargs) -> RoutingDecision:
        """Route a query to appropriate memory layers."""
        pass


class ResultMerger(ABC):
    """Abstract interface for result merging strategies."""
    
    @abstractmethod
    def merge(self, results: List[UnifiedResult]) -> MergedResult:
        """Merge results from multiple memory layers."""
        pass


class EventHandler(ABC):
    """Abstract interface for event handling."""
    
    @abstractmethod
    def handle_event(self, event: CoordinationEvent) -> bool:
        """Handle a coordination event."""
        pass


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class KeywordQueryRouter(QueryRouter):
    """Routes queries based on keyword analysis."""
    
    def __init__(self):
        self.working_keywords = {"conversation", "message", "chat", "talk", "said"}
        self.short_term_keywords = {"recent", "happened", "event", "experience", "tried"}
        self.long_term_keywords = {"fact", "knowledge", "definition", "concept", "entity"}
        self.hybrid_keywords = {"related", "connected", "linked", "similar"}
    
    def route(self, query: str, **kwargs) -> RoutingDecision:
        """Route query based on keyword matching."""
        q = query.lower()
        scores = {
            MemoryLayer.WORKING: sum(1 for kw in self.working_keywords if kw in q),
            MemoryLayer.SHORT_TERM: sum(1 for kw in self.short_term_keywords if kw in q),
            MemoryLayer.LONG_TERM: sum(1 for kw in self.long_term_keywords if kw in q),
            MemoryLayer.HYBRID: sum(1 for kw in self.hybrid_keywords if kw in q)
        }
        
        # Determine target layers
        max_score = max(scores.values())
        if max_score == 0:
            # No specific keywords, search all layers
            target_layers = list(MemoryLayer)
            pattern = CoordinationPattern.PARALLEL
            confidence = 0.3
        else:
            target_layers = [layer for layer, score in scores.items() if score == max_score]
            pattern = CoordinationPattern.SEQUENTIAL if len(target_layers) == 1 else CoordinationPattern.PARALLEL
            confidence = max_score / (len(q.split()) + 1)
        
        return RoutingDecision(
            query=query,
            target_layers=target_layers,
            pattern=pattern,
            confidence=confidence,
            metadata={"scores": scores}
        )


class RelevanceResultMerger(ResultMerger):
    """Merges results based on relevance scoring."""
    
    def merge(self, results: List[UnifiedResult]) -> MergedResult:
        """Merge results with relevance-based ranking."""
        all_items = []
        sources = []
        
        for result in results:
            all_items.extend(result.items)
            if result.layer:
                sources.append(result.layer)
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_items = []
        for item in all_items:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_items.append(item)
        
        # Sort by relevance (descending) then timestamp (descending)
        unique_items.sort(key=lambda x: (x.relevance, x.timestamp), reverse=True)
        
        return MergedResult(
            query=results[0].query if results else "",
            items=unique_items,
            sources=list(set(sources)),
            metadata={"original_count": len(all_items), "unique_count": len(unique_items)}
        )


class DefaultEventHandler(EventHandler):
    """Default event handler for coordination events."""
    
    def __init__(self):
        self.handled_events: List[CoordinationEvent] = []
    
    def handle_event(self, event: CoordinationEvent) -> bool:
        """Handle coordination event with basic logging."""
        self.handled_events.append(event)
        event.handled = True
        return True


# ---------------------------------------------------------------------------
# Memory Coordinator
# ---------------------------------------------------------------------------

class MemoryCoordinator:
    """Coordinates interactions between different memory types."""
    
    def __init__(
        self,
        unified_memory: UnifiedMemory,
        router: Optional[QueryRouter] = None,
        merger: Optional[ResultMerger] = None,
        event_handler: Optional[EventHandler] = None
    ):
        self.memory = unified_memory
        self.router = router or KeywordQueryRouter()
        self.merger = merger or RelevanceResultMerger()
        self.event_handler = event_handler or DefaultEventHandler()
        
        # Synchronization state
        self.sync_states: Dict[MemoryLayer, SyncState] = {
            layer: SyncState(layer=layer) for layer in MemoryLayer
        }
        self.sync_lock = threading.Lock()
        
        # Event queue
        self.event_queue: List[CoordinationEvent] = []
        self.event_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "queries_routed": 0,
            "results_merged": 0,
            "events_handled": 0,
            "sync_operations": 0
        }
    
    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------
    
    def route_query(self, query: str, **kwargs) -> RoutingDecision:
        """Route a query to appropriate memory layers."""
        try:
            decision = self.router.route(query, **kwargs)
            self.stats["queries_routed"] += 1
            
            # Emit routing event
            self._emit_event(CoordinationEvent(
                event_type=EventType.QUERY_EXECUTED,
                source_layer=MemoryLayer.HYBRID,
                target_layers=decision.target_layers,
                data={"query": query, "confidence": decision.confidence}
            ))
            
            return decision
        except Exception as e:
            raise RoutingError(f"Failed to route query '{query}': {e}")
    
    def execute_coordinated_query(
        self,
        query: str,
        pattern: Optional[CoordinationPattern] = None,
        **kwargs
    ) -> MergedResult:
        """Execute a coordinated query across memory layers."""
        # Route the query
        decision = self.route_query(query, **kwargs)
        if pattern:
            decision.pattern = pattern
        
        # Execute based on pattern
        if decision.pattern == CoordinationPattern.SEQUENTIAL:
            return self._execute_sequential(decision, **kwargs)
        elif decision.pattern == CoordinationPattern.PARALLEL:
            return self._execute_parallel(decision, **kwargs)
        elif decision.pattern == CoordinationPattern.HIERARCHICAL:
            return self._execute_hierarchical(decision, **kwargs)
        elif decision.pattern == CoordinationPattern.ADAPTIVE:
            return self._execute_adaptive(decision, **kwargs)
        else:
            return self._execute_sequential(decision, **kwargs)
    
    def _execute_sequential(self, decision: RoutingDecision, **kwargs) -> MergedResult:
        """Execute query sequentially across layers."""
        results = []
        start_time = datetime.now()
        
        for layer in decision.target_layers:
            try:
                result = self.memory.query(decision.query, layer=layer, **kwargs)
                results.append(result)
            except Exception as e:
                # Continue with other layers on error
                continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        merged = self.merger.merge(results)
        merged.pattern = CoordinationPattern.SEQUENTIAL
        merged.execution_time = execution_time
        
        self.stats["results_merged"] += 1
        return merged
    
    def _execute_parallel(self, decision: RoutingDecision, **kwargs) -> MergedResult:
        """Execute query in parallel across layers."""
        results = []
        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=len(decision.target_layers)) as executor:
            # Submit queries to all layers
            future_to_layer = {
                executor.submit(self.memory.query, decision.query, layer=layer, **kwargs): layer
                for layer in decision.target_layers
            }
            
            # Collect results
            for future in as_completed(future_to_layer):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Continue with other layers on error
                    continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        merged = self.merger.merge(results)
        merged.pattern = CoordinationPattern.PARALLEL
        merged.execution_time = execution_time
        
        self.stats["results_merged"] += 1
        return merged
    
    def _execute_hierarchical(self, decision: RoutingDecision, **kwargs) -> MergedResult:
        """Execute query in hierarchical order (working -> short -> long -> hybrid)."""
        layer_priority = [
            MemoryLayer.WORKING,
            MemoryLayer.SHORT_TERM,
            MemoryLayer.LONG_TERM,
            MemoryLayer.HYBRID
        ]
        
        # Sort target layers by priority
        sorted_layers = [layer for layer in layer_priority if layer in decision.target_layers]
        decision.target_layers = sorted_layers
        
        return self._execute_sequential(decision, **kwargs)
    
    def _execute_adaptive(self, decision: RoutingDecision, **kwargs) -> MergedResult:
        """Execute query adaptively based on characteristics."""
        # Simple adaptive logic: use parallel for multi-layer, sequential for single
        if len(decision.target_layers) > 1:
            decision.pattern = CoordinationPattern.PARALLEL
            return self._execute_parallel(decision, **kwargs)
        else:
            decision.pattern = CoordinationPattern.SEQUENTIAL
            return self._execute_sequential(decision, **kwargs)
    
    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------
    
    def merge_results(
        self,
        results: List[UnifiedResult],
        strategy: str = "relevance"
    ) -> MergedResult:
        """Merge results from multiple queries."""
        try:
            merged = self.merger.merge(results)
            self.stats["results_merged"] += 1
            return merged
        except Exception as e:
            raise MergingError(f"Failed to merge results: {e}")
    
    def rank_items(
        self,
        items: List[MemoryItem],
        strategy: RankingStrategy = RankingStrategy.RELEVANCE
    ) -> List[MemoryItem]:
        """Rank memory items by specified strategy."""
        if strategy == RankingStrategy.RELEVANCE:
            return sorted(items, key=lambda x: x.relevance, reverse=True)
        elif strategy == RankingStrategy.TIMESTAMP:
            return sorted(items, key=lambda x: x.timestamp, reverse=True)
        elif strategy == RankingStrategy.SOURCE_PRIORITY:
            # Define source priority order
            source_priority = {
                MemorySource.EVENT_STORE: 1,
                MemorySource.EXPERIENCE_TRACKER: 2,
                MemorySource.FACT_STORE: 3,
                MemorySource.KNOWLEDGE_GRAPH: 4
            }
            return sorted(items, key=lambda x: (source_priority.get(x.source, 5), -x.relevance))
        else:
            return items
    
    # ------------------------------------------------------------------
    # State synchronization
    # ------------------------------------------------------------------
    
    def synchronize_state(self, layers: Optional[List[MemoryLayer]] = None) -> Dict[MemoryLayer, bool]:
        """Synchronize state across memory layers."""
        if layers is None:
            layers = list(MemoryLayer)
        
        results = {}
        
        with self.sync_lock:
            for layer in layers:
                try:
                    # Get current state
                    current_state = self._get_layer_state(layer)
                    
                    # Check for conflicts
                    conflicts = self._detect_conflicts(layer, current_state)
                    
                    if conflicts:
                        self._emit_event(CoordinationEvent(
                            event_type=EventType.CONFLICT_DETECTED,
                            source_layer=layer,
                            data={"conflicts": conflicts}
                        ))
                        results[layer] = False
                    else:
                        # Update sync state
                        self.sync_states[layer].last_sync = datetime.now()
                        self.sync_states[layer].version += 1
                        self.sync_states[layer].checksum = self._calculate_checksum(current_state)
                        self.sync_states[layer].conflicts.clear()
                        results[layer] = True
                    
                    self.stats["sync_operations"] += 1
                    
                except Exception as e:
                    results[layer] = False
        
        return results
    
    def _get_layer_state(self, layer: MemoryLayer) -> Dict[str, Any]:
        """Get current state of a memory layer."""
        if layer == MemoryLayer.WORKING:
            return {
                "messages": self.memory.conversation.get_message_count(),
                "context_components": len(self.memory.context.components),
                "tokens_used": self.memory.token_manager.tracker.get_total_usage()
            }
        elif layer == MemoryLayer.SHORT_TERM:
            return {
                "events": self.memory.events.count,
                "experiences": self.memory.experiences.count
            }
        elif layer == MemoryLayer.LONG_TERM:
            return {
                "facts": self.memory.facts.count,
                "nodes": self.memory.graph.node_count,
                "edges": self.memory.graph.edge_count
            }
        elif layer == MemoryLayer.HYBRID:
            return self.memory.hybrid.stats()
        else:
            return {}
    
    def _detect_conflicts(self, layer: MemoryLayer, current_state: Dict[str, Any]) -> List[str]:
        """Detect conflicts in layer state."""
        conflicts = []
        sync_state = self.sync_states[layer]
        
        # Simple conflict detection based on unexpected state changes
        if sync_state.version > 0:
            current_checksum = self._calculate_checksum(current_state)
            if current_checksum != sync_state.checksum:
                # State changed without coordination
                conflicts.append(f"Unexpected state change in {layer.value}")
        
        return conflicts
    
    def _calculate_checksum(self, state: Dict[str, Any]) -> str:
        """Calculate checksum for state."""
        import hashlib
        state_str = str(sorted(state.items()))
        return hashlib.md5(state_str.encode()).hexdigest()
    
    # ------------------------------------------------------------------
    # Event coordination
    # ------------------------------------------------------------------
    
    def _emit_event(self, event: CoordinationEvent) -> None:
        """Emit a coordination event."""
        with self.event_lock:
            self.event_queue.append(event)
        
        # Handle event immediately
        self.event_handler.handle_event(event)
        self.stats["events_handled"] += 1
    
    def process_events(self) -> int:
        """Process pending coordination events."""
        processed = 0
        
        with self.event_lock:
            unhandled_events = [e for e in self.event_queue if not e.handled]
            
            for event in unhandled_events:
                try:
                    if self.event_handler.handle_event(event):
                        event.handled = True
                        processed += 1
                except Exception as e:
                    # Log error but continue processing
                    continue
        
        return processed
    
    def get_pending_events(self) -> List[CoordinationEvent]:
        """Get list of pending events."""
        with self.event_lock:
            return [e for e in self.event_queue if not e.handled]
    
    def clear_events(self) -> None:
        """Clear all events from queue."""
        with self.event_lock:
            self.event_queue.clear()
    
    # ------------------------------------------------------------------
    # Coordination patterns
    # ------------------------------------------------------------------
    
    def coordinate_store(
        self,
        content: str,
        target_layers: Optional[List[MemoryLayer]] = None,
        **kwargs
    ) -> Dict[MemoryLayer, MemoryItem]:
        """Coordinate storing content across multiple layers."""
        if target_layers is None:
            target_layers = [MemoryLayer.HYBRID]  # Default to hybrid
        
        results = {}
        
        for layer in target_layers:
            try:
                if layer == MemoryLayer.WORKING:
                    # Store as message
                    item = self.memory.store(content, role=kwargs.get("role", "user"))
                elif layer == MemoryLayer.HYBRID:
                    # Use unified store
                    item = self.memory.store(content, **kwargs)
                else:
                    # Store in specific layer via unified interface
                    item = self.memory.store(content, **kwargs)
                
                results[layer] = item
                
                # Emit storage event
                self._emit_event(CoordinationEvent(
                    event_type=EventType.MEMORY_STORED,
                    source_layer=layer,
                    data={"content": content, "item_id": item.id}
                ))
                
            except Exception as e:
                # Continue with other layers
                continue
        
        return results
    
    def coordinate_update(
        self,
        item_id: str,
        updates: Dict[str, Any],
        target_layers: Optional[List[MemoryLayer]] = None
    ) -> Dict[MemoryLayer, bool]:
        """Coordinate updating content across layers."""
        results = {}
        
        # Emit update event
        self._emit_event(CoordinationEvent(
            event_type=EventType.MEMORY_UPDATED,
            source_layer=MemoryLayer.HYBRID,
            target_layers=target_layers or [],
            data={"item_id": item_id, "updates": updates}
        ))
        
        # Note: Actual update implementation would depend on specific layer capabilities
        # This is a placeholder for coordination logic
        
        return results
    
    # ------------------------------------------------------------------
    # Statistics and monitoring
    # ------------------------------------------------------------------
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            **self.stats,
            "sync_states": {
                layer.value: {
                    "last_sync": state.last_sync.isoformat(),
                    "version": state.version,
                    "conflicts": len(state.conflicts)
                }
                for layer, state in self.sync_states.items()
            },
            "pending_events": len(self.get_pending_events()),
            "total_events": len(self.event_queue)
        }
    
    def reset_stats(self) -> None:
        """Reset coordination statistics."""
        self.stats = {
            "queries_routed": 0,
            "results_merged": 0,
            "events_handled": 0,
            "sync_operations": 0
        }
        self.clear_events()