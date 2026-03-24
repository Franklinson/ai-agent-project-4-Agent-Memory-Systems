"""Unified memory system integrating all memory types.

Provides a single interface over:
- Working memory: ConversationManager, ContextManager, TokenBudgetManager
- Short-term memory: EventStore, ExperienceTracker
- Long-term memory: FactStore, KnowledgeGraph
- Hybrid memory: HybridMemory with MemoryRouter
- Cross-type queries: CrossTypeQueryEngine

Supports automatic type detection, intelligent routing, and seamless access.
"""

import sys
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Add paths for all memory systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_47'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_49'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_50'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'day_51'))

from conversation_manager import ConversationManager, TruncationStrategy
from context_manager import ContextManager, Priority
from token_counter import TokenBudgetManager
from event_store import EventStore, EventType
from experience_tracker import ExperienceTracker, Outcome
from fact_store import FactStore, FactType
from knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from hybrid_memory import HybridMemory, MemoryType, MemorySource, MemoryItem
from memory_router import MemoryRouter, KeywordRoutingStrategy
from cross_type_queries import CrossTypeQueryEngine, QueryPattern, RankingStrategy


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UnifiedMemoryError(Exception):
    """Base exception for unified memory operations."""


class MemoryTypeError(UnifiedMemoryError):
    """Raised when memory type operations fail."""


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class MemoryLayer(str, Enum):
    WORKING = "working"      # Active conversation, context, tokens
    SHORT_TERM = "short_term"  # Recent events, experiences
    LONG_TERM = "long_term"   # Facts, knowledge graph
    HYBRID = "hybrid"        # Cross-type integration


@dataclass
class UnifiedResult:
    """Unified result from any memory operation."""
    query: str
    items: List[MemoryItem] = field(default_factory=list)
    layer: Optional[MemoryLayer] = None
    pattern: Optional[QueryPattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# UnifiedMemory
# ---------------------------------------------------------------------------

class UnifiedMemory:
    """Unified interface over all memory types and layers."""
    
    def __init__(
        self,
        total_budget: int = 8000,
        response_reserve: int = 1000,
        max_messages: int = 20,
        max_context_tokens: Optional[int] = None,
        enable_coordination: bool = True,
    ):
        # Working memory (conversation, context, tokens)
        self.token_manager = TokenBudgetManager(total_budget, response_reserve=response_reserve)
        self.conversation = ConversationManager(max_messages=max_messages)
        self.context = ContextManager(
            self.token_manager, 
            self.conversation,
            max_context_tokens or self.token_manager.get_available_for_context()
        )
        
        # Memory subsystems
        self.events = EventStore()
        self.experiences = ExperienceTracker()
        self.facts = FactStore()
        self.graph = KnowledgeGraph()
        
        # Hybrid integration
        self.hybrid = HybridMemory(
            event_store=self.events,
            experience_tracker=self.experiences,
            fact_store=self.facts,
            knowledge_graph=self.graph
        )
        
        # Intelligent routing
        self.router = MemoryRouter(self.hybrid, KeywordRoutingStrategy())
        
        # Cross-type queries
        self.cross_queries = CrossTypeQueryEngine(self.hybrid)
        
        # Memory coordination (optional)
        if enable_coordination:
            from memory_coordinator import MemoryCoordinator
            self.coordinator = MemoryCoordinator(self)
        else:
            self.coordinator = None
    
    # ------------------------------------------------------------------
    # Unified store interface
    # ------------------------------------------------------------------
    
    def store(
        self,
        content: str,
        memory_type: Optional[str] = None,
        **kwargs
    ) -> MemoryItem:
        """Store content with automatic type detection."""
        if memory_type == "message" or "role" in kwargs:
            role = kwargs.get("role", "user")
            metadata = kwargs.get("metadata", {})
            self.conversation.add_message(role, content, metadata)
            # Return a synthetic MemoryItem for consistency
            return MemoryItem(
                id=f"msg_{len(self.conversation.messages)}",
                memory_type=MemoryType.EPISODIC,
                source=MemorySource.EVENT_STORE,
                content=f"{role}: {content}",
                timestamp=datetime.now(),
                metadata={"layer": MemoryLayer.WORKING}
            )
        
        elif memory_type == "event" or "event_type" in kwargs:
            event_type = kwargs.pop("event_type", EventType.CUSTOM)
            return self.hybrid.store_event(event_type, data={"content": content}, **kwargs)
        
        elif memory_type == "experience" or "outcome" in kwargs:
            action = kwargs.pop("action", "action")
            outcome = kwargs.pop("outcome", Outcome.SUCCESS)
            return self.hybrid.store_experience(action, outcome, feedback=content, **kwargs)
        
        elif memory_type == "fact" or all(k in kwargs for k in ["subject", "predicate"]):
            subject = kwargs.pop("subject", "unknown")
            predicate = kwargs.pop("predicate", "relates_to")
            return self.hybrid.store_fact(subject, predicate, content, **kwargs)
        
        elif memory_type == "node" or "node_type" in kwargs:
            node_type = kwargs.pop("node_type", NodeType.ENTITY)
            return self.hybrid.store_node(content, node_type, **kwargs)
        
        else:
            # Default: store as fact
            return self.hybrid.store_fact("user_input", "contains", content)
    
    def get(self, item_id: str, layer: Optional[MemoryLayer] = None) -> MemoryItem:
        """Retrieve item by ID with optional layer scoping."""
        if layer == MemoryLayer.WORKING:
            # Check conversation messages
            try:
                msg_idx = int(item_id.split("_")[1]) - 1
                if 0 <= msg_idx < len(self.conversation.messages):
                    msg = self.conversation.messages[msg_idx]
                    return MemoryItem(
                        id=item_id,
                        memory_type=MemoryType.EPISODIC,
                        source=MemorySource.EVENT_STORE,
                        content=f"{msg.role}: {msg.content}",
                        timestamp=msg.timestamp,
                        metadata={"layer": MemoryLayer.WORKING}
                    )
            except (ValueError, IndexError):
                pass
        
        # Delegate to hybrid memory
        return self.hybrid.get(item_id)
    
    def delete(self, item_id: str, layer: Optional[MemoryLayer] = None) -> bool:
        """Delete item with layer awareness."""
        if layer == MemoryLayer.WORKING:
            # Cannot delete individual messages, only clear all
            return False
        return self.hybrid.delete(item_id)
    
    # ------------------------------------------------------------------
    # Unified query interface
    # ------------------------------------------------------------------
    
    def query(
        self,
        query_text: str,
        layer: Optional[MemoryLayer] = None,
        pattern: Optional[QueryPattern] = None,
        use_coordination: bool = False,
        **kwargs
    ) -> UnifiedResult:
        """Query across all memory layers with intelligent routing."""
        
        # Use coordination if enabled and requested
        if use_coordination and self.coordinator:
            from memory_coordinator import CoordinationPattern
            coord_pattern = None
            if pattern == QueryPattern.COMBINED:
                coord_pattern = CoordinationPattern.PARALLEL
            elif pattern == QueryPattern.SEQUENTIAL:
                coord_pattern = CoordinationPattern.SEQUENTIAL
            
            coord_result = self.coordinator.execute_coordinated_query(
                query_text, pattern=coord_pattern, **kwargs
            )
            
            # Convert to UnifiedResult
            return UnifiedResult(
                query=query_text,
                items=coord_result.items,
                layer=MemoryLayer.HYBRID,
                pattern=pattern,
                metadata={
                    "coordination_pattern": coord_result.pattern.value,
                    "execution_time": coord_result.execution_time,
                    "sources": [s.value for s in coord_result.sources]
                }
            )
        
        if layer == MemoryLayer.WORKING:
            return self._query_working(query_text, **kwargs)
        elif layer == MemoryLayer.SHORT_TERM:
            return self._query_short_term(query_text, **kwargs)
        elif layer == MemoryLayer.LONG_TERM:
            return self._query_long_term(query_text, **kwargs)
        elif pattern:
            return self._query_cross_type(query_text, pattern, **kwargs)
        else:
            # Intelligent routing
            result = self.router.route(query_text, **kwargs)
            return UnifiedResult(
                query=query_text,
                items=result.items,
                layer=MemoryLayer.HYBRID,
                metadata={"sources": result.sources_queried}
            )
    
    def _query_working(self, query_text: str, **kwargs) -> UnifiedResult:
        """Query working memory (conversation, context)."""
        items = []
        q = query_text.lower()
        
        # Search conversation messages
        for i, msg in enumerate(self.conversation.messages):
            if q in msg.content.lower() or q in msg.role.lower():
                item = MemoryItem(
                    id=f"msg_{i+1}",
                    memory_type=MemoryType.EPISODIC,
                    source=MemorySource.EVENT_STORE,
                    content=f"{msg.role}: {msg.content}",
                    timestamp=msg.timestamp,
                    metadata={"layer": MemoryLayer.WORKING}
                )
                items.append(item)
        
        return UnifiedResult(
            query=query_text,
            items=items,
            layer=MemoryLayer.WORKING
        )
    
    def _query_short_term(self, query_text: str, **kwargs) -> UnifiedResult:
        """Query short-term memory (events, experiences)."""
        result = self.hybrid.query(
            query_text,
            memory_type=MemoryType.EPISODIC,
            **kwargs
        )
        return UnifiedResult(
            query=query_text,
            items=result.items,
            layer=MemoryLayer.SHORT_TERM,
            metadata={"sources": result.sources_queried}
        )
    
    def _query_long_term(self, query_text: str, **kwargs) -> UnifiedResult:
        """Query long-term memory (facts, knowledge graph)."""
        result = self.hybrid.query(
            query_text,
            memory_type=MemoryType.SEMANTIC,
            **kwargs
        )
        return UnifiedResult(
            query=query_text,
            items=result.items,
            layer=MemoryLayer.LONG_TERM,
            metadata={"sources": result.sources_queried}
        )
    
    def _query_cross_type(self, query_text: str, pattern: QueryPattern, **kwargs) -> UnifiedResult:
        """Execute cross-type query patterns."""
        if pattern == QueryPattern.COMBINED:
            result = self.cross_queries.combined(query_text, **kwargs)
        elif pattern == QueryPattern.SEQUENTIAL:
            result = self.cross_queries.sequential(query_text, **kwargs)
        elif pattern == QueryPattern.PARALLEL:
            # Convert to SubQuery format if needed
            from cross_type_queries import SubQuery
            sub_queries = kwargs.get("sub_queries", [
                SubQuery(query_text, MemoryType.EPISODIC, "episodic"),
                SubQuery(query_text, MemoryType.SEMANTIC, "semantic")
            ])
            result = self.cross_queries.parallel(sub_queries, **kwargs)
        else:
            result = self.cross_queries.combined(query_text, **kwargs)
        
        return UnifiedResult(
            query=query_text,
            items=result.items,
            pattern=pattern,
            metadata=result.metadata
        )
    
    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    
    def add_to_context(
        self,
        name: str,
        content: str,
        priority: Priority = Priority.MEDIUM,
        **kwargs
    ) -> None:
        """Add content to working context."""
        self.context.add_component(name, content, priority, **kwargs)
    
    def build_context(self, include_memory: bool = True, **kwargs) -> str:
        """Build complete context including memory retrieval."""
        if include_memory and kwargs.get("query"):
            # Retrieve relevant memory
            query = kwargs.pop("query")  # Remove query from kwargs to avoid conflicts
            memory_result = self.query(query, **kwargs)
            if memory_result.items:
                memory_content = "\n".join(
                    f"- {item.content}" for item in memory_result.items[:5]
                )
                self.context.add_component(
                    "retrieved_memory",
                    f"Relevant memory:\n{memory_content}",
                    Priority.HIGH
                )
        
        return self.context.build_context()
    
    def build_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Build message list for LLM APIs."""
        return self.context.build_messages(**kwargs)
    
    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    
    def clear_working_memory(self) -> None:
        """Clear working memory (conversation, context)."""
        self.conversation.clear()
        self.context.clear()
        self.token_manager.reset_tracking()
    
    def clear_short_term_memory(self) -> None:
        """Clear short-term memory (events, experiences)."""
        # Note: EventStore and ExperienceTracker don't have clear methods
        # This would need to be implemented in those classes
        pass
    
    def clear_all_memory(self) -> None:
        """Clear all memory layers."""
        self.clear_working_memory()
        self.clear_short_term_memory()
        # Note: FactStore and KnowledgeGraph don't have clear methods
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "working": {
                "messages": self.conversation.get_message_count(),
                "context_components": len(self.context.components),
                "tokens_used": self.token_manager.tracker.get_total_usage(),
                "tokens_remaining": self.token_manager.tracker.get_remaining_budget()
            },
            "hybrid": self.hybrid.stats(),
            "routing": self.router.stats()
        }
    
    # ------------------------------------------------------------------
    # Advanced features
    # ------------------------------------------------------------------
    
    def link_memories(
        self,
        episodic_id: str,
        semantic_id: str,
        relationship: str = ""
    ) -> None:
        """Create cross-memory links."""
        self.hybrid.cross_link(episodic_id, semantic_id, relationship)
    
    def get_related_memories(self, item_id: str) -> List[MemoryItem]:
        """Get memories related to an item."""
        return self.hybrid.get_linked_items(item_id)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage and return statistics."""
        # Optimize context
        self.context.handle_overflow()
        
        # Get optimization stats
        return {
            "context_optimized": True,
            "overflow_detected": self.context.detect_overflow(),
            "memory_stats": self.get_memory_stats()
        }
    
    def export_memory(self, layer: Optional[MemoryLayer] = None) -> Dict[str, Any]:
        """Export memory data for backup/analysis."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_memory_stats()
        }
        
        if layer is None or layer == MemoryLayer.WORKING:
            export_data["working"] = {
                "messages": self.conversation.get_history(),
                "context": [c.to_dict() for c in self.context.components]
            }
        
        if layer is None or layer in (MemoryLayer.SHORT_TERM, MemoryLayer.LONG_TERM, MemoryLayer.HYBRID):
            # Export hybrid memory items
            all_items = self.hybrid.get_all()
            export_data["items"] = [
                {
                    "id": item.id,
                    "type": item.memory_type.value,
                    "source": item.source.value,
                    "content": item.content,
                    "timestamp": item.timestamp.isoformat(),
                    "metadata": item.metadata
                }
                for item in all_items
            ]
        
        return export_data