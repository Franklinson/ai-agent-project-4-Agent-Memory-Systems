"""Example usage of the memory coordination system."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_coordinator import (
    MemoryCoordinator, KeywordQueryRouter, RelevanceResultMerger, DefaultEventHandler,
    CoordinationPattern, EventType, CoordinationEvent
)
from unified_memory import UnifiedMemory, MemoryLayer
from context_manager import Priority
from event_store import EventType as ESEventType
from experience_tracker import Outcome
from fact_store import FactType
from knowledge_graph import NodeType
from cross_type_queries import RankingStrategy


def main():
    """Demonstrate memory coordination capabilities."""
    print("=== Memory Coordination System Demo ===\n")
    
    # Initialize unified memory and coordinator
    memory = UnifiedMemory(
        total_budget=4000,
        response_reserve=500,
        max_messages=10
    )
    
    coordinator = MemoryCoordinator(
        unified_memory=memory,
        router=KeywordQueryRouter(),
        merger=RelevanceResultMerger(),
        event_handler=DefaultEventHandler()
    )
    
    print("1. Coordinated Storage Across Layers...")
    
    # Store content across multiple layers
    store_results = coordinator.coordinate_store(
        "Python is a powerful programming language",
        target_layers=[MemoryLayer.WORKING, MemoryLayer.HYBRID],
        role="user"
    )
    
    print(f"✓ Stored in {len(store_results)} layers:")
    for layer, item in store_results.items():
        print(f"  - {layer.value}: {item.id}")
    
    # Store more diverse content
    memory.store("Started learning Python", memory_type="event", event_type=ESEventType.ACTION)
    memory.store("Python tutorial completed", memory_type="experience", 
                action="learning", outcome=Outcome.SUCCESS, score=0.9)
    memory.store("programming language", memory_type="fact", 
                subject="Python", predicate="is_a", fact_type=FactType.DEFINITION)
    memory.store("Django", memory_type="node", node_type=NodeType.ENTITY)
    
    print("\n2. Query Routing Analysis...")
    
    # Test different query types
    queries = [
        "What did we talk about?",  # Working memory
        "What happened recently?",  # Short-term memory
        "What is the definition of Python?",  # Long-term memory
        "What is related to programming?",  # Hybrid memory
        "Tell me about Python programming"  # Multiple layers
    ]
    
    for query in queries:
        decision = coordinator.route_query(query)
        print(f"✓ '{query}'")
        print(f"  → Layers: {[layer.value for layer in decision.target_layers]}")
        print(f"  → Pattern: {decision.pattern.value}")
        print(f"  → Confidence: {decision.confidence:.2f}")
    
    print("\n3. Coordinated Query Execution...")
    
    # Execute queries with different coordination patterns
    patterns = [
        CoordinationPattern.SEQUENTIAL,
        CoordinationPattern.PARALLEL,
        CoordinationPattern.HIERARCHICAL,
        CoordinationPattern.ADAPTIVE
    ]
    
    for pattern in patterns:
        result = coordinator.execute_coordinated_query(
            "Python programming",
            pattern=pattern
        )
        print(f"✓ {pattern.value}: {result.count} items in {result.execution_time:.3f}s")
        print(f"  → Sources: {[source.value for source in result.sources]}")
    
    print("\n4. Result Merging and Ranking...")
    
    # Execute query and demonstrate ranking
    result = coordinator.execute_coordinated_query("Python", pattern=CoordinationPattern.PARALLEL)
    
    if result.items:
        print(f"✓ Found {result.count} items")
        
        # Rank by different strategies
        strategies = [
            RankingStrategy.RELEVANCE,
            RankingStrategy.TIMESTAMP,
            RankingStrategy.SOURCE_PRIORITY
        ]
        
        for strategy in strategies:
            ranked = coordinator.rank_items(result.items, strategy)
            if ranked:
                print(f"  → {strategy.value}: {ranked[0].content[:50]}...")
    
    print("\n5. State Synchronization...")
    
    # Synchronize state across layers
    sync_results = coordinator.synchronize_state()
    
    print("✓ Synchronization results:")
    for layer, success in sync_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {layer.value}: {'Success' if success else 'Failed'}")
    
    # Check sync states
    stats = coordinator.get_coordination_stats()
    print(f"✓ Sync operations: {stats['sync_operations']}")
    
    print("\n6. Event Coordination...")
    
    # Process any pending events
    pending_before = len(coordinator.get_pending_events())
    processed = coordinator.process_events()
    pending_after = len(coordinator.get_pending_events())
    
    print(f"✓ Events processed: {processed}")
    print(f"✓ Pending events: {pending_before} → {pending_after}")
    
    # Show event types handled
    if hasattr(coordinator.event_handler, 'handled_events'):
        event_types = set(e.event_type for e in coordinator.event_handler.handled_events)
        print(f"✓ Event types handled: {[et.value for et in event_types]}")
    
    print("\n7. Advanced Coordination Patterns...")
    
    # Demonstrate cross-layer coordination
    print("Cross-layer fact linking:")
    
    # Store related content in different layers
    event_result = coordinator.coordinate_store(
        "Used Python for web development",
        target_layers=[MemoryLayer.HYBRID],
        memory_type="event",
        event_type=ESEventType.ACTION
    )
    
    fact_result = coordinator.coordinate_store(
        "web framework",
        target_layers=[MemoryLayer.HYBRID],
        memory_type="fact",
        subject="Django",
        predicate="is_a"
    )
    
    # Link memories if both were stored successfully
    if event_result and fact_result:
        event_item = list(event_result.values())[0]
        fact_item = list(fact_result.values())[0]
        
        try:
            memory.link_memories(event_item.id, fact_item.id, "used_for")
            related = memory.get_related_memories(event_item.id)
            print(f"✓ Linked {len(related)} related memories")
        except Exception as e:
            print(f"✗ Linking failed: {e}")
    
    print("\n8. Performance and Statistics...")
    
    # Get comprehensive statistics
    coord_stats = coordinator.get_coordination_stats()
    memory_stats = memory.get_memory_stats()
    
    print("Coordination Statistics:")
    print(f"  - Queries routed: {coord_stats['queries_routed']}")
    print(f"  - Results merged: {coord_stats['results_merged']}")
    print(f"  - Events handled: {coord_stats['events_handled']}")
    print(f"  - Sync operations: {coord_stats['sync_operations']}")
    print(f"  - Total events: {coord_stats['total_events']}")
    
    print("Memory Statistics:")
    print(f"  - Working memory: {memory_stats['working']['messages']} messages")
    print(f"  - Events: {memory_stats['hybrid']['events']}")
    print(f"  - Experiences: {memory_stats['hybrid']['experiences']}")
    print(f"  - Facts: {memory_stats['hybrid']['facts']}")
    print(f"  - Graph nodes: {memory_stats['hybrid']['graph_nodes']}")
    
    print("\n9. Error Handling and Recovery...")
    
    # Test error handling
    try:
        # Attempt invalid operation
        result = coordinator.execute_coordinated_query(
            "",  # Empty query
            pattern=CoordinationPattern.PARALLEL
        )
        print(f"✓ Handled empty query gracefully: {result.count} items")
    except Exception as e:
        print(f"✓ Error handled: {type(e).__name__}")
    
    # System should still be functional
    test_result = coordinator.execute_coordinated_query("test")
    print(f"✓ System remains functional: {test_result.count} items")
    
    print("\n10. Cleanup and Reset...")
    
    # Reset statistics
    coordinator.reset_stats()
    reset_stats = coordinator.get_coordination_stats()
    
    print(f"✓ Statistics reset:")
    print(f"  - Queries routed: {reset_stats['queries_routed']}")
    print(f"  - Events handled: {reset_stats['events_handled']}")
    print(f"  - Total events: {reset_stats['total_events']}")
    
    print("\n=== Demo Complete ===")
    print("Memory coordination system successfully demonstrated:")
    print("✓ Query routing with multiple strategies")
    print("✓ Result merging and ranking")
    print("✓ State synchronization across layers")
    print("✓ Event coordination and handling")
    print("✓ Multiple coordination patterns")
    print("✓ Error handling and recovery")
    print("✓ Performance monitoring and statistics")


if __name__ == "__main__":
    main()