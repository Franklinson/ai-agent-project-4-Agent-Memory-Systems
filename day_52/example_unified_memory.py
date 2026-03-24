"""Example usage of the unified memory system."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unified_memory import UnifiedMemory, MemoryLayer
from context_manager import Priority
from event_store import EventType
from experience_tracker import Outcome
from fact_store import FactType
from knowledge_graph import NodeType
from cross_type_queries import QueryPattern


def main():
    """Demonstrate unified memory system capabilities."""
    print("=== Unified Memory System Demo ===\n")
    
    # Initialize unified memory
    memory = UnifiedMemory(
        total_budget=4000,
        response_reserve=500,
        max_messages=10
    )
    
    print("1. Storing different types of memories...")
    
    # Store conversation
    memory.store("I want to learn Python programming", role="user")
    memory.store("I'd be happy to help you learn Python!", role="assistant")
    
    # Store events
    event_item = memory.store(
        "Started Python tutorial",
        memory_type="event",
        event_type=EventType.ACTION,
        participants=["user"]
    )
    
    # Store experiences
    exp_item = memory.store(
        "Completed first Python exercise successfully",
        memory_type="experience",
        action="python_exercise",
        outcome=Outcome.SUCCESS,
        score=0.85,
        tags=["learning", "python"]
    )
    
    # Store facts
    fact_item = memory.store(
        "programming language",
        memory_type="fact",
        subject="Python",
        predicate="is_a",
        fact_type=FactType.DEFINITION
    )
    
    # Store knowledge graph nodes
    node_item = memory.store(
        "Django",
        memory_type="node",
        node_type=NodeType.ENTITY
    )
    
    print(f"✓ Stored {memory.conversation.get_message_count()} messages")
    print(f"✓ Stored {memory.events.count} events")
    print(f"✓ Stored {memory.experiences.count} experiences")
    print(f"✓ Stored {memory.facts.count} facts")
    print(f"✓ Stored {memory.graph.node_count} nodes")
    
    print("\n2. Querying different memory layers...")
    
    # Query working memory
    working_result = memory.query("Python", layer=MemoryLayer.WORKING)
    print(f"✓ Working memory: {working_result.count} items")
    
    # Query short-term memory
    short_result = memory.query("Python", layer=MemoryLayer.SHORT_TERM)
    print(f"✓ Short-term memory: {short_result.count} items")
    
    # Query long-term memory
    long_result = memory.query("Python", layer=MemoryLayer.LONG_TERM)
    print(f"✓ Long-term memory: {long_result.count} items")
    
    # Intelligent routing
    routed_result = memory.query("What is Python?")
    print(f"✓ Intelligent routing: {routed_result.count} items")
    
    print("\n3. Cross-memory linking...")
    
    # Link episodic and semantic memories
    memory.link_memories(
        event_item.id,
        fact_item.id,
        "learned_about"
    )
    
    related = memory.get_related_memories(event_item.id)
    print(f"✓ Linked memories: {len(related)} related items")
    
    print("\n4. Context building with memory retrieval...")
    
    # Add system prompt
    memory.add_to_context(
        "system",
        "You are a helpful Python programming tutor.",
        Priority.CRITICAL
    )
    
    # Build context with automatic memory retrieval
    context = memory.build_context(
        include_memory=True,
        query="Python programming help"
    )
    
    print(f"✓ Built context: {len(context)} characters")
    print("Context preview:")
    print(context[:200] + "..." if len(context) > 200 else context)
    
    print("\n5. Cross-type queries...")
    
    # Combined query
    combined_result = memory.query(
        "Python",
        pattern=QueryPattern.COMBINED,
        limit=5
    )
    print(f"✓ Combined query: {combined_result.count} items")
    
    print("\n6. Memory statistics...")
    
    stats = memory.get_memory_stats()
    print("Memory usage:")
    print(f"  - Working: {stats['working']['messages']} messages, "
          f"{stats['working']['tokens_used']} tokens used")
    print(f"  - Events: {stats['hybrid']['events']}")
    print(f"  - Experiences: {stats['hybrid']['experiences']}")
    print(f"  - Facts: {stats['hybrid']['facts']}")
    print(f"  - Graph nodes: {stats['hybrid']['graph_nodes']}")
    
    print("\n7. Memory optimization...")
    
    # Add large context to trigger optimization
    for i in range(5):
        memory.add_to_context(
            f"large_component_{i}",
            "x" * 500,  # Large content
            Priority.LOW
        )
    
    optimization_result = memory.optimize_memory()
    print(f"✓ Memory optimized: overflow detected = {optimization_result['overflow_detected']}")
    
    print("\n8. Memory export...")
    
    export_data = memory.export_memory()
    print(f"✓ Exported {len(export_data['items'])} memory items")
    print(f"✓ Export includes: {list(export_data.keys())}")
    
    print("\n=== Demo Complete ===")
    print("The unified memory system successfully integrated all memory types!")


if __name__ == "__main__":
    main()