# AI Agent Memory Systems

A comprehensive collection of memory management systems and tools for AI agents, including conversation history management, token counting, budgeting, context window management with prioritization, abstract memory storage interfaces, semantic memory fact storage, knowledge graph organization, and RAG-based knowledge retrieval.

## Project Overview

This project implements various memory architectures and management systems for AI agents, focusing on practical implementations of conversation management, token budgeting, context prioritization, memory storage abstraction, memory optimization strategies, semantic memory with fact storage, knowledge graph organization, and retrieval-augmented generation.

## Structure

### Day 46: Memory Architecture Documentation
- **memory_architecture.md**: Comprehensive memory architecture for agent systems
  - Storage layers (working, short-term, long-term memory)
  - Retrieval mechanisms (exact match, similarity search, query-based, hybrid)
  - Update strategies and management patterns
- **memory_types_comparison.md**: Comparison of different memory types
- **use_cases_analysis.md**: Real-world use cases and applications

### Day 47: Conversation, Token & Context Management
- **conversation_manager.py**: Conversation history management with truncation strategies
  - Message storage with metadata and timestamps
  - Sliding window truncation
  - Summary-based truncation
  - Selective truncation (importance-based)
  - History retrieval and formatting
  
- **token_counter.py**: Token counting and budget management
  - Token counting using tiktoken
  - Budget tracking by component
  - Usage monitoring and alerts
  - Budget allocation
  - Context optimization
  - Response reserve management

- **context_manager.py**: Context window management with prioritization
  - Context building from multiple components
  - 4-level priority system (Critical, High, Medium, Low)
  - Overflow detection and handling
  - Dynamic component selection
  - Importance scoring
  - Graceful degradation
  - Integration with conversation and token managers

### Day 49: Episodic Event Store, Conversation Memory & Experience Tracker
- **event_store.py**: Event storage with temporal ordering and relationships
  - Event types: Action, Observation, Decision, Communication, Error, Custom
  - Event structure with timestamp, participants, context, data
  - Temporal ordering via binary-search insertion
  - Bidirectional event relationships
  - Retrieval by time range, event type, participant, or relationship
  - Timeline reconstruction and limiting
  - Event sequence sorting
  - Temporal pattern detection (interval analysis)
  - Delete with relationship cleanup

- **conversation_memory.py**: Multi-conversation memory with context tracking
  - Conversation lifecycle (create, get, delete)
  - Message sequences with ordering and metadata
  - Context maintenance across turns (merge-based updates)
  - Topic tracking (per-message and manual)
  - Retrieval by user, time range, topic, or content search
  - Multi-turn handling: turn pairs, message references, context window building

- **experience_tracker.py**: Experience tracking for agent learning
  - Experience logging with action, outcome, context, feedback, score, tags
  - Outcome types: Success, Failure, Partial, Unknown
  - Retrieval by action, outcome, tag, time range
  - Pattern recognition: success rate, average score, outcome distribution
  - Common tag analysis and score trend tracking
  - Lesson extraction with recommendations
  - Action comparison across multiple strategies
  - Improvement detection via sliding window score analysis

### Day 50: Semantic Memory Fact Store & Knowledge Graph
- **fact_store.py**: Subject-predicate-object triple store for semantic memory
  - Fact types: Assertion, Definition, Relation, Attribute, Rule
  - Fact structure with subject, predicate, object, properties, confidence
  - Indexed storage for fast retrieval by subject, predicate, object, type
  - Multi-field query interface with AND logic and confidence filtering
  - Bidirectional relationship management between facts
  - BFS traversal of related facts with configurable depth
  - CRUD operations with index and relationship cleanup
  - Custom error hierarchy (FactStoreError, FactNotFoundError)

- **knowledge_graph.py**: Knowledge graph for organizing facts and concepts
  - Node types: Entity, Concept, Event, Attribute, Category
  - Edge types: IS_A, HAS, PART_OF, RELATED_TO, CAUSES, DEPENDS_ON, CUSTOM
  - Directed edges with weight, label, and properties
  - Indexed storage for fast node/edge lookup by type, label
  - Adjacency tracking (outgoing/incoming edges per node)
  - Neighbor queries with direction and edge type filtering
  - BFS traversal with configurable depth, direction, and edge type
  - Shortest path finding (BFS) with edge type and depth constraints
  - Pattern matching: find (source, edge, target) triples by type
  - Subgraph extraction by node set
  - Node removal cascades to connected edges
  - Custom error hierarchy (KnowledgeGraphError, NodeNotFoundError, EdgeNotFoundError)

- **rag_integration.py**: RAG integration for semantic memory retrieval
  - Pluggable embedding provider interface (EmbeddingProvider ABC)
  - Built-in HashEmbedding for zero-dependency development and testing
  - In-memory vector store with cosine similarity search
  - Batch embedding and document ingestion
  - Metadata filtering on vector search
  - Pre-computed vector search support
  - RAGRetriever: indexes facts and knowledge graph nodes as vectors
  - Similarity search with automatic fact/node resolution
  - Related fact/node expansion via FactStore and KnowledgeGraph links
  - Formatted context generation for prompt augmentation
  - Full prompt builder with system prompt, retrieved knowledge, and query
  - Works standalone (text-only) or integrated with FactStore/KnowledgeGraph
  - Custom error hierarchy (RAGError, DocumentNotFoundError)

### Day 48: Memory Storage Interface
- **memory_storage.py**: Abstract storage interface with in-memory backend
  - Abstract base class (MemoryStorageInterface) for pluggable backends
  - Save memory with metadata and auto-generated IDs
  - Retrieve memory by ID
  - Update content and metadata with timestamp tracking
  - Delete memory with confirmation
  - List and filter memories by metadata
  - Custom error hierarchy (MemoryStorageError, MemoryNotFoundError, MemoryConflictError)

- **sql_store.py**: SQL storage backend using SQLAlchemy + SQLite
  - Implements MemoryStorageInterface for full compatibility
  - Connection management with configurable database URL
  - Automatic table creation on initialization
  - SQLite WAL mode for better concurrent access
  - Persistent storage across application restarts
  - Duplicate ID conflict detection (MemoryConflictError)
  - Flexible query method (by content substring, ID, or metadata)
  - Transaction management with rollback on errors

- **persistence_manager.py**: Coordinates storage operations with durability guarantees
  - Works with any MemoryStorageInterface backend
  - Retry logic with configurable attempts and exponential backoff
  - Write confirmation (read-after-write verification)
  - LRU cache for fast repeated retrievals
  - Bulk save with graceful degradation (partial failure tolerance)
  - Operation stats tracking (saves, retrievals, updates, deletes, retries, failures)
  - Structured logging (operations, errors, performance timings)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Conversation Management

```python
from day_47.conversation_manager import ConversationManager, TruncationStrategy

# Create manager with sliding window
manager = ConversationManager(max_messages=10)

# Add messages
manager.add_message("user", "Hello!")
manager.add_message("assistant", "Hi there!")

# Get formatted context
context = manager.format_for_context()
```

### Token Counting & Budgeting

```python
from day_47.token_counter import TokenBudgetManager

# Initialize with budget
manager = TokenBudgetManager(
    total_budget=8000,
    response_reserve=2000
)

# Count and track tokens
tokens = manager.count_and_track("system", "You are helpful")

# Check budget
if manager.can_fit_in_budget(500):
    print("Content fits in budget")

# Get summary
summary = manager.get_budget_summary()
```

### Context Window Management

```python
from day_47.context_manager import ContextManager, Priority

# Initialize
context_manager = ContextManager(token_manager, conv_manager)

# Add components with priorities
context_manager.add_system_prompt("You are helpful")
context_manager.add_conversation_history()
context_manager.add_component("context", "Important info", Priority.HIGH)

# Build context with automatic overflow handling
context = context_manager.build_context()
messages = context_manager.build_messages()
```

### Episodic Event Store

```python
from day_49.event_store import EventStore, EventType
from datetime import datetime, timedelta

store = EventStore()

# Store events
e1 = store.store(EventType.ACTION, data={"action": "login"}, participants=["alice"])
e2 = store.store(EventType.OBSERVATION, data={"saw": "dashboard"},
                 participants=["alice"], related_event_ids={e1.id})

# Retrieve
store.by_type(EventType.ACTION)           # all actions
store.by_participant("alice")             # alice's events
store.get_related(e1.id)                  # related events

# Temporal queries
store.timeline(limit=10)                  # last 10 events
store.event_sequence([e2.id, e1.id])      # sorted by time
store.temporal_pattern(EventType.ACTION)  # intervals between actions
```

### Experience Tracker

```python
from day_49.experience_tracker import ExperienceTracker, Outcome

tracker = ExperienceTracker()

# Log experiences
tracker.log("search", Outcome.SUCCESS, score=0.9, tags=["web"])
tracker.log("search", Outcome.FAILURE, score=0.2, feedback="timeout")
tracker.log("search", Outcome.SUCCESS, score=0.8, tags=["web", "api"])

# Pattern recognition
tracker.success_rate("search")            # 0.666...
tracker.average_score("search")           # 0.633...
tracker.outcome_distribution("search")    # {"success": 2, "failure": 1}
tracker.score_trend("search")             # [0.9, 0.2, 0.8]

# Learning
lesson = tracker.extract_lesson("search") # Lesson with recommendation
tracker.is_improving("search", window=3)  # True/False/None
tracker.compare_actions(["search", "summarize"])  # sorted by success rate
```

### Conversation Memory

```python
from day_49.conversation_memory import ConversationMemory

mem = ConversationMemory()

# Create conversation and add messages
conv = mem.create_conversation("alice", context={"lang": "en"})
m1 = mem.add_message(conv.id, "user", "Hello!", topics=["greeting"])
m2 = mem.add_message(conv.id, "assistant", "Hi there!")
mem.add_message(conv.id, "user", "Back to that", references=[m1.id])

# Context maintenance
mem.update_context(conv.id, {"mood": "happy"})
mem.add_topics(conv.id, ["python"])

# Retrieval
mem.by_user("alice")                      # all alice's conversations
mem.by_topic("greeting")                  # conversations about greetings
mem.search_content("hello")               # search message content
mem.by_time_range(start, end)             # by time window

# Multi-turn helpers
mem.get_turn_pairs(conv.id)               # user/assistant pairs
mem.get_referenced_messages(conv.id, m.id) # follow references
mem.build_context_window(conv.id, last_n=5) # formatted context
```

### Memory Storage (In-Memory)

```python
from day_48.memory_storage import InMemoryStorage

storage = InMemoryStorage()

# Save
mem = storage.save("important fact", metadata={"type": "note"})

# Retrieve
mem = storage.retrieve(mem.id)

# Update
mem = storage.update(mem.id, content="updated fact", metadata={"priority": "high"})

# List with filter
notes = storage.list_memories(filter_metadata={"type": "note"})

# Delete
storage.delete(mem.id)
```

### RAG Integration

```python
from day_50.fact_store import FactStore, FactType
from day_50.knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from day_50.rag_integration import HashEmbedding, VectorStore, RAGRetriever

# Set up components
fact_store = FactStore()
kg = KnowledgeGraph()
embedder = HashEmbedding(dimension=128)  # swap for sentence-transformers in prod
vs = VectorStore(embedder)
rag = RAGRetriever(vs, fact_store=fact_store, knowledge_graph=kg)

# Populate semantic memory
f1 = fact_store.store("Python", "is_a", "programming language",
                      fact_type=FactType.DEFINITION)
py = kg.add_node("Python", NodeType.ENTITY)

# Index into vector store
rag.index_fact(f1)
rag.index_node(py)
rag.index_text("Python was created by Guido van Rossum")

# Similarity search + retrieval
result = rag.retrieve("What is Python?", top_k=5, include_related=True)
print(result.context)          # formatted facts & concepts
print(result.facts)            # resolved Fact objects
print(result.nodes)            # resolved Node objects

# Build a RAG-augmented prompt
prompt = rag.augment_prompt(
    "What is Python?",
    system_prompt="You are a helpful assistant.",
    top_k=5,
)
```

### Knowledge Graph

```python
from day_50.knowledge_graph import KnowledgeGraph, NodeType, EdgeType

graph = KnowledgeGraph()

# Add nodes (entities, concepts)
python = graph.add_node("Python", NodeType.ENTITY)
pl = graph.add_node("Programming Language", NodeType.CONCEPT)
django = graph.add_node("Django", NodeType.ENTITY)
wf = graph.add_node("Web Framework", NodeType.CONCEPT)

# Add directed edges (relationships)
graph.add_edge(python.id, pl.id, EdgeType.IS_A)
graph.add_edge(django.id, wf.id, EdgeType.IS_A)
graph.add_edge(django.id, python.id, EdgeType.DEPENDS_ON)

# Query nodes and edges
graph.nodes_by_type(NodeType.ENTITY)              # all entities
graph.edges_from(python.id)                        # outgoing edges
graph.find_nodes(node_type=NodeType.CONCEPT)       # query with filters
graph.find_edges(edge_type=EdgeType.IS_A)          # all IS_A edges

# Neighbors and traversal
graph.neighbors(python.id, direction="both")       # connected nodes
graph.bfs(django.id, max_depth=2, direction="out") # BFS by depth

# Path finding
path = graph.find_path(django.id, pl.id)           # shortest path

# Pattern matching
graph.match_pattern(source_type=NodeType.ENTITY,
                    edge_type=EdgeType.IS_A,
                    target_type=NodeType.CONCEPT)   # matching triples

# Subgraph extraction
nodes, edges = graph.subgraph({python.id, pl.id, django.id})
```

### Semantic Fact Store

```python
from day_50.fact_store import FactStore, FactType

store = FactStore()

# Store facts as subject-predicate-object triples
f1 = store.store("Python", "is_a", "programming language", fact_type=FactType.DEFINITION)
f2 = store.store("Python", "created_by", "Guido van Rossum")
f3 = store.store("Python", "has_feature", "dynamic typing",
                 fact_type=FactType.ATTRIBUTE, confidence=0.95)

# Retrieve by field
store.by_subject("Python")                # all Python facts
store.by_predicate("is_a")                # all "is_a" facts
store.by_type(FactType.DEFINITION)         # all definitions

# Multi-field query
store.query(subject="Python", predicate="is_a")           # AND logic
store.query(fact_type=FactType.ATTRIBUTE, min_confidence=0.9)

# Relationship management
store.link(f1.id, f2.id)                  # bidirectional link
store.get_related(f1.id)                  # related facts
store.traverse(f1.id, max_depth=2)        # BFS traversal
store.unlink(f1.id, f2.id)               # remove link

# Update and delete
store.update(f1.id, properties={"version": "3.12"})
store.delete(f3.id)                       # cleans indexes & relationships
```

### SQL Storage (Persistent)

```python
from day_48.sql_store import SQLStorage

# SQLite (default) — data persists to file
storage = SQLStorage(db_url="sqlite:///memories.db")

# Or in-memory SQLite for testing
storage = SQLStorage(db_url="sqlite:///:memory:")

# Same CRUD interface as InMemoryStorage
mem = storage.save("important fact", metadata={"type": "note"})
mem = storage.retrieve(mem.id)
mem = storage.update(mem.id, content="updated", metadata={"priority": "high"})
storage.delete(mem.id)

# SQL-specific: flexible query method
results = storage.query(content="important")          # substring search
results = storage.query(type="note")                   # metadata filter
results = storage.query(content="fact", type="note")   # combined

storage.close()
```

### Persistence Manager

```python
from day_48.persistence_manager import PersistenceManager
from day_48.sql_store import SQLStorage

# Use with any storage backend
storage = SQLStorage(db_url="sqlite:///memories.db")
pm = PersistenceManager(storage=storage, max_retries=3)

# CRUD with retry + write confirmation
mem = pm.save("important fact", metadata={"type": "note"})
mem = pm.retrieve(mem.id)   # cache-aware
mem = pm.update(mem.id, content="updated")
pm.delete(mem.id)

# Bulk save (tolerates partial failures)
results = pm.bulk_save([
    {"content": "fact 1", "metadata": {"type": "note"}},
    {"content": "fact 2", "memory_id": "custom-id"},
])

# Operation stats
print(pm.stats)  # {saves: 2, retrievals: 1, ...}
```

## Features

### Conversation Management
- ✅ Message storage with metadata
- ✅ Multiple truncation strategies
- ✅ Role-based filtering
- ✅ Context formatting
- ✅ Timestamp tracking

### Token Management
- ✅ Accurate token counting (tiktoken)
- ✅ Component-based budget tracking
- ✅ Usage alerts and monitoring
- ✅ Budget allocation
- ✅ Context optimization
- ✅ Response reserve

### Context Window Management
- ✅ Priority-based component selection
- ✅ Overflow detection and handling
- ✅ Importance scoring
- ✅ Graceful degradation
- ✅ Dynamic selection
- ✅ Integration with conversation and token managers

### Episodic Event Store
- ✅ Multiple event types (Action, Observation, Decision, Communication, Error, Custom)
- ✅ Temporal ordering maintained on insert
- ✅ Retrieval by time range, type, participant
- ✅ Bidirectional event relationships
- ✅ Timeline reconstruction and event sequences
- ✅ Temporal pattern detection
- ✅ Delete with relationship cleanup

### Conversation Memory
- ✅ Multi-conversation storage with user linking
- ✅ Message sequences with ordering and metadata
- ✅ Context maintenance across turns
- ✅ Topic tracking (per-message and manual)
- ✅ Retrieval by user, time range, topic, content
- ✅ Multi-turn turn pairs and message references
- ✅ Context window building

### Experience Tracker
- ✅ Experience logging with outcomes, scores, feedback, tags
- ✅ Multiple outcome types (Success, Failure, Partial, Unknown)
- ✅ Retrieval by action, outcome, tag, time range
- ✅ Success rate and average score calculation
- ✅ Outcome distribution analysis
- ✅ Common tag extraction
- ✅ Score trend tracking
- ✅ Lesson extraction with recommendations
- ✅ Multi-action comparison
- ✅ Improvement detection (sliding window)

### Memory Storage
- ✅ Abstract storage interface (pluggable backends)
- ✅ CRUD operations (save, retrieve, update, delete)
- ✅ Metadata filtering
- ✅ Timestamp tracking
- ✅ Custom error hierarchy
- ✅ In-memory backend implementation

### SQL Storage Backend
- ✅ SQLAlchemy + SQLite persistent storage
- ✅ Configurable database URL
- ✅ Automatic table creation
- ✅ Duplicate ID conflict detection
- ✅ Flexible query method (content, ID, metadata)
- ✅ Transaction management with rollback
- ✅ Connection management and cleanup
- ✅ Data persistence across restarts

### Persistence Manager
- ✅ Backend-agnostic (works with any MemoryStorageInterface)
- ✅ Retry logic with exponential backoff
- ✅ Write confirmation (read-after-write)
- ✅ LRU cache with configurable size
- ✅ Bulk save with graceful degradation
- ✅ Operation stats tracking
- ✅ Structured logging (operations, errors, performance)

### Semantic Fact Store
- ✅ Subject-predicate-object triple storage
- ✅ Multiple fact types (Assertion, Definition, Relation, Attribute, Rule)
- ✅ Indexed retrieval by subject, predicate, object, type
- ✅ Multi-field query with AND logic
- ✅ Confidence-based filtering
- ✅ Bidirectional relationship management
- ✅ BFS relationship traversal with depth control
- ✅ CRUD with index and relationship cleanup
- ✅ Custom error hierarchy

### Knowledge Graph
- ✅ Typed nodes (Entity, Concept, Event, Attribute, Category)
- ✅ Typed directed edges (IS_A, HAS, PART_OF, RELATED_TO, CAUSES, DEPENDS_ON, CUSTOM)
- ✅ Weighted edges with labels and properties
- ✅ Indexed node/edge lookup by type and label
- ✅ Directional neighbor queries with edge type filtering
- ✅ BFS traversal with depth, direction, and edge type control
- ✅ Shortest path finding with constraints
- ✅ Pattern matching (source, edge, target triples)
- ✅ Subgraph extraction
- ✅ Cascading node removal
- ✅ Custom error hierarchy

### RAG Integration
- ✅ Pluggable embedding provider interface
- ✅ Built-in hash-based embedding (zero external dependencies)
- ✅ In-memory vector store with cosine similarity
- ✅ Batch document ingestion
- ✅ Top-k similarity search with min-score filtering
- ✅ Metadata filtering on search
- ✅ Pre-computed vector search
- ✅ Fact and knowledge graph node indexing
- ✅ Automatic fact/node resolution from search results
- ✅ Related fact/node expansion
- ✅ Formatted context generation
- ✅ RAG-augmented prompt building
- ✅ Works standalone or integrated with FactStore/KnowledgeGraph
- ✅ Custom error hierarchy

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run Day 47 tests
cd day_47
python -m pytest test_conversation_manager.py -v
python -m pytest test_token_counter.py -v
python -m pytest test_context_manager.py -v

# Run Day 49 tests
cd ../day_49
python -m pytest test_event_store.py -v
python -m pytest test_conversation_memory.py -v
python -m pytest test_experience_tracker.py -v

# Run Day 48 tests
cd ../day_48
python -m pytest test_memory_storage.py -v
python -m pytest test_sql_store.py -v
python -m pytest test_persistence_manager.py -v

# Run Day 50 tests
cd ../day_50
python -m pytest test_fact_store.py -v
python -m pytest test_knowledge_graph.py -v
python -m pytest test_rag_integration.py -v
```

## Running Examples

```bash
source venv/bin/activate
cd day_47
python examples_token_counter.py
python examples_context_manager.py
```

## Use Cases

1. **Chatbot Context Management**: Maintain conversation history within token limits
2. **Agent Memory Systems**: Store and retrieve relevant context for AI agents
3. **Cost Control**: Track and limit token usage for API cost management
4. **Multi-turn Dialogues**: Preserve important context while managing memory
5. **LLM Application Development**: Build token-aware applications
6. **Priority-Based Context**: Ensure critical information is always included
7. **Overflow Handling**: Gracefully handle context that exceeds model limits
8. **Multi-Component Systems**: Coordinate system prompts, history, tools, and context
9. **Pluggable Storage**: Swap storage backends without changing application code
10. **Semantic Memory**: Store and query knowledge as subject-predicate-object triples
11. **Knowledge Graphs**: Organize entities and concepts with typed relationships, traversal, and pattern matching
12. **RAG Retrieval**: Augment prompts with semantically relevant knowledge from facts and graphs

## Best Practices

1. **Set appropriate budgets**: Allocate tokens based on model context windows
2. **Use truncation strategies**: Choose the right strategy for your use case
3. **Monitor usage**: Track token usage by component for optimization
4. **Reserve for responses**: Always reserve tokens for model responses
5. **Optimize context**: Use context optimization for large contexts
6. **Reset between sessions**: Clear history and tracking for new conversations
7. **Use priority levels**: Assign appropriate priorities to context components
8. **Handle overflow gracefully**: Use automatic overflow handling
9. **Integrate managers**: Use conversation, token, and context managers together

## Dependencies

- Python 3.8+
- tiktoken: Token counting for OpenAI models
- SQLAlchemy: SQL database toolkit
- chromadb: Vector database for RAG integration
- openai: OpenAI API client for embeddings
- pytest: Testing framework

## Project Structure

```
ai-agent-project-4 Agent Memory Systems/
├── day_46/                          # Memory architecture documentation
│   ├── memory_architecture.md
│   ├── memory_types_comparison.md
│   └── use_cases_analysis.md
├── day_47/                          # Conversation, Token & Context
│   ├── conversation_manager.py      # Conversation history management
│   ├── token_counter.py             # Token counting & budgeting
│   ├── context_manager.py           # Context window management
│   ├── test_conversation_manager.py # Tests
│   ├── test_token_counter.py        # Tests
│   ├── test_context_manager.py      # Tests
│   ├── examples_token_counter.py    # Usage examples
│   ├── examples_context_manager.py  # Usage examples
│   └── README.md                    # Detailed documentation
├── day_48/                          # Memory Storage Interface
│   ├── memory_storage.py            # Abstract interface + in-memory backend
│   ├── sql_store.py                 # SQL backend (SQLAlchemy + SQLite)
│   ├── persistence_manager.py       # Persistence coordination layer
│   ├── test_memory_storage.py       # In-memory storage tests
│   ├── test_sql_store.py            # SQL storage tests
│   └── test_persistence_manager.py  # Persistence manager tests
├── day_49/                          # Episodic Event Store, Conversation Memory & Experience Tracker
│   ├── event_store.py               # Event storage with temporal ordering
│   ├── test_event_store.py          # Event store tests
│   ├── conversation_memory.py       # Multi-conversation memory management
│   ├── test_conversation_memory.py  # Conversation memory tests
│   ├── experience_tracker.py        # Experience tracking for learning
│   └── test_experience_tracker.py   # Experience tracker tests
├── day_50/                          # Semantic Memory, Knowledge Graph & RAG
│   ├── fact_store.py                # Subject-predicate-object triple store
│   ├── test_fact_store.py           # Fact store tests
│   ├── knowledge_graph.py           # Knowledge graph with typed nodes & edges
│   ├── test_knowledge_graph.py      # Knowledge graph tests
│   ├── rag_integration.py           # RAG: embedding, vector store, retrieval
│   └── test_rag_integration.py      # RAG integration tests
├── venv/                            # Virtual environment
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Contributing

When adding new features:
1. Check for venv before running environment commands
2. Update this README with new functionality
3. Add tests for new implementations
4. Document usage with examples

## License

This project is for educational and research purposes.

## Future Enhancements

- [ ] Event store persistence backend
- [x] Vector-based memory retrieval (RAG integration)
- [x] Persistent storage backends (SQLAlchemy + SQLite)
- [x] Conversation memory with multi-turn context tracking
- [ ] Memory consolidation strategies
- [ ] Multi-agent memory sharing
- [ ] Advanced context compression
- [x] Experience tracking with pattern recognition
- [ ] Memory importance scoring
- [x] Semantic memory fact store with relationship traversal
- [x] Knowledge graph for organizing facts and concepts
