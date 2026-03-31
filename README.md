# AI Agent Memory Systems

A comprehensive collection of memory management systems and tools for AI agents, including conversation history management, token counting, budgeting, context window management with prioritization, abstract memory storage interfaces, semantic memory fact storage, knowledge graph organization, RAG-based knowledge retrieval, hybrid episodic-semantic memory integration, intelligent memory type routing, cross-type query support with combined, sequential, and parallel query patterns, and intelligent memory type selection with multiple algorithms and optimization.

## Project Overview

This project implements various memory architectures and management systems for AI agents, focusing on practical implementations of conversation management, token budgeting, context prioritization, memory storage abstraction, memory optimization strategies, semantic memory with fact storage, knowledge graph organization, retrieval-augmented generation, hybrid memory combining episodic and semantic subsystems, intelligent memory type routing with adaptive learning, cross-type query execution with result merging and relevance ranking, and intelligent memory type selection with rule-based, query-based, pattern-based, and hybrid algorithms.

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

### Day 53: Vector Database for Semantic Memory
- **vector_store.py**: Vector database implementation using ChromaDB
  - Database connection management (in-memory and persistent)
  - Collection creation and management with metadata
  - Document CRUD operations with vector embeddings
  - Similarity search with metadata filtering
  - Search by pre-computed vectors
  - Document counting and collection inspection
  - Comprehensive error handling
  - Support for custom embedding providers
  - Persistent storage across sessions
  - Batch document operations

- **simple_test.py**: Basic functionality tests
  - Collection management operations
  - Document CRUD with pre-computed embeddings
  - Metadata handling and validation
  - Error handling verification

- **example_usage.py**: Usage examples and demonstrations
  - Basic vector store operations
  - Persistent storage examples
  - Metadata filtering demonstrations
  - Best practices and patterns

- **embedding_generator.py**: Embedding generation for semantic memory
  - Text preprocessing with configurable options (lowercase, special char removal, length limits)
  - Multiple embedding providers (Sentence Transformers, OpenAI)
  - Pluggable provider interface for extensibility
  - Memory and disk-based caching with LRU eviction
  - Batch processing with configurable batch sizes
  - Comprehensive error handling and validation
  - Performance monitoring and statistics
  - Factory function for easy setup
  - Support for custom preprocessing pipelines

- **test_embedding_generator.py**: Comprehensive test suite
  - Text preprocessing tests (normalization, character removal, length validation)
  - Caching tests (memory, disk, LRU eviction, statistics)
  - Embedding generation tests (single, batch, error handling)
  - Provider tests (Sentence Transformers, OpenAI, mocking)
  - Factory function tests with various configurations
  - Integration tests with preprocessing and caching

- **example_usage.py**: Usage examples and demonstrations
  - Sentence Transformers examples with local models
  - OpenAI API examples with different models
  - Custom configuration examples (preprocessing, caching)
  - Semantic memory integration examples
  - Performance comparison between models
  - Similarity search demonstrations

- **demo.py**: Simple demonstration script
  - Basic embedding generation workflow
  - Caching demonstration
  - Batch processing examples
  - Text preprocessing showcase
  - Performance statistics display
- **unified_memory.py**: Unified interface integrating all memory types
  - Single API over working, short-term, long-term, and hybrid memory
  - Automatic type detection and intelligent routing
  - Seamless access across conversation, context, tokens, events, experiences, facts, and knowledge graph
  - Cross-memory linking and related item retrieval
  - Context building with automatic memory retrieval
  - Memory optimization and export capabilities
  - Layer-aware querying (working, short-term, long-term, hybrid)
  - Cross-type query patterns (combined, sequential, parallel)
  - Unified store/get/delete interface with type abstraction
  - Error handling with graceful degradation
  - Comprehensive memory statistics and management

- **memory_coordinator.py**: Coordination mechanisms for memory types
  - Query routing with keyword-based and adaptive strategies
  - Result merging with relevance-based ranking and deduplication
  - State synchronization across memory layers with conflict detection
  - Event coordination with threading-safe event queue
  - Multiple coordination patterns (sequential, parallel, hierarchical, adaptive)
  - Cross-layer storage coordination
  - Performance monitoring and statistics
  - Error handling and recovery mechanisms

- **type_selector.py**: Intelligent memory type selection with multiple algorithms
  - Query analysis: extracts temporal, factual, experiential, relational, conversational signals
  - Information need detection (recall, lookup, exploration, temporal, experiential)
  - Question detection and temporal reference identification
  - RuleBasedSelector: deterministic keyword/signal threshold rules
  - QueryBasedSelector: scores each (layer, type) pair and picks the best
  - PatternBasedSelector: learns from past query→result patterns
  - HybridSelector: combines multiple selectors via weighted voting
  - AutoSelector: automatic selection with fallback and caching
  - SelectionOptimizer: tracks outcomes and tunes selector performance
  - Custom error hierarchy (TypeSelectorError, SelectionFailedError)

### Day 51: Hybrid Memory System & Memory Router
- **hybrid_memory.py**: Unified interface combining episodic and semantic memory
  - Integrates EventStore, ExperienceTracker (episodic) with FactStore, KnowledgeGraph (semantic)
  - Unified store/get/delete across all subsystems
  - Type detection: auto-detect whether an ID is episodic or semantic
  - Source detection: identify which subsystem owns an item
  - Query routing with memory type scoping (episodic, semantic, or both)
  - Rich query filters: event type, participant, action, outcome, tag, subject, predicate, fact type, confidence, node type, node label, time range
  - Result merging with timestamp-based sorting and limit
  - QueryResult helpers: filter by source or memory type
  - Cross-memory linking between episodic and semantic items
  - Linked item retrieval (bidirectional)
  - Cross-link cleanup on delete
  - Subsystem stats aggregation
  - get_all with type filtering and limit
  - MemoryItem wrapper with unified fields (id, type, source, content, timestamp, metadata, relevance)
  - Custom error hierarchy (HybridMemoryError, MemoryNotFoundError)

- **memory_router.py**: Intelligent memory type routing with query analysis
  - Query analysis: keyword-based intent detection (temporal, factual, experiential, relational, ambiguous)
  - Signal scoring: normalized scores across intent categories
  - Structured filter extraction: auto-detects event types, outcomes, fact types, node types from query text
  - Pluggable routing strategies via RoutingStrategy ABC
  - KeywordRoutingStrategy: routes based on dominant keyword-intent signal
  - WeightedRoutingStrategy: configurable confidence threshold with BOTH fallback
  - AdaptiveRoutingStrategy: learns from query outcomes to improve routing over time
  - Adaptive learning: tracks (intent → memory_type) result counts, adjusts weights via reward signals
  - Exploration vs exploitation: configurable exploration rate for uncertain intents
  - MemoryRouter orchestrator: wraps HybridMemory with automatic analysis → strategy → execution pipeline
  - Override support: caller can override any routing decision with explicit kwargs
  - route_with_analysis: returns result + analysis + decision for inspection
  - Audit log: full history of routed queries with analysis, decision, and result count
  - Routing stats: total queries, average results, intent distribution
  - Custom error hierarchy (RoutingError)

- **cross_type_queries.py**: Cross-type query support with multiple execution patterns
  - Combined queries: query both episodic and semantic simultaneously, merge and rank
  - Sequential queries: query one type first, bridge results to inform a second query
  - Parallel queries: execute multiple sub-queries concurrently via ThreadPoolExecutor
  - Multi-hop queries: chain multiple sequential hops with bridge functions between each
  - Result merging with ID-based deduplication
  - Relevance scoring: text-match boosting + base relevance + source priority tiebreak
  - Three ranking strategies: relevance, timestamp, source_priority
  - SubQuery dataclass for structured parallel query definitions
  - MergedResult with pattern, items, sub_results, execution time, and metadata
  - Default bridge function: extracts subject/label/action keywords from primary results
  - Custom bridge function support for domain-specific sequential logic
  - Formatted result output with source labels and relevance scores
  - Custom error hierarchy (CrossQueryError)

### Day 54: User Profile System
- **user_profile.py**: User profile system with data structures
  - Profile data structure with user identifier, basic information, preferences, settings, metadata
  - Multiple profile types (Basic, Premium, Admin, Guest) with type-specific validation
  - Preference management with validation (language, timezone, theme, notifications, privacy)
  - Profile updates with version control and validation
  - Schema validation, data validation, type checking, constraint checking
  - ProfileManager for CRUD operations (create, get, update, delete, search)
  - JSON serialization/deserialization with datetime handling
  - Profile statistics and export/import functionality
  - Custom error hierarchy (ValidationError, ProfileNotFoundError)
  - Pluggable validator system (BasicProfileValidator, PremiumProfileValidator)

- **test_user_profile.py**: Comprehensive test suite
  - User preferences tests (defaults, validation)
  - Profile creation and validation tests
  - Premium profile validation tests
  - Preference and profile update tests
  - Login recording tests
  - Serialization/deserialization tests
  - Profile manager CRUD tests
  - Search and statistics tests
  - Export/import tests
  - Validator tests

- **example_usage.py**: Usage examples and demonstrations
  - Basic profile creation and management
  - Premium profile with enhanced validation
  - Validation error handling examples
  - Profile manager functionality
  - Serialization examples
  - Multiple profile types demonstration
  - Advanced features with custom data

- **demo.py**: Simple demonstration script
  - Quick profile system overview
  - Profile creation and management
  - Preference updates and activity recording
  - Statistics and serialization demo

- **profile_storage.py**: Secure profile storage with privacy and security
  - Profile storage with encryption using Fernet (AES 128)
  - Access control with token-based authentication and authorization
  - Role-based permissions (Read, Write, Delete, Admin)
  - Audit logging with SQLite database
  - Master key management for encryption
  - Resource-level access control (users can only access own profiles)
  - Token expiration and cleanup
  - IP address tracking in audit logs
  - Comprehensive error handling with custom exceptions
  - CRUD operations with security validation

- **test_profile_storage.py**: Comprehensive security tests
  - Encryption manager tests (encryption, decryption, key management)
  - Access controller tests (authentication, authorization, permissions)
  - Audit logger tests (logging, activity retrieval)
  - Secure storage integration tests
  - Access control enforcement tests
  - Audit logging integration tests
  - Security scenario tests

- **example_secure_storage.py**: Security usage examples
  - Basic secure storage operations
  - Access control demonstrations
  - Audit logging examples
  - Encryption demonstrations
  - Security scenarios and edge cases
  - Performance monitoring examples

- **demo_secure_storage.py**: Simple security demonstration
  - Authentication and authorization demo
  - Encrypted storage showcase
  - Access control enforcement
  - Audit logging demonstration
  - Admin operations example

- **profile_retrieval.py**: Efficient profile retrieval with caching
  - Multi-level caching system (LRU, TTL caches)
  - Connection pooling for optimized database access
  - Multiple retrieval patterns (direct, batch, filtered, preference, metadata)
  - Query optimization with database indexes
  - Performance monitoring and metrics collection
  - Cache invalidation and management
  - Thread-safe operations with concurrent access support
  - Database optimization tools (ANALYZE, VACUUM)

- **test_profile_retrieval.py**: Comprehensive retrieval tests
  - Cache implementation tests (LRU, TTL)
  - Connection pool tests
  - Profile retrieval pattern tests
  - Performance monitoring tests
  - Concurrent access tests
  - Query optimization tests
  - Cache strategy tests

- **example_profile_retrieval.py**: Retrieval usage examples
  - Basic retrieval operations
  - Advanced search and filtering
  - Caching performance demonstrations
  - Performance monitoring examples
  - Query optimization examples

- **demo_profile_retrieval.py**: Simple retrieval demonstration
  - Quick retrieval system overview
  - Caching performance comparison
  - Search operations showcase
  - Performance metrics display

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

### Unified Memory System

```python
from day_52.unified_memory import UnifiedMemory, MemoryLayer
from day_47.context_manager import Priority
from day_49.event_store import EventType
from day_49.experience_tracker import Outcome
from day_50.fact_store import FactType
from day_50.knowledge_graph import NodeType
from day_51.cross_type_queries import QueryPattern

# Initialize unified memory system
memory = UnifiedMemory(
    total_budget=4000,
    response_reserve=500,
    max_messages=10
)

# Store different types of content with automatic type detection
memory.store("Hello!", role="user")  # → working memory (conversation)
memory.store(
    "User logged in",
    memory_type="event",
    event_type=EventType.ACTION
)  # → short-term memory (events)
memory.store(
    "Task completed successfully",
    memory_type="experience",
    action="task",
    outcome=Outcome.SUCCESS
)  # → short-term memory (experiences)
memory.store(
    "programming language",
    memory_type="fact",
    subject="Python",
    predicate="is_a"
)  # → long-term memory (facts)
memory.store(
    "Django",
    memory_type="node",
    node_type=NodeType.ENTITY
)  # → long-term memory (knowledge graph)

# Query different memory layers
working_result = memory.query("hello", layer=MemoryLayer.WORKING)
short_result = memory.query("login", layer=MemoryLayer.SHORT_TERM)
long_result = memory.query("Python", layer=MemoryLayer.LONG_TERM)

# Intelligent routing (automatic layer selection)
routed_result = memory.query("What is Python?")  # → routes to semantic memory
temporal_result = memory.query("When did login happen?")  # → routes to episodic memory

# Cross-type query patterns
combined_result = memory.query("Python", pattern=QueryPattern.COMBINED)
sequential_result = memory.query("coding", pattern=QueryPattern.SEQUENTIAL)

# Context building with memory retrieval
memory.add_to_context("system", "You are helpful", Priority.CRITICAL)
context = memory.build_context(include_memory=True, query="Python help")
messages = memory.build_messages()

# Cross-memory linking
event_item = memory.store("Used Python", memory_type="event", event_type=EventType.ACTION)
fact_item = memory.store("programming language", memory_type="fact", subject="Python", predicate="is_a")
memory.link_memories(event_item.id, fact_item.id, "relates_to")
related = memory.get_related_memories(event_item.id)

# Memory management
stats = memory.get_memory_stats()
optimization = memory.optimize_memory()
export_data = memory.export_memory()
memory.clear_working_memory()
```

### Intelligent Type Selection

```python
from day_52.type_selector import (
    analyse_query_characteristics, AutoSelector, SelectionOptimizer,
    RuleBasedSelector, QueryBasedSelector, PatternBasedSelector, HybridSelector,
)

# Analyse query characteristics
chars = analyse_query_characteristics("When did the login event happen?")
print(chars.temporal_score)       # high temporal signal
print(chars.information_need)     # InformationNeed.TEMPORAL
print(chars.is_question)          # True

# Automatic selection with fallback
auto = AutoSelector()
sel = auto.select("What is the definition of Python?")
print(sel.layer)          # MemoryLayer.LONG_TERM
print(sel.memory_type)    # MemoryType.SEMANTIC
print(sel.confidence)     # 0.0–1.0
print(sel.algorithm)      # "hybrid"

# Use a specific algorithm
rule_sel = AutoSelector(algorithm=RuleBasedSelector()).select("recent events")
query_sel = AutoSelector(algorithm=QueryBasedSelector()).select("recent events")

# Optimization with outcome tracking
optimizer = SelectionOptimizer()
sel = optimizer.select("What is Python?")
optimizer.record_outcome("What is Python?", sel, result_count=5)
print(optimizer.stats())          # {total_selections: 1, avg_results: 5.0, ...}
print(optimizer.best_algorithm()) # best-performing algorithm name
```

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

### Efficient Profile Retrieval

```python
from day_54.profile_retrieval import ProfileRetrieval, QueryType
from day_54.user_profile import UserProfile, ProfileType

# Initialize retrieval system with caching
retrieval = ProfileRetrieval("profiles.db", {
    "profile_cache_size": 100,
    "preference_cache_size": 50,
    "query_cache_size": 20,
    "preference_ttl": 1800  # 30 minutes
})

# Store profiles
profile = UserProfile(
    user_id="alice",
    username="alice",
    email="alice@example.com",
    age=28,
    profile_type=ProfileType.PREMIUM,
    full_name="Alice Johnson"
)
retrieval.store_profile(profile)

# Direct profile lookup with caching
profile = retrieval.get_profile("alice", use_cache=True)
print(f"Retrieved: {profile.username}")

# Batch retrieval (optimized)
user_ids = ["alice", "bob", "charlie"]
batch_profiles = retrieval.get_profiles_batch(user_ids, use_cache=True)
print(f"Retrieved {len(batch_profiles)} profiles")

# Advanced search with caching
results = retrieval.search_profiles({
    "profile_type": "premium",
    "age_min": 25,
    "language": "en",
    "theme": "dark"
}, use_cache=True)
print(f"Found {len(results)} matching profiles")

# Preference lookup (cached)
prefs = retrieval.get_preferences("alice", use_cache=True)
print(f"Theme: {prefs['theme']}, Language: {prefs['language']}")

# Lightweight metadata query
metadata = retrieval.get_profile_metadata("alice")
print(f"Version: {metadata['version']}, Type: {metadata['profile_type']}")

# Performance metrics
metrics = retrieval.get_performance_metrics()
summary = metrics.get_summary()

print(f"Cache hit rates: {summary['cache_hit_rates']}")
print(f"Query avg times: {summary['query_avg_times']}")
print(f"Total queries: {summary['total_queries']}")

# Cache statistics
for cache_name, stats in metrics.cache_stats.items():
    print(f"{cache_name}: {stats.hit_rate:.1%} hit rate, {stats.size} items")

# Database optimization
retrieval.optimize_database()

# Cleanup
retrieval.close()
```

### Secure Profile Storage

```python
from day_54.profile_storage import SecureProfileStorage, AccessLevel, AuditAction
from day_54.user_profile import UserProfile, ProfileType

# Create secure storage with encryption
storage = SecureProfileStorage(
    db_path="secure_profiles.db",
    master_key="my_secure_master_key_123"
)

# Authenticate users
alice_token = storage.authenticate("alice", "password", ip_address="192.168.1.100")
admin_token = storage.authenticate("admin_user", "password", ip_address="192.168.1.101")

# Create and store encrypted profile
profile = UserProfile(
    user_id="alice",
    username="alice",
    email="alice@example.com",
    bio="Sensitive personal information",
    custom_data={"secret": "confidential data"}
)

# Store profile (automatically encrypted)
storage.store_profile(alice_token, profile, ip_address="192.168.1.100")

# Retrieve profile (automatically decrypted)
retrieved = storage.retrieve_profile(alice_token, "alice", ip_address="192.168.1.100")
print(f"Retrieved: {retrieved.username}")

# Access control - Alice cannot access other profiles
try:
    storage.retrieve_profile(alice_token, "bob")  # Fails
except AuthorizationError:
    print("Access denied - users can only access own profiles")

# Admin can access any profile
admin_retrieved = storage.retrieve_profile(admin_token, "alice")
print(f"Admin accessed: {admin_retrieved.username}")

# Update profile
profile.update_profile(bio="Updated sensitive information")
storage.update_profile(alice_token, profile, ip_address="192.168.1.100")

# Admin operations
all_profiles = storage.list_profiles(admin_token)  # Admin only
print(f"Total profiles: {len(all_profiles)}")

# Audit logging
audit_entries = storage.get_audit_log(alice_token)
print(f"Alice's activity: {len(audit_entries)} entries")

for entry in audit_entries[:3]:
    print(f"  {entry.action.value} at {entry.timestamp} from {entry.ip_address}")

# Security features
master_key = storage.get_master_key()  # For backup
print(f"Master key: {master_key[:20]}...")

# Logout and cleanup
storage.logout(alice_token, ip_address="192.168.1.100")
storage.cleanup_expired_tokens()
```

### User Profile System

```python
from day_54.user_profile import UserProfile, ProfileManager, ProfileType, ValidationError

# Create profile manager
manager = ProfileManager()

# Create different profile types
basic_profile = manager.create_profile(
    username="alice",
    email="alice@example.com",
    age=25
)

premium_profile = manager.create_profile(
    ProfileType.PREMIUM,
    username="bob",
    email="bob@example.com",
    full_name="Bob Smith",
    age=30,
    custom_data={"subscription_tier": "premium"}
)

# Update preferences
manager.update_preferences(
    basic_profile.user_id,
    theme="dark",
    language="es",
    notifications=False
)

# Update profile
manager.update_profile(
    premium_profile.user_id,
    bio="Senior developer",
    tags=["developer", "premium"]
)

# Record activity
basic_profile.record_login()
premium_profile.record_login()

# Search profiles
young_users = manager.search_profiles(age=25)
premium_users = manager.list_profiles(ProfileType.PREMIUM)

# Get statistics
stats = manager.get_stats()
print(f"Total profiles: {stats['total_profiles']}")
print(f"Active users: {stats['active_users']}")
print(f"By type: {stats['by_type']}")

# Serialization
json_data = basic_profile.to_json()
restored_profile = UserProfile.from_json(json_data)

# Validation handling
try:
    invalid_profile = UserProfile(
        username="ab",  # Too short
        email="invalid"  # Invalid format
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# Export/import profiles
export_data = manager.export_profiles()
new_manager = ProfileManager()
new_manager.import_profiles(export_data)
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

### Embedding Generation

```python
from day_53.embedding_generator import create_embedding_generator

# Create generator with Sentence Transformers (local)
generator = create_embedding_generator(
    provider_type="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
    cache_dir="./embeddings_cache",
    lowercase=True,
    max_length=512
)

# Single text embedding
result = generator.generate("Python is a programming language")
print(f"Text: {result.text}")
print(f"Model: {result.model}")
print(f"Dimension: {result.dimension}")
print(f"Processing time: {result.processing_time:.4f}s")
print(f"Cached: {result.cached}")
print(f"Embedding: {result.embedding[:5]}...")  # First 5 dimensions

# Batch processing
texts = [
    "Machine learning algorithms",
    "Natural language processing",
    "Computer vision systems"
]

batch_result = generator.generate_batch(texts, batch_size=2)
print(f"Processed {len(batch_result.texts)} texts")
print(f"Total time: {batch_result.total_processing_time:.4f}s")
print(f"Cache hits: {batch_result.cache_hits}")
print(f"Cache misses: {batch_result.cache_misses}")

# Cache statistics
stats = generator.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory cache size: {stats['memory_cache_size']}")

# OpenAI provider (requires API key)
openai_generator = create_embedding_generator(
    provider_type="openai",
    model_name="text-embedding-3-small",
    api_key="your-api-key",
    cache_dir="./openai_cache"
)

result = openai_generator.generate("AI transforms industries")
print(f"OpenAI embedding dimension: {result.dimension}")  # 1536
```

### Custom Embedding Configuration

```python
from day_53.embedding_generator import (
    EmbeddingGenerator, TextPreprocessor, EmbeddingCache, SentenceTransformerProvider
)

# Custom preprocessor
preprocessor = TextPreprocessor(
    lowercase=False,  # Keep original case
    remove_special_chars=True,  # Remove special characters
    min_length=3,
    max_length=256
)

# Custom cache with smaller size
cache = EmbeddingCache(
    cache_dir="./custom_cache",
    max_cache_size=50
)

# Create provider and generator
provider = SentenceTransformerProvider("all-mpnet-base-v2")
generator = EmbeddingGenerator(provider, preprocessor, cache)

# Generate with custom configuration
result = generator.generate("  Custom TEXT Processing!  ")
print(f"Processed: '{result.text}'")  # Custom TEXT Processing!
```

### Semantic Memory Integration

```python
from day_53.embedding_generator import create_embedding_generator
from day_50.fact_store import FactStore, FactType
from day_50.knowledge_graph import KnowledgeGraph, NodeType

# Create embedding generator
generator = create_embedding_generator(
    provider_type="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)

# Create semantic memory components
fact_store = FactStore()
kg = KnowledgeGraph()

# Store facts and generate embeddings
fact = fact_store.store(
    "Python", "is_a", "programming language",
    fact_type=FactType.DEFINITION
)

# Generate embedding for the fact
fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
embedding_result = generator.generate(fact_text)

print(f"Fact: {fact_text}")
print(f"Embedding dimension: {embedding_result.dimension}")
print(f"Processing time: {embedding_result.processing_time:.4f}s")

# Store knowledge graph nodes with embeddings
node = kg.add_node("Python", NodeType.ENTITY)
node_embedding = generator.generate(node.label)

# Batch process multiple semantic items
semantic_texts = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks process data"
]

batch_result = generator.generate_batch(semantic_texts)
print(f"Generated {len(batch_result.embeddings)} semantic embeddings")
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

### Cross-Type Queries

```python
from day_51.cross_type_queries import (
    CrossTypeQueryEngine, SubQuery, RankingStrategy, QueryPattern,
    merge_items, rank_items, format_results,
)
from day_51.hybrid_memory import HybridMemory, MemoryType

hm = HybridMemory()
# ... populate with events, experiences, facts, nodes ...
engine = CrossTypeQueryEngine(hm)

# Combined: query both types at once, merge and rank
result = engine.combined("Python", ranking=RankingStrategy.RELEVANCE, limit=10)
print(result.count)                # total merged items
print(format_results(result.items)) # formatted output

# Sequential: episodic first, then use results to query semantic
result = engine.sequential(
    "login", primary_type=MemoryType.EPISODIC,
    # default bridge extracts keywords from primary results
)
print(result.metadata["bridge_kwargs"])  # derived secondary query hints

# Sequential with custom bridge
def my_bridge(items):
    return {"query_text": "Django", "subject": "Django"}
result = engine.sequential("Python", bridge_fn=my_bridge)

# Parallel: multiple independent sub-queries concurrently
result = engine.parallel([
    SubQuery("Python", MemoryType.SEMANTIC, label="facts"),
    SubQuery("login", MemoryType.EPISODIC, label="events"),
    SubQuery("search", MemoryType.EPISODIC, label="experiences"),
], limit=20)

# Multi-hop: chain queries across memory types
result = engine.multi_hop("Python", hops=[
    {"memory_type": MemoryType.SEMANTIC},
    {"memory_type": MemoryType.EPISODIC, "bridge_fn": my_bridge},
])
```

### Memory Router (Intelligent Routing)

```python
from day_51.memory_router import (
    MemoryRouter, analyse_query,
    KeywordRoutingStrategy, WeightedRoutingStrategy, AdaptiveRoutingStrategy,
)
from day_51.hybrid_memory import HybridMemory

hm = HybridMemory()
# ... populate with events, experiences, facts, nodes ...

# Basic routing — auto-analyses query and routes to the right subsystem
router = MemoryRouter(hm)
result = router.route("When did the login event happen?")  # → episodic
result = router.route("What is Python?")                   # → semantic
result = router.route("How did search perform?")           # → episodic (experiences)

# Inspect analysis without executing
analysis = router.analyse("What is related to Django?")
print(analysis.intent)              # QueryIntent.RELATIONAL
print(analysis.signals)             # {temporal: 0.0, factual: 0.2, ...}
print(analysis.detected_filters)    # {node_type: NodeType.ENTITY, ...}

# Full inspection: result + analysis + decision
result, analysis, decision = router.route_with_analysis("What is Python?")
print(decision.memory_type)         # MemoryType.SEMANTIC
print(decision.strategy_name)       # "keyword"

# Switch strategies at runtime
router.strategy = WeightedRoutingStrategy(threshold=0.5)
router.strategy = AdaptiveRoutingStrategy(learning_rate=0.1)

# Override routing decisions
result = router.route("anything", memory_type=MemoryType.SEMANTIC)

# Routing stats
print(router.stats())  # {total_queries: 3, avg_results: 2.0, ...}
```

### Hybrid Memory (Episodic + Semantic)

```python
from day_51.hybrid_memory import HybridMemory, MemoryType
from day_49.event_store import EventType
from day_49.experience_tracker import Outcome
from day_50.fact_store import FactType
from day_50.knowledge_graph import NodeType, EdgeType

hm = HybridMemory()

# Store across subsystems via unified API
ev = hm.store_event(EventType.ACTION, data={"action": "login"}, participants=["alice"])
exp = hm.store_experience("search", Outcome.SUCCESS, score=0.9, tags=["web"])
fact = hm.store_fact("Python", "is_a", "programming language", fact_type=FactType.DEFINITION)
node = hm.store_node("Python", NodeType.ENTITY)

# Unified get/delete (auto-detects subsystem)
item = hm.get(ev.id)
hm.detect_type(fact.id)          # MemoryType.SEMANTIC
hm.detect_source(ev.id)          # MemorySource.EVENT_STORE

# Query with routing
result = hm.query("Python")                                    # both types
result = hm.query("", memory_type=MemoryType.EPISODIC)         # episodic only
result = hm.query("", subject="Python", memory_type=MemoryType.SEMANTIC)
result = hm.query("", action="search", outcome=Outcome.SUCCESS)

# Cross-memory linking
link = hm.cross_link(ev.id, fact.id, relationship="triggered_by")
linked = hm.get_linked_items(ev.id)  # [fact MemoryItem]

# Stats
hm.stats()  # {events: 1, experiences: 1, facts: 1, graph_nodes: 1, ...}
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

### Unified Memory System
- ✅ Single API over all memory types (working, short-term, long-term, hybrid)
- ✅ Automatic type detection and intelligent routing
- ✅ Seamless access across conversation, context, tokens, events, experiences, facts, and knowledge graph
- ✅ Cross-memory linking and related item retrieval
- ✅ Context building with automatic memory retrieval
- ✅ Memory optimization and export capabilities
- ✅ Layer-aware querying (working, short-term, long-term, hybrid)
- ✅ Cross-type query patterns (combined, sequential, parallel)
- ✅ Unified store/get/delete interface with type abstraction
- ✅ Error handling with graceful degradation
- ✅ Comprehensive memory statistics and management

### Memory Coordination
- ✅ Query routing with keyword-based and adaptive strategies
- ✅ Result merging with relevance-based ranking and deduplication
- ✅ State synchronization across memory layers with conflict detection
- ✅ Event coordination with threading-safe event queue
- ✅ Multiple coordination patterns (sequential, parallel, hierarchical, adaptive)
- ✅ Cross-layer storage coordination
- ✅ Performance monitoring and statistics
- ✅ Error handling and recovery mechanisms

### Intelligent Type Selection
- ✅ Query characteristic analysis (temporal, factual, experiential, relational, conversational signals)
- ✅ Information need detection (recall, lookup, exploration, temporal, experiential)
- ✅ Question detection and temporal reference identification
- ✅ RuleBasedSelector: deterministic threshold-based rules
- ✅ QueryBasedSelector: scored (layer, type) pair selection with alternatives
- ✅ PatternBasedSelector: learns from past query→result patterns with heuristic fallback
- ✅ HybridSelector: weighted voting across multiple selectors
- ✅ AutoSelector: automatic selection with caching and configurable fallback
- ✅ SelectionOptimizer: outcome tracking, per-algorithm performance stats, best algorithm detection
- ✅ Pluggable algorithm interface (SelectionAlgorithm ABC)
- ✅ Custom error hierarchy

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

### Efficient Profile Retrieval
- ✅ Multi-level caching system (LRU and TTL caches)
- ✅ Connection pooling for optimized database access
- ✅ Multiple retrieval patterns (direct, batch, filtered, preference, metadata)
- ✅ Query optimization with database indexes and ANALYZE
- ✅ Performance monitoring and comprehensive metrics collection
- ✅ Cache invalidation and intelligent cache management
- ✅ Thread-safe operations with concurrent access support
- ✅ Database optimization tools (ANALYZE, VACUUM)
- ✅ Configurable cache strategies and TTL settings
- ✅ Batch operations for improved performance
- ✅ Query result caching with automatic invalidation
- ✅ Performance decorators for automatic monitoring
- ✅ Connection pool statistics and monitoring
- ✅ Cache hit rate optimization and statistics

### Secure Profile Storage
- ✅ Profile storage with AES-128 encryption using Fernet
- ✅ Token-based authentication and authorization system
- ✅ Role-based access control (Read, Write, Delete, Admin permissions)
- ✅ Resource-level access control (users can only access own profiles)
- ✅ Comprehensive audit logging with SQLite database
- ✅ IP address tracking and activity monitoring
- ✅ Master key management for encryption key backup
- ✅ Token expiration and automatic cleanup
- ✅ Secure CRUD operations with validation
- ✅ Custom security exception hierarchy
- ✅ Admin-only operations (list all profiles, access any profile)
- ✅ Authentication failure logging and monitoring
- ✅ Data encryption at rest (sensitive data not readable in database)
- ✅ Session management with logout and token revocation

### User Profile System
- ✅ Profile data structure with user identifier, basic information, preferences, settings, metadata
- ✅ Multiple profile types (Basic, Premium, Admin, Guest) with type-specific validation
- ✅ Preference management with validation (language, timezone, theme, notifications, privacy)
- ✅ Profile updates with version control and validation
- ✅ Schema validation, data validation, type checking, constraint checking
- ✅ ProfileManager for CRUD operations (create, get, update, delete, search)
- ✅ JSON serialization/deserialization with datetime handling
- ✅ Profile statistics and export/import functionality
- ✅ Custom error hierarchy (ValidationError, ProfileNotFoundError)
- ✅ Pluggable validator system with type-specific validation rules
- ✅ Login tracking and activity recording
- ✅ Custom data and preference storage
- ✅ Profile search and filtering capabilities
- ✅ Version control for profile updates

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

### Hybrid Memory System
- ✅ Unified interface over episodic and semantic subsystems
- ✅ Unified store/get/delete across EventStore, ExperienceTracker, FactStore, KnowledgeGraph
- ✅ Automatic type detection (episodic vs semantic)
- ✅ Automatic source detection (which subsystem owns an ID)
- ✅ Query routing by memory type (episodic, semantic, or both)
- ✅ Rich query filters (event type, participant, action, outcome, tag, subject, predicate, fact type, confidence, node type, node label, time range)
- ✅ Result merging with timestamp sorting and limit
- ✅ Cross-memory linking between episodic and semantic items
- ✅ Bidirectional linked item retrieval
- ✅ Cross-link cleanup on delete
- ✅ Subsystem stats aggregation
- ✅ MemoryItem unified wrapper
- ✅ Custom error hierarchy

### Cross-Type Queries
- ✅ Combined queries: simultaneous episodic + semantic with merged results
- ✅ Sequential queries: primary → bridge → secondary with result accumulation
- ✅ Parallel queries: concurrent sub-query execution via ThreadPoolExecutor
- ✅ Multi-hop queries: chained sequential hops with bridge functions
- ✅ Result merging with ID-based deduplication
- ✅ Three ranking strategies (relevance, timestamp, source_priority)
- ✅ Text-match relevance scoring with source priority tiebreak
- ✅ Custom bridge function support for sequential queries
- ✅ Default bridge: keyword extraction from primary results
- ✅ SubQuery dataclass for structured parallel definitions
- ✅ MergedResult with execution time tracking and metadata
- ✅ Formatted result output
- ✅ Custom error hierarchy

### Memory Router
- ✅ Query analysis with intent detection (temporal, factual, experiential, relational, ambiguous)
- ✅ Normalized signal scoring across intent categories
- ✅ Structured filter extraction from query text (event types, outcomes, fact types, node types)
- ✅ Pluggable routing strategies via abstract base class
- ✅ KeywordRoutingStrategy: dominant-signal routing
- ✅ WeightedRoutingStrategy: confidence-threshold routing with BOTH fallback
- ✅ AdaptiveRoutingStrategy: learns from query outcomes with configurable learning/exploration rates
- ✅ MemoryRouter orchestrator with automatic analysis → strategy → execution pipeline
- ✅ Override support for explicit routing control
- ✅ Full inspection mode (route_with_analysis)
- ✅ Audit log with query history
- ✅ Routing statistics (total queries, average results, intent distribution)
- ✅ Runtime strategy switching
- ✅ Custom error hierarchy

### Vector Database
- ✅ ChromaDB integration for vector storage
- ✅ Collection management with metadata
- ✅ Document CRUD operations with embeddings
- ✅ Similarity search with metadata filtering
- ✅ Search by pre-computed vectors
- ✅ Persistent storage support
- ✅ Batch document operations
- ✅ Custom embedding provider interface
- ✅ Comprehensive error handling
- ✅ Document counting and inspection

### Embedding Generation
- ✅ Text preprocessing with configurable options (lowercase, special char removal, length limits)
- ✅ Multiple embedding providers (Sentence Transformers, OpenAI)
- ✅ Pluggable provider interface for extensibility
- ✅ Memory and disk-based caching with LRU eviction
- ✅ Batch processing with configurable batch sizes
- ✅ Comprehensive error handling and validation
- ✅ Performance monitoring and statistics
- ✅ Factory function for easy setup
- ✅ Support for custom preprocessing pipelines
- ✅ Cache hit/miss tracking and statistics
- ✅ Automatic model loading and dimension detection
- ✅ Support for both numpy arrays and regular lists
- ✅ Graceful handling of missing dependencies

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

# Run Day 54 tests
cd ../day_54
python -m pytest test_user_profile.py -v
python -m pytest test_profile_storage.py -v
python -m pytest test_profile_retrieval.py -v

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

# Run Day 51 tests
cd ../day_51
python -m pytest test_hybrid_memory.py -v
python -m pytest test_memory_router.py -v
python -m pytest test_cross_type_queries.py -v

# Run Day 52 tests
cd ../day_52
python -m pytest test_unified_memory.py -v
python -m pytest test_memory_coordinator.py -v
python -m pytest test_type_selector.py -v
```

## Running Examples

```bash
source venv/bin/activate
cd day_54
python demo.py
python demo_secure_storage.py
python demo_profile_retrieval.py
python example_usage.py
python example_secure_storage.py
python example_profile_retrieval.py

cd ../day_47
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
13. **Hybrid Memory**: Unified querying across episodic events/experiences and semantic facts/graphs with cross-linking
14. **Intelligent Routing**: Automatic query analysis and routing to the right memory subsystem with adaptive learning
16. **User Profile Management**: Store and manage user profiles with preferences, validation, and multiple profile types
17. **Secure Profile Storage**: Encrypted profile storage with access control, audit logging, and security features
18. **Efficient Profile Retrieval**: High-performance profile retrieval with multi-level caching and query optimization
19. **Profile-Based Personalization**: Customize agent behavior based on user profiles and preferences

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
- cryptography: Encryption library for secure storage
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
├── day_54/                          # User Profile System
│   ├── user_profile.py              # User profile data structures & management
│   ├── profile_storage.py           # Secure profile storage with encryption
│   ├── profile_retrieval.py         # Efficient profile retrieval with caching
│   ├── test_user_profile.py         # User profile tests
│   ├── test_profile_storage.py      # Secure storage tests
│   ├── test_profile_retrieval.py    # Profile retrieval tests
│   ├── example_usage.py             # Profile usage examples
│   ├── example_secure_storage.py    # Security examples
│   ├── example_profile_retrieval.py # Retrieval examples
│   ├── demo.py                      # Simple profile demonstration
│   ├── demo_secure_storage.py       # Simple security demonstration
│   └── demo_profile_retrieval.py    # Simple retrieval demonstration
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
├── day_52/                          # Unified Memory System
│   ├── unified_memory.py            # Unified interface integrating all memory types
│   ├── memory_coordinator.py        # Coordination mechanisms for memory types
│   ├── test_unified_memory.py        # Unified memory tests
│   ├── test_memory_coordinator.py   # Memory coordinator tests
│   ├── type_selector.py              # Intelligent memory type selection
│   ├── test_type_selector.py         # Type selector tests
│   ├── example_unified_memory.py     # Unified memory usage examples
│   └── example_memory_coordinator.py # Memory coordinator usage examples
├── day_51/                          # Hybrid Memory, Router & Cross-Type Queries
│   ├── hybrid_memory.py             # Unified episodic + semantic memory
│   ├── test_hybrid_memory.py        # Hybrid memory tests
│   ├── memory_router.py             # Intelligent memory type routing
│   ├── test_memory_router.py        # Memory router tests
│   ├── cross_type_queries.py        # Cross-type query patterns & merging
│   └── test_cross_type_queries.py   # Cross-type query tests
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
- [x] Hybrid memory combining episodic and semantic subsystems
- [x] Intelligent memory type routing with adaptive learning
- [x] Cross-type query support with combined, sequential, and parallel patterns
- [x] Unified memory system integrating all memory types with single interface
- [x] Intelligent memory type selection with multiple algorithms and optimization
