# Day 49: Episodic Event Store, Conversation Memory & Experience Tracker

Three complementary memory modules for AI agents — episodic events, multi-conversation management, and experience-based learning.

## Modules

### Event Store (`event_store.py`)

Stores episodic events with temporal ordering and bidirectional relationships.

**Event types:** Action, Observation, Decision, Communication, Error, Custom

```python
from event_store import EventStore, EventType

store = EventStore()

# Store events with relationships
e1 = store.store(EventType.ACTION, data={"action": "login"}, participants=["alice"])
e2 = store.store(EventType.OBSERVATION, data={"saw": "dashboard"},
                 participants=["alice"], related_event_ids={e1.id})

# Retrieval
store.by_type(EventType.ACTION)           # all actions
store.by_participant("alice")             # alice's events
store.by_time_range(start, end)           # time window
store.get_related(e1.id)                  # related events

# Temporal queries
store.timeline(limit=10)                  # last 10 events
store.event_sequence([e2.id, e1.id])      # sorted by time
store.temporal_pattern(EventType.ACTION)  # intervals between actions

# Manage
store.link(e1.id, e2.id)                 # link events manually
store.delete(e1.id)                       # delete with relationship cleanup
```

**Key features:**
- Binary-search insertion maintains temporal order
- Bidirectional event relationships (auto-linked on store)
- Temporal pattern detection (interval analysis)
- Timeline reconstruction and event sequence sorting

---

### Conversation Memory (`conversation_memory.py`)

Multi-conversation storage with message sequences, context tracking, and topic management.

```python
from conversation_memory import ConversationMemory

mem = ConversationMemory()

# Create conversation and add messages
conv = mem.create_conversation("alice", context={"lang": "en"})
m1 = mem.add_message(conv.id, "user", "Hello!", topics=["greeting"])
m2 = mem.add_message(conv.id, "assistant", "Hi there!")
mem.add_message(conv.id, "user", "Back to that", references=[m1.id])

# Context maintenance
mem.update_context(conv.id, {"mood": "happy"})  # merges with existing
mem.add_topics(conv.id, ["python"])

# Retrieval
mem.by_user("alice")                      # all alice's conversations
mem.by_topic("greeting")                  # by topic
mem.search_content("hello")               # case-insensitive content search
mem.by_time_range(start, end)             # by time window

# Multi-turn helpers
mem.get_turn_pairs(conv.id)               # user/assistant pairs
mem.get_referenced_messages(conv.id, m.id) # follow message references
mem.build_context_window(conv.id, last_n=5) # formatted context string
```

**Key features:**
- Conversation lifecycle (create, get, delete) linked to users
- Message ordering with metadata and cross-references
- Merge-based context updates across turns
- Topic tracking per-message and manually
- Context window building for LLM input

---

### Experience Tracker (`experience_tracker.py`)

Logs agent experiences with outcomes, recognises patterns, extracts lessons, and tracks improvement.

**Outcome types:** Success, Failure, Partial, Unknown

```python
from experience_tracker import ExperienceTracker, Outcome

tracker = ExperienceTracker()

# Log experiences
tracker.log("search", Outcome.SUCCESS, score=0.9, tags=["web"])
tracker.log("search", Outcome.FAILURE, score=0.2, feedback="timeout")
tracker.log("search", Outcome.SUCCESS, score=0.8, tags=["web", "api"])

# Retrieval
tracker.by_action("search")              # all search experiences
tracker.by_outcome(Outcome.SUCCESS)       # all successes
tracker.by_tag("web")                     # by tag
tracker.recent(5)                         # last 5 experiences

# Pattern recognition
tracker.success_rate("search")            # 0.666...
tracker.average_score("search")           # 0.633...
tracker.outcome_distribution("search")    # {"success": 2, "failure": 1}
tracker.common_tags("search")             # ["web", "api"]
tracker.score_trend("search")             # [0.9, 0.2, 0.8]

# Learning
lesson = tracker.extract_lesson("search") # Lesson with recommendation
tracker.compare_actions(["search", "summarize"])  # sorted by success rate
tracker.is_improving("search", window=3)  # True / False / None

# Post-hoc feedback
tracker.add_feedback(exp.id, "worked well", score=0.95)
```

**Key features:**
- Score validation (0.0–1.0) and duplicate ID detection
- Pattern recognition: success rate, average score, outcome distribution
- Common tag analysis and chronological score trends
- Lesson extraction with tiered recommendations (≥80% → continue, ≥50% → review, <50% → change)
- Multi-action comparison sorted by success rate
- Improvement detection via sliding window score analysis

## Running Tests

```bash
source venv/bin/activate
cd day_49

# All tests
python -m pytest -v

# Individual modules
python -m pytest test_event_store.py -v
python -m pytest test_conversation_memory.py -v
python -m pytest test_experience_tracker.py -v
```

## File Structure

```
day_49/
├── event_store.py               # Event storage with temporal ordering
├── test_event_store.py          # 29 tests
├── conversation_memory.py       # Multi-conversation memory management
├── test_conversation_memory.py  # 38 tests
├── experience_tracker.py        # Experience tracking for learning
├── test_experience_tracker.py   # 41 tests
└── README.md                    # This file
```

## How They Work Together

These three modules address different aspects of agent memory:

| Module | Purpose | Stores |
|---|---|---|
| Event Store | What happened | Discrete events with temporal relationships |
| Conversation Memory | What was said | Multi-turn dialogues with context |
| Experience Tracker | What was learned | Outcomes, patterns, and lessons |

An agent could use all three: log events as they occur, track conversations with users, and learn from experience outcomes to improve future decisions.
