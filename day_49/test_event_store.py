"""Tests for the episodic event store."""

import pytest
from datetime import datetime, timedelta
from event_store import EventStore, EventType, Event, EventNotFoundError, EventStoreError


@pytest.fixture
def store():
    return EventStore()


@pytest.fixture
def populated_store(store):
    """Store with 5 events across types and participants."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    store.store(EventType.ACTION, data={"action": "login"}, participants=["alice"],
                timestamp=base, event_id="e1")
    store.store(EventType.OBSERVATION, data={"saw": "dashboard"}, participants=["alice"],
                timestamp=base + timedelta(minutes=5), event_id="e2", related_event_ids={"e1"})
    store.store(EventType.DECISION, data={"chose": "report"}, participants=["alice", "bob"],
                timestamp=base + timedelta(minutes=10), event_id="e3")
    store.store(EventType.COMMUNICATION, data={"msg": "hello"}, participants=["bob"],
                timestamp=base + timedelta(minutes=15), event_id="e4")
    store.store(EventType.ERROR, data={"error": "timeout"}, participants=["alice"],
                timestamp=base + timedelta(minutes=20), event_id="e5")
    return store


# --- Storage ---

class TestStorage:
    def test_store_event(self, store):
        event = store.store(EventType.ACTION, data={"key": "val"}, participants=["agent"])
        assert isinstance(event, Event)
        assert event.event_type == EventType.ACTION
        assert event.data == {"key": "val"}
        assert store.count == 1

    def test_store_with_custom_id(self, store):
        event = store.store(EventType.OBSERVATION, event_id="custom-1")
        assert event.id == "custom-1"

    def test_store_duplicate_id_raises(self, store):
        store.store(EventType.ACTION, event_id="dup")
        with pytest.raises(EventStoreError, match="already exists"):
            store.store(EventType.ACTION, event_id="dup")

    def test_store_with_context(self, store):
        event = store.store(EventType.CUSTOM, context={"env": "prod"})
        assert event.context == {"env": "prod"}

    def test_store_with_invalid_relation_raises(self, store):
        with pytest.raises(EventNotFoundError, match="Related event"):
            store.store(EventType.ACTION, related_event_ids={"nonexistent"})

    def test_store_defaults(self, store):
        event = store.store(EventType.ACTION)
        assert event.participants == []
        assert event.context == {}
        assert event.data == {}
        assert event.related_event_ids == set()


# --- Retrieval ---

class TestRetrieval:
    def test_get_event(self, populated_store):
        event = populated_store.get("e1")
        assert event.id == "e1"
        assert event.data["action"] == "login"

    def test_get_nonexistent_raises(self, store):
        with pytest.raises(EventNotFoundError):
            store.get("nope")

    def test_by_type(self, populated_store):
        actions = populated_store.by_type(EventType.ACTION)
        assert len(actions) == 1
        assert actions[0].id == "e1"

    def test_by_participant(self, populated_store):
        alice_events = populated_store.by_participant("alice")
        assert [e.id for e in alice_events] == ["e1", "e2", "e3", "e5"]

    def test_by_time_range(self, populated_store):
        base = datetime(2025, 1, 1, 12, 0, 0)
        results = populated_store.by_time_range(base, base + timedelta(minutes=10))
        assert [e.id for e in results] == ["e1", "e2", "e3"]

    def test_by_type_empty(self, store):
        assert store.by_type(EventType.ACTION) == []

    def test_by_participant_empty(self, store):
        assert store.by_participant("nobody") == []


# --- Relationships ---

class TestRelationships:
    def test_store_with_relationship(self, populated_store):
        e1 = populated_store.get("e1")
        e2 = populated_store.get("e2")
        assert "e2" in e1.related_event_ids  # bidirectional
        assert "e1" in e2.related_event_ids

    def test_link_events(self, populated_store):
        populated_store.link("e3", "e4")
        assert "e4" in populated_store.get("e3").related_event_ids
        assert "e3" in populated_store.get("e4").related_event_ids

    def test_get_related(self, populated_store):
        related = populated_store.get_related("e1")
        assert len(related) == 1
        assert related[0].id == "e2"

    def test_link_nonexistent_raises(self, store):
        store.store(EventType.ACTION, event_id="a")
        with pytest.raises(EventNotFoundError):
            store.link("a", "missing")


# --- Temporal Queries ---

class TestTemporalQueries:
    def test_timeline_full(self, populated_store):
        tl = populated_store.timeline()
        assert [e.id for e in tl] == ["e1", "e2", "e3", "e4", "e5"]

    def test_timeline_limit(self, populated_store):
        tl = populated_store.timeline(limit=2)
        assert [e.id for e in tl] == ["e4", "e5"]

    def test_event_sequence(self, populated_store):
        seq = populated_store.event_sequence(["e5", "e1", "e3"])
        assert [e.id for e in seq] == ["e1", "e3", "e5"]

    def test_temporal_pattern(self, store):
        base = datetime(2025, 1, 1)
        store.store(EventType.ACTION, participants=["a"], timestamp=base, event_id="t1")
        store.store(EventType.ACTION, participants=["a"], timestamp=base + timedelta(seconds=10), event_id="t2")
        store.store(EventType.ACTION, participants=["a"], timestamp=base + timedelta(seconds=25), event_id="t3")
        intervals = store.temporal_pattern(EventType.ACTION)
        assert intervals == [10.0, 15.0]

    def test_temporal_pattern_with_participant(self, store):
        base = datetime(2025, 1, 1)
        store.store(EventType.ACTION, participants=["a"], timestamp=base)
        store.store(EventType.ACTION, participants=["b"], timestamp=base + timedelta(seconds=5))
        store.store(EventType.ACTION, participants=["a"], timestamp=base + timedelta(seconds=20))
        intervals = store.temporal_pattern(EventType.ACTION, participant="a")
        assert intervals == [20.0]

    def test_temporal_pattern_insufficient_events(self, store):
        store.store(EventType.ACTION)
        assert store.temporal_pattern(EventType.ACTION) == []

    def test_temporal_order_maintained_with_out_of_order_inserts(self, store):
        base = datetime(2025, 6, 1)
        store.store(EventType.ACTION, timestamp=base + timedelta(hours=2), event_id="late")
        store.store(EventType.ACTION, timestamp=base, event_id="early")
        store.store(EventType.ACTION, timestamp=base + timedelta(hours=1), event_id="mid")
        tl = store.timeline()
        assert [e.id for e in tl] == ["early", "mid", "late"]


# --- Delete ---

class TestDelete:
    def test_delete_event(self, populated_store):
        assert populated_store.delete("e5")
        assert populated_store.count == 4
        with pytest.raises(EventNotFoundError):
            populated_store.get("e5")

    def test_delete_cleans_relationships(self, populated_store):
        populated_store.delete("e2")
        e1 = populated_store.get("e1")
        assert "e2" not in e1.related_event_ids

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(EventNotFoundError):
            store.delete("nope")


# --- Event Types ---

class TestEventTypes:
    def test_all_event_types(self, store):
        for et in EventType:
            event = store.store(et, data={"type": et.value})
            assert event.event_type == et

    def test_custom_event_type(self, store):
        event = store.store(EventType.CUSTOM, data={"custom": True})
        assert event.event_type == EventType.CUSTOM
