"""Episodic event store for storing and querying temporal events."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid


class EventStoreError(Exception):
    """Base exception for event store operations."""


class EventNotFoundError(EventStoreError):
    """Raised when an event is not found."""


class EventType(str, Enum):
    ACTION = "action"
    OBSERVATION = "observation"
    DECISION = "decision"
    COMMUNICATION = "communication"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Event:
    """A single episodic event."""
    id: str
    event_type: EventType
    timestamp: datetime
    participants: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    related_event_ids: Set[str] = field(default_factory=set)


class EventStore:
    """Stores episodic events with temporal ordering and relationship tracking."""

    def __init__(self):
        self._events: Dict[str, Event] = {}
        self._timeline: List[str] = []  # event IDs in temporal order

    @property
    def count(self) -> int:
        return len(self._events)

    def _insert_in_order(self, event: Event):
        """Insert event ID into timeline maintaining temporal order."""
        ts = event.timestamp
        # Binary search for insertion point
        lo, hi = 0, len(self._timeline)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._events[self._timeline[mid]].timestamp <= ts:
                lo = mid + 1
            else:
                hi = mid
        self._timeline.insert(lo, event.id)

    def store(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        participants: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        related_event_ids: Optional[Set[str]] = None,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
    ) -> Event:
        """Store a new event. Returns the created Event."""
        eid = event_id or str(uuid.uuid4())
        if eid in self._events:
            raise EventStoreError(f"Event '{eid}' already exists")

        # Validate related events exist
        relations = related_event_ids or set()
        for rid in relations:
            if rid not in self._events:
                raise EventNotFoundError(f"Related event '{rid}' not found")

        event = Event(
            id=eid,
            event_type=event_type,
            timestamp=timestamp or datetime.now(),
            participants=participants or [],
            context=context or {},
            data=data or {},
            related_event_ids=relations,
        )
        self._events[eid] = event
        self._insert_in_order(event)

        # Add bidirectional relationships
        for rid in relations:
            self._events[rid].related_event_ids.add(eid)

        return event

    def get(self, event_id: str) -> Event:
        """Retrieve a single event by ID."""
        if event_id not in self._events:
            raise EventNotFoundError(f"Event '{event_id}' not found")
        return self._events[event_id]

    def link(self, event_id_1: str, event_id_2: str):
        """Create a bidirectional relationship between two events."""
        e1 = self.get(event_id_1)
        e2 = self.get(event_id_2)
        e1.related_event_ids.add(event_id_2)
        e2.related_event_ids.add(event_id_1)

    def get_related(self, event_id: str) -> List[Event]:
        """Get all events related to the given event, in temporal order."""
        event = self.get(event_id)
        related = [self._events[rid] for rid in event.related_event_ids if rid in self._events]
        return sorted(related, key=lambda e: e.timestamp)

    # --- Retrieval methods ---

    def by_time_range(self, start: datetime, end: datetime) -> List[Event]:
        """Retrieve events within a timestamp range (inclusive)."""
        return [
            self._events[eid] for eid in self._timeline
            if start <= self._events[eid].timestamp <= end
        ]

    def by_type(self, event_type: EventType) -> List[Event]:
        """Retrieve events of a specific type in temporal order."""
        return [
            self._events[eid] for eid in self._timeline
            if self._events[eid].event_type == event_type
        ]

    def by_participant(self, participant: str) -> List[Event]:
        """Retrieve events involving a specific participant in temporal order."""
        return [
            self._events[eid] for eid in self._timeline
            if participant in self._events[eid].participants
        ]

    # --- Temporal queries ---

    def timeline(self, limit: Optional[int] = None) -> List[Event]:
        """Reconstruct full timeline (or last N events)."""
        ids = self._timeline if limit is None else self._timeline[-limit:]
        return [self._events[eid] for eid in ids]

    def event_sequence(self, event_ids: List[str]) -> List[Event]:
        """Return the given events sorted in temporal order."""
        events = [self.get(eid) for eid in event_ids]
        return sorted(events, key=lambda e: e.timestamp)

    def temporal_pattern(self, event_type: EventType, participant: Optional[str] = None) -> List[float]:
        """Return intervals (in seconds) between consecutive matching events.

        Useful for detecting temporal patterns like frequency of actions.
        """
        matches = self.by_type(event_type)
        if participant:
            matches = [e for e in matches if participant in e.participants]
        if len(matches) < 2:
            return []
        return [
            (matches[i + 1].timestamp - matches[i].timestamp).total_seconds()
            for i in range(len(matches) - 1)
        ]

    def delete(self, event_id: str) -> bool:
        """Delete an event and clean up relationships."""
        event = self.get(event_id)
        for rid in event.related_event_ids:
            if rid in self._events:
                self._events[rid].related_event_ids.discard(event_id)
        self._timeline.remove(event_id)
        del self._events[event_id]
        return True
