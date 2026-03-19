"""Semantic memory fact store using subject-predicate-object triples."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid


class FactStoreError(Exception):
    """Base exception for fact store operations."""


class FactNotFoundError(FactStoreError):
    """Raised when a fact is not found."""


class FactType(Enum):
    ASSERTION = "assertion"
    DEFINITION = "definition"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    RULE = "rule"


@dataclass
class Fact:
    """A subject-predicate-object triple with metadata."""
    id: str
    subject: str
    predicate: str
    object: str
    fact_type: FactType = FactType.ASSERTION
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FactStore:
    """Stores and queries semantic facts as subject-predicate-object triples."""

    def __init__(self):
        self._facts: Dict[str, Fact] = {}
        # Indexes for fast lookup
        self._by_subject: Dict[str, Set[str]] = {}
        self._by_predicate: Dict[str, Set[str]] = {}
        self._by_object: Dict[str, Set[str]] = {}
        self._by_type: Dict[FactType, Set[str]] = {}
        # Relationships: fact_id -> set of related fact_ids
        self._relationships: Dict[str, Set[str]] = {}

    @property
    def count(self) -> int:
        return len(self._facts)

    # --- Storage ---

    def store(self, subject: str, predicate: str, object_: str,
              fact_type: FactType = FactType.ASSERTION,
              properties: Optional[Dict[str, Any]] = None,
              confidence: float = 1.0,
              fact_id: Optional[str] = None,
              related_fact_ids: Optional[Set[str]] = None) -> Fact:
        fid = fact_id or str(uuid.uuid4())
        if fid in self._facts:
            raise FactStoreError(f"Fact '{fid}' already exists")

        fact = Fact(
            id=fid, subject=subject, predicate=predicate, object=object_,
            fact_type=fact_type, properties=properties or {},
            confidence=confidence,
        )
        self._facts[fid] = fact
        self._index_add(fact)

        if related_fact_ids:
            for rid in related_fact_ids:
                if rid in self._facts:
                    self._link(fid, rid)

        return fact

    def get(self, fact_id: str) -> Fact:
        if fact_id not in self._facts:
            raise FactNotFoundError(f"Fact '{fact_id}' not found")
        return self._facts[fact_id]

    def update(self, fact_id: str, subject: Optional[str] = None,
               predicate: Optional[str] = None, object_: Optional[str] = None,
               properties: Optional[Dict[str, Any]] = None,
               confidence: Optional[float] = None) -> Fact:
        fact = self.get(fact_id)
        self._index_remove(fact)

        if subject is not None:
            fact.subject = subject
        if predicate is not None:
            fact.predicate = predicate
        if object_ is not None:
            fact.object = object_
        if properties is not None:
            fact.properties.update(properties)
        if confidence is not None:
            fact.confidence = confidence
        fact.updated_at = datetime.now()

        self._index_add(fact)
        return fact

    def delete(self, fact_id: str) -> bool:
        fact = self.get(fact_id)
        self._index_remove(fact)
        # Clean up relationships
        for rid in list(self._relationships.get(fact_id, [])):
            self._relationships.get(rid, set()).discard(fact_id)
        self._relationships.pop(fact_id, None)
        del self._facts[fact_id]
        return True

    # --- Retrieval ---

    def by_subject(self, subject: str) -> List[Fact]:
        ids = self._by_subject.get(subject, set())
        return sorted([self._facts[i] for i in ids], key=lambda f: f.created_at)

    def by_predicate(self, predicate: str) -> List[Fact]:
        ids = self._by_predicate.get(predicate, set())
        return sorted([self._facts[i] for i in ids], key=lambda f: f.created_at)

    def by_object(self, object_: str) -> List[Fact]:
        ids = self._by_object.get(object_, set())
        return sorted([self._facts[i] for i in ids], key=lambda f: f.created_at)

    def by_type(self, fact_type: FactType) -> List[Fact]:
        ids = self._by_type.get(fact_type, set())
        return sorted([self._facts[i] for i in ids], key=lambda f: f.created_at)

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None,
              object_: Optional[str] = None, fact_type: Optional[FactType] = None,
              min_confidence: Optional[float] = None) -> List[Fact]:
        """Query facts by any combination of fields. All conditions are ANDed."""
        candidates: Optional[Set[str]] = None

        for key, index in [
            (subject, self._by_subject),
            (predicate, self._by_predicate),
            (object_, self._by_object),
        ]:
            if key is not None:
                ids = index.get(key, set())
                candidates = ids if candidates is None else candidates & ids

        if fact_type is not None:
            ids = self._by_type.get(fact_type, set())
            candidates = ids if candidates is None else candidates & ids

        if candidates is None:
            candidates = set(self._facts.keys())

        results = [self._facts[i] for i in candidates]
        if min_confidence is not None:
            results = [f for f in results if f.confidence >= min_confidence]
        return sorted(results, key=lambda f: f.created_at)

    # --- Relationship management ---

    def link(self, fact_id_a: str, fact_id_b: str) -> None:
        self.get(fact_id_a)
        self.get(fact_id_b)
        self._link(fact_id_a, fact_id_b)

    def unlink(self, fact_id_a: str, fact_id_b: str) -> None:
        self._relationships.get(fact_id_a, set()).discard(fact_id_b)
        self._relationships.get(fact_id_b, set()).discard(fact_id_a)

    def get_related(self, fact_id: str) -> List[Fact]:
        self.get(fact_id)  # validate exists
        ids = self._relationships.get(fact_id, set())
        return sorted([self._facts[i] for i in ids if i in self._facts],
                       key=lambda f: f.created_at)

    def traverse(self, fact_id: str, max_depth: int = 2) -> Dict[int, List[Fact]]:
        """BFS traversal of related facts up to max_depth."""
        self.get(fact_id)
        visited: Set[str] = {fact_id}
        current_level = {fact_id}
        result: Dict[int, List[Fact]] = {}

        for depth in range(1, max_depth + 1):
            next_level: Set[str] = set()
            for fid in current_level:
                for rid in self._relationships.get(fid, set()):
                    if rid not in visited and rid in self._facts:
                        next_level.add(rid)
                        visited.add(rid)
            if not next_level:
                break
            result[depth] = sorted([self._facts[i] for i in next_level],
                                    key=lambda f: f.created_at)
            current_level = next_level

        return result

    # --- Internal helpers ---

    def _link(self, a: str, b: str) -> None:
        self._relationships.setdefault(a, set()).add(b)
        self._relationships.setdefault(b, set()).add(a)

    def _index_add(self, fact: Fact) -> None:
        self._by_subject.setdefault(fact.subject, set()).add(fact.id)
        self._by_predicate.setdefault(fact.predicate, set()).add(fact.id)
        self._by_object.setdefault(fact.object, set()).add(fact.id)
        self._by_type.setdefault(fact.fact_type, set()).add(fact.id)

    def _index_remove(self, fact: Fact) -> None:
        self._by_subject.get(fact.subject, set()).discard(fact.id)
        self._by_predicate.get(fact.predicate, set()).discard(fact.id)
        self._by_object.get(fact.object, set()).discard(fact.id)
        self._by_type.get(fact.fact_type, set()).discard(fact.id)
