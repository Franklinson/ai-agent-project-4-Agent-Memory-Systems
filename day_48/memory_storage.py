"""Memory storage interface that abstracts storage details for AI agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


class MemoryStorageError(Exception):
    """Base exception for memory storage operations."""


class MemoryNotFoundError(MemoryStorageError):
    """Raised when a memory is not found."""


class MemoryConflictError(MemoryStorageError):
    """Raised when an update conflicts with existing state."""


@dataclass
class Memory:
    """A single memory entry with metadata."""
    id: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryStorageInterface(ABC):
    """Abstract base class defining the storage interface."""

    @abstractmethod
    def save(self, content: Any, metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None) -> Memory:
        """Store a memory. Returns the saved Memory."""

    @abstractmethod
    def retrieve(self, memory_id: str) -> Memory:
        """Retrieve a memory by ID. Raises MemoryNotFoundError if not found."""

    @abstractmethod
    def update(self, memory_id: str, content: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        """Update an existing memory. Raises MemoryNotFoundError or MemoryConflictError."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True on success. Raises MemoryNotFoundError if not found."""

    @abstractmethod
    def list_memories(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """List memories, optionally filtered by metadata."""


class InMemoryStorage(MemoryStorageInterface):
    """In-memory storage backend implementation."""

    def __init__(self):
        self._store: Dict[str, Memory] = {}

    def save(self, content: Any, metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None) -> Memory:
        try:
            mid = memory_id or str(uuid.uuid4())
            memory = Memory(id=mid, content=content, metadata=metadata or {})
            self._store[mid] = memory
            return memory
        except Exception as e:
            raise MemoryStorageError(f"Failed to save memory: {e}")

    def retrieve(self, memory_id: str) -> Memory:
        if memory_id not in self._store:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found")
        return self._store[memory_id]

    def update(self, memory_id: str, content: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        if memory_id not in self._store:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found")
        memory = self._store[memory_id]
        if content is not None:
            memory.content = content
        if metadata is not None:
            memory.metadata.update(metadata)
        memory.updated_at = datetime.now().isoformat()
        return memory

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._store:
            raise MemoryNotFoundError(f"Memory '{memory_id}' not found")
        del self._store[memory_id]
        return True

    def list_memories(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Memory]:
        if not filter_metadata:
            return list(self._store.values())
        return [
            m for m in self._store.values()
            if all(m.metadata.get(k) == v for k, v in filter_metadata.items())
        ]
