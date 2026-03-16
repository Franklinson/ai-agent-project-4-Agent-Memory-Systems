"""Persistence manager that coordinates storage operations with durability guarantees."""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from memory_storage import (
    Memory, MemoryStorageInterface, InMemoryStorage,
    MemoryStorageError, MemoryNotFoundError,
)

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    """Raised when a persistence operation fails after all retries."""


class PersistenceManager:
    """Coordinates storage operations with retry logic, caching, and logging."""

    def __init__(
        self,
        storage: Optional[MemoryStorageInterface] = None,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        cache_size: int = 128,
    ):
        self._storage = storage or InMemoryStorage()
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache: Dict[str, Memory] = {}
        self._cache_size = cache_size
        self._stats = {"saves": 0, "retrievals": 0, "updates": 0, "deletes": 0, "retries": 0, "failures": 0}
        logger.info("PersistenceManager initialized (backend=%s, max_retries=%d)", type(self._storage).__name__, max_retries)

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def _retry(self, operation: str, func, *args, **kwargs):
        """Execute func with retry logic for transient errors."""
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info("%s succeeded on attempt %d", operation, attempt)
                return result
            except MemoryNotFoundError:
                raise
            except MemoryStorageError as e:
                last_error = e
                self._stats["retries"] += 1
                logger.warning("%s attempt %d/%d failed: %s", operation, attempt, self._max_retries, e)
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)

        self._stats["failures"] += 1
        logger.error("%s failed after %d attempts: %s", operation, self._max_retries, last_error)
        raise PersistenceError(f"{operation} failed after {self._max_retries} retries: {last_error}")

    def _cache_put(self, memory: Memory):
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[memory.id] = memory

    def _cache_remove(self, memory_id: str):
        self._cache.pop(memory_id, None)

    def save(self, content: Any, metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None) -> Memory:
        """Save memory with durability guarantee (retry + write confirmation)."""
        start = time.time()
        memory = self._retry("save", self._storage.save, content, metadata, memory_id)

        # Write confirmation: verify the save persisted
        try:
            confirmed = self._storage.retrieve(memory.id)
            self._cache_put(confirmed)
            self._stats["saves"] += 1
            logger.info("save confirmed id=%s (%.3fs)", memory.id, time.time() - start)
            return confirmed
        except MemoryStorageError as e:
            logger.error("save write confirmation failed for id=%s: %s", memory.id, e)
            self._stats["failures"] += 1
            raise PersistenceError(f"Write confirmation failed: {e}")

    def retrieve(self, memory_id: str) -> Memory:
        """Retrieve memory, checking cache first."""
        start = time.time()
        if memory_id in self._cache:
            self._stats["retrievals"] += 1
            logger.debug("cache hit id=%s", memory_id)
            return self._cache[memory_id]

        memory = self._retry("retrieve", self._storage.retrieve, memory_id)
        self._cache_put(memory)
        self._stats["retrievals"] += 1
        logger.info("retrieve id=%s (%.3fs)", memory_id, time.time() - start)
        return memory

    def update(self, memory_id: str, content: Any = None, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        """Update memory with retry and cache invalidation."""
        start = time.time()
        memory = self._retry("update", self._storage.update, memory_id, content, metadata)
        self._cache_put(memory)
        self._stats["updates"] += 1
        logger.info("update id=%s (%.3fs)", memory_id, time.time() - start)
        return memory

    def delete(self, memory_id: str) -> bool:
        """Delete memory with retry and cache cleanup."""
        start = time.time()
        result = self._retry("delete", self._storage.delete, memory_id)
        self._cache_remove(memory_id)
        self._stats["deletes"] += 1
        logger.info("delete id=%s (%.3fs)", memory_id, time.time() - start)
        return result

    def list_memories(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """List memories from storage."""
        return self._retry("list", self._storage.list_memories, filter_metadata)

    def bulk_save(self, items: List[Dict[str, Any]]) -> List[Memory]:
        """Save multiple memories. Returns list of successfully saved memories (graceful degradation)."""
        saved = []
        for item in items:
            try:
                mem = self.save(item["content"], item.get("metadata"), item.get("memory_id"))
                saved.append(mem)
            except (PersistenceError, MemoryStorageError) as e:
                logger.error("bulk_save skipped item: %s", e)
        logger.info("bulk_save completed: %d/%d succeeded", len(saved), len(items))
        return saved
