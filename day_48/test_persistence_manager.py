"""Tests for persistence manager."""

import logging
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from memory_storage import (
    InMemoryStorage, Memory, MemoryStorageInterface,
    MemoryStorageError, MemoryNotFoundError,
)
from persistence_manager import PersistenceManager, PersistenceError


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def pm(storage):
    return PersistenceManager(storage=storage, max_retries=3, retry_delay=0.01)


class TestSave:
    def test_save_and_confirm(self, pm):
        mem = pm.save("hello", metadata={"type": "test"})
        assert mem.content == "hello"
        assert mem.metadata == {"type": "test"}
        assert pm.stats["saves"] == 1

    def test_save_with_custom_id(self, pm):
        mem = pm.save("data", memory_id="custom-1")
        assert mem.id == "custom-1"

    def test_save_persists_to_storage(self, pm, storage):
        mem = pm.save("data", memory_id="p1")
        assert storage.retrieve("p1").content == "data"


class TestRetrieve:
    def test_retrieve_existing(self, pm):
        pm.save("data", memory_id="r1")
        mem = pm.retrieve("r1")
        assert mem.content == "data"
        assert pm.stats["retrievals"] == 1

    def test_retrieve_not_found(self, pm):
        with pytest.raises(MemoryNotFoundError):
            pm.retrieve("nonexistent")

    def test_retrieve_uses_cache(self, pm, storage):
        pm.save("data", memory_id="c1")
        pm.retrieve("c1")  # populates cache
        storage.delete("c1")  # remove from backend
        mem = pm.retrieve("c1")  # should come from cache
        assert mem.content == "data"


class TestUpdate:
    def test_update_content(self, pm):
        pm.save("old", memory_id="u1")
        updated = pm.update("u1", content="new")
        assert updated.content == "new"
        assert pm.stats["updates"] == 1

    def test_update_refreshes_cache(self, pm, storage):
        pm.save("old", memory_id="u2")
        pm.retrieve("u2")  # cache it
        pm.update("u2", content="new")
        storage.delete("u2")  # remove from backend
        assert pm.retrieve("u2").content == "new"  # cache has updated value

    def test_update_not_found(self, pm):
        with pytest.raises(MemoryNotFoundError):
            pm.update("nonexistent", content="x")


class TestDelete:
    def test_delete_existing(self, pm):
        pm.save("data", memory_id="d1")
        assert pm.delete("d1") is True
        assert pm.stats["deletes"] == 1

    def test_delete_clears_cache(self, pm):
        pm.save("data", memory_id="d2")
        pm.retrieve("d2")  # cache it
        pm.delete("d2")
        with pytest.raises(MemoryNotFoundError):
            pm.retrieve("d2")

    def test_delete_not_found(self, pm):
        with pytest.raises(MemoryNotFoundError):
            pm.delete("nonexistent")


class TestListMemories:
    def test_list_all(self, pm):
        pm.save("a", memory_id="l1")
        pm.save("b", memory_id="l2")
        assert len(pm.list_memories()) == 2

    def test_list_with_filter(self, pm):
        pm.save("a", metadata={"type": "note"}, memory_id="f1")
        pm.save("b", metadata={"type": "task"}, memory_id="f2")
        results = pm.list_memories(filter_metadata={"type": "note"})
        assert len(results) == 1


class TestRetryLogic:
    def test_retries_on_transient_error(self):
        mock_storage = MagicMock(spec=MemoryStorageInterface)
        mem = Memory(id="x", content="data")
        mock_storage.save.side_effect = [MemoryStorageError("transient"), mem]
        mock_storage.retrieve.return_value = mem

        pm = PersistenceManager(storage=mock_storage, max_retries=3, retry_delay=0.01)
        result = pm.save("data")
        assert result.id == "x"
        assert mock_storage.save.call_count == 2
        assert pm.stats["retries"] == 1

    def test_fails_after_max_retries(self):
        mock_storage = MagicMock(spec=MemoryStorageInterface)
        mock_storage.save.side_effect = MemoryStorageError("persistent failure")

        pm = PersistenceManager(storage=mock_storage, max_retries=2, retry_delay=0.01)
        with pytest.raises(PersistenceError, match="failed after 2 retries"):
            pm.save("data")
        assert pm.stats["failures"] == 1

    def test_no_retry_on_not_found(self):
        mock_storage = MagicMock(spec=MemoryStorageInterface)
        mock_storage.retrieve.side_effect = MemoryNotFoundError("gone")

        pm = PersistenceManager(storage=mock_storage, max_retries=3, retry_delay=0.01)
        with pytest.raises(MemoryNotFoundError):
            pm.retrieve("gone")
        assert mock_storage.retrieve.call_count == 1  # no retries


class TestBulkSave:
    def test_bulk_save_all_succeed(self, pm):
        items = [
            {"content": "a", "memory_id": "b1"},
            {"content": "b", "memory_id": "b2"},
        ]
        results = pm.bulk_save(items)
        assert len(results) == 2

    def test_bulk_save_partial_failure(self):
        mock_storage = MagicMock(spec=MemoryStorageInterface)
        mem = Memory(id="ok", content="good")
        mock_storage.save.side_effect = [mem, MemoryStorageError("fail"), mem]
        mock_storage.retrieve.return_value = mem

        pm = PersistenceManager(storage=mock_storage, max_retries=1, retry_delay=0.01)
        items = [
            {"content": "good"},
            {"content": "bad"},
            {"content": "good"},
        ]
        results = pm.bulk_save(items)
        assert len(results) == 2  # 1 failed, 2 succeeded


class TestCacheEviction:
    def test_cache_evicts_oldest(self):
        pm = PersistenceManager(storage=InMemoryStorage(), max_retries=1, retry_delay=0, cache_size=2)
        pm.save("a", memory_id="e1")
        pm.save("b", memory_id="e2")
        pm.save("c", memory_id="e3")  # should evict e1 from cache
        assert "e1" not in pm._cache
        assert "e3" in pm._cache


class TestLogging:
    def test_save_logs_confirmation(self, storage, caplog):
        pm = PersistenceManager(storage=storage, max_retries=1, retry_delay=0)
        with caplog.at_level(logging.INFO, logger="persistence_manager"):
            pm.save("data", memory_id="log1")
        assert any("save confirmed id=log1" in r.message for r in caplog.records)

    def test_retry_logs_warning(self, caplog):
        mock_storage = MagicMock(spec=MemoryStorageInterface)
        mem = Memory(id="x", content="data")
        mock_storage.save.side_effect = [MemoryStorageError("fail"), mem]
        mock_storage.retrieve.return_value = mem

        pm = PersistenceManager(storage=mock_storage, max_retries=2, retry_delay=0.01)
        with caplog.at_level(logging.WARNING, logger="persistence_manager"):
            pm.save("data")
        assert any("attempt 1/2 failed" in r.message for r in caplog.records)


class TestDefaultBackend:
    def test_defaults_to_in_memory(self):
        pm = PersistenceManager()
        mem = pm.save("test")
        assert pm.retrieve(mem.id).content == "test"


class TestStats:
    def test_stats_tracking(self, pm):
        pm.save("a", memory_id="s1")
        pm.retrieve("s1")
        pm.update("s1", content="b")
        pm.delete("s1")
        assert pm.stats == {"saves": 1, "retrievals": 1, "updates": 1, "deletes": 1, "retries": 0, "failures": 0}
