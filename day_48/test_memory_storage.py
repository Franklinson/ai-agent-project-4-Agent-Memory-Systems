"""Tests for memory storage interface."""

import pytest
from memory_storage import (
    InMemoryStorage, Memory, MemoryNotFoundError,
    MemoryConflictError, MemoryStorageError
)


@pytest.fixture
def storage():
    return InMemoryStorage()


class TestSave:
    def test_save_returns_memory(self, storage):
        mem = storage.save("hello", {"type": "greeting"})
        assert isinstance(mem, Memory)
        assert mem.content == "hello"
        assert mem.metadata == {"type": "greeting"}
        assert mem.id is not None

    def test_save_with_custom_id(self, storage):
        mem = storage.save("data", memory_id="custom-1")
        assert mem.id == "custom-1"

    def test_save_without_metadata(self, storage):
        mem = storage.save("data")
        assert mem.metadata == {}

    def test_save_sets_timestamps(self, storage):
        mem = storage.save("data")
        assert mem.created_at is not None
        assert mem.updated_at is not None


class TestRetrieve:
    def test_retrieve_existing(self, storage):
        saved = storage.save("data", memory_id="r1")
        retrieved = storage.retrieve("r1")
        assert retrieved.content == "data"
        assert retrieved.id == saved.id

    def test_retrieve_not_found(self, storage):
        with pytest.raises(MemoryNotFoundError):
            storage.retrieve("nonexistent")


class TestUpdate:
    def test_update_content(self, storage):
        storage.save("old", memory_id="u1")
        updated = storage.update("u1", content="new")
        assert updated.content == "new"

    def test_update_metadata(self, storage):
        storage.save("data", metadata={"a": 1}, memory_id="u2")
        updated = storage.update("u2", metadata={"b": 2})
        assert updated.metadata == {"a": 1, "b": 2}

    def test_update_changes_updated_at(self, storage):
        mem = storage.save("data", memory_id="u3")
        original_updated = mem.updated_at
        updated = storage.update("u3", content="new")
        assert updated.updated_at >= original_updated

    def test_update_not_found(self, storage):
        with pytest.raises(MemoryNotFoundError):
            storage.update("nonexistent", content="x")


class TestDelete:
    def test_delete_existing(self, storage):
        storage.save("data", memory_id="d1")
        assert storage.delete("d1") is True
        with pytest.raises(MemoryNotFoundError):
            storage.retrieve("d1")

    def test_delete_not_found(self, storage):
        with pytest.raises(MemoryNotFoundError):
            storage.delete("nonexistent")


class TestListMemories:
    def test_list_all(self, storage):
        storage.save("a", memory_id="l1")
        storage.save("b", memory_id="l2")
        assert len(storage.list_memories()) == 2

    def test_list_empty(self, storage):
        assert storage.list_memories() == []

    def test_list_with_filter(self, storage):
        storage.save("a", metadata={"type": "note"}, memory_id="f1")
        storage.save("b", metadata={"type": "task"}, memory_id="f2")
        storage.save("c", metadata={"type": "note"}, memory_id="f3")
        results = storage.list_memories(filter_metadata={"type": "note"})
        assert len(results) == 2
        assert all(m.metadata["type"] == "note" for m in results)

    def test_list_filter_no_match(self, storage):
        storage.save("a", metadata={"type": "note"})
        results = storage.list_memories(filter_metadata={"type": "missing"})
        assert results == []
