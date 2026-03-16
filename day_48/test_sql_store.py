"""Tests for SQL storage backend."""

import os
import pytest
from memory_storage import Memory, MemoryNotFoundError, MemoryConflictError, MemoryStorageError
from sql_store import SQLStorage


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.db"
    store = SQLStorage(db_url=f"sqlite:///{db_path}")
    yield store
    store.close()


class TestConnection:
    def test_creates_database(self, tmp_path):
        db_path = tmp_path / "new.db"
        store = SQLStorage(db_url=f"sqlite:///{db_path}")
        assert db_path.exists()
        store.close()

    def test_in_memory_database(self):
        store = SQLStorage(db_url="sqlite:///:memory:")
        mem = store.save("test")
        assert store.retrieve(mem.id).content == "test"
        store.close()


class TestSave:
    def test_save_returns_memory(self, storage):
        mem = storage.save("hello", metadata={"type": "greeting"})
        assert isinstance(mem, Memory)
        assert mem.content == "hello"
        assert mem.metadata == {"type": "greeting"}

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

    def test_save_duplicate_id_raises_conflict(self, storage):
        storage.save("first", memory_id="dup")
        with pytest.raises(MemoryConflictError):
            storage.save("second", memory_id="dup")


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
        assert storage.retrieve("u1").content == "new"

    def test_update_metadata(self, storage):
        storage.save("data", metadata={"a": 1}, memory_id="u2")
        updated = storage.update("u2", metadata={"b": 2})
        assert updated.metadata == {"a": 1, "b": 2}

    def test_update_changes_updated_at(self, storage):
        mem = storage.save("data", memory_id="u3")
        updated = storage.update("u3", content="new")
        assert updated.updated_at >= mem.updated_at

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
        assert storage.list_memories(filter_metadata={"type": "missing"}) == []


class TestQuery:
    def test_query_by_content(self, storage):
        storage.save("hello world", memory_id="q1")
        storage.save("goodbye", memory_id="q2")
        results = storage.query(content="hello")
        assert len(results) == 1
        assert results[0].id == "q1"

    def test_query_by_id(self, storage):
        storage.save("data", memory_id="q3")
        results = storage.query(id="q3")
        assert len(results) == 1

    def test_query_by_metadata(self, storage):
        storage.save("a", metadata={"priority": "high"}, memory_id="q4")
        storage.save("b", metadata={"priority": "low"}, memory_id="q5")
        results = storage.query(priority="high")
        assert len(results) == 1
        assert results[0].id == "q4"

    def test_query_combined(self, storage):
        storage.save("important note", metadata={"type": "note"}, memory_id="q6")
        storage.save("important task", metadata={"type": "task"}, memory_id="q7")
        results = storage.query(content="important", type="note")
        assert len(results) == 1
        assert results[0].id == "q6"

    def test_query_no_results(self, storage):
        storage.save("data")
        assert storage.query(content="nonexistent") == []


class TestPersistence:
    def test_data_persists_across_instances(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'persist.db'}"
        store1 = SQLStorage(db_url=db_url)
        store1.save("persistent data", memory_id="p1", metadata={"key": "val"})
        store1.close()

        store2 = SQLStorage(db_url=db_url)
        mem = store2.retrieve("p1")
        assert mem.content == "persistent data"
        assert mem.metadata == {"key": "val"}
        store2.close()
