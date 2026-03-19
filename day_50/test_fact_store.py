"""Tests for the semantic memory fact store."""

import pytest
from datetime import datetime, timedelta
from fact_store import (
    FactStore, FactType, Fact,
    FactStoreError, FactNotFoundError,
)


@pytest.fixture
def store():
    return FactStore()


@pytest.fixture
def populated_store(store):
    store.store("Python", "is_a", "programming language", fact_type=FactType.DEFINITION, fact_id="f1")
    store.store("Python", "created_by", "Guido van Rossum", fact_type=FactType.ASSERTION, fact_id="f2")
    store.store("Python", "has_feature", "dynamic typing", fact_type=FactType.ATTRIBUTE, fact_id="f3")
    store.store("Java", "is_a", "programming language", fact_type=FactType.DEFINITION, fact_id="f4")
    store.store("Python", "related_to", "Java", fact_type=FactType.RELATION, fact_id="f5", confidence=0.7)
    return store


# --- Storage ---

class TestStore:
    def test_store_basic(self, store):
        fact = store.store("Earth", "is_a", "planet")
        assert fact.subject == "Earth"
        assert fact.predicate == "is_a"
        assert fact.object == "planet"
        assert fact.fact_type == FactType.ASSERTION
        assert fact.confidence == 1.0
        assert store.count == 1

    def test_store_with_all_options(self, store):
        fact = store.store(
            "cat", "has", "fur",
            fact_type=FactType.ATTRIBUTE,
            properties={"source": "observation"},
            confidence=0.95,
            fact_id="custom-id",
        )
        assert fact.id == "custom-id"
        assert fact.fact_type == FactType.ATTRIBUTE
        assert fact.properties == {"source": "observation"}
        assert fact.confidence == 0.95

    def test_store_with_related_facts(self, store):
        f1 = store.store("A", "is", "B", fact_id="r1")
        f2 = store.store("B", "is", "C", fact_id="r2", related_fact_ids={"r1"})
        related = store.get_related("r2")
        assert len(related) == 1
        assert related[0].id == "r1"

    def test_store_duplicate_id_raises(self, store):
        store.store("A", "is", "B", fact_id="dup")
        with pytest.raises(FactStoreError, match="already exists"):
            store.store("C", "is", "D", fact_id="dup")

    def test_store_all_fact_types(self, store):
        for ft in FactType:
            store.store("s", "p", "o", fact_type=ft)
        assert store.count == len(FactType)


class TestGet:
    def test_get_existing(self, populated_store):
        fact = populated_store.get("f1")
        assert fact.subject == "Python"

    def test_get_missing_raises(self, store):
        with pytest.raises(FactNotFoundError):
            store.get("nonexistent")


class TestUpdate:
    def test_update_subject(self, populated_store):
        fact = populated_store.update("f1", subject="Python3")
        assert fact.subject == "Python3"
        assert len(populated_store.by_subject("Python3")) == 1
        assert "f1" not in populated_store._by_subject.get("Python", set())

    def test_update_properties_merge(self, populated_store):
        populated_store.update("f1", properties={"version": "3.12"})
        populated_store.update("f1", properties={"stable": True})
        fact = populated_store.get("f1")
        assert fact.properties == {"version": "3.12", "stable": True}

    def test_update_confidence(self, populated_store):
        populated_store.update("f5", confidence=0.9)
        assert populated_store.get("f5").confidence == 0.9

    def test_update_missing_raises(self, store):
        with pytest.raises(FactNotFoundError):
            store.update("nope", subject="X")


class TestDelete:
    def test_delete_existing(self, populated_store):
        assert populated_store.delete("f1")
        assert populated_store.count == 4
        with pytest.raises(FactNotFoundError):
            populated_store.get("f1")

    def test_delete_cleans_indexes(self, store):
        store.store("A", "is", "B", fact_id="x")
        store.delete("x")
        assert store.by_subject("A") == []
        assert store.by_predicate("is") == []
        assert store.by_object("B") == []

    def test_delete_cleans_relationships(self, store):
        store.store("A", "is", "B", fact_id="a")
        store.store("C", "is", "D", fact_id="b")
        store.link("a", "b")
        store.delete("a")
        assert store.get_related("b") == []

    def test_delete_missing_raises(self, store):
        with pytest.raises(FactNotFoundError):
            store.delete("nope")


# --- Retrieval ---

class TestRetrieval:
    def test_by_subject(self, populated_store):
        facts = populated_store.by_subject("Python")
        assert len(facts) == 4  # f1, f2, f3, f5

    def test_by_predicate(self, populated_store):
        facts = populated_store.by_predicate("is_a")
        assert len(facts) == 2
        subjects = {f.subject for f in facts}
        assert subjects == {"Python", "Java"}

    def test_by_object(self, populated_store):
        facts = populated_store.by_object("programming language")
        assert len(facts) == 2

    def test_by_type(self, populated_store):
        facts = populated_store.by_type(FactType.DEFINITION)
        assert len(facts) == 2

    def test_by_subject_empty(self, store):
        assert store.by_subject("nothing") == []


class TestQuery:
    def test_query_single_field(self, populated_store):
        results = populated_store.query(subject="Python")
        assert len(results) == 4

    def test_query_multiple_fields(self, populated_store):
        results = populated_store.query(subject="Python", predicate="is_a")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_query_with_type(self, populated_store):
        results = populated_store.query(fact_type=FactType.ATTRIBUTE)
        assert len(results) == 1
        assert results[0].predicate == "has_feature"

    def test_query_with_min_confidence(self, populated_store):
        results = populated_store.query(subject="Python", min_confidence=0.8)
        assert all(f.confidence >= 0.8 for f in results)
        assert "f5" not in {f.id for f in results}

    def test_query_no_filters_returns_all(self, populated_store):
        results = populated_store.query()
        assert len(results) == 5

    def test_query_no_match(self, populated_store):
        results = populated_store.query(subject="Rust")
        assert results == []

    def test_query_all_filters(self, populated_store):
        results = populated_store.query(
            subject="Python", predicate="is_a", object_="programming language",
            fact_type=FactType.DEFINITION, min_confidence=0.5,
        )
        assert len(results) == 1
        assert results[0].id == "f1"


# --- Relationships ---

class TestRelationships:
    def test_link_and_get_related(self, populated_store):
        populated_store.link("f1", "f4")
        related = populated_store.get_related("f1")
        assert any(f.id == "f4" for f in related)
        # Bidirectional
        related_back = populated_store.get_related("f4")
        assert any(f.id == "f1" for f in related_back)

    def test_unlink(self, populated_store):
        populated_store.link("f1", "f2")
        populated_store.unlink("f1", "f2")
        assert all(f.id != "f2" for f in populated_store.get_related("f1"))

    def test_link_invalid_fact_raises(self, store):
        store.store("A", "is", "B", fact_id="a")
        with pytest.raises(FactNotFoundError):
            store.link("a", "nonexistent")

    def test_get_related_empty(self, populated_store):
        assert populated_store.get_related("f4") == []

    def test_traverse_depth_1(self, store):
        store.store("A", "r", "1", fact_id="a")
        store.store("B", "r", "2", fact_id="b")
        store.store("C", "r", "3", fact_id="c")
        store.link("a", "b")
        store.link("b", "c")

        result = store.traverse("a", max_depth=1)
        assert 1 in result
        assert len(result[1]) == 1
        assert result[1][0].id == "b"
        assert 2 not in result

    def test_traverse_depth_2(self, store):
        store.store("A", "r", "1", fact_id="a")
        store.store("B", "r", "2", fact_id="b")
        store.store("C", "r", "3", fact_id="c")
        store.link("a", "b")
        store.link("b", "c")

        result = store.traverse("a", max_depth=2)
        assert 1 in result
        assert 2 in result
        assert result[1][0].id == "b"
        assert result[2][0].id == "c"

    def test_traverse_no_relations(self, store):
        store.store("A", "r", "1", fact_id="a")
        result = store.traverse("a", max_depth=3)
        assert result == {}

    def test_traverse_cycle(self, store):
        store.store("A", "r", "1", fact_id="a")
        store.store("B", "r", "2", fact_id="b")
        store.link("a", "b")
        # BFS won't revisit nodes
        result = store.traverse("a", max_depth=5)
        assert len(result) == 1
        assert len(result[1]) == 1
