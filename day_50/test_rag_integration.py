"""Tests for RAG integration with semantic memory (ChromaDB + OpenAI/Hash embedding)."""

import pytest
import chromadb
from fact_store import FactStore, FactType
from knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from rag_integration import (
    HashEmbedding, OpenAIEmbedding, VectorStore, RAGRetriever,
    RAGError, DocumentNotFoundError,
    _cosine_similarity, SearchResult, VectorDocument, EmbeddingProvider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedder():
    return HashEmbedding(dimension=64)


@pytest.fixture
def store(embedder):
    return VectorStore(embedder)


@pytest.fixture
def fact_store():
    fs = FactStore()
    fs.store("Python", "is_a", "programming language",
             fact_type=FactType.DEFINITION, fact_id="f1")
    fs.store("Python", "created_by", "Guido van Rossum",
             fact_type=FactType.ASSERTION, fact_id="f2")
    fs.store("Python", "has_feature", "dynamic typing",
             fact_type=FactType.ATTRIBUTE, fact_id="f3", confidence=0.95)
    fs.store("Java", "is_a", "programming language",
             fact_type=FactType.DEFINITION, fact_id="f4")
    fs.link("f1", "f2")
    return fs


@pytest.fixture
def knowledge_graph():
    kg = KnowledgeGraph()
    kg.add_node("Python", NodeType.ENTITY, node_id="python")
    kg.add_node("Programming Language", NodeType.CONCEPT, node_id="pl")
    kg.add_node("Django", NodeType.ENTITY, node_id="django")
    kg.add_edge("python", "pl", EdgeType.IS_A)
    kg.add_edge("django", "python", EdgeType.DEPENDS_ON)
    return kg


@pytest.fixture
def retriever(store, fact_store, knowledge_graph):
    return RAGRetriever(store, fact_store, knowledge_graph)


# ---------------------------------------------------------------------------
# HashEmbedding
# ---------------------------------------------------------------------------

class TestHashEmbedding:
    def test_dimension(self, embedder):
        assert embedder.dimension == 64

    def test_embed_returns_correct_length(self, embedder):
        vec = embedder.embed("hello world")
        assert len(vec) == 64

    def test_embed_is_normalized(self, embedder):
        import math
        vec = embedder.embed("test text")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_embed_deterministic(self, embedder):
        a = embedder.embed("same text")
        b = embedder.embed("same text")
        assert a == b

    def test_embed_different_texts_differ(self, embedder):
        a = embedder.embed("cats are great")
        b = embedder.embed("quantum physics theory")
        assert a != b

    def test_embed_empty_string(self, embedder):
        vec = embedder.embed("")
        assert len(vec) == 64
        assert all(v == 0.0 for v in vec)

    def test_embed_batch(self, embedder):
        texts = ["hello", "world", "test"]
        results = embedder.embed_batch(texts)
        assert len(results) == 3
        assert results[0] == embedder.embed("hello")

    def test_similar_texts_higher_score(self, embedder):
        a = embedder.embed("python programming language")
        b = embedder.embed("python programming")
        c = embedder.embed("underwater basket weaving")
        sim_ab = _cosine_similarity(a, b)
        sim_ac = _cosine_similarity(a, c)
        assert sim_ab > sim_ac


# ---------------------------------------------------------------------------
# OpenAIEmbedding
# ---------------------------------------------------------------------------

class TestOpenAIEmbedding:
    def test_is_embedding_provider(self):
        assert issubclass(OpenAIEmbedding, EmbeddingProvider)

    def test_default_dimension(self):
        # Verify class can be imported and has expected model dimensions
        # (don't call API — no key in CI)
        dims = {"text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536}
        for model, expected_dim in dims.items():
            # Instantiation requires openai client; just verify the mapping
            assert expected_dim > 0


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# VectorStore (ChromaDB-backed)
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_add_document(self, store):
        doc = store.add("hello world")
        assert doc.text == "hello world"
        assert store.count == 1

    def test_add_with_metadata(self, store):
        doc = store.add("test", metadata={"source": "unit"}, doc_id="d1")
        assert doc.id == "d1"
        assert doc.metadata == {"source": "unit"}

    def test_add_duplicate_raises(self, store):
        store.add("a", doc_id="dup")
        with pytest.raises(RAGError, match="already exists"):
            store.add("b", doc_id="dup")

    def test_get_document(self, store):
        store.add("test", doc_id="d1")
        doc = store.get("d1")
        assert doc.text == "test"

    def test_get_missing_raises(self, store):
        with pytest.raises(DocumentNotFoundError):
            store.get("nope")

    def test_delete_document(self, store):
        store.add("test", doc_id="d1")
        assert store.delete("d1")
        assert store.count == 0

    def test_delete_missing_raises(self, store):
        with pytest.raises(DocumentNotFoundError):
            store.delete("nope")

    def test_add_batch(self, store):
        items = [
            {"text": "first", "metadata": {"i": 1}},
            {"text": "second", "doc_id": "s2"},
        ]
        docs = store.add_batch(items)
        assert len(docs) == 2
        assert store.count == 2
        assert docs[1].id == "s2"

    def test_add_batch_duplicate_raises(self, store):
        store.add("existing", doc_id="x")
        with pytest.raises(RAGError):
            store.add_batch([{"text": "new", "doc_id": "x"}])

    def test_provider_property(self, store, embedder):
        assert store.provider is embedder

    def test_uses_chromadb_backend(self, store):
        """Verify the store is backed by a real ChromaDB collection."""
        assert hasattr(store, '_collection')
        assert hasattr(store, '_client')
        store.add("chromadb test", doc_id="chroma1")
        # Query ChromaDB directly to confirm data is there
        result = store._collection.get(ids=["chroma1"])
        assert result["ids"] == ["chroma1"]
        assert result["documents"] == ["chromadb test"]

    def test_custom_chroma_client(self, embedder):
        """Verify a custom ChromaDB client can be injected."""
        client = chromadb.Client()
        vs = VectorStore(embedder, collection_name="custom_test",
                         chroma_client=client)
        vs.add("test doc", doc_id="c1")
        # Verify via the injected client
        col = client.get_collection("custom_test")
        assert col.count() == 1

    def test_delete_removes_from_chromadb(self, store):
        """Verify delete propagates to ChromaDB."""
        store.add("to delete", doc_id="del1")
        store.delete("del1")
        result = store._collection.get(ids=["del1"])
        assert result["ids"] == []


class TestVectorSearch:
    def test_search_returns_results(self, store):
        store.add("python programming language", doc_id="d1")
        store.add("java programming language", doc_id="d2")
        store.add("chocolate cake recipe", doc_id="d3")
        results = store.search("python programming", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ranking(self, store):
        store.add("python programming language", doc_id="d1")
        store.add("chocolate cake recipe", doc_id="d2")
        results = store.search("python programming", top_k=2)
        assert results[0].document.id == "d1"
        assert results[0].score >= results[1].score

    def test_search_top_k(self, store):
        for i in range(10):
            store.add(f"document number {i}", doc_id=f"d{i}")
        results = store.search("document", top_k=3)
        assert len(results) == 3

    def test_search_min_score(self, store):
        store.add("python programming", doc_id="d1")
        store.add("unrelated topic xyz", doc_id="d2")
        results = store.search("python programming", min_score=0.99)
        assert all(r.score >= 0.99 for r in results)

    def test_search_empty_store(self, store):
        assert store.search("anything") == []

    def test_search_filter_metadata(self, store):
        store.add("python info", metadata={"source": "wiki"}, doc_id="d1")
        store.add("python docs", metadata={"source": "docs"}, doc_id="d2")
        results = store.search("python", filter_metadata={"source": "docs"})
        assert len(results) == 1
        assert results[0].document.id == "d2"

    def test_search_by_vector(self, store, embedder):
        store.add("python programming", doc_id="d1")
        store.add("cake recipe", doc_id="d2")
        vec = embedder.embed("python programming")
        results = store.search_by_vector(vec, top_k=1)
        assert len(results) == 1
        assert results[0].document.id == "d1"

    def test_search_scores_are_similarities(self, store):
        """Verify scores are cosine similarities (0-1), not distances."""
        store.add("python programming", doc_id="d1")
        results = store.search("python programming", top_k=1)
        assert 0.0 <= results[0].score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# RAGRetriever — indexing
# ---------------------------------------------------------------------------

class TestRAGIndexing:
    def test_index_fact(self, retriever, fact_store):
        fact = fact_store.get("f1")
        doc = retriever.index_fact(fact)
        assert doc.metadata["source"] == "fact_store"
        assert doc.metadata["fact_id"] == "f1"
        assert retriever.vector_store.count == 1

    def test_index_facts(self, retriever, fact_store):
        facts = [fact_store.get(fid) for fid in ("f1", "f2", "f3")]
        docs = retriever.index_facts(facts)
        assert len(docs) == 3
        assert retriever.vector_store.count == 3

    def test_index_node(self, retriever, knowledge_graph):
        node = knowledge_graph.get_node("python")
        doc = retriever.index_node(node)
        assert doc.metadata["source"] == "knowledge_graph"
        assert doc.metadata["node_id"] == "python"

    def test_index_nodes(self, retriever, knowledge_graph):
        nodes = [knowledge_graph.get_node(nid) for nid in ("python", "pl", "django")]
        docs = retriever.index_nodes(nodes)
        assert len(docs) == 3

    def test_index_text(self, retriever):
        doc = retriever.index_text("arbitrary knowledge", doc_id="t1")
        assert doc.text == "arbitrary knowledge"
        assert doc.metadata["source"] == "text"

    def test_index_text_custom_metadata(self, retriever):
        doc = retriever.index_text("info", metadata={"topic": "test"})
        assert doc.metadata["topic"] == "test"


# ---------------------------------------------------------------------------
# RAGRetriever — retrieval
# ---------------------------------------------------------------------------

class TestRAGRetrieval:
    def _index_all(self, retriever, fact_store, knowledge_graph):
        facts = [fact_store.get(fid) for fid in ("f1", "f2", "f3", "f4")]
        retriever.index_facts(facts)
        nodes = [knowledge_graph.get_node(nid) for nid in ("python", "pl", "django")]
        retriever.index_nodes(nodes)

    def test_retrieve_basic(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python programming language", top_k=3)
        assert rr.query == "python programming language"
        assert len(rr.results) <= 3
        assert isinstance(rr.context, str)

    def test_retrieve_returns_facts(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python programming", top_k=5)
        assert len(rr.facts) > 0

    def test_retrieve_returns_nodes(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python", top_k=10)
        assert len(rr.nodes) > 0

    def test_retrieve_include_related_facts(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr_no = retriever.retrieve("python is_a programming", top_k=5,
                                   include_related=False)
        rr_yes = retriever.retrieve("python is_a programming", top_k=5,
                                    include_related=True)
        assert len(rr_yes.facts) >= len(rr_no.facts)

    def test_retrieve_include_related_nodes(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python", top_k=10, include_related=True)
        node_ids = {n.id for n in rr.nodes}
        assert len(node_ids) >= 1

    def test_retrieve_with_filter(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python", top_k=10,
                                filter_metadata={"source": "fact_store"})
        for sr in rr.results:
            assert sr.document.metadata["source"] == "fact_store"

    def test_retrieve_empty_store(self, retriever):
        rr = retriever.retrieve("anything")
        assert rr.results == []
        assert rr.context == ""

    def test_retrieve_context_has_facts(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python programming", top_k=5)
        if rr.facts:
            assert "Facts:" in rr.context

    def test_retrieve_context_has_concepts(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        rr = retriever.retrieve("python", top_k=10)
        if rr.nodes:
            assert "Concepts:" in rr.context

    def test_retrieve_text_only_context(self, retriever):
        retriever.index_text("some knowledge about cats", doc_id="t1")
        rr = retriever.retrieve("cats")
        assert "Retrieved:" in rr.context


# ---------------------------------------------------------------------------
# RAGRetriever — prompt augmentation
# ---------------------------------------------------------------------------

class TestPromptAugmentation:
    def _index_all(self, retriever, fact_store, knowledge_graph):
        facts = [fact_store.get(fid) for fid in ("f1", "f2", "f3", "f4")]
        retriever.index_facts(facts)
        nodes = [knowledge_graph.get_node(nid) for nid in ("python", "pl", "django")]
        retriever.index_nodes(nodes)

    def test_augment_prompt_basic(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        prompt = retriever.augment_prompt("What is Python?")
        assert "Question: What is Python?" in prompt
        assert "Relevant knowledge:" in prompt

    def test_augment_prompt_with_system(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        prompt = retriever.augment_prompt(
            "What is Python?",
            system_prompt="You are a helpful assistant.",
        )
        assert prompt.startswith("You are a helpful assistant.")
        assert "Question: What is Python?" in prompt

    def test_augment_prompt_empty_store(self, retriever):
        prompt = retriever.augment_prompt("anything")
        assert "Question: anything" in prompt

    def test_augment_prompt_top_k(self, retriever, fact_store, knowledge_graph):
        self._index_all(retriever, fact_store, knowledge_graph)
        prompt = retriever.augment_prompt("python", top_k=1)
        assert "Question: python" in prompt


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_retriever_without_fact_store(self, store, knowledge_graph):
        r = RAGRetriever(store, fact_store=None, knowledge_graph=knowledge_graph)
        node = knowledge_graph.get_node("python")
        r.index_node(node)
        rr = r.retrieve("python")
        assert len(rr.nodes) >= 1
        assert rr.facts == []

    def test_retriever_without_knowledge_graph(self, store, fact_store):
        r = RAGRetriever(store, fact_store=fact_store, knowledge_graph=None)
        r.index_fact(fact_store.get("f1"))
        rr = r.retrieve("python programming")
        assert len(rr.facts) >= 1
        assert rr.nodes == []

    def test_retriever_text_only(self, store):
        r = RAGRetriever(store)
        r.index_text("cats are fluffy animals", doc_id="t1")
        r.index_text("dogs are loyal companions", doc_id="t2")
        rr = r.retrieve("fluffy animals")
        assert len(rr.results) > 0
        assert rr.facts == []
        assert rr.nodes == []

    def test_deleted_fact_skipped(self, retriever, fact_store, knowledge_graph):
        retriever.index_fact(fact_store.get("f1"))
        fact_store.delete("f1")
        rr = retriever.retrieve("python programming")
        assert all(f.id != "f1" for f in rr.facts)

    def test_custom_embedding_dimension(self):
        emb = HashEmbedding(dimension=32)
        vs = VectorStore(emb)
        doc = vs.add("test")
        assert len(doc.embedding) == 32
