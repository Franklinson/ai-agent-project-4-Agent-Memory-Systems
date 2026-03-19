"""RAG integration for semantic memory — embedding, vector storage, similarity search, retrieval augmentation.

Uses ChromaDB as the vector database and supports pluggable embedding providers
(HashEmbedding for dev/testing, OpenAIEmbedding for production).
"""

import hashlib
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import chromadb

from fact_store import Fact, FactStore
from knowledge_graph import KnowledgeGraph, Node


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RAGError(Exception):
    """Base exception for RAG operations."""


class DocumentNotFoundError(RAGError):
    """Raised when a document is not found."""


# ---------------------------------------------------------------------------
# Embedding provider interface
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract interface for generating text embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of produced vectors."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for *text*."""

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Override for batch-optimised backends."""
        return [self.embed(t) for t in texts]


class HashEmbedding(EmbeddingProvider):
    """Deterministic hash-based embedding — no external dependencies.

    Produces a fixed-dimension vector by hashing character n-grams.
    Useful for testing and development; swap for OpenAIEmbedding
    or a sentence-transformers provider in production.
    """

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self._dimension
        text_lower = text.lower().strip()
        if not text_lower:
            return vec
        for n in range(1, 4):
            for i in range(len(text_lower) - n + 1):
                gram = text_lower[i:i + n]
                h = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
                idx = h % self._dimension
                sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using the OpenAI API.

    Requires a valid OPENAI_API_KEY environment variable or explicit api_key.

    Usage:
        embedder = OpenAIEmbedding(api_key="sk-...")
        vec = embedder.embed("hello world")
    """

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model
        self._dimensions = {"text-embedding-3-small": 1536,
                            "text-embedding-3-large": 3072,
                            "text-embedding-ada-002": 1536}
        self._dimension = self._dimensions.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Vector store (ChromaDB-backed)
# ---------------------------------------------------------------------------

@dataclass
class VectorDocument:
    """A document stored in the vector store."""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """A single similarity-search result."""
    document: VectorDocument
    score: float


class VectorStore:
    """ChromaDB-backed vector store with pluggable embedding provider.

    Uses ChromaDB for persistent/in-memory vector indexing and HNSW-based
    approximate nearest-neighbor search. Embeddings are generated by the
    provided EmbeddingProvider and passed to ChromaDB as pre-computed vectors.
    """

    def __init__(self, embedding_provider: EmbeddingProvider,
                 collection_name: Optional[str] = None,
                 chroma_client: Optional[chromadb.ClientAPI] = None):
        self._provider = embedding_provider
        self._client = chroma_client or chromadb.Client()
        name = collection_name or f"rag_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})
        # Local mirror for fast get/count and metadata we control
        self._documents: Dict[str, VectorDocument] = {}

    @property
    def count(self) -> int:
        return len(self._documents)

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None,
            doc_id: Optional[str] = None) -> VectorDocument:
        did = doc_id or str(uuid.uuid4())
        if did in self._documents:
            raise RAGError(f"Document '{did}' already exists")
        emb = self._provider.embed(text)
        meta = metadata or {}
        # ChromaDB metadata values must be str, int, float, or bool
        chroma_meta = {k: v for k, v in meta.items()
                       if isinstance(v, (str, int, float, bool))}
        self._collection.add(ids=[did], embeddings=[emb],
                             documents=[text],
                             metadatas=[chroma_meta] if chroma_meta else None)
        doc = VectorDocument(id=did, text=text, embedding=emb, metadata=meta)
        self._documents[did] = doc
        return doc

    def add_batch(self, items: List[Dict[str, Any]]) -> List[VectorDocument]:
        """Add multiple documents. Each item: {text, metadata?, doc_id?}."""
        texts = [it["text"] for it in items]
        embeddings = self._provider.embed_batch(texts)
        ids = []
        metas = []
        docs = []
        for it, emb in zip(items, embeddings):
            did = it.get("doc_id") or str(uuid.uuid4())
            if did in self._documents:
                raise RAGError(f"Document '{did}' already exists")
            meta = it.get("metadata", {})
            chroma_meta = {k: v for k, v in meta.items()
                           if isinstance(v, (str, int, float, bool))}
            ids.append(did)
            metas.append(chroma_meta or None)
            doc = VectorDocument(id=did, text=it["text"], embedding=emb,
                                 metadata=meta)
            self._documents[did] = doc
            docs.append(doc)
        has_meta = any(m is not None for m in metas)
        self._collection.add(ids=ids, embeddings=embeddings,
                             documents=texts,
                             metadatas=metas if has_meta else None)
        return docs

    def get(self, doc_id: str) -> VectorDocument:
        if doc_id not in self._documents:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")
        return self._documents[doc_id]

    def delete(self, doc_id: str) -> bool:
        self.get(doc_id)  # validate exists
        self._collection.delete(ids=[doc_id])
        del self._documents[doc_id]
        return True

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Return the *top_k* most similar documents to *query*."""
        if not self._documents:
            return []
        q_emb = self._provider.embed(query)
        return self._search_by_embedding(q_emb, top_k, min_score, filter_metadata)

    def search_by_vector(self, embedding: List[float], top_k: int = 5,
                         min_score: float = 0.0) -> List[SearchResult]:
        """Search using a pre-computed embedding vector."""
        if not self._documents:
            return []
        return self._search_by_embedding(embedding, top_k, min_score)

    def _search_by_embedding(self, embedding: List[float], top_k: int,
                             min_score: float,
                             filter_metadata: Optional[Dict[str, Any]] = None
                             ) -> List[SearchResult]:
        """Query ChromaDB and convert results to SearchResult list."""
        where = None
        if filter_metadata:
            conditions = [
                {k: {"$eq": v}} for k, v in filter_metadata.items()
                if isinstance(v, (str, int, float, bool))
            ]
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}

        n = min(top_k, len(self._documents))
        if n == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding], n_results=n, where=where,
            include=["distances"])

        scored: List[SearchResult] = []
        for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
            # ChromaDB cosine distance = 1 - similarity
            similarity = 1.0 - distance
            if similarity >= min_score and doc_id in self._documents:
                scored.append(SearchResult(
                    document=self._documents[doc_id], score=similarity))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored


# ---------------------------------------------------------------------------
# RAG retriever — ties vector store to semantic memory
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Knowledge retrieved and formatted for prompt augmentation."""
    query: str
    results: List[SearchResult]
    context: str
    facts: List[Fact] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)


class RAGRetriever:
    """Retrieval-augmented generation over FactStore / KnowledgeGraph."""

    def __init__(self, vector_store: VectorStore,
                 fact_store: Optional[FactStore] = None,
                 knowledge_graph: Optional[KnowledgeGraph] = None):
        self._vs = vector_store
        self._fs = fact_store
        self._kg = knowledge_graph
        self._doc_to_fact: Dict[str, str] = {}
        self._doc_to_node: Dict[str, str] = {}

    @property
    def vector_store(self) -> VectorStore:
        return self._vs

    # --- Indexing ---

    def index_fact(self, fact: Fact, doc_id: Optional[str] = None) -> VectorDocument:
        """Embed a fact and store it in the vector store."""
        text = f"{fact.subject} {fact.predicate} {fact.object}"
        did = doc_id or f"fact:{fact.id}"
        doc = self._vs.add(text, metadata={
            "source": "fact_store", "fact_id": fact.id,
            "fact_type": fact.fact_type.value,
            "confidence": fact.confidence,
        }, doc_id=did)
        self._doc_to_fact[did] = fact.id
        return doc

    def index_facts(self, facts: List[Fact]) -> List[VectorDocument]:
        return [self.index_fact(f) for f in facts]

    def index_node(self, node: Node, doc_id: Optional[str] = None) -> VectorDocument:
        """Embed a knowledge-graph node and store it."""
        text = f"{node.label} ({node.node_type.value})"
        if node.properties:
            text += " " + " ".join(f"{k}: {v}" for k, v in node.properties.items())
        did = doc_id or f"node:{node.id}"
        doc = self._vs.add(text, metadata={
            "source": "knowledge_graph", "node_id": node.id,
            "node_type": node.node_type.value,
        }, doc_id=did)
        self._doc_to_node[did] = node.id
        return doc

    def index_nodes(self, nodes: List[Node]) -> List[VectorDocument]:
        return [self.index_node(n) for n in nodes]

    def index_text(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                   doc_id: Optional[str] = None) -> VectorDocument:
        """Index arbitrary text."""
        return self._vs.add(text, metadata=metadata or {"source": "text"}, doc_id=doc_id)

    # --- Retrieval ---

    def retrieve(self, query: str, top_k: int = 5,
                 min_score: float = 0.0,
                 include_related: bool = False,
                 filter_metadata: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Search the vector store and optionally expand with related facts/nodes."""
        results = self._vs.search(query, top_k=top_k, min_score=min_score,
                                  filter_metadata=filter_metadata)
        facts: List[Fact] = []
        nodes: List[Node] = []
        seen_facts: Set[str] = set()
        seen_nodes: Set[str] = set()

        for sr in results:
            did = sr.document.id
            if did in self._doc_to_fact and self._fs:
                fid = self._doc_to_fact[did]
                if fid not in seen_facts:
                    try:
                        facts.append(self._fs.get(fid))
                        seen_facts.add(fid)
                    except Exception:
                        pass
                    if include_related:
                        for rf in self._fs.get_related(fid):
                            if rf.id not in seen_facts:
                                facts.append(rf)
                                seen_facts.add(rf.id)
            if did in self._doc_to_node and self._kg:
                nid = self._doc_to_node[did]
                if nid not in seen_nodes:
                    try:
                        nodes.append(self._kg.get_node(nid))
                        seen_nodes.add(nid)
                    except Exception:
                        pass
                    if include_related:
                        for nb in self._kg.neighbors(nid, direction="both"):
                            if nb.id not in seen_nodes:
                                nodes.append(nb)
                                seen_nodes.add(nb.id)

        context = self._format_context(results, facts, nodes)
        return RetrievalResult(query=query, results=results,
                               context=context, facts=facts, nodes=nodes)

    def augment_prompt(self, query: str, system_prompt: str = "",
                       top_k: int = 5, min_score: float = 0.0,
                       include_related: bool = False) -> str:
        """Build a RAG-augmented prompt string."""
        rr = self.retrieve(query, top_k=top_k, min_score=min_score,
                           include_related=include_related)
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if rr.context:
            parts.append(f"Relevant knowledge:\n{rr.context}")
        parts.append(f"Question: {query}")
        return "\n\n".join(parts)

    # --- Formatting ---

    @staticmethod
    def _format_context(results: List[SearchResult],
                        facts: List[Fact],
                        nodes: List[Node]) -> str:
        lines: List[str] = []
        if facts:
            lines.append("Facts:")
            for f in facts:
                lines.append(f"- {f.subject} {f.predicate} {f.object} "
                             f"[{f.fact_type.value}, confidence={f.confidence}]")
        if nodes:
            lines.append("Concepts:")
            for n in nodes:
                lines.append(f"- {n.label} ({n.node_type.value})")
        if not facts and not nodes and results:
            lines.append("Retrieved:")
            for sr in results:
                lines.append(f"- {sr.document.text} (score={sr.score:.3f})")
        return "\n".join(lines)
