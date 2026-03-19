# Day 50: Semantic Memory, Knowledge Graph & RAG

Semantic memory layer for AI agents — store knowledge as facts and graphs, then retrieve it via similarity search for prompt augmentation.

## Modules

| Module | Purpose |
|---|---|
| `fact_store.py` | Subject-predicate-object triple store with indexed retrieval and relationship traversal |
| `knowledge_graph.py` | Typed nodes and directed edges with BFS traversal, path finding, and pattern matching |
| `rag_integration.py` | Pluggable embedding, in-memory vector store, similarity search, and RAG prompt building |

## Quick Start

### Fact Store

```python
from fact_store import FactStore, FactType

store = FactStore()

# Store facts as triples
f1 = store.store("Python", "is_a", "programming language", fact_type=FactType.DEFINITION)
f2 = store.store("Python", "created_by", "Guido van Rossum")
f3 = store.store("Python", "has_feature", "dynamic typing",
                 fact_type=FactType.ATTRIBUTE, confidence=0.95)

# Retrieve
store.by_subject("Python")                          # all Python facts
store.by_predicate("is_a")                          # all "is_a" facts
store.by_type(FactType.DEFINITION)                   # all definitions
store.query(subject="Python", predicate="is_a")      # AND query
store.query(fact_type=FactType.ATTRIBUTE, min_confidence=0.9)

# Relationships
store.link(f1.id, f2.id)                             # bidirectional link
store.get_related(f1.id)                             # linked facts
store.traverse(f1.id, max_depth=2)                   # BFS traversal

# Update and delete
store.update(f1.id, properties={"version": "3.12"})
store.delete(f3.id)                                  # cleans indexes & links
```

### Knowledge Graph

```python
from knowledge_graph import KnowledgeGraph, NodeType, EdgeType

graph = KnowledgeGraph()

# Nodes
python = graph.add_node("Python", NodeType.ENTITY)
pl = graph.add_node("Programming Language", NodeType.CONCEPT)
django = graph.add_node("Django", NodeType.ENTITY)

# Directed edges
graph.add_edge(python.id, pl.id, EdgeType.IS_A)
graph.add_edge(django.id, python.id, EdgeType.DEPENDS_ON)

# Queries
graph.nodes_by_type(NodeType.ENTITY)
graph.edges_from(python.id)
graph.find_nodes(node_type=NodeType.CONCEPT)
graph.find_edges(edge_type=EdgeType.IS_A)

# Traversal
graph.neighbors(python.id, direction="both")
graph.bfs(django.id, max_depth=2, direction="out")
graph.find_path(django.id, pl.id)

# Pattern matching
graph.match_pattern(source_type=NodeType.ENTITY,
                    edge_type=EdgeType.IS_A,
                    target_type=NodeType.CONCEPT)

# Subgraph
nodes, edges = graph.subgraph({python.id, pl.id, django.id})
```

### RAG Integration

```python
from fact_store import FactStore, FactType
from knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from rag_integration import HashEmbedding, VectorStore, RAGRetriever

# Set up
fact_store = FactStore()
kg = KnowledgeGraph()
embedder = HashEmbedding(dimension=128)
vs = VectorStore(embedder)
rag = RAGRetriever(vs, fact_store=fact_store, knowledge_graph=kg)

# Populate
f1 = fact_store.store("Python", "is_a", "programming language",
                      fact_type=FactType.DEFINITION)
py = kg.add_node("Python", NodeType.ENTITY)

# Index into vector store
rag.index_fact(f1)
rag.index_node(py)
rag.index_text("Python was created by Guido van Rossum")

# Retrieve
result = rag.retrieve("What is Python?", top_k=5, include_related=True)
print(result.context)   # formatted facts & concepts
print(result.facts)     # resolved Fact objects
print(result.nodes)     # resolved Node objects

# Build RAG-augmented prompt
prompt = rag.augment_prompt(
    "What is Python?",
    system_prompt="You are a helpful assistant.",
    top_k=5,
)
```

## API Reference

### FactStore

| Method | Description |
|---|---|
| `store(subject, predicate, object_, ...)` | Store a fact triple |
| `get(fact_id)` | Retrieve a fact by ID |
| `update(fact_id, ...)` | Update fact fields |
| `delete(fact_id)` | Delete fact, clean indexes and links |
| `by_subject(subject)` | Facts with matching subject |
| `by_predicate(predicate)` | Facts with matching predicate |
| `by_object(object_)` | Facts with matching object |
| `by_type(fact_type)` | Facts with matching type |
| `query(subject?, predicate?, object_?, fact_type?, min_confidence?)` | Multi-field AND query |
| `link(id_a, id_b)` | Create bidirectional relationship |
| `unlink(id_a, id_b)` | Remove relationship |
| `get_related(fact_id)` | Get directly linked facts |
| `traverse(fact_id, max_depth)` | BFS traversal of linked facts |

**Fact types:** ASSERTION, DEFINITION, RELATION, ATTRIBUTE, RULE

### KnowledgeGraph

| Method | Description |
|---|---|
| `add_node(label, node_type, ...)` | Add a node |
| `get_node(node_id)` | Get node by ID |
| `update_node(node_id, ...)` | Update label/properties |
| `remove_node(node_id)` | Remove node and all connected edges |
| `add_edge(source_id, target_id, edge_type, ...)` | Add a directed edge |
| `get_edge(edge_id)` | Get edge by ID |
| `update_edge(edge_id, ...)` | Update label/weight/properties |
| `remove_edge(edge_id)` | Remove an edge |
| `nodes_by_type(node_type)` | All nodes of a type |
| `nodes_by_label(label)` | All nodes with a label |
| `find_nodes(node_type?, label?, filter_fn?)` | Multi-field AND query |
| `edges_by_type(edge_type)` | All edges of a type |
| `edges_from(node_id)` | Outgoing edges |
| `edges_to(node_id)` | Incoming edges |
| `find_edges(edge_type?, source_id?, target_id?, min_weight?)` | Multi-field AND query |
| `neighbors(node_id, direction, edge_type?)` | Connected nodes |
| `bfs(start_id, max_depth, edge_type?, direction)` | BFS traversal by depth |
| `find_path(start_id, end_id, edge_type?, max_depth)` | Shortest path (BFS) |
| `match_pattern(source_type?, edge_type?, target_type?)` | Find matching triples |
| `subgraph(node_ids)` | Extract induced subgraph |

**Node types:** ENTITY, CONCEPT, EVENT, ATTRIBUTE, CATEGORY
**Edge types:** IS_A, HAS, PART_OF, RELATED_TO, CAUSES, DEPENDS_ON, CUSTOM

### RAG Integration

| Method | Description |
|---|---|
| `VectorStore.add(text, metadata?, doc_id?)` | Embed and store a document |
| `VectorStore.add_batch(items)` | Batch add documents |
| `VectorStore.get(doc_id)` | Get document by ID |
| `VectorStore.delete(doc_id)` | Delete a document |
| `VectorStore.search(query, top_k, min_score?, filter_metadata?)` | Similarity search |
| `VectorStore.search_by_vector(embedding, top_k, min_score?)` | Search with pre-computed vector |
| `RAGRetriever.index_fact(fact)` | Index a FactStore fact |
| `RAGRetriever.index_facts(facts)` | Batch index facts |
| `RAGRetriever.index_node(node)` | Index a KnowledgeGraph node |
| `RAGRetriever.index_nodes(nodes)` | Batch index nodes |
| `RAGRetriever.index_text(text, metadata?)` | Index arbitrary text |
| `RAGRetriever.retrieve(query, top_k, min_score?, include_related?, filter_metadata?)` | Search and resolve facts/nodes |
| `RAGRetriever.augment_prompt(query, system_prompt?, top_k, ...)` | Build RAG-augmented prompt |

**Embedding providers:**
- `HashEmbedding(dimension)` — deterministic hash-based embedding, zero dependencies, for dev/testing
- `OpenAIEmbedding(model, api_key)` — production embeddings via OpenAI API (text-embedding-3-small, etc.)
- Implement `EmbeddingProvider` ABC to plug in sentence-transformers, Cohere, etc.

**Vector store:**
- Backed by ChromaDB with HNSW cosine similarity index
- Accepts a custom `chromadb.Client` or creates an in-memory client by default
- Pre-computed embeddings passed to ChromaDB (no model download required)

## Architecture

```
┌─────────────┐     ┌──────────────────┐
│  FactStore  │────▶│                  │
│  (triples)  │     │   RAGRetriever   │──▶ RetrievalResult
└─────────────┘     │                  │      • query
                    │  index_fact()    │      • results (scored)
┌─────────────┐     │  index_node()    │      • context (formatted)
│ Knowledge   │────▶│  index_text()    │      • facts (resolved)
│   Graph     │     │  retrieve()      │      • nodes (resolved)
│ (nodes/edges│     │  augment_prompt()│
└─────────────┘     └───────┬──────────┘
                            │
                    ┌───────▼──────────┐
                    │   VectorStore    │
                    │  (embeddings +   │
                    │  cosine search)  │
                    └───────┬──────────┘
                            │
                    ┌───────▼──────────┐
                    │ EmbeddingProvider │
                    │  (HashEmbedding   │
                    │   or custom)      │
                    └──────────────────┘
```

The RAGRetriever works in three modes:
1. **Standalone** — index and search arbitrary text, no FactStore/KnowledgeGraph needed
2. **With FactStore** — index facts as vectors, resolve back to Fact objects on retrieval
3. **Fully integrated** — index both facts and graph nodes, expand results via relationships and neighbors

## Error Hierarchy

```
FactStoreError
└── FactNotFoundError

KnowledgeGraphError
├── NodeNotFoundError
└── EdgeNotFoundError

RAGError
└── DocumentNotFoundError
```

## Running Tests

```bash
source venv/bin/activate
cd day_50

# All tests
python -m pytest -v

# Individual modules
python -m pytest test_fact_store.py -v         # 35 tests
python -m pytest test_knowledge_graph.py -v    # 65 tests
python -m pytest test_rag_integration.py -v    # 53 tests
```

## Test Coverage

| Module | Tests | Areas |
|---|---|---|
| `test_fact_store.py` | 35 | CRUD, indexed retrieval, multi-field query, confidence filtering, relationships, BFS traversal, cycle handling |
| `test_knowledge_graph.py` | 65 | Node/edge CRUD, type/label queries, neighbor directions, BFS, path finding, pattern matching, subgraph extraction, cascading delete |
| `test_rag_integration.py` | 53 | Embedding (normalization, determinism, similarity), vector store CRUD, search (ranking, top-k, min-score, metadata filter, pre-computed), indexing (facts, nodes, text), retrieval (resolution, related expansion, filtering, context formatting), prompt augmentation, edge cases (missing backends, deleted facts) |

## Dependencies

- Python 3.8+
- chromadb: Vector database (HNSW-based approximate nearest-neighbor search)
- openai: OpenAI API client (for OpenAIEmbedding provider)
- pytest: Testing framework

The `HashEmbedding` provider requires no external dependencies and is used by default for development and testing.

## Swapping Embedding Providers

### OpenAI (built-in)

```python
from rag_integration import OpenAIEmbedding, VectorStore, RAGRetriever

embedder = OpenAIEmbedding(api_key="sk-...")  # or set OPENAI_API_KEY env var
vs = VectorStore(embedder)
rag = RAGRetriever(vs, fact_store=fact_store, knowledge_graph=kg)
```

### Custom provider

The `EmbeddingProvider` ABC makes it easy to plug in production-grade embeddings:

```python
from rag_integration import EmbeddingProvider

class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()
```

Then pass it to VectorStore:

```python
embedder = SentenceTransformerEmbedding()
vs = VectorStore(embedder)
rag = RAGRetriever(vs, fact_store=fact_store, knowledge_graph=kg)
```
