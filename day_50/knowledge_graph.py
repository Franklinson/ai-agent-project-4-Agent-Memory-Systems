"""Knowledge graph for organizing facts and concepts as nodes and edges."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid


class KnowledgeGraphError(Exception):
    """Base exception for knowledge graph operations."""


class NodeNotFoundError(KnowledgeGraphError):
    """Raised when a node is not found."""


class EdgeNotFoundError(KnowledgeGraphError):
    """Raised when an edge is not found."""


class NodeType(Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    EVENT = "event"
    ATTRIBUTE = "attribute"
    CATEGORY = "category"


class EdgeType(Enum):
    IS_A = "is_a"
    HAS = "has"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    CUSTOM = "custom"


@dataclass
class Node:
    """A node (entity or concept) in the knowledge graph."""
    id: str
    label: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Edge:
    """A directed edge (relationship) between two nodes."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    label: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class KnowledgeGraph:
    """Stores entities and concepts as nodes with typed, directed edges."""

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        # Indexes
        self._by_node_type: Dict[NodeType, Set[str]] = {}
        self._by_label: Dict[str, Set[str]] = {}
        self._by_edge_type: Dict[EdgeType, Set[str]] = {}
        # Adjacency: node_id -> set of edge_ids
        self._outgoing: Dict[str, Set[str]] = {}
        self._incoming: Dict[str, Set[str]] = {}

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # --- Node CRUD ---

    def add_node(self, label: str, node_type: NodeType = NodeType.ENTITY,
                 properties: Optional[Dict[str, Any]] = None,
                 node_id: Optional[str] = None) -> Node:
        nid = node_id or str(uuid.uuid4())
        if nid in self._nodes:
            raise KnowledgeGraphError(f"Node '{nid}' already exists")
        node = Node(id=nid, label=label, node_type=node_type,
                    properties=properties or {})
        self._nodes[nid] = node
        self._by_node_type.setdefault(node_type, set()).add(nid)
        self._by_label.setdefault(label, set()).add(nid)
        return node

    def get_node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    def update_node(self, node_id: str, label: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> Node:
        node = self.get_node(node_id)
        if label is not None and label != node.label:
            self._by_label.get(node.label, set()).discard(node_id)
            node.label = label
            self._by_label.setdefault(label, set()).add(node_id)
        if properties is not None:
            node.properties.update(properties)
        node.updated_at = datetime.now()
        return node

    def remove_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        # Remove all connected edges
        for eid in list(self._outgoing.get(node_id, [])):
            self._remove_edge_internal(eid)
        for eid in list(self._incoming.get(node_id, [])):
            self._remove_edge_internal(eid)
        # Remove from indexes
        self._by_node_type.get(node.node_type, set()).discard(node_id)
        self._by_label.get(node.label, set()).discard(node_id)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        del self._nodes[node_id]
        return True

    # --- Edge CRUD ---

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType = EdgeType.RELATED_TO,
                 label: str = "", weight: float = 1.0,
                 properties: Optional[Dict[str, Any]] = None,
                 edge_id: Optional[str] = None) -> Edge:
        self.get_node(source_id)
        self.get_node(target_id)
        eid = edge_id or str(uuid.uuid4())
        if eid in self._edges:
            raise KnowledgeGraphError(f"Edge '{eid}' already exists")
        edge = Edge(id=eid, source_id=source_id, target_id=target_id,
                    edge_type=edge_type, label=label, weight=weight,
                    properties=properties or {})
        self._edges[eid] = edge
        self._by_edge_type.setdefault(edge_type, set()).add(eid)
        self._outgoing.setdefault(source_id, set()).add(eid)
        self._incoming.setdefault(target_id, set()).add(eid)
        return edge

    def get_edge(self, edge_id: str) -> Edge:
        if edge_id not in self._edges:
            raise EdgeNotFoundError(f"Edge '{edge_id}' not found")
        return self._edges[edge_id]

    def update_edge(self, edge_id: str, label: Optional[str] = None,
                    weight: Optional[float] = None,
                    properties: Optional[Dict[str, Any]] = None) -> Edge:
        edge = self.get_edge(edge_id)
        if label is not None:
            edge.label = label
        if weight is not None:
            edge.weight = weight
        if properties is not None:
            edge.properties.update(properties)
        return edge

    def remove_edge(self, edge_id: str) -> bool:
        self.get_edge(edge_id)  # validate exists
        self._remove_edge_internal(edge_id)
        return True

    # --- Node queries ---

    def nodes_by_type(self, node_type: NodeType) -> List[Node]:
        ids = self._by_node_type.get(node_type, set())
        return sorted([self._nodes[i] for i in ids], key=lambda n: n.created_at)

    def nodes_by_label(self, label: str) -> List[Node]:
        ids = self._by_label.get(label, set())
        return sorted([self._nodes[i] for i in ids], key=lambda n: n.created_at)

    def find_nodes(self, node_type: Optional[NodeType] = None,
                   label: Optional[str] = None,
                   filter_fn: Optional[Callable[[Node], bool]] = None) -> List[Node]:
        """Query nodes by type, label, and/or custom filter. All conditions ANDed."""
        candidates: Optional[Set[str]] = None
        if node_type is not None:
            ids = self._by_node_type.get(node_type, set())
            candidates = ids if candidates is None else candidates & ids
        if label is not None:
            ids = self._by_label.get(label, set())
            candidates = ids if candidates is None else candidates & ids
        if candidates is None:
            candidates = set(self._nodes.keys())
        results = [self._nodes[i] for i in candidates]
        if filter_fn:
            results = [n for n in results if filter_fn(n)]
        return sorted(results, key=lambda n: n.created_at)

    # --- Edge queries ---

    def edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        ids = self._by_edge_type.get(edge_type, set())
        return sorted([self._edges[i] for i in ids], key=lambda e: e.created_at)

    def edges_from(self, node_id: str) -> List[Edge]:
        self.get_node(node_id)
        return sorted([self._edges[i] for i in self._outgoing.get(node_id, set())],
                       key=lambda e: e.created_at)

    def edges_to(self, node_id: str) -> List[Edge]:
        self.get_node(node_id)
        return sorted([self._edges[i] for i in self._incoming.get(node_id, set())],
                       key=lambda e: e.created_at)

    def find_edges(self, edge_type: Optional[EdgeType] = None,
                   source_id: Optional[str] = None,
                   target_id: Optional[str] = None,
                   min_weight: Optional[float] = None) -> List[Edge]:
        """Query edges by type, source, target, and/or min weight. All ANDed."""
        candidates: Optional[Set[str]] = None
        if edge_type is not None:
            ids = self._by_edge_type.get(edge_type, set())
            candidates = ids if candidates is None else candidates & ids
        if source_id is not None:
            ids = self._outgoing.get(source_id, set())
            candidates = ids if candidates is None else candidates & ids
        if target_id is not None:
            ids = self._incoming.get(target_id, set())
            candidates = ids if candidates is None else candidates & ids
        if candidates is None:
            candidates = set(self._edges.keys())
        results = [self._edges[i] for i in candidates]
        if min_weight is not None:
            results = [e for e in results if e.weight >= min_weight]
        return sorted(results, key=lambda e: e.created_at)

    # --- Neighbors ---

    def neighbors(self, node_id: str, direction: str = "both",
                  edge_type: Optional[EdgeType] = None) -> List[Node]:
        """Get neighboring nodes. direction: 'out', 'in', or 'both'."""
        self.get_node(node_id)
        neighbor_ids: Set[str] = set()
        if direction in ("out", "both"):
            for eid in self._outgoing.get(node_id, set()):
                e = self._edges[eid]
                if edge_type is None or e.edge_type == edge_type:
                    neighbor_ids.add(e.target_id)
        if direction in ("in", "both"):
            for eid in self._incoming.get(node_id, set()):
                e = self._edges[eid]
                if edge_type is None or e.edge_type == edge_type:
                    neighbor_ids.add(e.source_id)
        return sorted([self._nodes[i] for i in neighbor_ids],
                       key=lambda n: n.created_at)

    # --- Traversal ---

    def bfs(self, start_id: str, max_depth: int = 3,
            edge_type: Optional[EdgeType] = None,
            direction: str = "out") -> Dict[int, List[Node]]:
        """BFS from start node, returning nodes grouped by depth."""
        self.get_node(start_id)
        visited: Set[str] = {start_id}
        current = {start_id}
        result: Dict[int, List[Node]] = {}
        for depth in range(1, max_depth + 1):
            next_level: Set[str] = set()
            for nid in current:
                for nb in self.neighbors(nid, direction=direction, edge_type=edge_type):
                    if nb.id not in visited:
                        next_level.add(nb.id)
                        visited.add(nb.id)
            if not next_level:
                break
            result[depth] = sorted([self._nodes[i] for i in next_level],
                                    key=lambda n: n.created_at)
            current = next_level
        return result

    def find_path(self, start_id: str, end_id: str,
                  edge_type: Optional[EdgeType] = None,
                  max_depth: int = 10) -> Optional[List[Node]]:
        """BFS shortest path from start to end. Returns list of nodes or None."""
        self.get_node(start_id)
        self.get_node(end_id)
        if start_id == end_id:
            return [self._nodes[start_id]]
        visited: Set[str] = {start_id}
        queue: deque[Tuple[str, List[str]]] = deque([(start_id, [start_id])])
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= max_depth:
                continue
            for nb in self.neighbors(current, direction="out", edge_type=edge_type):
                if nb.id == end_id:
                    return [self._nodes[i] for i in path + [nb.id]]
                if nb.id not in visited:
                    visited.add(nb.id)
                    queue.append((nb.id, path + [nb.id]))
        return None

    # --- Pattern matching ---

    def match_pattern(self, source_type: Optional[NodeType] = None,
                      edge_type: Optional[EdgeType] = None,
                      target_type: Optional[NodeType] = None) -> List[Tuple[Node, Edge, Node]]:
        """Find all (source, edge, target) triples matching the given types."""
        results: List[Tuple[Node, Edge, Node]] = []
        edges = self.edges_by_type(edge_type) if edge_type else list(self._edges.values())
        for e in edges:
            src = self._nodes[e.source_id]
            tgt = self._nodes[e.target_id]
            if source_type is not None and src.node_type != source_type:
                continue
            if target_type is not None and tgt.node_type != target_type:
                continue
            results.append((src, e, tgt))
        return sorted(results, key=lambda t: t[1].created_at)

    def subgraph(self, node_ids: Set[str]) -> Tuple[List[Node], List[Edge]]:
        """Extract the subgraph induced by the given node IDs."""
        nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
        edges = [e for e in self._edges.values()
                 if e.source_id in node_ids and e.target_id in node_ids]
        return (sorted(nodes, key=lambda n: n.created_at),
                sorted(edges, key=lambda e: e.created_at))

    # --- Internal ---

    def _remove_edge_internal(self, edge_id: str) -> None:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return
        self._by_edge_type.get(edge.edge_type, set()).discard(edge_id)
        self._outgoing.get(edge.source_id, set()).discard(edge_id)
        self._incoming.get(edge.target_id, set()).discard(edge_id)
