"""Tests for the knowledge graph."""

import pytest
from knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType, Node, Edge,
    KnowledgeGraphError, NodeNotFoundError, EdgeNotFoundError,
)


@pytest.fixture
def graph():
    return KnowledgeGraph()


@pytest.fixture
def lang_graph(graph):
    """A small language/framework graph for testing."""
    graph.add_node("Python", NodeType.ENTITY, node_id="python")
    graph.add_node("Java", NodeType.ENTITY, node_id="java")
    graph.add_node("Programming Language", NodeType.CONCEPT, node_id="pl")
    graph.add_node("Django", NodeType.ENTITY, node_id="django")
    graph.add_node("Web Framework", NodeType.CONCEPT, node_id="wf")
    graph.add_node("dynamic typing", NodeType.ATTRIBUTE, node_id="dyn")

    graph.add_edge("python", "pl", EdgeType.IS_A, edge_id="e1")
    graph.add_edge("java", "pl", EdgeType.IS_A, edge_id="e2")
    graph.add_edge("django", "wf", EdgeType.IS_A, edge_id="e3")
    graph.add_edge("django", "python", EdgeType.DEPENDS_ON, edge_id="e4")
    graph.add_edge("python", "dyn", EdgeType.HAS, edge_id="e5")
    return graph


# --- Node CRUD ---

class TestNodeCRUD:
    def test_add_node(self, graph):
        node = graph.add_node("Python", NodeType.ENTITY)
        assert node.label == "Python"
        assert node.node_type == NodeType.ENTITY
        assert graph.node_count == 1

    def test_add_node_with_properties(self, graph):
        node = graph.add_node("Python", NodeType.ENTITY,
                              properties={"version": "3.12"}, node_id="py")
        assert node.id == "py"
        assert node.properties == {"version": "3.12"}

    def test_add_duplicate_id_raises(self, graph):
        graph.add_node("A", NodeType.ENTITY, node_id="dup")
        with pytest.raises(KnowledgeGraphError, match="already exists"):
            graph.add_node("B", NodeType.ENTITY, node_id="dup")

    def test_get_node(self, lang_graph):
        node = lang_graph.get_node("python")
        assert node.label == "Python"

    def test_get_missing_raises(self, graph):
        with pytest.raises(NodeNotFoundError):
            graph.get_node("nope")

    def test_update_node_label(self, lang_graph):
        lang_graph.update_node("python", label="Python3")
        assert lang_graph.get_node("python").label == "Python3"
        assert lang_graph.nodes_by_label("Python3")[0].id == "python"
        assert lang_graph.nodes_by_label("Python") == []

    def test_update_node_properties(self, lang_graph):
        lang_graph.update_node("python", properties={"version": "3.12"})
        lang_graph.update_node("python", properties={"stable": True})
        props = lang_graph.get_node("python").properties
        assert props == {"version": "3.12", "stable": True}

    def test_remove_node(self, lang_graph):
        initial_edges = lang_graph.edge_count
        lang_graph.remove_node("python")
        assert lang_graph.node_count == 5
        # Edges connected to python (e1, e4, e5) should be removed
        assert lang_graph.edge_count == initial_edges - 3
        with pytest.raises(NodeNotFoundError):
            lang_graph.get_node("python")

    def test_remove_missing_raises(self, graph):
        with pytest.raises(NodeNotFoundError):
            graph.remove_node("nope")

    def test_all_node_types(self, graph):
        for nt in NodeType:
            graph.add_node(nt.value, nt)
        assert graph.node_count == len(NodeType)


# --- Edge CRUD ---

class TestEdgeCRUD:
    def test_add_edge(self, lang_graph):
        assert lang_graph.edge_count == 5

    def test_add_edge_with_options(self, graph):
        graph.add_node("A", NodeType.ENTITY, node_id="a")
        graph.add_node("B", NodeType.ENTITY, node_id="b")
        edge = graph.add_edge("a", "b", EdgeType.CAUSES, label="triggers",
                              weight=0.8, properties={"reason": "test"},
                              edge_id="e1")
        assert edge.label == "triggers"
        assert edge.weight == 0.8
        assert edge.properties == {"reason": "test"}
        assert edge.edge_type == EdgeType.CAUSES

    def test_add_edge_missing_node_raises(self, graph):
        graph.add_node("A", NodeType.ENTITY, node_id="a")
        with pytest.raises(NodeNotFoundError):
            graph.add_edge("a", "missing", EdgeType.RELATED_TO)

    def test_add_duplicate_edge_id_raises(self, lang_graph):
        with pytest.raises(KnowledgeGraphError, match="already exists"):
            lang_graph.add_edge("python", "java", EdgeType.RELATED_TO, edge_id="e1")

    def test_get_edge(self, lang_graph):
        edge = lang_graph.get_edge("e1")
        assert edge.source_id == "python"
        assert edge.target_id == "pl"

    def test_get_missing_edge_raises(self, graph):
        with pytest.raises(EdgeNotFoundError):
            graph.get_edge("nope")

    def test_update_edge(self, lang_graph):
        lang_graph.update_edge("e1", label="instance_of", weight=0.9,
                               properties={"verified": True})
        edge = lang_graph.get_edge("e1")
        assert edge.label == "instance_of"
        assert edge.weight == 0.9
        assert edge.properties["verified"] is True

    def test_remove_edge(self, lang_graph):
        assert lang_graph.remove_edge("e1")
        assert lang_graph.edge_count == 4
        with pytest.raises(EdgeNotFoundError):
            lang_graph.get_edge("e1")

    def test_remove_missing_edge_raises(self, graph):
        with pytest.raises(EdgeNotFoundError):
            graph.remove_edge("nope")

    def test_all_edge_types(self, graph):
        graph.add_node("A", NodeType.ENTITY, node_id="a")
        graph.add_node("B", NodeType.ENTITY, node_id="b")
        for i, et in enumerate(EdgeType):
            graph.add_edge("a", "b", et, edge_id=f"e{i}")
        assert graph.edge_count == len(EdgeType)


# --- Node queries ---

class TestNodeQueries:
    def test_nodes_by_type(self, lang_graph):
        entities = lang_graph.nodes_by_type(NodeType.ENTITY)
        assert len(entities) == 3
        labels = {n.label for n in entities}
        assert labels == {"Python", "Java", "Django"}

    def test_nodes_by_label(self, lang_graph):
        results = lang_graph.nodes_by_label("Python")
        assert len(results) == 1
        assert results[0].id == "python"

    def test_nodes_by_label_empty(self, lang_graph):
        assert lang_graph.nodes_by_label("Rust") == []

    def test_find_nodes_by_type(self, lang_graph):
        results = lang_graph.find_nodes(node_type=NodeType.CONCEPT)
        assert len(results) == 2

    def test_find_nodes_by_label(self, lang_graph):
        results = lang_graph.find_nodes(label="Django")
        assert len(results) == 1

    def test_find_nodes_with_filter(self, lang_graph):
        lang_graph.update_node("python", properties={"version": "3.12"})
        results = lang_graph.find_nodes(
            filter_fn=lambda n: "version" in n.properties
        )
        assert len(results) == 1
        assert results[0].id == "python"

    def test_find_nodes_combined(self, lang_graph):
        results = lang_graph.find_nodes(node_type=NodeType.ENTITY, label="Python")
        assert len(results) == 1

    def test_find_nodes_no_match(self, lang_graph):
        results = lang_graph.find_nodes(node_type=NodeType.ENTITY, label="Programming Language")
        assert results == []

    def test_find_nodes_no_filters(self, lang_graph):
        results = lang_graph.find_nodes()
        assert len(results) == 6


# --- Edge queries ---

class TestEdgeQueries:
    def test_edges_by_type(self, lang_graph):
        is_a = lang_graph.edges_by_type(EdgeType.IS_A)
        assert len(is_a) == 3

    def test_edges_from(self, lang_graph):
        edges = lang_graph.edges_from("python")
        assert len(edges) == 2  # e1 (is_a pl), e5 (has dyn)

    def test_edges_to(self, lang_graph):
        edges = lang_graph.edges_to("pl")
        assert len(edges) == 2  # e1, e2

    def test_find_edges_by_type(self, lang_graph):
        results = lang_graph.find_edges(edge_type=EdgeType.DEPENDS_ON)
        assert len(results) == 1
        assert results[0].source_id == "django"

    def test_find_edges_by_source(self, lang_graph):
        results = lang_graph.find_edges(source_id="django")
        assert len(results) == 2  # e3, e4

    def test_find_edges_by_target(self, lang_graph):
        results = lang_graph.find_edges(target_id="python")
        assert len(results) == 1  # e4

    def test_find_edges_combined(self, lang_graph):
        results = lang_graph.find_edges(source_id="python", edge_type=EdgeType.IS_A)
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_find_edges_min_weight(self, graph):
        graph.add_node("A", NodeType.ENTITY, node_id="a")
        graph.add_node("B", NodeType.ENTITY, node_id="b")
        graph.add_edge("a", "b", weight=0.5, edge_id="lo")
        graph.add_edge("a", "b", weight=0.9, edge_id="hi")
        results = graph.find_edges(min_weight=0.8)
        assert len(results) == 1
        assert results[0].id == "hi"

    def test_find_edges_no_filters(self, lang_graph):
        assert len(lang_graph.find_edges()) == 5


# --- Neighbors ---

class TestNeighbors:
    def test_neighbors_out(self, lang_graph):
        nbs = lang_graph.neighbors("python", direction="out")
        ids = {n.id for n in nbs}
        assert ids == {"pl", "dyn"}

    def test_neighbors_in(self, lang_graph):
        nbs = lang_graph.neighbors("python", direction="in")
        ids = {n.id for n in nbs}
        assert ids == {"django"}

    def test_neighbors_both(self, lang_graph):
        nbs = lang_graph.neighbors("python", direction="both")
        ids = {n.id for n in nbs}
        assert ids == {"pl", "dyn", "django"}

    def test_neighbors_with_edge_type(self, lang_graph):
        nbs = lang_graph.neighbors("python", direction="out", edge_type=EdgeType.IS_A)
        assert len(nbs) == 1
        assert nbs[0].id == "pl"

    def test_neighbors_empty(self, lang_graph):
        assert lang_graph.neighbors("dyn", direction="out") == []


# --- Traversal ---

class TestTraversal:
    def test_bfs_depth_1(self, lang_graph):
        result = lang_graph.bfs("django", max_depth=1, direction="out")
        assert 1 in result
        ids = {n.id for n in result[1]}
        assert ids == {"wf", "python"}

    def test_bfs_depth_2(self, lang_graph):
        result = lang_graph.bfs("django", max_depth=2, direction="out")
        assert 1 in result
        assert 2 in result
        depth2_ids = {n.id for n in result[2]}
        assert "pl" in depth2_ids  # django -> python -> pl

    def test_bfs_with_edge_type(self, lang_graph):
        result = lang_graph.bfs("django", max_depth=3, edge_type=EdgeType.IS_A,
                                direction="out")
        assert 1 in result
        ids = {n.id for n in result[1]}
        assert ids == {"wf"}
        assert 2 not in result  # wf has no outgoing IS_A

    def test_bfs_no_neighbors(self, lang_graph):
        result = lang_graph.bfs("dyn", max_depth=3, direction="out")
        assert result == {}

    def test_bfs_cycle(self, graph):
        graph.add_node("A", NodeType.ENTITY, node_id="a")
        graph.add_node("B", NodeType.ENTITY, node_id="b")
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        result = graph.bfs("a", max_depth=5, direction="out")
        assert len(result) == 1
        assert len(result[1]) == 1


# --- Path finding ---

class TestPathFinding:
    def test_find_path_direct(self, lang_graph):
        path = lang_graph.find_path("python", "pl")
        assert path is not None
        assert [n.id for n in path] == ["python", "pl"]

    def test_find_path_multi_hop(self, lang_graph):
        path = lang_graph.find_path("django", "pl")
        # django -> python -> pl
        assert path is not None
        ids = [n.id for n in path]
        assert ids[0] == "django"
        assert ids[-1] == "pl"
        assert len(ids) == 3

    def test_find_path_same_node(self, lang_graph):
        path = lang_graph.find_path("python", "python")
        assert path is not None
        assert len(path) == 1
        assert path[0].id == "python"

    def test_find_path_none(self, lang_graph):
        # pl has no outgoing edges, so no path from pl to django
        path = lang_graph.find_path("pl", "django")
        assert path is None

    def test_find_path_with_edge_type(self, lang_graph):
        # django -> python via DEPENDS_ON, python -> pl via IS_A
        # With IS_A filter, no path from django to pl
        path = lang_graph.find_path("django", "pl", edge_type=EdgeType.IS_A)
        assert path is None

    def test_find_path_max_depth(self, lang_graph):
        path = lang_graph.find_path("django", "pl", max_depth=1)
        assert path is None  # needs 2 hops

    def test_find_path_missing_node_raises(self, graph):
        with pytest.raises(NodeNotFoundError):
            graph.find_path("a", "b")


# --- Pattern matching ---

class TestPatternMatching:
    def test_match_by_edge_type(self, lang_graph):
        triples = lang_graph.match_pattern(edge_type=EdgeType.IS_A)
        assert len(triples) == 3

    def test_match_by_source_type(self, lang_graph):
        triples = lang_graph.match_pattern(source_type=NodeType.ENTITY)
        assert len(triples) == 5  # all edges have entity sources

    def test_match_by_target_type(self, lang_graph):
        triples = lang_graph.match_pattern(target_type=NodeType.CONCEPT)
        assert len(triples) == 3  # e1, e2, e3

    def test_match_combined(self, lang_graph):
        triples = lang_graph.match_pattern(
            source_type=NodeType.ENTITY,
            edge_type=EdgeType.IS_A,
            target_type=NodeType.CONCEPT,
        )
        assert len(triples) == 3
        for src, edge, tgt in triples:
            assert src.node_type == NodeType.ENTITY
            assert edge.edge_type == EdgeType.IS_A
            assert tgt.node_type == NodeType.CONCEPT

    def test_match_no_results(self, lang_graph):
        triples = lang_graph.match_pattern(
            source_type=NodeType.CONCEPT,
            edge_type=EdgeType.CAUSES,
        )
        assert triples == []

    def test_match_no_filters(self, lang_graph):
        triples = lang_graph.match_pattern()
        assert len(triples) == 5


# --- Subgraph ---

class TestSubgraph:
    def test_subgraph(self, lang_graph):
        nodes, edges = lang_graph.subgraph({"python", "pl", "java"})
        assert len(nodes) == 3
        # Only edges between these three: e1 (python->pl), e2 (java->pl)
        assert len(edges) == 2

    def test_subgraph_single_node(self, lang_graph):
        nodes, edges = lang_graph.subgraph({"python"})
        assert len(nodes) == 1
        assert len(edges) == 0

    def test_subgraph_missing_ids_ignored(self, lang_graph):
        nodes, edges = lang_graph.subgraph({"python", "nonexistent"})
        assert len(nodes) == 1

    def test_subgraph_empty(self, lang_graph):
        nodes, edges = lang_graph.subgraph(set())
        assert nodes == []
        assert edges == []
