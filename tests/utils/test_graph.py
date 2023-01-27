from collections import defaultdict

import pytest

from concave_uhull.utils.geometry import euclidean_distance
from concave_uhull.utils.graph import add_edge, remove_edge, shortest_path


@pytest.fixture
def square_edges():
    """TODO"""
    return [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
    ]


def test_add_edge(square_edges):
    """Test add edge to graph"""
    # create data structure for graph, edge weights and edges
    graph = defaultdict(set)
    weight = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(
            graph_adjacency_list=graph,
            edge_weights=weight,
            edge_source=source,
            edge_target=target,
            weight_function=euclidean_distance,
        )

    # graph must have 4 nodes
    assert len(graph) == 4

    # there should be no edge between (0.0, 0.0) and (1.0, 1.0)
    assert (1.0, 1.0) not in graph[(0.0, 0.0)]

    # weight of (0.0, 0.0) - (1.0, 0.0) should be 1.0
    assert weight[(0.0, 0.0)][(1.0, 0.0)] == 1.0


def test_add_edge_assertion_error(square_edges):
    """function should throw an assertion error when trying
    to add edge that already exists"""
    # create data structure for graph, edge weights and edges
    graph = defaultdict(set)
    weight = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(
            graph_adjacency_list=graph,
            edge_weights=weight,
            edge_source=source,
            edge_target=target,
            weight_function=euclidean_distance,
        )

    # try adding existing edge
    source, target = (0.0, 0.0), (1.0, 0.0)
    with pytest.raises(AssertionError, match="already exists"):
        add_edge(
            graph_adjacency_list=graph,
            edge_weights=weight,
            edge_source=source,
            edge_target=target,
            weight_function=euclidean_distance,
        )


def test_remove_edge(square_edges):
    """Test edge removal"""
    # create data structure for graph, edge weights and edges
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph_adjacency_list, edge_weights, source, target, euclidean_distance)

    # remove edge
    edge_source, edge_target = (0.0, 0.0), (1.0, 0.0)
    remove_edge(
        graph_adjacency_list,
        edge_weights,
        edge_source,
        edge_target,
    )

    # there must be no edge (0.0, 0.0) - (0.0, 1.0)
    assert edge_target not in graph_adjacency_list[edge_source]

    # there must be no weight associated with the edge (0.0, 0.0) - (0.0, 1.0)
    assert edge_target not in edge_weights[edge_source]


def test_remove_edge_assertion_error(square_edges):
    """Function should throw an assertion error when
    trying to remove an edge that does not exist in
    the graph"""
    # create data structure for graph, edge weights and edges
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph_adjacency_list, edge_weights, source, target, euclidean_distance)

    # remove edge
    edge_source, edge_target = (0.0, 0.0), (1.0, 0.0)
    remove_edge(
        graph_adjacency_list,
        edge_weights,
        edge_source,
        edge_target,
    )

    # try to remove nonexistent edge
    with pytest.raises(AssertionError, match="No edge"):
        remove_edge(
            graph_adjacency_list,
            edge_weights,
            edge_source,
            edge_target,
        )


def test_shortest_path(square_edges):
    """Tests to get the shortest path between nodes"""
    # create data structure for graph, edge weights and edges
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph_adjacency_list, edge_weights, source, target, euclidean_distance)

    # get the shortest path between nodes
    edge_source, edge_target = (0.0, 0.0), (1.0, 0.0)
    path = shortest_path(graph_adjacency_list, edge_weights, edge_source, edge_target)

    # there is edge connecting the nodes, so the shortest path is formed by the nodes
    # themselves.
    assert path == [edge_source, edge_target]

    # removing the edge, the shortest path will be formed by all the points,
    # as it contains all other remaining edges.
    remove_edge(
        graph_adjacency_list,
        edge_weights,
        edge_source,
        edge_target,
    )
    path = shortest_path(graph_adjacency_list, edge_weights, edge_source, edge_target)
    assert path == [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]


def test_shortest_path_assertion_error(square_edges):
    """Function throws assertion error in two cases: when there is no path in
    the graph connecting the two points or when one of the nodes (or both) are
    not in the graph."""
    # create data structure for graph, edge weights and edges
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph_adjacency_list, edge_weights, source, target, euclidean_distance)

    # add edge to make the graph disconnected
    edge_source, edge_target = (0.25, 0.25), (0.75, 0.75)
    add_edge(
        graph_adjacency_list, edge_weights, edge_source, edge_target, euclidean_distance
    )

    # Try to find the shortest path between nodes of one connected component and another
    # (does not exist)
    with pytest.raises(AssertionError, match="There is no path"):
        shortest_path(
            graph_adjacency_list=graph_adjacency_list,
            edge_weights=edge_weights,
            edge_source=(0.0, 0.0),
            edge_target=edge_target,
        )

    # Try to find the shortest path between nodes that do not belong to the graph (impossible)
    with pytest.raises(AssertionError, match="Impossible to find path"):
        shortest_path(
            graph_adjacency_list=graph_adjacency_list,
            edge_weights=edge_weights,
            edge_source=(11.0, 11.0),
            edge_target=edge_target,
        )
