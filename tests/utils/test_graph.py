from collections import defaultdict

import pytest

from alpha_shape.utils.geometry import euclidean_distance
from alpha_shape.utils.graph import add_edge, remove_edge


@pytest.fixture
def square_edges():
    return [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
    ]


def test_add_edge(square_edges):
    """"""
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
    """"""
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
    """"""
    # create data structure for graph, edge weights and edges
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph_adjacency_list, edge_weights, source, target, euclidean_distance)

    # remove edge
    remove_edge(
        graph_adjacency_list,
        edge_weights,
        edge_source=(0.0, 0.0),
        edge_target=(0.0, 1.0),
    )

    # there must be no edge (0.0, 0.0) - (0.0, 1.0)
    assert (0.0, 1.0) not in graph_adjacency_list[(0.0, 0.0)]

    # there must be no weight associated with the edge (0.0, 0.0) - (0.0, 1.0)
    assert (0.0, 1.0) not in edge_weights[(0.0, 0.0)]


def test_remove_edge_assertion_error(square_edges):
    """"""
    # create data structure for graph, edge weights and edges
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph_adjacency_list, edge_weights, source, target, euclidean_distance)

    # remove edge
    remove_edge(
        graph_adjacency_list,
        edge_weights,
        edge_source=(0.0, 0.0),
        edge_target=(0.0, 1.0),
    )

    with pytest.raises(AssertionError, match="No edge"):
        remove_edge(
            graph_adjacency_list,
            edge_weights,
            edge_source=(0.0, 0.0),
            edge_target=(0.0, 1.0),
        )
