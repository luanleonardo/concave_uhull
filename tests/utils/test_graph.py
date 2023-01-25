from collections import defaultdict

import pytest

from alpha_shape.utils.geometry import euclidean_distance
from alpha_shape.utils.graph import add_edge


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
        add_edge(graph, weight, source, target, euclidean_distance)

    # graph must have 4 nodes
    assert len(graph) == 4

    # there should be no edge between (0.0, 0.0) and (1.0, 1.0)
    assert (1.0, 1.0) not in graph[(0.0, 0.0)]

    # weight of (0.0, 0.0) - (1.0, 0.0) should be 1.0
    assert weight[(0.0, 0.0)][(1.0, 0.0)] == 1.0
