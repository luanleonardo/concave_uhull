from collections import defaultdict

from alpha_shape.utils.geometry import euclidean_distance
from alpha_shape.utils.graph import add_edge


def test_add_edge():
    """"""
    # create data structure for graph, edge weights and edges
    graph = defaultdict(set)
    weight = defaultdict(dict)
    square_edges = [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
    ]

    # define graph from edges
    for source, target in square_edges:
        add_edge(graph, weight, source, target, euclidean_distance)

    # graph must have 4 nodes
    assert len(graph) == 4

    # there should be no edge between (0.0, 0.0) and (1.0, 1.0)
    assert (1.0, 1.0) not in graph[(0.0, 0.0)]

    # weight of (0.0, 0.0) - (1.0, 0.0) should be 1.0
    assert weight[(0.0, 0.0)][(1.0, 0.0)] == 1.0
