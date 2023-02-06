import pytest

from concave_uhull.utils.geometry import euclidean_distance
from concave_uhull.utils.graph import (
    Graph,
    dijkstra_algorithm,
    shortest_path_algorithm,
)


@pytest.fixture
def square_edges():
    """
    Defines a set of points that form a square of side 1.0.
    """
    return [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
    ]


def test_add_edge_method(square_edges):
    """
    Tests the add edge method of the Graph class.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # graph must have 4 nodes
    assert len(graph) == 4

    # set of nodes
    assert graph.nodes == {(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)}

    # there should be no connection between (0.0, 0.0) and (1.0, 1.0)
    assert (0.0, 0.0) not in graph[(1.0, 1.0)]
    assert (1.0, 1.0) not in graph[(0.0, 0.0)]

    # weight of edge (0.0, 0.0) - (1.0, 0.0) should be 1.0
    assert graph.weight[(0.0, 0.0)][(1.0, 0.0)] == 1.0


@pytest.mark.parametrize(
    "edge_source,edge_target",
    [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0))],
)
def test_add_edge_method_assertion_error(
    square_edges, edge_source, edge_target
):
    """
    Method should throw an assertion error when trying to add edge
    that already exists.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # try adding existing edge
    with pytest.raises(AssertionError, match="already exists"):
        graph.add_edge(
            edge_source=edge_source,
            edge_target=edge_target,
            edge_weight=euclidean_distance(edge_source, edge_target),
        )


@pytest.mark.parametrize(
    "edge_source,edge_target",
    [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0))],
)
def test_remove_edge_method(square_edges, edge_source, edge_target):
    """
    Test class edge removal method.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # remove edge
    graph.remove_edge(
        edge_source=edge_source,
        edge_target=edge_target,
    )

    # there must be no connection between nodes
    assert edge_target not in graph[edge_source]

    # there must be no weight associated with the edge
    assert edge_target not in graph.weight[edge_source]


@pytest.mark.parametrize(
    "edge_source,edge_target",
    [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0))],
)
def test_remove_edge_method_assertion_error(
    square_edges, edge_source, edge_target
):
    """
    Method should throw an assertion error when trying to remove
    an edge that does not exist in the graph.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # remove edge
    graph.remove_edge(
        edge_source=edge_source,
        edge_target=edge_target,
    )

    # try to remove nonexistent edge
    with pytest.raises(AssertionError, match="No edge"):
        graph.remove_edge(
            edge_source=edge_source,
            edge_target=edge_target,
        )


@pytest.mark.parametrize(
    "edge_source,edge_target,expected_dist",
    [
        ((0.0, 0.0), (0.0, 1.0), 1.0),
        ((0.0, 0.0), (1.0, 1.0), 2.0),
        ((0.0, 0.0), (0.25, 0.25), float("inf")),
        ((0.0, 0.0), (0.75, 0.75), float("inf")),
    ],
)
def test_dijkstra_algorithm(
    square_edges, edge_source, edge_target, expected_dist
):
    """
    Test shortest path dijkstra algorithm. Distance/cost is infinite when
    there is no path between nodes.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # add edge to make the graph disconnected
    source, target = (0.25, 0.25), (0.75, 0.75)
    graph.add_edge(
        edge_source=source,
        edge_target=target,
        edge_weight=euclidean_distance(source, target),
    )

    # get the shortest path distances
    distances, _ = dijkstra_algorithm(
        graph=graph, edge_source=edge_source, edge_target=edge_target
    )

    # when there is no path between nodes, expected distance is inf.
    assert distances[edge_target] == expected_dist


def test_shortest_path_to_graph_class(square_edges):
    """
    Tests to get the shortest path between nodes.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # get the shortest path between nodes
    edge_source, edge_target = (0.0, 0.0), (1.0, 0.0)
    path = shortest_path_algorithm(
        graph=graph, edge_source=edge_source, edge_target=edge_target
    )

    # there is edge connecting the nodes, so the shortest path is formed by the nodes
    # themselves.
    assert path == [edge_source, edge_target]

    # removing the edge, the shortest path will be formed by all the points,
    # as it contains all other remaining edges.
    graph.remove_edge(
        edge_source=edge_source,
        edge_target=edge_target,
    )
    path = shortest_path_algorithm(
        graph=graph, edge_source=edge_source, edge_target=edge_target
    )
    assert path == [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]


def test_shortest_path_to_graph_class_assertion_error(square_edges):
    """
    Function throws assertion error in two cases: when there is no path in
    the graph connecting the two points or when one of the nodes (or both) are
    not in the graph.
    """
    # create instance of graph class
    graph = Graph()

    # define graph from edges
    for source, target in square_edges:
        graph.add_edge(
            edge_source=source,
            edge_target=target,
            edge_weight=euclidean_distance(source, target),
        )

    # add edge to make the graph disconnected
    edge_source, edge_target = (0.25, 0.25), (0.75, 0.75)
    graph.add_edge(
        edge_source=edge_source,
        edge_target=edge_target,
        edge_weight=euclidean_distance(edge_source, edge_target),
    )

    # Try to find the shortest path between nodes of one connected component
    # and another (does not exist)
    with pytest.raises(AssertionError, match="There is no path"):
        shortest_path_algorithm(
            graph=graph,
            edge_source=(0.0, 0.0),
            edge_target=edge_target,
        )

    # Try to find the shortest path between nodes that do not belong to the
    # graph (impossible)
    with pytest.raises(AssertionError, match="Impossible to find path"):
        shortest_path_algorithm(
            graph=graph,
            edge_source=(11.0, 11.0),
            edge_target=edge_target,
        )
