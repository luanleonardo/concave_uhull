from collections import defaultdict
from heapq import heappop, heappush
from typing import Callable, Dict, List, Tuple

from concave_uhull.utils.geometry import haversine_distance


def add_edge(
    graph_adjacency_list: defaultdict,
    edge_weights: defaultdict,
    edge_source: Tuple,
    edge_target: Tuple,
    weight_function: Callable = haversine_distance,
) -> None:
    """
    Adds nodes and edges to the undirected graph's adjacency list, as well as
    calculates the weight of the added edge.

    Parameters
    ----------
    graph_adjacency_list
        A defaultdict(set) representing the adjacency list of the undirected graph.
    edge_weights
        A defaultdict(dict) representing the edge weight matrix.
    edge_source
        Tuple with the coordinates of the source of the edge.
    edge_target
        Tuple with the coordinates of the target of the edge.
    weight_function
        Function that receives two tuples of node coordinates and obtains a
        weight for the edge formed by the nodes. By default, we use the
        Haversine distance function, as we assume that the coordinates of the
        nodes are of the form (lng, lat).

    Returns
    -------
    None
        Returns None

    Raises
    ------
    AssertionError
        If the edge already exists in the adjacency list of the undirected graph.
    """
    # assertions
    assert (
        edge_target not in graph_adjacency_list[edge_source]
    ), f"Edge ({edge_source}, {edge_target}) already exists"
    assert (
        edge_source not in graph_adjacency_list[edge_target]
    ), f"Edge ({edge_target}, {edge_source}) already exists"

    # add edge to graph
    graph_adjacency_list[edge_source].add(edge_target)
    graph_adjacency_list[edge_target].add(edge_source)

    # compute edge weight
    edge_weights[edge_source][edge_target] = weight_function(edge_source, edge_target)
    edge_weights[edge_target][edge_source] = edge_weights[edge_source][edge_target]


def remove_edge(
    graph_adjacency_list: defaultdict,
    edge_weights: defaultdict,
    edge_source: Tuple,
    edge_target: Tuple,
) -> None:
    """
    Remove edge from undirected graph adjacency list. In addition, it removes
    the cost associated with the edge removed from the cost matrix.

    Parameters
    ----------
    graph_adjacency_list
        A defaultdict(set) representing the adjacency list of the undirected graph.
    edge_weights
        A defaultdict(dict) representing the edge weight matrix.
    edge_source
        Tuple with the coordinates of the source of the edge.
    edge_target
        Tuple with the coordinates of the target of the edge.

    Returns
    -------
    None
        Returns None

    Raises
    ------
    AssertionError
        If the edge does not exist in the adjacency list of the undirected graph.
    """
    # assertions
    assert (
        edge_target in graph_adjacency_list[edge_source]
    ), f"No edge ({edge_source}, {edge_target}) to remove"
    assert (
        edge_source in graph_adjacency_list[edge_target]
    ), f"No edge ({edge_target}, {edge_source}) to remove"

    # remove edge from graph
    graph_adjacency_list[edge_source].remove(edge_target)
    graph_adjacency_list[edge_target].remove(edge_source)

    # delete edge weight
    del edge_weights[edge_source][edge_target]
    del edge_weights[edge_target][edge_source]


def _dijkstra(
    graph_adjacency_list: defaultdict,
    edge_weights: defaultdict,
    edge_source: Tuple,
    edge_target: Tuple,
) -> Tuple[Dict, Dict]:
    """
    Dijkstra's algorithm for the shortest path problem between a single source
    and all destinations with edges of non-negative weights. The funtions allows
    the computation of the shortest paths to each and every destination, if a
    particular destination is not specified when the function is invoked.

    Parameters
    ----------
    graph_adjacency_list
        A defaultdict(set) representing the adjacency list of the undirected graph.
    edge_weights
        A defaultdict(dict) representing the edge weight matrix.
    edge_source
        Tuple with the coordinates of the source of the edge.
    edge_target
        Tuple with the coordinates of the target of the edge.

    Returns
    -------
    Tuple[Dict, Dict]
        distances:
            Dictionary where each key represents a destination node and the value represents
            the shortest path distance/cost between the source node and the key node.

        predecessors:
            Dictionary where each key represents a target node and the value represents the
            predecessor node on the shortest path between the source node and the key node.
    """
    nodes = graph_adjacency_list.keys()
    predecessors = {node: None for node in nodes}
    visited = {node: False for node in nodes}
    distances = {node: float("inf") for node in nodes}
    distances[edge_source] = 0
    heap = [(0, edge_source)]
    while heap:
        distance_node, node = heappop(heap)
        if not visited[node]:
            visited[node] = True
            if node == edge_target:
                break
            for neighbor in graph_adjacency_list[node]:
                distance_neighbor = distance_node + edge_weights[node][neighbor]
                if distance_neighbor < distances[neighbor]:
                    distances[neighbor] = distance_neighbor
                    predecessors[neighbor] = node
                    heappush(heap, (distance_neighbor, neighbor))
    return distances, predecessors


def shortest_path(
    graph_adjacency_list: defaultdict,
    edge_weights: defaultdict,
    edge_source: Tuple,
    edge_target: Tuple,
) -> List[Tuple]:
    """
    It uses Dijkstra's algorithm to obtain the shortest path between the source node
    and the destination node. The obtained path is represented by a list of coordinates
    of the nodes, where the first coordinate of the list is the origin node and the
    last coordinate of the list is the destination node.

    Parameters
    ----------
    graph_adjacency_list
        A defaultdict(set) representing the adjacency list of the undirected graph.
    edge_weights
        A defaultdict(dict) representing the edge weight matrix.
    edge_source
        Tuple with the coordinates of the source of the edge.
    edge_target
        Tuple with the coordinates of the target of the edge.

    Returns
    -------
    List[Tuple]
        A list of coordinates of the nodes, where the first coordinate of the list is
        the origin node and the last coordinate of the list is the destination node.

    Raises
    ------
    AssertionError
        If the source node or destination node does not belong to the graph.
        If there is no path between source node and destination node.
    """
    # assertion about both nodes belong to the graph
    assert (
        edge_source in graph_adjacency_list and edge_target in graph_adjacency_list
    ), "Impossible to find path between nodes that do not belong to the graph"

    # get path cost and predecessor nodes using dijkstra's algorithm
    distances, predecessors = _dijkstra(
        graph_adjacency_list, edge_weights, edge_source, edge_target
    )

    # assertion about no path connecting the nodes
    assert distances[edge_target] != float(
        "inf"
    ), f"There is no path connecting node {edge_source} to node {edge_target}"

    # get the shortest path
    path = [edge_target]
    current_edge = predecessors[edge_target]
    path.append(current_edge)
    while current_edge != edge_source:
        current_edge = predecessors[current_edge]
        path.append(current_edge)

    return path[::-1]
