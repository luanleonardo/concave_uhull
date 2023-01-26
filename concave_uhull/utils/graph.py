from collections import defaultdict
from heapq import heappop, heappush
from typing import Callable, Tuple

from concave_uhull.utils.geometry import haversine_distance


def add_edge(
    graph_adjacency_list: defaultdict,
    edge_weights: defaultdict,
    edge_source: Tuple,
    edge_target: Tuple,
    weight_function: Callable = haversine_distance,
):
    """
    TODO
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
):
    """
    TODO
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


def _dijkstra(graph_adjacency_list, edge_weights, edge_source, edge_target):
    """
    TODO
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


def shortest_path(graph_adjacency_list, edge_weights, edge_source, edge_target):
    """
    TODO
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
