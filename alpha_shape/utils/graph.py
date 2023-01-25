from collections import defaultdict
from heapq import heappop, heappush

from alpha_shape.utils.geometry import haversine_distance


def add_edge(
    graph_adjacency_list: defaultdict,
    edge_weights: defaultdict,
    edge_source: tuple,
    edge_target: tuple,
    weight_function=haversine_distance,
):
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
    edge_source: tuple,
    edge_target: tuple,
):
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
    distances, predecessors = _dijkstra(
        graph_adjacency_list, edge_weights, edge_source, edge_target
    )
    if len(predecessors) == 0:
        return []
    edge_path = [edge_target]
    current_edge = predecessors[edge_target]
    edge_path.append(current_edge)
    while current_edge != edge_source:
        current_edge = predecessors[current_edge]
        edge_path.append(current_edge)
    return edge_path[::-1]
