from collections import defaultdict
from heapq import heappop, heappush

from alpha_shape.utils.geometry import haversine_distance


def add_edge(
    graph: defaultdict,
    weight: defaultdict,
    source: tuple,
    target: tuple,
    distance=haversine_distance,
):
    # assertions
    assert target not in graph[source], f"edge ({source}, {target}) already exists"
    assert source not in graph[target], f"edge ({target}, {source}) already exists"

    # add edge to graph
    graph[source].add(target)
    graph[target].add(source)

    # compute edge weight
    weight[source][target] = distance(source, target)
    weight[target][source] = weight[source][target]


def remove_edge(graph: defaultdict, weight: defaultdict, source: tuple, target: tuple):
    # assertions
    assert target in graph[source], f"no edge ({source}, {target}) to remove"
    assert source in graph[target], f"no edge ({target}, {source}) to remove"

    # remove edge from graph
    graph[source].remove(target)
    graph[target].remove(source)

    # delete edge weight
    del weight[source][target]
    del weight[target][source]


def _dijkstra(graph, weight, source, target):
    nodes = graph.keys()
    predecessors = {node: None for node in nodes}
    visited = {node: False for node in nodes}
    distances = {node: float("inf") for node in nodes}
    distances[source] = 0
    heap = [(0, source)]
    while heap:
        distance_node, node = heappop(heap)
        if not visited[node]:
            visited[node] = True
            if node == target:
                break
            for neighbor in graph[node]:
                distance_neighbor = distance_node + weight[node][neighbor]
                if distance_neighbor < distances[neighbor]:
                    distances[neighbor] = distance_neighbor
                    predecessors[neighbor] = node
                    heappush(heap, (distance_neighbor, neighbor))
    return distances, predecessors


def shortest_path(graph, weight, source, target):
    distances, predecessors = _dijkstra(graph, weight, source, target)
    if len(predecessors) == 0:
        return []
    edge_path = [target]
    current_edge = predecessors[target]
    edge_path.append(current_edge)
    while current_edge != source:
        current_edge = predecessors[current_edge]
        edge_path.append(current_edge)
    return edge_path[::-1]
