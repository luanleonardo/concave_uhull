from heapq import heappop, heappush

import numpy as np


def _add_edge(graph, weight, source, target):
    # assertions
    assert target not in graph[source], f"edge ({source}, {target}) already exists"
    assert source not in graph[target], f"edge ({target}, {source}) already exists"

    # add edge to graph
    graph[source].add(target)
    graph[target].add(source)

    # compute edge weight
    weight[source][target] = np.hypot(source[0] - target[0], source[1] - target[1])
    weight[target][source] = weight[source][target]


def _remove_edge(graph, weight, source, target):
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
    prec = {node: None for node in nodes}
    black = {node: False for node in nodes}
    dist = {node: float("inf") for node in nodes}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        dist_node, node = heappop(heap)
        if not black[node]:
            black[node] = True
            if node == target:
                break
            for neighbor in graph[node]:
                dist_neighbor = dist_node + weight[node][neighbor]
                if dist_neighbor < dist[neighbor]:
                    dist[neighbor] = dist_neighbor
                    prec[neighbor] = node
                    heappush(heap, (dist_neighbor, neighbor))
    return dist, prec


def shortest_path(graph, weight, source, target):
    dist, path_dict = _dijkstra(graph, weight, source, target)
    if len(path_dict) == 0:
        return []
    edge_path = [target]
    current_edge = path_dict[target]
    edge_path.append(current_edge)
    while current_edge != source:
        current_edge = path_dict[current_edge]
        edge_path.append(current_edge)
    return edge_path[::-1]
