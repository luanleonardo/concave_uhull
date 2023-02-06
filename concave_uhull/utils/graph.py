from collections import defaultdict
from heapq import heappop, heappush
from typing import Dict, List, Tuple


class Graph:
    """
    A utility class for edge-induced graph structure, supports edge addition
    and removal operations.
    """

    def __init__(self):
        self.adjacency_set = defaultdict(set)
        self.weight = defaultdict(dict)

    def __getitem__(self, node: Tuple) -> set:
        return self.adjacency_set[node]

    def __len__(self) -> int:
        return len(self.adjacency_set)

    @property
    def nodes(self) -> set:
        return set(self.adjacency_set.keys())

    def add_edge(
        self, edge_source: Tuple, edge_target: Tuple, edge_weight: float
    ) -> None:
        """
        Adds nodes and edges to the undirected graph's adjacency set, as well as
        calculates the weight of the added edge.

        Parameters
        ----------
        edge_source
            Tuple with the coordinates of the source of the edge.
        edge_target
            Tuple with the coordinates of the target of the edge.
        edge_weight
            A weight for the edge formed by the nodes.

        Returns
        -------
        None
            Returns None

        Raises
        ------
        AssertionError
            If the edge already exists in the adjacency list of the undirected graph.
        """
        # assertions about edge existence
        assert (
            edge_target not in self.adjacency_set[edge_source]
        ), f"Edge ({edge_source}, {edge_target}) already exists"
        assert (
            edge_source not in self.adjacency_set[edge_target]
        ), f"Edge ({edge_target}, {edge_source}) already exists"

        # add edge to graph
        self.adjacency_set[edge_source].add(edge_target)
        self.adjacency_set[edge_target].add(edge_source)

        # update edge weight
        self.weight[edge_source][edge_target] = edge_weight
        self.weight[edge_target][edge_source] = edge_weight

    def remove_edge(
        self,
        edge_source: Tuple,
        edge_target: Tuple,
    ) -> None:
        """
        Remove edge from undirected graph adjacency set. In addition, it removes
        the cost associated with the edge removed from the cost matrix.

        Parameters
        ----------
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
            edge_target in self.adjacency_set[edge_source]
        ), f"No edge ({edge_source}, {edge_target}) to remove"
        assert (
            edge_source in self.adjacency_set[edge_target]
        ), f"No edge ({edge_target}, {edge_source}) to remove"

        # remove edge from graph
        self.adjacency_set[edge_source].remove(edge_target)
        self.adjacency_set[edge_target].remove(edge_source)

        # delete edge weight
        del self.weight[edge_source][edge_target]
        del self.weight[edge_target][edge_source]


def dijkstra_algorithm(
    graph: Graph,
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
    graph
        An instance of the Graph class, an undirected weighted graph represented
        by adjacency set.
    edge_source
        Tuple with the coordinates of the source of the edge.
    edge_target
        Tuple with the coordinates of the target of the edge.

    Returns
    -------
    Tuple[Dict, Dict]
        distance:
            Dictionary where each key represents a destination node and the value represents
            the shortest path distance/cost between the source node and the key node.
        predecessors:
            Dictionary where each key represents a target node and the value represents the
            predecessor node on the shortest path between the source node and the key node.
    """
    explored = set()
    distance = {
        node: float("inf") if node != edge_source else 0.0
        for node in graph.nodes
    }
    heap = [(distance[edge_source], edge_source)]
    predecessors = dict()
    while heap:
        distance_node, node = heappop(heap)
        if node == edge_target:
            break
        explored.add(node)
        for neighbor in graph[node]:
            distance_neighbor = distance_node + graph.weight[node][neighbor]
            if distance_neighbor < distance[neighbor]:
                distance[neighbor] = distance_neighbor
                heappush(heap, (distance[neighbor], neighbor))
                predecessors[neighbor] = node
    return distance, predecessors


def shortest_path_algorithm(
    graph: Graph,
    edge_source: Tuple,
    edge_target: Tuple,
) -> List[Tuple]:
    """
    It uses Dijkstra's algorithm to obtain the shortest path between the source node
    and the destination node. The obtained path is represented by a list of coordinates
    of the nodes, where the first coordinate of the list is the source node and the
    last coordinate of the list is the target node.

    Parameters
    ----------
    graph
        An instance of the Graph class, an undirected weighted graph represented
        by adjacency set.
    edge_source
        Tuple with the coordinates of the source of the edge.
    edge_target
        Tuple with the coordinates of the target of the edge.

    Returns
    -------
    List[Tuple]
        A list of coordinates of the nodes, where the first coordinate of the list is
        the source node and the last coordinate of the list is the target node.

    Raises
    ------
    AssertionError
        If the source node or destination node does not belong to the graph.
        If there is no path between source node and destination node.
    """
    # assertion about both nodes belong to the graph
    assertion_msg = (
        "Impossible to find path between nodes that do not belong to the graph"
    )
    assert edge_source in graph.nodes, assertion_msg
    assert edge_target in graph.nodes, assertion_msg

    # get path cost and predecessor nodes using dijkstra's algorithm
    distances, predecessors = dijkstra_algorithm(
        graph=graph, edge_source=edge_source, edge_target=edge_target
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
