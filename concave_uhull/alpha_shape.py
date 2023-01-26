from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np

from concave_uhull.utils.geometry import (
    area_of_polygon,
    delaunay_triangulation,
    haversine_distance,
)
from concave_uhull.utils.graph import add_edge, remove_edge, shortest_path


def _get_alpha_triangulation(
    coordinates_points: List[Tuple],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List:
    """
    TODO

    References
    ----------
        [1] Tukey's fences, https://en.wikipedia.org/wiki/Outlier#Tukey's_fences
    """
    # get Delauney triangulation
    triangulation = delaunay_triangulation(coordinates_points)

    #
    lengths = []
    for p1, p2, p3 in triangulation:
        # get lengths of sides of triangle using the formula for the distance
        # between points on a Cartesian plane
        lengths.append(distance(p1, p2))
        lengths.append(distance(p2, p3))
        lengths.append(distance(p3, p1))

    # sets the Turkey fence to the given alpha
    q25, q75 = np.quantile(lengths, [0.25, 0.75])
    intr_qr = q75 - q25
    min_acceptable_length = q25 - (alpha * intr_qr)
    max_acceptable_length = q75 + (alpha * intr_qr)

    #
    def _is_alpha_triangule(triangule, min_length, max_length):
        """
        TODO
        """
        # get lengths of sides of triangle
        s1 = distance(triangule[0], triangule[1])
        s2 = distance(triangule[1], triangule[2])
        s3 = distance(triangule[2], triangule[0])
        return all([min_length < length < max_length for length in [s1, s2, s3]])

    #
    return list(
        filter(
            lambda triangule: _is_alpha_triangule(
                triangule, min_acceptable_length, max_acceptable_length
            ),
            triangulation,
        )
    )


def _get_alpha_shape_edges(
    coordinates_points: List[Tuple],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List[Tuple]:
    """
    TODO
    """
    #
    alpha_triangulation = _get_alpha_triangulation(coordinates_points, alpha, distance)

    #
    alpha_shape_edges_set = set()

    #
    def _save_edge(edges_saved, edge_source, edge_target):
        """
        TODO
        """
        edge = (edge_source, edge_target)
        edge_reversed = (edge_target, edge_source)

        if edge in edges_saved or edge_reversed in edges_saved:
            assert (
                edge_reversed in edges_saved
            ), "Can't go twice over same directed edge right?"
            edges_saved.remove(edge_reversed)
            return

        edges_saved.add(edge)

    #
    for p1, p2, p3 in alpha_triangulation:
        _save_edge(alpha_shape_edges_set, p1, p2)
        _save_edge(alpha_shape_edges_set, p2, p3)
        _save_edge(alpha_shape_edges_set, p3, p1)

    # list of indices of edge points that define the concave hull
    return list(alpha_shape_edges_set)


def alpha_shape_polygons(
    coordinates_points: List[Tuple],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List[List[Tuple]]:
    """
    TODO
    """
    #
    alpha_shape_edges = _get_alpha_shape_edges(
        coordinates_points=coordinates_points, alpha=alpha, distance=distance
    )

    #
    graph_adjacency_list = defaultdict(set)
    edge_weights = defaultdict(dict)

    #
    nodes_to_explore = set()
    for edge_source, edge_target in alpha_shape_edges:
        add_edge(
            graph_adjacency_list=graph_adjacency_list,
            edge_weights=edge_weights,
            edge_source=edge_source,
            edge_target=edge_target,
            weight_function=distance,
        )
        nodes_to_explore.add(edge_source)
        nodes_to_explore.add(edge_target)

    #
    alpha_shape_polygons_list = []
    while nodes_to_explore:

        #
        edge_source = nodes_to_explore.pop()
        edge_target = next(iter(graph_adjacency_list[edge_source]), None)

        #
        if edge_target is None:
            continue

        #
        remove_edge(
            graph_adjacency_list=graph_adjacency_list,
            edge_weights=edge_weights,
            edge_source=edge_source,
            edge_target=edge_target,
        )

        #
        polygon_vertices = shortest_path(
            graph_adjacency_list, edge_weights, edge_source, edge_target
        )
        for vertice in polygon_vertices:
            if vertice in nodes_to_explore:
                nodes_to_explore.remove(vertice)

        #
        polygon_vertices.append(edge_source)
        alpha_shape_polygons_list.append(polygon_vertices)

    #
    alpha_shape_polygons_list.sort(key=area_of_polygon, reverse=True)

    #
    return alpha_shape_polygons_list
