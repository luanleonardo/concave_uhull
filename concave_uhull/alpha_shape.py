from typing import Callable, List, Tuple

import numpy as np

from concave_uhull.utils.geometry import haversine_distance


def _get_length_acceptance_range(
    points_coordinates: List[Tuple[float, float]],
    points_delaunay_triangulation: List[List[float]],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> Tuple[float, float]:
    """TODO

    References
    ----------
        [1] Tukey's fences, https://en.wikipedia.org/wiki/Outlier#Tukey's_fences
    """
    lengths = []
    # i1, i2, i3 = indices of corner points of the triangle
    for i1, i2, i3 in points_delaunay_triangulation:

        # get the coordinates of the points that form the triangle
        p1 = points_coordinates[i1]
        p2 = points_coordinates[i2]
        p3 = points_coordinates[i3]

        # get lengths of sides of triangle using the formula for the distance
        # between points on a Cartesian plane
        lengths.append(distance(p1, p2))
        lengths.append(distance(p2, p3))
        lengths.append(distance(p3, p1))

    # sets the Turkey fence to the given alpha
    q25, q75 = np.quantile(lengths, [0.25, 0.75])
    intr_qr = q75 - q25
    min_lenght = q25 - (alpha * intr_qr)
    max_lenght = q75 + (alpha * intr_qr)

    # length acceptance range as tuple
    return min_lenght, max_lenght


def _get_alpha_shape_edges_from_triangulation(
    points_coordinates: List[Tuple[float, float]],
    points_delaunay_triangulation: List[List[float]],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List[Tuple[int, int]]:
    """
    TODO
    """
    min_lenght, max_lenght = _get_length_acceptance_range(
        points_coordinates, points_delaunay_triangulation, alpha, distance
    )

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points, if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            edges.remove((j, i))
            return
        edges.add((i, j))

    # indices of edge points that define the concave hull
    alpha_shape_edges = set()

    # i1, i2, i3 = indices of corner points of the triangle
    for i1, i2, i3 in points_delaunay_triangulation:

        # get the coordinates of the points that form the triangle
        p1 = points_coordinates[i1]
        p2 = points_coordinates[i2]
        p3 = points_coordinates[i3]

        # get lengths of sides of triangle using the formula for the distance
        # between points on a Cartesian plane
        d1 = distance(p1, p2)
        d2 = distance(p2, p3)
        d3 = distance(p3, p1)

        # ignore triangles outside the range of acceptable lengths
        if any([length < min_lenght or max_lenght < length for length in [d1, d2, d3]]):
            continue

        # saves indices of points of acceptable triangles, but only outermost edges
        add_edge(alpha_shape_edges, i1, i2)
        add_edge(alpha_shape_edges, i2, i3)
        add_edge(alpha_shape_edges, i3, i1)

    # list of indices of edge points that define the concave hull
    return list(alpha_shape_edges)
