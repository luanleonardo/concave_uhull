from typing import Callable, Dict, List, Set, Tuple

import numpy as np

from concave_uhull.utils.geometry import (
    area_of_polygon,
    delaunay_triangulation,
    haversine_distance,
)
from concave_uhull.utils.graph import Graph, shortest_path_algorithm


def _get_alpha_triangulation(
    coordinates_points: List[Tuple],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List[Tuple]:
    """
    Provides an alpha triangulation of the coordinates points given. The triangulation has
    the following property: the lengths of the sides of each triangle are within a special
    interval, called the Turkey fence, whose 'width' of the interval is determined using
    the given alpha parameter. The length of each side of the triangles is calculated
    using the given distance function.

    Parameters
    ----------
    coordinates_points
        List of point coordinates. Coordinates are represented by tuples of two numerical values.

    alpha
        Float value responsible for determining the 'width' of Tukey's fence.

    distance
        Function that receives two tuples of coordinates of vertices and obtains a
        measure of distance between the vertices. By default, we use the Haversine
        distance function, as we assume that the coordinates of the vertices are of
        the form (lng, lat).

    Returns
    -------
    List[Tuple]
        A list of alpha triangle vertices coordinates. A triangle is considered to be
        alpha if the length of all its sides is within the Tukey fence of 'width'
        determined by alpha.

    References
    ----------
    .. [1] Tukey's fences, https://en.wikipedia.org/wiki/Outlier#Tukey's_fences
    .. [2] Identifying Outliers: IQR Method, https://online.stat.psu.edu/stat200/lesson/3/3.2
    .. [3] Why “1.5” in IQR Method of Outlier Detection?,
    https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097

    Notes
    -----
    The function performs the following steps to obtain an alpha triangulation:
        1. Get Delauney triangulation;
        2. Get triangle information, such as vertex coordinates and side lengths;
        3. Get the Tukey's fence for the given alpha (a.k.a alpha fence);
        4. Return only alpha triangles.
    """
    # Step 1: get Delauney triangulation;
    triangulation = delaunay_triangulation(coordinates_points)

    # Step 2: get triangle information, such as vertex coordinates and side
    # lengths;
    triangulation_info: Dict = {
        "lengths_list": [],
        "coordinates_and_lengths": [],
    }
    for coordinates in triangulation:
        # coordinates of each vertex of the triangle
        p1, p2, p3 = coordinates

        # lengths of triangle sides
        lengths = distance(p1, p2), distance(p2, p3), distance(p3, p1)

        # save triangle information
        triangulation_info["lengths_list"].extend(lengths)
        triangulation_info["coordinates_and_lengths"].append(
            (coordinates, lengths)
        )

    # Step 3: get the Tukey's fence for the given alpha (a.k.a alpha fence);
    q25, q75 = np.quantile(triangulation_info["lengths_list"], [0.25, 0.75])
    intr_qr = q75 - q25
    min_acceptable_length = q25 - (alpha * intr_qr)
    max_acceptable_length = q75 + (alpha * intr_qr)

    # Step 4: return only alpha triangles, that is, triangles whose side lengths are
    # inside the alpha fence.
    def _is_alpha_triangule(triangle_sides_length):
        """
        The triangle is considered alpha, if the length of all its
        sides are within the alpha fence.
        """
        return all(
            min_acceptable_length < length < max_acceptable_length
            for length in triangle_sides_length
        )

    return [
        coordinates
        for coordinates, lengths in triangulation_info[
            "coordinates_and_lengths"
        ]
        if _is_alpha_triangule(lengths)
    ]


def _get_alpha_shape_edges(
    coordinates_points: List[Tuple],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List[Tuple]:
    """
    Gets a list of the boundary edges of each alpha triangle, in an alpha triangulation of
    the given point coordinates. Edges are represented by tuples of tuples, which represent
    the coordinates of each extreme vertex of the edge, from the source vertice to the
    destination (target) vertice.

    Parameters
    ----------
    coordinates_points
        List of point coordinates. Coordinates are represented by tuples of two numerical values.

    alpha
        Float value responsible for determining the 'width' of Tukey's fence.

    distance
        Function that receives two tuples of coordinates of vertices and obtains a
        measure of distance between the vertices. By default, we use the Haversine
        distance function, as we assume that the coordinates of the vertices are of
        the form (lng, lat).

    Returns
    -------
    List[Tuple]
        A list of the boundary edges of each alpha triangle, in an alpha triangulation of
        the given point coordinates. Edges are represented by tuples of tuples, which represent
        the coordinates of each extreme vertex of the edge, from the source vertice to the
        destination (target) vertice

    See Also
    --------
    _get_alpha_triangulation : Provides an alpha triangulation of the coordinates points given.

    Notes
    -----
    The function performs the following steps to obtain the edges:
        1. Get alpha triangulation of the coordinates points given;
        2. returns only the boundary edges of each alpha triangle from the obtained
        alpha triangulation for the given set of point coordinates.
    """
    # Step 1: get alpha triangulation;
    alpha_triangulation = _get_alpha_triangulation(
        coordinates_points=coordinates_points, alpha=alpha, distance=distance
    )

    # Step 2: returns only the boundary edges of each alpha triangle from the
    # obtained alpha triangulation for the given set of point coordinates.
    def _save_boundary_edges(edges_saved, edge_source, edge_target):
        """
        Saves only boundary edges.

        Notes
        -----
        Edges that are not boundaries will be shared by two triangles, as both
        have the same orientation it is guaranteed that we will pass through
        these edges in both directions. To identify a non-boundary edge, it is
        sufficient to check whether the edge or its reverse has been saved in
        the set and, if so, remove it. Following these steps, only the border
        edges will remain in the set.
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

    alpha_shape_edges_set: Set[Tuple] = set()
    for p1, p2, p3 in alpha_triangulation:
        _save_boundary_edges(alpha_shape_edges_set, p1, p2)
        _save_boundary_edges(alpha_shape_edges_set, p2, p3)
        _save_boundary_edges(alpha_shape_edges_set, p3, p1)
    return list(alpha_shape_edges_set)


def get_alpha_shape_polygons(
    coordinates_points: List[Tuple],
    alpha: float = 1.5,
    distance: Callable = haversine_distance,
) -> List[List[Tuple]]:
    """
    Provides a list of polygons, sorted in descending order by their areas, representing the
    concave hull of the given set of coordinates. The implemented algorithm uses a strategy
    based on the alpha shape algorithm, which is obtained from a special triangulation of the
    set of coordinates. This triangulation is strongly influenced by the value of the alpha
    parameter and the given distance function.

    Parameters
    ----------
    coordinates_points
        List of point coordinates. Coordinates are represented by tuples of two numerical values.

    alpha
        Float value responsible for determining the 'width' of Tukey's fence.

    distance
        Function that receives two tuples of coordinates of vertices and obtains a
        measure of distance between the vertices. By default, we use the Haversine
        distance function, as we assume that the coordinates of the vertices are of
        the form (lng, lat).

    Returns
    -------
    List[List[Tuple]]
        Returns list of alpha shape polygons in descending order by polygon area.

    See Also
    --------
    _get_alpha_shape_edges : Gets a list of the boundary edges of each alpha triangle, in an
        alpha triangulation of the given point coordinates.

    Notes
    -----
    The function performs the following steps to obtain the alpha shape polygon list:

        1. Gets a list of the boundary edges of each alpha triangle, in an alpha
        triangulation of the given point coordinates.

        2. Defines an undirected graph, induced by the boundary alpha vertices and
        non-negative edge weights computed with the distance function.

        3. Create alpha shape polygon list with following substeps:
            3.1 A random edge is selected, its extreme points memorized and the edge
            removed from the graph.

            3.2 The shortest path from one memorized extreme point to the other
            is obtained. With this path, we form a polygon of the alpha shape by adding
            the first point to the end of the path.

            3.3 After that all waypoints are removed from the set of points to be
            explored. And then add the obtained polygon to the polygon list of the
            alpha shape.

        4. Returns list of alpha shape polygons in descending order by polygon area.

    References
    ----------
    .. [1] D. Kalinina et. al., "Computing concave hull with closed curve smoothing:
    performance, concaveness measure and applications",
    https://doi.org/10.1016/j.procs.2018.08.258
    .. [2] D. Kalinina et. al., "Concave Hull GitHub repository.",
    https://github.com/dkalinina/Concave_Hull.
    """
    # Step 1: Gets a list of the boundary edges of each alpha triangle, in an alpha
    # triangulation of the given point coordinates.
    alpha_shape_edges = _get_alpha_shape_edges(
        coordinates_points=coordinates_points, alpha=alpha, distance=distance
    )

    # Step 2: Defines an undirected graph, induced by the boundary alpha vertices and
    # non-negative edge weights computed with the distance function.
    graph = Graph()
    nodes_to_explore = set()
    for edge_source, edge_target in alpha_shape_edges:
        graph.add_edge(
            edge_source=edge_source,
            edge_target=edge_target,
            edge_weight=distance(edge_source, edge_target),
        )
        nodes_to_explore.add(edge_source)
        nodes_to_explore.add(edge_target)

    # Step 3: Create alpha shape polygon list with following substeps:
    alpha_shape_polygons_list = []
    while nodes_to_explore:

        # Step 3.1: A random edge is selected, its extreme points memorized and
        # the edge removed from the graph.
        edge_source = nodes_to_explore.pop()
        edge_target = next(iter(graph[edge_source]), None)
        if edge_target is None:
            continue
        graph.remove_edge(
            edge_source=edge_source,
            edge_target=edge_target,
        )

        # Step 3.2: The shortest path from one memorized extreme point to the
        # other is obtained. With this path, we form a polygon of the alpha shape
        # by adding the first point to the end of the path.
        polygon_vertices = shortest_path_algorithm(
            graph=graph, edge_source=edge_source, edge_target=edge_target
        )
        polygon_vertices.append(edge_source)

        # Step 3.3:  After that all waypoints are removed from the set of points
        # to be explored. And then add the obtained polygon to the polygon list of
        # the alpha shape.
        for vertice in polygon_vertices:
            if vertice in nodes_to_explore:
                nodes_to_explore.remove(vertice)
        alpha_shape_polygons_list.append(polygon_vertices)

    # Step 4: Returns list of alpha shape polygons in descending order by
    # polygon area.
    alpha_shape_polygons_list.sort(key=area_of_polygon, reverse=True)
    return alpha_shape_polygons_list
