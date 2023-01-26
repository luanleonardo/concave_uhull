import numpy as np
import pytest

from concave_uhull.utils.geometry import (
    area_of_polygon,
    delaunay_triangulation,
    euclidean_distance,
    haversine_distance,
)


@pytest.fixture
def coordinates_points():
    return [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]


def test_euclidean_distance():
    """
    Calculate the Euclidean distance between coordinates

    References
    ----------
    [1] Euclidean distance, https://en.wikipedia.org/wiki/Euclidean_distance
    """
    x = (4, 0)
    y = (0, 3)

    assert euclidean_distance(x, y) == 5


def test_haversine_distance():
    """
    Calculate the distance between the Ezeiza Airport (Buenos
    Aires, Argentina) and the Charles de Gaulle Airport (Paris, France)

    References
    ----------
    [1] Haversine distance, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    """
    bsas = (-58.5166646, -34.83333)
    paris = (2.53844117956, 49.0083899664)
    expected_result_in_km = 11099.54035582
    result_in_km = haversine_distance(bsas, paris)

    assert np.isclose(result_in_km, expected_result_in_km, atol=1e-4)


def test_delaunay_triangulation(coordinates_points):
    """TODO"""
    # get Delauney triangulation,
    triangulation = delaunay_triangulation(coordinates_points)

    # with points of a square it is only possible to obtain two triangles
    assert len(triangulation) == 2


def test_area_of_polygon(coordinates_points):
    """TODO"""
    # define square polygon vertices
    square_polygon_vertices = coordinates_points + [(0.0, 0.0)]

    # compute area of square polygon
    square_area = area_of_polygon(square_polygon_vertices)

    # with points of a square it is only possible to obtain two triangles
    assert square_area == 1.0
