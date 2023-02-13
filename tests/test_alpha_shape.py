import numpy as np
import pytest

from concave_uhull.alpha_shape import get_alpha_shape_polygons
from concave_uhull.geometry import area_of_polygon, euclidean_distance


@pytest.fixture
def coordinates_square_set():
    """Coordinates of a set of 5k points, similar to a square of side 4."""
    np.random.seed(0)
    x = 4 * np.random.rand(5000)
    y = 4 * np.random.rand(5000)
    return list(zip(x, y))


@pytest.fixture
def circular_crown_set(coordinates_square_set):
    """Coordinates of a circular crown like set, formed from the difference between
    concentric circles in (2.0, 2.0) of areas 2 * pi and pi."""
    # center of circles
    center_circles = (2.0, 2.0)

    # check if the point is on the circular crown
    def _is_circular_crown_point(point, center):
        return 1.0 < euclidean_distance(point, center) < np.sqrt(2.0)

    # returns points on the circular crown
    return list(
        filter(
            lambda point: _is_circular_crown_point(point, center_circles),
            coordinates_square_set,
        )
    )


def tests_get_alpha_shape_polygons_in_square_set(coordinates_square_set):
    """Test get alpha shapes polygons in the set similar to a square of side 4."""
    # get alpha shape polygons of the square set
    polygons = get_alpha_shape_polygons(
        coordinates_square_set, distance=euclidean_distance
    )

    # at least one alpha form must be returned
    assert len(polygons) > 0

    # the largest area alpha shape should have an area close to that of a
    # square of side 4
    largest_area_polygon = polygons[0]
    assert np.isclose(area_of_polygon(largest_area_polygon), 16.0, atol=0.5)


def tests_get_alpha_shape_polygons_in_circular_crown_set(circular_crown_set):
    """Test get alpha shape polygons in the circular crown set."""
    # get alpha shape polygons from circular crown set
    polygons = get_alpha_shape_polygons(
        circular_crown_set, distance=euclidean_distance
    )

    # at least two alpha shapes must be returned, one for the outermost points
    # (similar to the circle with the largest area 2pi) and another shape for the
    # innermost points (similar to the circle with the smallest area pi).
    assert len(polygons) >= 2

    # the largest-area alpha shape must have an area less than 2pi (area of the
    # largest circle) and greater than pi (area of the smallest circle).
    largest_area_polygon = polygons[0]
    largest_area = area_of_polygon(largest_area_polygon)
    assert np.pi < largest_area < 2 * np.pi

    # the second largest-area alpha shape must have an area smaller than the first
    # (obvious) and greater than pi (area of the smaller circle).
    second_largest_area_polygon = polygons[1]
    second_largest_area = area_of_polygon(second_largest_area_polygon)
    assert np.pi < second_largest_area < largest_area
