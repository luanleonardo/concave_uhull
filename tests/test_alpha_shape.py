import numpy as np
import pytest

from concave_uhull.alpha_shape import alpha_shape_polygons
from concave_uhull.utils.geometry import area_of_polygon, euclidean_distance


@pytest.fixture
def coordinates_square_set():
    """Coordendas de um conjunto de 5k pontos, similar a um quadrado de lado 4."""
    np.random.seed(0)
    x = 4 * np.random.rand(5000)
    y = 4 * np.random.rand(5000)
    return list(zip(x, y))


@pytest.fixture
def circular_crown_set(coordinates_square_set):
    """Coordendas de um conjunto similar a uma coroa circular, formada da diferença
    entre círculos concêntricos em (2.0, 2.0) de áreas 2 * pi e pi."""
    # centro dos circulos
    center_circles = (2.0, 2.0)

    # checa se ponto esta na coroa circular
    def _is_circular_crown_point(point, center):
        return 1.0 < euclidean_distance(point, center) < np.sqrt(2.0)

    # retorna pontos na coroa circular
    return list(
        filter(
            lambda point: _is_circular_crown_point(point, center_circles),
            coordinates_square_set,
        )
    )


def tests_alpha_shape_polygons_in_square_set(coordinates_square_set):
    """Testa formas alphas no conjunto similar a um quadrado de lado 4."""
    # obtem as formas alpha do conjunto quadrado
    polygons = alpha_shape_polygons(coordinates_square_set, distance=euclidean_distance)

    # ao menos uma forma alpha deve ser retornada
    assert len(polygons) > 0

    # a forma alpha de maior área deve ter área próxima a de um quadrado de lado 4.
    largest_area_polygon = polygons[0]
    assert np.isclose(area_of_polygon(largest_area_polygon), 16.0, atol=0.5)


def tests_alpha_shape_polygons_in_circular_crown_set(circular_crown_set):
    """Testa formas alphas no conjunto similar a coroa circular."""
    # obtem as formas alpha do conjunto quadrado
    polygons = alpha_shape_polygons(circular_crown_set, distance=euclidean_distance)

    # ao menos duas formas alpha devem ser retornadas, uma para os pontos mais externos
    # (similar ao círculo de maior área 2pi) e outra forma para os pontos mais internos
    # (similar ao círculo de menor área pi).
    assert len(polygons) >= 2

    # a forma alpha de maior área deve área menor do que 2pi (área do maior círculo)
    # e maior do que pi (área do círculo menor).
    largest_area_polygon = polygons[0]
    largest_area = area_of_polygon(largest_area_polygon)
    assert np.pi < largest_area < 2 * np.pi

    # a segunda forma alpha de maior área deve ter área menor que a primeira (óbvio)
    # e maior do que pi (área do círculo menor).
    second_largest_area_polygon = polygons[1]
    second_largest_area = area_of_polygon(second_largest_area_polygon)
    assert np.pi < second_largest_area < largest_area
