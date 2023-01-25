import numpy as np

from alpha_shape.utils.geometry import euclidean_distance, haversine_distance


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
