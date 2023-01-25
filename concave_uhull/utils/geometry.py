from typing import Tuple

import numpy as np


def euclidean_distance(coord1: Tuple, coord2: Tuple) -> float:
    """
    Calculate the Euclidean distance between coordinates

    References
    ----------
        [1] Euclidean distance, https://en.wikipedia.org/wiki/Euclidean_distance
    """
    return np.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])


def haversine_distance(coord1: Tuple, coord2: Tuple) -> float:
    """
    Calculate the Haversine distance between coordinates.

    The Haversine (or great circle) distance is the angular distance
    between two points on the surface of a sphere. The first coordinate of
    each point is assumed to be the longitude, the second is the latitude.

    Returns
    -------
        Haversine distance between coordinates in kilometers.

    References
    ----------
        [1] Haversine formula, https://en.wikipedia.org/wiki/Haversine_formula
    """
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    # radius of Earth in kilometers
    radius_earth = 6371000.0 / 1000.0

    # Haversine Formula
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.square(np.sin(delta_phi / 2.0)) + np.cos(phi_1) * np.cos(phi_2) * np.square(
        np.sin(delta_lambda / 2.0)
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    # output distance in kilometers
    return radius_earth * c
