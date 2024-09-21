"""Energy-distance test of two samples.

Name: "ed"
display_name="Energy-distance"
allowed_feature_types=["num"]

Import:

    >>> from stattests import energy_dist_test

Properties:
- only for numerical features
- returns p-value

"""
from typing import Tuple

import pandas as pd
from scipy.stats import energy_distance


def _energy_dist(
    reference_data: pd.Series, 
    current_data: pd.Series, 
    #feature_type: str, 
    threshold: float=0.1
) -> Tuple[float, bool]:
    
    """Run the energy_distance test of two samples.
    Args:
        reference_data: reference data
        current_data: current data
        threshold: all values above this threshold propose a data drift
    Returns:
        distance: energy distance
        test_result: whether the drift is detected
    """

    distance = energy_distance(reference_data, current_data)

    return distance, distance > threshold
