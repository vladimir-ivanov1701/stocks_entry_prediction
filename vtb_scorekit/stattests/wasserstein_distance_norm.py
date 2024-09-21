"""Wasserstein distance of two samples.

Name: "wasserstein"
display_name="Wasserstein distance (normed)"
allowed_feature_types=["num"]

Import:

    >>> from stattests import wasserstein_stat_test

Properties:
- only for numerical features
- returns p-value

"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _wasserstein_distance_norm(
    reference_data: pd.Series, 
    current_data: pd.Series, 
    #feature_type: str, 
    threshold: float=0.1
) -> Tuple[float, bool]:
    
    """Compute the first Wasserstein distance between two arrays normed by std of reference data
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: all values above this threshold means data drift
    Returns:
        wasserstein_distance_norm: normed Wasserstein distance
        test_result: whether the drift is detected
    """
    norm = max(np.std(reference_data), 0.001)
    wd_norm_value = stats.wasserstein_distance(reference_data, current_data) / norm

    return wd_norm_value, wd_norm_value >= threshold
