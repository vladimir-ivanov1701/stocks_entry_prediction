"""Kullback-Leibler divergence of two samples.

Name: "kl_div"
display_name="Kullback-Leibler divergence"
allowed_feature_types=["cat", "num"]

Import:

    >>> from stattests import kl_div_stat_test

Properties:
- only for categorical and numerical features
- returns divergence

"""
from typing import Tuple

import pandas as pd
from scipy import stats

from .utils import get_binned_data


def _kl_div(
    reference_data: pd.Series, 
    current_data: pd.Series, 
    feature_type: str='num',
    threshold: float=0.1, 
    n_bins: int = 30
) -> Tuple[float, bool]:
    
    """Compute the Kullback-Leibler divergence between two arrays
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: all values above this threshold means data drift
        n_bins: number of bins
    Returns:
        kl_div: calculated Kullback-Leibler divergence value
        test_result: whether the drift is detected
    """
    
    reference_percents, current_percents = get_binned_data(reference_data, current_data, feature_type, n_bins)
    kl_div_value = stats.entropy(reference_percents, current_percents)

    return kl_div_value, kl_div_value >= threshold
