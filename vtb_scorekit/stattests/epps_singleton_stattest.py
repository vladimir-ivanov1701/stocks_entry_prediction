"""Epps-Singleton test of two samples.

Name: "es"
display_name="Epps-Singleton"
allowed_feature_types=["num"]

Import:

    >>> from stattests import epps_singleton_test

Properties:
- only for numerical features
- returns p-value
- default threshold 0.05

"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import epps_singleton_2samp
from scipy.stats import iqr


def _epps_singleton(
    reference_data: pd.Series,
    current_data: pd.Series,
    #feature_type: str,
    threshold: float=0.05,
) -> Tuple:
    
    """Run the Epps-Singleton (ES) test of two samples.
    Args:
        reference_data: reference data
        current_data: current data
        threshold: level of significance (default will be 0.05)
    Returns:
        p_value: p-value based on the asymptotic chi2-distribution.
        test_result: whether the drift is detected
    """
    
    # raised error if iqr is zero
    iqr_value = iqr(np.hstack((reference_data, current_data)))
    if iqr_value == 0:
        return "Low variance", 0
    p_value = epps_singleton_2samp(reference_data, current_data)[1]

    return p_value, p_value < threshold
