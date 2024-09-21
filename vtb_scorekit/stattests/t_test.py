"""T test of two samples.

Name: "t_test"
display_name="t_test"
allowed_feature_types=["num"]

Import:

    >>> from stattests import t_test

Properties:
- only for numerical features
- returns p-value

"""
from typing import Tuple

import pandas as pd
from scipy.stats import ttest_ind


def _t_test2samp(
    reference_data: pd.Series, 
    current_data: pd.Series, 
    #feature_type: str, 
    threshold: float=0.05
) -> Tuple[float, bool]:
    
    """Compute the two-sample t test between reference and current
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: level of significance (default will be 0.05)
    Returns:
        p_value: two-tailed p-value
        test_result: whether drift is detected
    """

    p_value = ttest_ind(reference_data, current_data)[1]
    
    return p_value, p_value < threshold
