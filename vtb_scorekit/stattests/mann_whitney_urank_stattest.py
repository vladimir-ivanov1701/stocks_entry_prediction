"""Mann-Whitney U-rank test of two samples.

Name: "mannw"
display_name="Mann-Whitney U-rank test"
allowed_feature_types=["num"]

Import:

    >>> from stattests import mann_whitney_u_stat_test

Properties:
- only for numerical features
- returns p-value

"""
from typing import Tuple

import pandas as pd
from scipy.stats import mannwhitneyu


def _mannwhitneyu_rank(
    reference_data: pd.Series,
    current_data: pd.Series,
    #feature_type: str,
    threshold: float=0.05,
) -> Tuple[float, bool]:
    
    """Perform the Mann-Whitney U-rank test between two arrays
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: all values above this threshold means data drift
    Returns:
        pvalue: the two-tailed p-value for the test depending on alternative and method
        test_result: whether the drift is detected
    """

    p_value = mannwhitneyu(x=reference_data, y=current_data)[1]

    return p_value, p_value < threshold
