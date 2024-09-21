"""Anderson-Darling test of two samples.

Name: "anderson"
display_name="Anderson-Darling"
allowed_feature_types=["num"]

Import:

    >>> from stattests import anderson_darling_test

Properties:
- only for numerical features
- returns p-value

"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp


def _anderson_darling(
    reference_data: pd.Series,
    current_data: pd.Series,
    #feature_type: str,
    threshold: float=0.1,
) -> Tuple[float, bool]:
    
    p_value = anderson_ksamp([reference_data.values, current_data.values])[2]
    
    return p_value, p_value < threshold
