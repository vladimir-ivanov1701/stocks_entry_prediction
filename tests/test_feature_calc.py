import pandas as pd
from main.functions import prepare_columns

test_df = pd.DataFrame()
test_df = [
    [""]
]


def test_prepare_columns():
    df1 = test_df.copy()
    df1 = prepare_columns(df1)
    assert df1.columns == ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]
