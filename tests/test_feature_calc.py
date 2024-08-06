import pandas as pd

from main.functions import (add_fractals, add_pivot, calc_active_impulse,
                            calc_candle_color, calc_end_correction,
                            calc_superactive_impulse,
                            calc_sl_tp)

# создание тестового датасета
test_df = [
            [71940.0, 71960.0, 71938.0, 71941.0, 427.0, "2023-01-03 21:45:00"],
            [71940, 71943.0, 71935.0, 71940.0, 301.0, "2023-01-03 21:50:00"],
            [70383.0, 70392.0, 70205.0, 70286.0, 5659.0, "2023-01-03 09:05:00"]
        ]
test_df = pd.DataFrame(
    test_df, columns=[
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOL",
        "DATETIME"
        ]
    )


def test_calc_candle_color():
    '''
    Тест расчёта цвета свечи.
    '''

    df = test_df.copy()
    df = calc_candle_color(df)
    assert df["CANDLE_COLOR"][0] == "green", \
        "Candle color sould be green, please check calculation"
    assert df["CANDLE_COLOR"][1] == "green", \
        "Doji color should be green, please check calculation"
    assert df["CANDLE_COLOR"][2] == "red", \
        "Candle color shoud be red, check calculation"


def test_add_pivot():
    '''
    Тест расчёта Pivot (High + Low + Close) / 3.
    '''

    df = test_df.copy()
    df = add_pivot(df)
    assert df["PIVOT"][0] == 71946.33333, \
        f"Pivot is {df.PIVOT[0]}, expected value is 71946.33333 (string 0)"
    assert df["PIVOT"][1] == 71939.33333, \
        f"Pivot is {df.PIVOT[1]}, expected value is 71939.33333 (string 1)"
    assert df["PIVOT"][2] == 70294.33333, \
        f"Pivot is {df.PIVOT[2]}, expected value is 70294.33333 (string 2)"


def test_add_fractals():
    '''
    Тест расчёта фракталов.
    '''

    df = [
        [10, 15, 5, 9, 100, "2023-01-01 12:00:00", "red"],
        [9, 20, 8, 12, 128, "2023-01-01 12:05:00", "green"],
        [12, 18, 9, 11, 120, "2023-01-01 12:10:00", "red"],
        [11, 20, 9, 15, 100, "2023-01-01 12:15:00", "green"],
        [15, 20, 9, 16, 110, "2023-01-01 12:20:00", "green"],
        [16, 18, 10, 12, 100, "2023-01-01 12:25:00", "red"],
        [12, 18, 7, 11, 120, "2023-01-01 12:30:00", "red"],
        [11, 15, 8, 14, 150, "2023-01-01 12:30:00", "green"],
        [14, 16, 8, 15, 140, "2023-01-01 12:35:00", "green"],
        [15, 20, 13, 18, 160, "2023-01-01 12:35:00", "green"]
    ]
    df = pd.DataFrame(
        df, columns=[
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOL",
            "DATETIME",
            "CANDLE_COLOR"
        ])

    df = add_fractals(df)
    assert df["IS_FRACTAL_UP"][1] == 1, \
        "Fractal Up is 0 (string 1), expected 1. " \
        "Check calculation of Fractal Up var.1"
    assert df["IS_FRACTAL_UP"][4] == 1, \
        "Fractal Up is 0 (string 4), expected 1. "\
        "Check calculation of Fractal Up var.2"

    assert df["IS_FRACTAL_DOWN"][6] == 1, \
        "Fractal Down is 0 (string 6), expected 1. " \
        "Check calculation of Fractal Down var.1"
    assert df["IS_FRACTAL_DOWN"][8] == 1, \
        "Fractal Down is 0 (string 8), expected 1." \
        "Check calculation of Fractal Down var.2"

    for i in range(10):
        if i not in (1, 4, 6, 8):
            assert df["IS_FRACTAL_DOWN"][i] == 0, \
                f"Fractal Down is 1 (string {i}), expected 0."
            assert df["IS_FRACTAL_UP"][i] == 0, \
                f"Fractal Up is 1 (string {i}), expected 0."


def test_calc_end_correction():
    df = test_df.copy()
    df["CANDLE_COLOR"] = ["green", "green", "red"]
    df = calc_end_correction(df)

    end_corrs = [19, 3, 81]
    end_corrs_perc = [19.0, 999999.0, 0.83505]

    for i in range(3):
        assert df["END_CORRECTION"][i] == end_corrs[i], \
            f"Incorrect end correction (string {i})." \
            f"Value is {df.END_CORRECTION[i]}." \
            f"Expected {end_corrs[i]}"
        assert df["END_CORRECTION_PERC"][i] == end_corrs_perc[i], \
            f"Incorrect end correction percent (string {i}." \
            f"Expected {df.END_CORRECTION_PERC[i]}." \
            f"Expected {end_corrs_perc[i]})"


def test_calc_active_impulse():
    '''
    Тест расчета активных импульсов.
    '''

    df = [
        [10, 15, 8, 12, 100, "2023-01-01 12:00:00", 3, 1.5],
        [12, 20, 11, 19, 120, "2023-01-01 12:05:00", 1, 0.14286],
        [19, 20, 17, 18, 110, "2023-01-01 12:10:00", 1, 1],
        [18, 20, 12, 12, 150, "2023-01-01 12:15:00", 0, 0.0]
    ]
    df = pd.DataFrame(
        df, columns=[
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOL",
            "DATETIME",
            "END_CORRECTION",
            "END_CORRECTION_PERC"
        ])

    df = calc_active_impulse(df)

    assert df["UPGOING_ACTIVE_IMPULSE"][1] == 1, \
        "Upgoing active impulse (string 1) is 0, expected 1."
    assert df["UPGOING_ACTIVE_IMPULSE"][2] == 0, \
        "Upgoing active impulse (string 2) is 1, expected 0."
    assert df["DOWNGOING_ACTIVE_IMPULSE"][3] == 1, \
        "Downgoing active impulse (string 3) is 0, expected 1."


def test_calc_superactive_impulse():
    df = [
        [10, 15, 9, 14, 0, 0],
        [14, 30, 12, 30, 1, 0],
        [30, 31, 26, 28, 0, 0],
        [28, 29, 10, 15, 0, 1]
    ]
    df = pd.DataFrame(df, columns=[
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "UPGOING_ACTIVE_IMPULSE",
        "DOWNGOING_ACTIVE_IMPULSE"
        ]
    )

    df = calc_superactive_impulse(df)

    assert df["IS_SUPERACTIVE_IMPULSE"][1] == 1, \
        "Superactive impulse is 0 (string 1), expected 1."
    assert df["IS_SUPERACTIVE_IMPULSE"][2] == 0, \
        "Superactive impulse is 1 (string 2), expected 0."
    assert df["IS_SUPERACTIVE_IMPULSE"][3] == 1, \
        "Superactive impulse is 0 (string 3), expected 1."


def test_calc_sl_tp():
    sl, tp = calc_sl_tp("ED", sl_rub=100, tp_rub=200)

    assert sl == 0.0012, \
        f"SL is {sl}, expected 0.0012."
    assert tp == 0.0023, \
        f"TP is {tp}, expected 0.0023"
