import re

import numpy as np
import pandas as pd

from main.constants import MAX_SHIFT, MSAD_PERIODS, PIPS_COSTS, SL_RUB, TP_RUB


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Подготовка датасета. Переименование столбцов.
    '''

    df = df.rename(columns={c: re.sub("[<>]", "", c) for c in df.columns})
    df["DATE"] = pd.to_datetime(df["DATE"], format="%y%m%d")
    df["TIME"] = pd.to_datetime(df["TIME"], format="%H%M%S").dt.time
    df["DATETIME"] = pd.to_datetime(
        df["DATE"].astype(str) + ' ' + df["TIME"].astype(str)
    )

    df.drop(
        [
            "DATE",
            "TIME",
            "DATETIME_KEY",
            "TICKER",
            "PER"
        ],
        axis=1,
        inplace=True
    )

    for col in [
        "OPEN",
        "CLOSE",
        "HIGH",
        "LOW",
        "VOL"
    ]:
        df[col] = df[col].astype('float32')

    return df


def calc_candle_color(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция рассчитывает цвет свечи.
    - Close > Open - зелёный
    - Close < Open - красный
    Если Close == Open, то цвет определяется по последней цветной свече.
    '''

    df["CANDLE_COLOR"] = np.where(
        df["CLOSE"] > df["OPEN"],
        "green",
        np.where(
            df["CLOSE"] < df["OPEN"],
            "red",
            None
        )
    )
    df = df.ffill()
    return df


def add_pivot(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Рассчитывает кривую Pivot ((High + Low + Close) / 3).
    '''

    df['PIVOT'] = np.round((df['HIGH'] + df['LOW'] + df['CLOSE']) / 3, 5)
    return df


def add_movings(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Добавляет скользящие средние к датафрейму.
    Скользящие средние считаются по среднему и стандартному отклонению.
    '''

    col_list = df.drop(["DATETIME", "CANDLE_COLOR"], axis=1).columns
    for per in MSAD_PERIODS:
        for col in col_list:
            df[f"{col}_MA_mean"] = df[col].rolling(per).mean()
            df[f"{col}_MA_std"] = df[col].rolling(per).std()

    return df


def add_fractals(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция считает фракталы Up и Down.
    Условия фрактала Up, вар.1:
        - Candle[1] High > Candle[0] High
        - Candle[1] High > Candle[2] High
    Условия фрактала Up, вар.2:
        Candle[1] High == Candle[0] High
        Candle[2] High < Candle[1] High
        Candle[2] Close < Candle[2] Open (красная свеча).
    Условия фрактала Down, вар.1:
        Candle[1] Low < Candle[0] Low
        Candle[1] Low < Candle[2] Low
    Условия фрактала Down, вар.2:
        Candle[1] Low == Candle[0] Low
        Candle[1] Low < Candle[2] Low
        Candle[2] Close > Candle[2] Open (зелёная свеча).
    '''

    df["LOW_PREV"] = df.shift()["LOW"]
    df["LOW_NEXT"] = df.shift(-1)["LOW"]
    df["HIGH_PREV"] = df.shift()["HIGH"]
    df["HIGH_NEXT"] = df.shift(-1)["HIGH"]
    df["CANDLE_COLOR_PREV"] = df.shift()["CANDLE_COLOR"]
    df["CANDLE_COLOR_NEXT"] = df.shift(-1)["CANDLE_COLOR"]

    df["IS_FRACTAL_UP"] = np.where(
        df["HIGH"] > df["HIGH_PREV"],
        np.where(
            df["HIGH"] > df["HIGH_NEXT"],
            1,
            0
        ),
        np.where(
            df["HIGH"] == df["HIGH_PREV"],
            np.where(
                df["HIGH"] > df["HIGH_NEXT"],
                np.where(
                    df["CANDLE_COLOR_NEXT"] == "red",
                    1,
                    0
                ),
                0
            ),
            0,
        )
    )

    df["IS_FRACTAL_DOWN"] = np.where(
        df["LOW"] < df["LOW_PREV"],
        np.where(
            df["LOW"] < df["LOW_NEXT"],
            1,
            0
        ),
        np.where(
            df["LOW"] == df["LOW_PREV"],
            np.where(
                df["LOW"] < df["LOW_NEXT"],
                np.where(
                    df["CANDLE_COLOR_NEXT"] == "green",
                    1,
                    0
                ),
                0
            ),
            0,
        ),
    )

    df.drop(
        [
            "LOW_PREV",
            "LOW_NEXT",
            "HIGH_PREV",
            "HIGH_NEXT",
            "CANDLE_COLOR_PREV",
            "CANDLE_COLOR_NEXT"
        ],
        axis=1,
        inplace=True
    )
    return df


def calc_end_correction(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция рассчитывает конечную коррекцию в абсолютных значениях
    и в %. Верхняя тень - для зелёных свечей, нижняя тень - для красных свечей.
    Если Open == Close (дожи) - тогда конечная коррекция
    определяется цветом предыдущей свечи.
    '''

    df["END_CORRECTION"] = np.where(
        df["CANDLE_COLOR"] == "green",
        df["HIGH"] - df["CLOSE"],
        df["CLOSE"] - df["LOW"]
    )
    df["END_CORRECTION_PERC"] = np.round(
        np.where(
            df["CLOSE"] != df["OPEN"],
            df["END_CORRECTION"] / np.abs(df["CLOSE"] - df["OPEN"]),
            999999
        ),
        5
    )
    return df


def calc_active_impulse(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция рассчитывает параметры активного импульса.
    Восходящий активный импульс:
        - Candle[1] Open >= Candle[0] Open
        - Candle[1] Close >= Candle[0] Close
        - Candle[1] Close > Candle[1] Open
        - Candle[1] Body (Close - Open) >= Candle[0] Body]
    Нисходящий активный импульс:
        - Candle[1] Open <= Candle[0] Open
        - Candle[1] Close < Candle[0] Close
        - Candle[1] Close < Candle[1] Open
        - Candle[1] Body (Open - Close) >= Candle[0] Body
    Общие условия для обоих направлений:
        - Candle[1] End Correction Percent < Candle[0] End Correction Percent
        - Candle[1] End Correction <= Candle[1] Body * 0.2
        - Candle[1] Body >= Candle[1] Amplitude * 0.7
    '''

    df["OPEN_PREV"] = df.shift()["OPEN"]
    df["CLOSE_PREV"] = df.shift()["CLOSE"]
    df["END_CORR_PREV"] = df.shift()["END_CORRECTION"]
    df["END_CORR_PERC_PREV"] = df.shift()["END_CORRECTION_PERC"]

    df["ACTIVE_IMPULSE_COMMON"] = np.where(
        (df["END_CORRECTION_PERC"] < df["END_CORR_PERC_PREV"]),
        np.where(
            df["END_CORRECTION"] <= np.abs(df["CLOSE"] - df["OPEN"]) * 0.2,
            np.where(
                np.abs(df["CLOSE"] - df["OPEN"]) >=
                (df["HIGH"] - df["LOW"]) * 0.7,
                1,
                0
            ),
            0
        ),
        0
    )

    df["UPGOING_ACTIVE_IMPULSE"] = np.where(
        df["OPEN"] >= df["OPEN_PREV"],
        np.where(
            df["CLOSE"] >= df["CLOSE_PREV"],
            np.where(
                df["CLOSE"] > df["OPEN"],
                np.where(
                    (df["CLOSE"] - df["OPEN"]) >=
                    (df["CLOSE_PREV"] - df["OPEN_PREV"]),
                    np.where(
                        df["ACTIVE_IMPULSE_COMMON"] == 1,
                        1,
                        0
                    ),
                    0
                ),
                0
            ),
            0
        ),
        0
    )

    df["DOWNGOING_ACTIVE_IMPULSE"] = np.where(
        df["OPEN"] <= df["OPEN_PREV"],
        np.where(
            df["CLOSE"] < df["OPEN"],
            np.where(
                (df["OPEN"] - df["CLOSE"]) >=
                (df["OPEN_PREV"] - df["CLOSE_PREV"]),
                np.where(
                    df["ACTIVE_IMPULSE_COMMON"] == 1,
                    1,
                    0
                ),
                0
            ),
            0
        ),
        0
    )

    df.drop(
        [
            "OPEN_PREV",
            "CLOSE_PREV",
            "END_CORR_PREV",
            "END_CORR_PERC_PREV",
            "ACTIVE_IMPULSE_COMMON"
        ],
        axis=1,
        inplace=True)
    return df


def calc_superactive_impulse(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция считает сверхактивные импульсы (соотношение тел свечей > 3.5).
    '''

    df["OPEN_PREV"] = df.shift()["OPEN"]
    df["CLOSE_PREV"] = df.shift()["CLOSE"]
    df["IS_SUPERACTIVE_IMPULSE"] = np.where(
        (
            np.abs(df["CLOSE"] - df["OPEN"]) /
            np.abs(df["CLOSE_PREV"] - df["OPEN_PREV"])
        ) > 3.5,
        np.where(
            df["UPGOING_ACTIVE_IMPULSE"] == 1,
            1,
            np.where(
                df["DOWNGOING_ACTIVE_IMPULSE"] == 1,
                1,
                0
            ),
        ),
        0
    )

    df.drop(["OPEN_PREV", "CLOSE_PREV"], axis=1, inplace=True)
    return df


def calc_shift(df: pd.DataFrame) -> pd.DataFrame:

    col_list = df.drop(["DATETIME", "CANDLE_COLOR"], axis=1).columns

    for i in range(MAX_SHIFT):
        for col in col_list:
            df[f"{col}_SHIFT_{i}"] = df[col].shift(i)

    return df


def calc_sl_tp(
        futures_name: str,
        sl_rub: int = SL_RUB,
        tp_rub: int = TP_RUB
        ) -> float:
    '''
    Расчёт стоп-лосса в пунктах
    '''

    price_step = PIPS_COSTS[futures_name]["PRICE_STEP"]
    price_step_cost = PIPS_COSTS[futures_name]["PRICE_STEP_COST"]

    sl_pips = np.round(sl_rub * price_step / price_step_cost, 4)
    tp_pips = np.round(tp_rub * price_step / price_step_cost, 4)

    return sl_pips, tp_pips


def calc_targets(
    df: pd.DataFrame,
    futures_name: str,
    sl_rub: int = SL_RUB,
    tp_rub: int = TP_RUB
) -> pd.DataFrame:
    '''
    Расчет таргета.
    '''

    sl_pips, tp_pips = calc_sl_tp(futures_name, sl_rub, tp_rub)

    res_long, res_short = [0] * df.shape[0], [0] * df.shape[0]

    for i in range(df.shape[0] - 1):
        entry_price = df["CLOSE"][i]
        sl_price_long = entry_price - sl_pips
        tp_price_long = entry_price + tp_pips
        sl_price_short = entry_price + sl_pips
        tp_price_short = entry_price - tp_pips

        # calc target for long
        k = i + 1
        while k < df.shape[0]:
            low_price = df["LOW"][k]
            high_price = df["HIGH"][k]

            if low_price <= sl_price_long:
                res_long[i] = 0
                break
            elif high_price >= tp_price_long:
                res_long[i] = 1
                break
            else:
                k += 1

        # calc target for short
        k = i + 1
        while k < df.shape[0]:
            low_price = df["LOW"][k]
            high_price = df["HIGH"][k]

            if high_price >= sl_price_short:
                res_short[i] = 0
                break
            elif low_price <= tp_price_short:
                res_short[i] = 1
                break
            else:
                k += 1

    df["TARGET_LONG"] = res_long
    df["TARGET_SHORT"] = res_short

    return df


def calc_features(
    df: pd.DataFrame,
    futures_name: str,
    sl_rub: int = SL_RUB,
    tp_rub: int = TP_RUB
) -> pd.DataFrame:
    '''
    Расчет всех фич датасета.
    '''

    df = calc_candle_color(df)
    df = add_pivot(df)
    df = add_fractals(df)
    df = calc_end_correction(df)
    df = calc_active_impulse(df)
    df = calc_superactive_impulse(df)
    df = add_movings(df)
    df = calc_targets(df, futures_name, sl_rub, tp_rub)

    return df
