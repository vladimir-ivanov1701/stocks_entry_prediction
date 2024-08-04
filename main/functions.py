from .constants import MSAD_PERIODS
import pandas as pd
import re
import numpy as np


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

    for per in MSAD_PERIODS:
        df[f'MSAD_{per}_mean'] = df['PIVOT'].rolling(per).mean()
        df[f'MSAD_{per}_std'] = df['PIVOT'].rolling(per).std()
        df[f'VOL_MA_{per}_mean'] = df['VOL'].rolling(per).mean()
        df[f'VOL_MA_{per}_std'] = df['VOL'].rolling(per).std()

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
