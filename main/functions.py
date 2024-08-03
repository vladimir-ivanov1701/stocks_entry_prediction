from constants import MSAD_PERIODS
import pandas as pd
import re
import numpy as np
import math


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

    df['PIVOT'] = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
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

    df["IS_FRACTAL_DOWN"] = np.where(
        (
            df["LOW"] < df.shift()["LOW"] and
            df["LOW"] < df.shift(-1)["LOW"]
        ) or
        (
            df["LOW"] == df.shift()["LOW"] and
            df["LOW"] < df.shift(-1)["LOW"] and
            df.shift(-1)["CANDLE_COLOR"] == "green"
        ),
        1,
        0
    )
    df["IS_FRACTAL_UP"] = np.where(
        (
            df["HIGH"] > df.shift()["HIGH"] and
            df["HIGH"] > df.shift(-1)["HIGH"]
        ) or
        (
            df["HIGH"] == df.shift()["HIGH"] and
            df["HIGH"] > df.shift(-1)["HIGH"] and
            df.shift(-1)["CANDLE_COLOR"] == "red"
        )
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
    df["END_CORRECTION_PERC"] = (df["END_CORRECTION"] /
                                 math.abs(df["CLOSE"] - df["OPEN"]))
    return df
    '''
    len_df = df.shape[0]
    end_corr, end_corr_perc = [0] * len_df, [0] * len_df
    for i in range(0, len_df):
        k = 0
        while k <= i:
            corr = df['CLOSE'][i-k] - df['OPEN'][i-k]
            if corr > 0:
                end_corr[i] = df['HIGH'][i] - df['CLOSE'][i]
                end_corr_perc[i] = (
                    end_corr[i] / (df['CLOSE'][i] - df['OPEN'][i])
                ) if df['OPEN'][i] != df['CLOSE'][i] else 0
                break
            elif corr < 0:
                end_corr[i] = df['CLOSE'][i] - df['LOW'][i]
                end_corr_perc[i] = (
                    end_corr[i] / (df['OPEN'][i] - df['CLOSE'][i])
                ) if df['OPEN'][i] != df['CLOSE'][i] else 0
                break
            else:
                k += 1

    df['END_CORRECTION'] = end_corr
    df['END_CORRECTION_PERC'] = end_corr_perc
    return df
    '''
