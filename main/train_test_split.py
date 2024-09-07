import pandas as pd
import numpy as np
import math
import warnings
import random as rnd
from typing import List
from sklearn.model_selection import train_test_split
from main.logging import MyLogger

from constants import TEST_SIZE
warnings.filterwarnings('ignore')

logger = MyLogger()


class TrainTestSplitter():
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        index_col: str,
        logger: MyLogger
    ):
        '''
        Инициализация класса.
        '''

        self.data = df
        self.target_col = target_col
        self.index_col = index_col
        self.logger = logger
        self.stata = pd.DataFrame(
            columns=['Total', 'Defs', 'DefRate', 'Share']
        )
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def choose_train_test_split(
        self,
        df: pd.DataFrame,
        test_sample_size: float = TEST_SIZE
    ) -> pd.DataFrame:
        '''
        Функция подбирает оптимальное разбиение на трейн и тест.
        '''

        rnd.seed(0)

        self.logger.info('Выбираем лучшее распределение train test')
        min_delta = 1
        min_delta_state = -1
        for rand_state in range(1000):
            if rand_state % 100 == 0:
                self.logger.info(
                    f"Выбираем лучшее распределение train/test: {rand_state}"
                )
            x_train, x_test, y_train, y_test = self.stratified_split(
                data=df,
                target=self.target_col,
                list_of_vars_for_strat=[self.target_col],
                sort_by_var=self.index_col,
                size_of_test=test_sample_size,
                drop_technical=False,
                random_state=rand_state
            )
            # определяем, подходит ли разбиение
            # и какое различие в дефрейтах трейн/тест
            delta = abs(np.average(y_test) / np.average(y_train) - 1)
            share = len(y_test) / len(self.data)
            share_delta = abs(share / test_sample_size - 1)
            if share_delta < 0.005 and delta < min_delta:
                min_delta = delta
                min_delta_state = rand_state

        self.logger.info(f"Минимальная разница TR на train/test: {delta}")

        # заново бьем на основании лучшего разбиения и формируем индексы
        x_train, x_test, y_train, y_test = self.stratified_split(
            data=df,
            target=self.target_col,
            list_of_vars_for_strat=[self.target_col],
            sort_by_var=self.index_col,
            size_of_test=test_sample_size,
            drop_technical=False,
            random_state=min_delta_state
        )

        delta = abs(np.average(y_test) / np.average(y_train) - 1)
        share = len(y_test) / len(self.data)
        share_delta = abs(share / test_sample_size - 1)
        df['TrainTest'] = [math.nan for _ in range(len(df))]
        self.df.loc[
            self.df[self.index_col].isin(x_train[self.index_col]),
            'TrainTest'
        ] = 'train'
        self.df.loc[
            self.df[self.index_col].isin(x_test[self.index_col]),
            'TrainTest'
        ] = 'test'

        buff = df

        stata_df = self.get_train_test_stata(
            'Total',
            buff,
            df,
            self.stata,
            self.target_col
        )
        buff = df.loc[df['TrainTest'] == 'train']
        stata_df = self.get_train_test_stata(
            'Train',
            buff,
            df,
            stata_df,
            self.target_col
        )
        buff = self.data.loc[self.data['TrainTest'] == 'test']
        stata_df = self.get_train_test_stata(
            'Test',
            buff,
            df,
            stata_df,
            self.target_col
        )
        stata_df.loc['RandomState', 'Total'] = min_delta_state

        self.logger.info(f"train/test dr delta: {delta:.3%}")
        self.logger.info(f"test share delta: {share_delta:.3%}")

        return df, stata_df

    def stratified_split(
        self,
        data: pd.DataFrame,
        target: str,
        list_of_vars_for_strat: List,
        sort_by_var: str,
        size_of_test: float = TEST_SIZE,
        drop_technical: bool = False,
        random_state: int = 0
    ) -> pd.DataFrame:
        """
        Стратифицированно бьет данные на трейн и тест
        в разрезе уникальных идентификаторов.

        data - передаваемые данные pd.DataFrame
        target - название таргетной переменной
        list_of_vars_for_strat - список переменных,по которым
                                 производится стратификация
        sort_by_var - переменная, по которой производится группировка
                      (id клиентов/транзакций и пр). В этой переменной
                      не должно быть пропусков, но если пропуски
                      заполнить одним значением, метод "решит", что это
                      один и тот же клиент. Поэтому если есть пропуски,
                      то их следует заполнять уникальными, не встреченными
                      ранее в данных значениями
        size_of_test - размер тестовой выборки
        drop_technical - удалить ли "технические" переменные
                         (лист переменных, по которым делается стратификация,
                         группирующая переменная)
        """

        max_target = data.groupby(sort_by_var).aggregate({target: 'max'})
        max_target = max_target.reset_index()

        df = pd.merge(
            data,
            max_target,
            on=sort_by_var,
            suffixes=["", "_max"]
        )

        target1 = target+"_max"

        if len(list_of_vars_for_strat) == 0:
            list_of_vars_for_strat = [target1]
        if target in list_of_vars_for_strat:
            list_of_vars_for_strat.remove(target)
            list_of_vars_for_strat.append(target1)
        else:
            list_of_vars_for_strat.append(target1)

        for i in list_of_vars_for_strat:
            if i == list_of_vars_for_strat[0]:
                df['For_stratify'] = df[i].astype('str')
            else:
                df['For_stratify'] += df[i].astype('str')

        data_nodup = df[[
            sort_by_var,
            'For_stratify',
            target1
        ]].drop_duplicates(subset=sort_by_var)

        train, test, target_train, target_test = train_test_split(
            data_nodup,
            data_nodup[target1],
            test_size=size_of_test,
            stratify=data_nodup['For_stratify'],
            random_state=random_state
        )

        X_train = df[df[sort_by_var].isin(train[sort_by_var])].copy()
        train_index = X_train.index
        y_train = df.iloc[train_index][target].copy()
        X_test = df[df[sort_by_var].isin(test[sort_by_var])].copy()
        test_index = X_test.index
        y_test = df.iloc[test_index][target].copy()

        if drop_technical:
            X_train.drop(list_of_vars_for_strat, axis=1, inplace=True)
            X_train.drop(sort_by_var, axis=1, inplace=True)

            X_test.drop(list_of_vars_for_strat, axis=1, inplace=True)
            X_test.drop(sort_by_var, axis=1, inplace=True)

        else:
            X_train.drop(target1, axis=1, inplace=True)
            X_test.drop(target1, axis=1, inplace=True)

        X_train.drop(target, axis=1, inplace=True)
        X_train.drop('For_stratify', axis=1, inplace=True)

        X_test.drop(target, axis=1, inplace=True)
        X_test.drop('For_stratify', axis=1, inplace=True)

        return X_train, X_test, y_train, y_test

    def get_train_test_stata(
        self,
        wd,
        df,
        ini_data,
        stata_df,
        target_column
    ) -> pd.DataFrame:
        ttls = len(df)
        defs = sum(df[target_column].values)
        defrate = defs/ttls
        share = ttls/len(ini_data)

        stata_df.loc[wd, 'Total'] = ttls
        stata_df.loc[wd, 'Defs'] = defs
        stata_df.loc[wd, 'DefRate'] = defrate
        stata_df.loc[wd, 'Share'] = share

        return stata_df
