import pandas as pd
import numpy as np
import os
import math
import warnings
from time import time
import optbinning as obn
from   statsmodels.stats.proportion import proportion_confint
from main.logging import MyLogger
from main.constants import N_BINS
from typing import List

warnings.filterwarnings('ignore')


class OptBinner():
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        logger: MyLogger,
        n_bins: int = N_BINS
    ):
        '''
        Инициализация класса
        '''

        self.train_data = train_data
        self.test_data = test_data
        self.logger = logger
        self.n_bins = n_bins
        self.df_trends = pd.DataFrame(columns=['Factor', 'Trend'])
        self.res = []
        self.basic_cols = [
            'ID',
            'APP_NUM',
            'DT_OPEN',
            'CONTRACT_SUM',
            'INN',
            'default',
            'rep_date'
        ]

    def get_stats(
        self,
        x_list: int,
        y_list: List,
        n_cats: int,
        have_missing: bool
    ) -> List:
        '''
        Расчет статистик
        '''

        res = []
        lst = [x for x in range(n_cats)]
        lst += [max(x_list)] if max(x_list) > n_cats else []
        for i in lst:
            buff = y_list[x_list == i]
            a = sum(buff)
            b = len(buff)
            if b == 0:
                dr, lBound, uBound = math.nan, math.nan, math.nan
            else:
                dr = a/b
                lBound, uBound = proportion_confint(a, b, method='wilson')

            res.append([dr, b-a, a, lBound, uBound])

        if max(x_list) < n_cats and have_missing:
            res.append([math.nan, 0, 0, math.nan, math.nan])
        return res

    def getOrderCats(stats: List) -> List:
        '''
        Получаем порядок категорий по дефрейту (от наилучшего к наихудшему).
        Количество категорий передаем принудительно, чтобы игнорировать
        дефрейт миссинга.
        '''

        cats = [i for i in range(len(stats))]
        defr = [stats[i][0] for i in range(len(stats))]
        res = [x for _, x in sorted(zip(defr, cats))]
        return res

    def isGoodConfInterval(stats: List, n_breaks: int) -> List:
        '''
        Определяем, не пересекаются ли доверительные интервалы
        дефрейтов по соседним бинам.
        Количество категорий передаем принудительно,
        чтобы игнорировать дефрейт миссинга.
        Третье значение в каждом массиве статистики - нижняя граница
        доверительного интервала
        Четвёртое значение в каждом массиве статистики - верхняя граница
        доверительного интервала
        '''

        return all(
            [
                stats[bucket][3] > stats[bucket-1][4]
                or stats[bucket][4] < stats[bucket-1][3]
                for bucket in range(1, n_breaks)
            ]
        )

    def getWOEsByIndices(x_cats, woes):
        res = np.full(len(x_cats), 0)
        for i in np.unique(x_cats):
            res = np.where(x_cats == i, woes[i], res)
        return res

    def binning_factor(
            self,
            factor_code,
            trend,
            max_bins,
            x_train,
            y_train,
            x_test=[],
            y_test=[],
            doBoot=False,
            isCategorical=False,
            checkConf=True):
        '''
        factor_code - название фактора
        trend - направление корреляции с вероятностью дефолта
            'ascending' 'descending' 'auto'
        max_bins - максимальное количество корзин
        x_train - значения независимой переменной по обучающей выборке
        y_train - значения зависимой переменной по обучающей выборке
        x_test - значения независимой переменной по тестовой выборке
        y_test - значения зависимой переменной по тестовой выборке
        doBoot - делать бутстреп
        isCategorical - категориальный фактор  
        checkConf - проверка на пересечение доверительных интервалов
        '''

        x_train, y_train, x_test, y_test = np.array(x_train), \
            np.array(y_train), np.array(x_test), np.array(y_test)
        doTest = len(x_test) > 0

        if doBoot:
            print('doBoot не реализован')
            raise 1/0

        max_bins_ini = max_bins

        #  результирующие колонки
        cols = [
            'Factor',
            'nBreaks',
            'IV',
            'trainIntuitive',
            'testIntuitive',
            'goodBootShare'
        ] + \
            ['Break_' + str(x) for x in range(max_bins)] + \
            ['Break_Missing'] + \
            ['trainWOE_' + str(x) for x in range(max_bins)] + \
            ['trainWOE_Missing'] + \
            ['trainDefRate_' + str(x) for x in range(max_bins)] + \
            ['trainDefRate_Missing'] + \
            ['trainGoods_' + str(x) for x in range(max_bins)] + \
            ['trainGoods_Missing'] + \
            ['trainBads_' + str(x) for x in range(max_bins)] + \
            ['trainBads_Missing'] + \
            ['testDefRate_' + str(x) for x in range(max_bins)] + \
            ['testDefRate_Missing'] + \
            ['testGoods_' + str(x) for x in range(max_bins)] + \
            ['testGoods_Missing'] + \
            ['testBads_' + str(x) for x in range(max_bins)] + \
            ['testBads_Missing']

        # непосредственно биннинг
        while True:
            t2 = time()
            if not isCategorical:  # непрерывный фактор
                optb = obn.OptimalBinning(name = factor_code, 
                                            dtype = "numerical", 
                                            solver = "cp",
                                            monotonic_trend = trend,
                                            min_prebin_size = 0.05,
                                            max_n_bins = max_bins,
                                            cat_cutoff = 0.05,
                                            divergence = 'iv')
            else: # категориальный фактор
                optb = obn.OptimalBinning(name = factor_code, 
                                            dtype = "categorical", 
                                            solver = "mip",
                                            min_prebin_size = 0.02,
                                            max_n_bins = max_bins)
                
            
            try:
                optb.fit(x_train, y_train)
                doBuff = True
            except:
                train_cats = [0 for _ in range(len(x_train))]
                train_woes = [0 for _ in range(len(x_train))]
                test_cats = [0 for _ in range(len(x_test))] if doTest else []
                test_woes = [0 for _ in range(len(x_test))] if doTest else []
                buff = [factor_code] + [math.nan for _ in range(len(cols)-1)]
                doBuff = False
                break
            bins = optb.binning_table.build()
            bin_missing = bins[bins['Bin'].astype(str)=='Missing'].index[0]
            woes = bins['WoE'].values
            indices = bins.index.values
            # количество категорий
            if not isCategorical:
                nBreaks = len(optb.splits) + 1
            else:
                nBreaks = len(optb.splits)
            IV = bins.loc['Totals', 'IV']
            
            # получение категорий для каждого наблюдения  
            train_cats = optb.transform(x_train, metric='indices', metric_missing=bin_missing)
            # trainWOEs = optb.transform(xTrain, metric='woe') # некорректно работает для миссинга
            train_woes = self.getWOEsByIndices(train_cats, woes)
            if doTest:
                test_cats = optb.transform(x_test, metric='indices', metric_missing=bin_missing)
            # testWOEs = optb.transform(xTest, metric='woe') # некорректно работает для миссинга
                test_woes = self.getWOEsByIndices(test_cats, woes)
        
            # расчет дефрейта и прочей статистики по категориям
            # статистику можно взять и из готовых значений, но только по train, для test все равно считать
            haveMissing = max(train_cats) > nBreaks
            train_stats = self.get_stats(train_cats, y_train, nBreaks, haveMissing)
            if doTest:
                test_stats = self.get_stats(test_cats, y_test, nBreaks, haveMissing)
            else:
                test_stats = train_stats
            
            # получаем порядок категорий по дефрейту (от наилучшего к наихудшему)
            trainOrder = self.getOrderCats(train_stats, nBreaks)
            if doTest:
                testOrder = self.getOrderCats(test_stats, nBreaks)
            if checkConf:
                trainIntuitive = self.isGoodConfInterval(train_stats, nBreaks)
            else:
                trainIntuitive = True
            testIntuitive = (testOrder==trainOrder) if doTest else True

            # выходим из цикла
            if trainIntuitive and testIntuitive: # and bootIntuitive:
                break
            if nBreaks <= 2:
                break
            # дошли до этой строки - значит, дефрейты не интуитивны, но количество бакетов больше двух
            max_bins = nBreaks - 1 

        #  записываем результаты по фактору в результирующий массив
        if doBuff:
            bins_list = bins.values.tolist()
            # общая информация
            buff = [factor_code, nBreaks, IV, trainIntuitive, testIntuitive] + [1 - 1 if doBoot else math.nan]
            # границы биннинга
            a = 1 if not isCategorical else 0
            buff += [optb.splits[i] for i in range(nBreaks-a)] + [math.nan for i in range(max_bins_ini-nBreaks+a)] + ['Missing']
            # train WOE
            buff += [bins_list[i][6] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [bins_list[nBreaks+1][6] if haveMissing else math.nan]
            # train DefRate
            buff += [train_stats[i][0] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [train_stats[nBreaks][0] if haveMissing else math.nan]
            # train Goods
            buff += [train_stats[i][1] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [train_stats[nBreaks][1] if haveMissing else math.nan]
            # train Bads
            buff += [train_stats[i][2] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [train_stats[nBreaks][2] if haveMissing else math.nan]
            # test DefRate
            buff += [test_stats[i][0] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [test_stats[nBreaks][0] if haveMissing else math.nan]
            # test Goods
            buff += [test_stats[i][1] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [test_stats[nBreaks][1] if haveMissing else math.nan]
            # test Bads
            buff += [test_stats[i][2] for i in range(nBreaks)] + [math.nan for i in range(max_bins_ini-nBreaks)] + [test_stats[nBreaks][2] if haveMissing else math.nan]

        return cols, buff, train_cats, train_woes, test_cats if doTest else [], test_woes if doTest else []






cat_train = data_train[basic_cols].copy()
woe_train = cat_train.copy()
data_train.drop(basic_cols, 
                axis='columns',
                inplace = True)

cat_test = data_test[basic_cols].copy()
woe_test = cat_test.copy()
data_test.drop(basic_cols, 
                axis='columns',
                inplace = True)
