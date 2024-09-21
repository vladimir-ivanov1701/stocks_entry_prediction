# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import os
import gc
import copy
from ._utils import fig_to_excel, adjust_cell_width, add_suffix, rem_suffix, cross_split, is_cross_name, make_header, add_ds_folder
from .stattests import *
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from textwrap import wrap
from concurrent import futures
from functools import partial
import logging
import sys
from itertools import combinations
from tqdm import tqdm
from scipy.special import ndtri
import pkg_resources
try:
    import optbinning
except:
    print('WARNING! Optbinning not found.')

warnings.simplefilter('ignore')
plt.rc('font', family='Verdana', size=12)
try:
    plt.style.use([s for s in plt.style.available if 'darkgrid' in s][0])
except:
    pass
pd.set_option('display.precision', 3)
gc.enable()


class DataSamples:
    """
    Основной класс для хранения данных
    """
    def __init__(self, samples=None, target=None, features=None, cat_columns=None, min_nunique=20, time_column=None,
                 id_column=None, feature_descriptions=None, default_descriptions=False, train_name=None, special_bins=None, result_folder='',
                 n_jobs=1, random_state=0, samples_split=None, bootstrap_split=None, ci_analytic=True, ci_alpha=0.05,
                 compress=False, verbose=True, logger=20):
        """
        :param samples: выборка для разработки. Задается в виде словаря {название_сэмпла: датафрейм}, может содержать любое кол-во сэмплов
        :param target: целевая переменная
        :param features: список переменных. При None берутся все поля числового типа и нечисловые (кроме target, time_column, id_column) с кол-вом уникльных значений меньше min_nunique
        :param cat_columns: список категориальных переменных. При None категориальными считаются все переменные с кол-вом уникальных значений меньше min_nunique
        :param min_nunique: кол-во уникальных значений, до которого переменная считается категориальной при автоматическом определении
        :param time_column: дата среза
        :param id_column: уникальный в рамках среза айди наблюдения
        :param feature_descriptions: датафрейм с описанием переменных. Должен содержать индекс с названием переменных и любое кол-во полей с описанием, которые будут подтягиваться в отчеты
        :param default_descriptions: использовать дефолтные описания фичей из стандартных витрин
        :param train_name: название сэмпла обучающей выборки. При None берется первый сэмпл
        :param special_bins: словарь вида {название бина: значение}, каждое из значений которого помещается в отдельный бин
        :param result_folder: папка, в которую будут сохраняться все результаты работы с этим ДатаСэмплом
        :param n_jobs: кол-во используемых рабочих процессов, при -1 берется число, равное CPU_LIMIT
        :param random_state: сид для генератора случайных чисел, используется во всех остальных методах, где необходимо
        :param samples_split: словарь с параметрами для вызова метода self.samples_split
        :param bootstrap_split: словарь с параметрами для вызова метода self.bootstrap_split
        :param ci_analytic: всегда считать доверительные интервалы аналитически
        :param ci_alpha: уровень значимости
        :param compress: сократить использование памяти за счет преобразования типов данных переменных
                         float64 -> float32
                         int64   -> int32, int16, int8
                         object  -> category
        :param verbose: флаг вывода комментариев в процессе создания ДатаСэмпла
        :param logger: либо объект logging.Logger, либо числовое значение уровня логгирования:
                         10 - выводятся все сообщения с типами debug, info, warning, error
                         20 - выводятся все сообщения с типами info, warning, error
                         30 - выводятся все сообщения с типами warning, error
                         40 - выводятся сообщения только с типом error
                         50 - не выводятся никакие сообщения
        """
        if isinstance(result_folder, str):
            if result_folder and not os.path.exists(result_folder):
                os.makedirs(result_folder)
            self.result_folder = result_folder + ('/' if result_folder and not result_folder.endswith('/') else '')
        else:
            self.result_folder = ''
        if isinstance(logger, logging.Logger):
            self.logger = logger
            self.logger_level = self.logger.level
        elif isinstance(logger, (int, float)):
            self.logger = logging.getLogger()
            self.logger.setLevel(logger)
            if self.logger.hasHandlers():
                self.logger.handlers.clear()
            for dis_log in ['matplotlib', 'numexpr']:
                logging.getLogger(dis_log).setLevel(logging.ERROR)
            for log in [logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(self.result_folder + 'log.txt')] if logger < 50 else []):
                log.setFormatter(logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
                self.logger.addHandler(log)
            self.logger_level = self.logger.level
        else:
            self.logger = None
        if verbose:
            self.logger.info(make_header('Creating DataSamples', 150))

        self.samples = {name: sample for name, sample in samples.items() if not sample.empty} if samples is not None else None
        self.train_name = list(self.samples.keys())[0] if train_name is None and self.samples else train_name
        self.target = target
        if self.samples and self.target and self.train_name in self.samples[self.train_name]:
            extra_values = [x for x in self.samples[self.train_name][self.target].unique() if x not in [0, 1]]
            if extra_values and verbose:
                self.logger.warning(f'Target contains extra values {extra_values}')
        self.id_column = id_column
        self.time_column = time_column if time_column and self.samples and time_column in self.samples[self.train_name] else None
        if features is not None:
            self.features = list(features)
        elif self.samples is not None:
            self.features = [f for f in self.samples[self.train_name].columns
                             if f not in [self.target, self.id_column, self.time_column]]
        else:
            self.features = []
        if cat_columns is None:
            if self.samples is not None:
                self.cat_columns = [f for f in self.features if self.samples[self.train_name][f].nunique() < min_nunique or
                                    (features is not None and not pd.api.types.is_numeric_dtype(self.samples[self.train_name][f]))]
            else:
                self.cat_columns = []
        else:
            self.cat_columns = cat_columns
        self.special_bins = special_bins if special_bins is not None else {}
        if 'nan' not in self.special_bins:
            self.special_bins['nan'] = np.nan
        if self.samples:
            if features is None:
                self.features = [f for f in self.features if pd.api.types.is_numeric_dtype(
                    self.samples[self.train_name][f]) or f in self.cat_columns]
                if verbose:
                    self.logger.info(f'Selected {len(self.features)} features: {self.features}')

            if cat_columns is None:
                if verbose:
                    self.logger.info(f'Selected {len(self.cat_columns)} categorical features: {self.cat_columns}')

            f_inf = [f for f in self.features if
                     pd.api.types.is_numeric_dtype(self.samples[self.train_name][f]) and np.isinf(
                         self.samples[self.train_name][f]).any()]
            if f_inf:
                special_value = int(''.rjust(len(
                    str(round(self.samples[self.train_name][f_inf].replace([np.inf, -np.inf], 10).max().max()))) + 1,
                                             '9'))
                while special_value in self.special_bins.values():
                    special_value = special_value * 10 + 9
                self.special_bins['Infinity'] = special_value
                for name in self.samples:
                    self.samples[name][f_inf] = self.samples[name][f_inf].replace([np.inf, -np.inf], special_value)
                if verbose:
                    self.logger.warning(
                        f"Features {f_inf} contain infinities. They are replaced by {special_value} and special bin {{'Infinity': {special_value}}} is added.")

        if feature_descriptions is not None:
            self.feature_descriptions = feature_descriptions[~feature_descriptions.index.duplicated(keep='last')].copy()
            if self.features:
                self.feature_descriptions = self.feature_descriptions[self.feature_descriptions.index.isin(list(self.features))].copy()
        else:
            self.feature_descriptions = None
        if default_descriptions:
            dflt_descr = pd.read_excel(pkg_resources.resource_filename(__name__, 'templates/default_descriptions.xlsx')).set_index('Переменная')
            dflt_descr = dflt_descr[~dflt_descr.index.duplicated(keep='last')].copy()
            if self.features:
                dflt_descr = dflt_descr[dflt_descr.index.isin(list(self.features))].copy()
            self.features_bl = dflt_descr['Бизнес-логика']
            dflt_descr = dflt_descr.drop(['Бизнес-логика'], axis=1)
            if self.feature_descriptions is None:
                self.feature_descriptions = dflt_descr
            else:
                self.feature_descriptions = pd.concat([self.feature_descriptions, dflt_descr.rename({'Описание': self.feature_descriptions.columns[0]}, axis=1)])
                self.feature_descriptions = self.feature_descriptions[~self.feature_descriptions.index.duplicated(keep='first')].copy()
        else:
            self.features_bl = None
        if self.feature_descriptions is not None and self.features:
            self.logger.info(f'{len(self.feature_descriptions)}/{len(self.features)} feature descriptions found')
        self.feature_titles = {index: index + '\n\n' +
           '\n'.join(['\n'.join(wrap(l, 75)) for l in row.astype('str').values.flatten().tolist() if not pd.isnull(l) and l != 'nan'])
                               for index, row in self.feature_descriptions.iterrows()} if self.feature_descriptions is not None else {}

        if n_jobs != 1:
            try:
                n_jobs_max = int(float(os.environ.get('CPU_LIMIT')))
            except:
                n_jobs_max = os.cpu_count()
            if n_jobs > n_jobs_max:
                if verbose:
                    self.logger.warning(f'N_jobs exceeds CPU_LIMIT. Set to {n_jobs_max}')
                n_jobs = n_jobs_max
            elif n_jobs < 1:
                n_jobs = n_jobs_max
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_jobs_restr = 1
        self.min_nunique = min_nunique
        self.ci_analytic = ci_analytic
        self.ci_alpha = ci_alpha
        if compress:
            for name, sample in self.samples.items():
                self.reduce_mem_usage(sample)
        if bootstrap_split is not None and isinstance(bootstrap_split, dict):
            self.bootstrap_split(**bootstrap_split)
        else:
            self.bootstrap_base = None
            self.bootstrap = []
        if samples_split is not None and isinstance(samples_split, dict):
            self.samples_split(**samples_split)
        if self.samples and self.target and verbose:
            self.logger.info('DataSamples stats:\n' + pd.DataFrame([[name, sample.shape[0], sample[self.target].sum(), sample[self.target].sum() / sample.shape[0], f'{sample[self.time_column].min()} - {sample[self.time_column].max()}' if self.time_column is not None else 'NA']
                          for name, sample in {**self.samples, **({'Bootstrap base': self.bootstrap_base} if self.bootstrap_base is not None else {})}.items()], columns=['', 'amount', 'target', 'target_rate', 'period']).set_index('').T.to_string())

    def samples_split(self, df=None, test_size=0.3, validate_size=0, split_type='oos', stratify=True, id_column=None):
        """
        Разбивает датафрейма на сэмплы.
        При split_type:
             'oos' - создаются сэмплы 'Test' и 'Validate' (при validate_size > 0),
             'oot' - создается сэмпл 'Test oot'
        остальные сэмплы не затрагиваются
        :param df: датафрейм из которого нарезаются сэмплы. При None берется self.samples[self.train_name]
        :param test_size: размер сэмпла test
        :param validate_size: размер сэмпла validate
        :param split_type: тип разбиения 'oos' = 'out-of-sample', 'oot' = 'out-of-time'
        :param stratify: Применимо только для split_type='oos', может принимать значения:
                             False - не проводить стратификацию
                             True - стратификация по целевой переменной
                             list - список полей, по которым будет выполнена стратификация при сплите
        :param id_column: название поля с айди. Если задано, то все одинаковые айди распределяются в один сэмпл,
                          в случае split_type='oot' из теста исключаются айди, попавшие в трэйне. Размер теста при этом может стать сильно меньше test_size
        """
        if df is None:
            df = self.samples[self.train_name].copy()
        else:
            self.train_name = 'Train'

        if split_type == 'oos':
            if isinstance(stratify, list):
                stratify_field = 'stratify_tmp_field'
                if self.target in stratify:
                    stratify.remove(self.target)
                df[stratify_field] = df[stratify].astype('str').fillna('nan').agg('~'.join, axis=1)
                if id_column:
                    df[stratify_field] += '~' + df.groupby(id_column)[self.target].transform('max').astype('str')
                    for_split = df[[id_column, stratify_field]].copy().drop_duplicates(subset=id_column)
                else:
                    df[stratify_field] += '~' + df[self.target].astype('str')
                    for_split = df[[stratify_field]].copy()
                df = df.drop([stratify_field], axis=1)
            else:
                stratify_field = self.target
                if id_column:
                    for_split = df.groupby(by=id_column)[[self.target]].max()
                else:
                    for_split = df[[stratify_field]]
            split = {}
            split[self.train_name], split['Test'] = train_test_split(for_split, test_size=test_size, random_state=self.random_state, stratify=for_split[stratify_field] if stratify != False else None)
            if validate_size > 0:
                split[self.train_name], split['Validate'] = train_test_split(split[self.train_name], test_size=validate_size/(1 - test_size), random_state=self.random_state, stratify=split[self.train_name][stratify_field] if stratify != False else None)
            if id_column:
                dr = self.samples[self.train_name][self.target].mean()
                for name, sp in split.items():
                    tmp = df[df[id_column].isin(sp.index)]
                    if stratify != False and name != self.train_name:
                        dr_tmp = tmp[self.target].mean()
                        if dr_tmp > dr:
                            g = tmp[tmp[self.target] == 0]
                            tmp = pd.concat([g, tmp[tmp[self.target] == 1].sample(n=round(len(g) * dr / (1 - dr)), random_state=self.random_state)])
                        else:
                            b = tmp[tmp[self.target] == 1]
                            tmp = pd.concat([tmp[tmp[self.target] == 0].sample(n=round(len(b) * (1 - dr) / dr), random_state=self.random_state), b])
                    self.samples[name] = tmp
            else:
                self.samples.update({name: df.loc[list(sp.index)] for name, sp in split.items()})

        elif split_type == 'oot':
            if self.time_column is None:
                self.logger.error('Wich column contains time data? Please pay attention to time_column parameter')
                return None
            if validate_size > 0:
                self.logger.warning('Validation for oot is unavailable')
            else:
                tmp_dataset = df.copy().sort_values(by=self.time_column)
                test_threshold = list(tmp_dataset[self.time_column].drop_duplicates())[int(round((1 - test_size)*len(tmp_dataset[self.time_column].drop_duplicates()), 0))]
                self.samples.update({self.train_name: tmp_dataset[tmp_dataset[self.time_column] < test_threshold],
                                'Test oot': tmp_dataset[tmp_dataset[self.time_column] >= test_threshold]})
                if id_column is not None:
                    self.samples['Test oot'] = self.samples['Test oot'][~self.samples['Test oot'][id_column].isin(self.samples[self.train_name][id_column])]
        else:
            self.logger.error('Wrong split type. Please use oot or oos')
            return None
        self.logger.info('Actual parts of samples after samples split:\n' + pd.DataFrame([[name, round(sample.shape[0]/df.shape[0], 4)]
                          for name, sample in self.samples.items()], columns=['', 'part']).set_index('').T.to_string())

    def bootstrap_split(self, df=None, bootstrap_part=1, bootstrap_number=100, stratify=True, replace=True):
        """
        Создание подвыборок для бутстрэпа
        :param df: датафрейм, основа для нарезания подвыборок, при None используется self.samples[self.train_name]
        :param bootstrap_part: размер каждой подвыборки
        :param bootstrap_number: кол-во подвыборок
        :param stratify: стратификация каждой подвыборки по целевой переменной
        :param replace: разрешается ли повторять каждое наблюдение множество раз в подвыборке
        """
        if self.ci_analytic:
            self.logger.warning("Using bootstrap isn't possible with ci_analytic=True. Set ci_analytic=False.")
            self.ci_analytic = False
        self.bootstrap_base = df if df is not None else self.samples[self.train_name]
        if 'Infinity' in self.special_bins:
            self.bootstrap_base[self.features] = self.bootstrap_base[self.features].replace([np.inf, -np.inf], self.special_bins['Infinity'])
        if stratify:
            class0 = self.bootstrap_base[self.bootstrap_base[self.target] == 0]
            class1 = self.bootstrap_base[self.bootstrap_base[self.target] == 1]
        self.bootstrap = [self.bootstrap_base.index.get_indexer(        class1.sample(frac=bootstrap_part, replace=replace, random_state=self.random_state+bi).index\
                                                                .append(class0.sample(frac=bootstrap_part, replace=replace, random_state=self.random_state+bi).index)
                                                                if stratify else
                                                                self.bootstrap_base.sample(frac=bootstrap_part, replace=replace, random_state=self.random_state+bi).index)
                          for bi in range(bootstrap_number)]
        self.logger.info(f'{len(self.bootstrap)} bootstrap samples with {len(self.bootstrap[0])} observation each and '
                         f'{round(np.array([self.bootstrap_base.iloc[idx][self.target].mean() for idx in self.bootstrap]).mean(), 4)} mean target rate were created' )

    def to_df(self, sample_field='sample'):
        """
        Конвертирование ДатаСэмпла в датафрейм
        :param sample_field: добавляемое поле, в которое будет записано название сэмплов

        :return: датафрейм
        """
        return pd.concat([sample.copy().assign(**{sample_field: name}) if sample_field else sample for name, sample in self.samples.items()])

    def stats(self, out=None, gini_in_time=True, targettrend=None, stratify=None, test_size=0.3):
        """
        Вычисление статистики по сэмплам
        :param out: название эксель файла для сохранения статистики
        :param gini_in_time: флаг для расчета динамики джини по срезам. На больших выборках с бутстрэпом может занимать много времени
        :param targettrend: название папки и листа в файле для сохранения графиков TargetTrend. При None TargetTrend не считается
        :param stratify: список полей, по которым проводилась стратификация. Используется для проверки качества стратификации
        :param test_size: заданный размер тестовой выборки. Используется для проверки качества стратификации
        """
        def get_calculated_extended_dr_pivot_by_subsamples(
                X_1: pd.DataFrame,
                X_2: pd.DataFrame,
                strat_list,
                target_name: str,
                test_size: float = 0.3,
        ) -> pd.DataFrame:
            """
            Проверка статистик по итогам разбиения выборок.
            """

            # Группировка по train по стратам
            train_grouped_stat = (
                X_1.groupby(strat_list)[target_name]
                    .agg(["count", "sum"])
                    .rename(columns={"count": "n_obs", "sum": "n_bad"})
            )

            train_grouped_stat["n_good"] = (
                    train_grouped_stat["n_obs"] - train_grouped_stat["n_bad"]
            )
            train_grouped_stat.loc["ALL"] = train_grouped_stat.sum()
            train_grouped_stat["dr"] = train_grouped_stat["n_bad"] / train_grouped_stat["n_obs"]
            train_grouped_stat["ideal"] = 1 - test_size  # %train
            train_grouped_stat.columns = [i + "_train" for i in train_grouped_stat.columns]

            # Группировка по test по стратам
            test_grouped_stat = (
                X_2.groupby(strat_list)[target_name]
                    .agg(["count", "sum"])
                    .rename(columns={"count": "n_obs", "sum": "n_bad"})
            )

            test_grouped_stat["n_good"] = (
                    test_grouped_stat["n_obs"] - test_grouped_stat["n_bad"]
            )
            test_grouped_stat.loc["ALL"] = test_grouped_stat.sum()
            test_grouped_stat["dr"] = test_grouped_stat["n_bad"] / test_grouped_stat["n_obs"]
            test_grouped_stat["ideal"] = test_size  # %test
            test_grouped_stat.columns = [i + "_test" for i in test_grouped_stat.columns]

            # Объединяем train+test
            tr_test_grouped_stat = pd.concat([train_grouped_stat, test_grouped_stat], axis=1)

            # Считаем train+test
            tr_test_grouped_stat["n_obs_all"] = (
                    tr_test_grouped_stat["n_obs_train"] + tr_test_grouped_stat["n_obs_test"]
            )
            tr_test_grouped_stat["n_bad_all"] = (
                    tr_test_grouped_stat["n_bad_train"] + tr_test_grouped_stat["n_bad_test"]
            )
            tr_test_grouped_stat["n_good_all"] = (
                    tr_test_grouped_stat["n_obs_all"] - tr_test_grouped_stat["n_bad_all"]
            )

            # Считаем долю в % от all и отклонения от all
            for subsample_n in ["train", "test"]:
                tr_test_grouped_stat[f"p_obs_{subsample_n}"] = (
                        tr_test_grouped_stat[f"n_obs_{subsample_n}"]
                        / tr_test_grouped_stat["n_obs_all"]
                )
                tr_test_grouped_stat[f"p_good_{subsample_n}"] = (
                        tr_test_grouped_stat[f"n_good_{subsample_n}"]
                        / tr_test_grouped_stat["n_good_all"]
                )
                tr_test_grouped_stat[f"p_bad_{subsample_n}"] = (
                        tr_test_grouped_stat[f"n_bad_{subsample_n}"]
                        / tr_test_grouped_stat["n_bad_all"]
                )

                tr_test_grouped_stat[f"p_absdelta_obs_{subsample_n}"] = abs(
                    tr_test_grouped_stat[f"p_obs_{subsample_n}"]
                    - tr_test_grouped_stat[f"ideal_{subsample_n}"]
                )
                tr_test_grouped_stat[f"p_absdelta_good_{subsample_n}"] = abs(
                    tr_test_grouped_stat[f"p_good_{subsample_n}"]
                    - tr_test_grouped_stat[f"ideal_{subsample_n}"]
                )
                tr_test_grouped_stat[f"p_absdelta_bad_{subsample_n}"] = abs(
                    tr_test_grouped_stat[f"p_bad_{subsample_n}"]
                    - tr_test_grouped_stat[f"ideal_{subsample_n}"]
                )

            # Сортируем
            tr_test_grouped_stat = tr_test_grouped_stat[["n_obs_all", "n_bad_all", "n_good_all", "ideal_train", "n_obs_train",
                    "p_obs_train", "p_absdelta_obs_train", "n_good_train", "p_good_train", "p_absdelta_good_train",
                    "n_bad_train", "p_bad_train", "p_absdelta_bad_train", "dr_train", "ideal_test", "n_obs_test",
                    "p_obs_test", "p_absdelta_obs_test", "n_good_test", "p_good_test", "p_absdelta_good_test",
                    "n_bad_test", "p_bad_test", "p_absdelta_bad_test",  "dr_test"]]
            return tr_test_grouped_stat

        stats = pd.DataFrame([[name, sample.shape[0], sample[self.target].sum(), sample[self.target].sum() / sample.shape[0],
                               f'{sample[self.time_column].min()} - {sample[self.time_column].max()}' if self.time_column is not None else 'NA']
                            for name, sample in {**self.samples, **({'Bootstrap base': self.bootstrap_base} if self.bootstrap_base is not None else {})}.items()],
                            columns=['', 'amount', 'target', 'target_rate', 'period']).set_index('').T
        if out:
            with pd.ExcelWriter(self.result_folder + out, engine='xlsxwriter') as writer:
                stats.to_excel(writer, sheet_name='Sample stats')
                ws = writer.sheets['Sample stats']
                format1 = writer.book.add_format({'num_format': '0.00%'})
                ws.set_row(3, None, format1)
                ws.set_column(0, len(self.samples) + 1, 30)

                for name, sample in self.samples.items():
                    tmp = sample[self.features].describe(percentiles=[0.05, 0.5, 0.95], include='all').T
                    try:
                        f_stats = f_stats.merge(tmp, left_index=True, right_index=True, how='left')
                    except:
                        f_stats = tmp.copy()
                f_stats.columns = pd.MultiIndex.from_product([list(self.samples.keys()), tmp.columns])
                f_stats.to_excel(writer, sheet_name='Feature stats')
                gini_df = self.calc_gini(prebinning=True, add_description=True)
                self.corr_mat(description_df=gini_df, styler=True).to_excel(writer, sheet_name='Correlation')
                adjust_cell_width(writer.sheets['Correlation'], gini_df)
                gini_df.to_excel(writer, sheet_name='Gini stats')
                adjust_cell_width(writer.sheets['Gini stats'], gini_df)
                if gini_in_time and self.time_column is not None:
                    self.calc_gini_in_time(prebinning=True).to_excel(writer, sheet_name='Gini stats', startrow=len(self.features) + 3)
                if targettrend and isinstance(targettrend, str):
                    ws = writer.book.add_worksheet(targettrend)
                    for i, fig in enumerate(self.targettrend(quantiles=10)):
                        fig_to_excel(fig, ws, row=i * 20, col=0, scale=0.85)

                get_calculated_extended_dr_pivot_by_subsamples(
                    X_1=self.samples[self.train_name],
                    X_2=self.samples['Test'],
                    strat_list=[self.target] if stratify is None else stratify,
                    target_name=self.target,
                    test_size=test_size).to_excel(writer, sheet_name='Stratification check')
            self.logger.info(f'Statistics saved in the file {self.result_folder + out}.')
        self.logger.info('DataSamples stats:\n' + stats.to_string())

    def calc_gini(self, samples=None, features=None, fillna=None, add_description=False, abs=False, prebinning=False):
        """
        Вычисление джини всех переменных
        :param samples: список названий сэмплов для расчета. При None вычисляется на всех доступных сэмпплах
        :param features: список переменных для расчета. При None берется self.features
        :param fillna: значение для заполнения пропусков. При None пропуски не заполняются
        :param add_description: флаг для добавления в датафрейм описания перемнных из self.feature_descriptions
        :param abs: возвращать абсолютные значения джини
        :param prebinning: выполнить пребиннинг переменных перед вычислением джини

        :return: датафрейм с джини
        """
        if features is None:
            features = self.features
        if samples is None:
            samples = list(self.samples) + ['Bootstrap']
            main_sample = self.train_name
        else:
            main_sample = samples[0]

        if prebinning:
            try:
                optb = optbinning.BinningProcess(variable_names=features, categorical_variables=self.cat_columns,
                                                 max_n_prebins=10, min_prebin_size=0.05)
                optb.fit(self.samples[self.train_name][features], self.samples[self.train_name][self.target])
            except Exception as e:
                self.logger.error(e)
                self.logger.warning('Error during prebinning, set prebinning=False')
                prebinning = False
        ginis = {name: self.get_features_gini(sample[self.target],
                                              sample[[f for f in features if f in sample]] if not prebinning else optb.transform(sample[features]),
                                              fillna=fillna, abs_flag=abs)
                 for name, sample in self.samples.items() if name in samples}

        if not self.ci_analytic and self.bootstrap_base is not None and 'Bootstrap' in samples:
            bts_features = [f for f in features if f in self.bootstrap_base.columns]
            if prebinning:
                bootstrap_base = pd.concat([optb.transform(self.bootstrap_base[features]), self.bootstrap_base[self.target]], axis=1)
            else:
                bootstrap_base = self.bootstrap_base
            if bts_features:
                if self.n_jobs_restr > 1:
                    with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                        ginis_bootstrap = []
                        jobs = []
                        iterations = len(self.bootstrap)
                        idx_iter = iter(self.bootstrap)
                        while iterations:
                            for idx in idx_iter:
                                jobs.append(pool.submit(self.get_features_gini, y=bootstrap_base.iloc[idx][self.target],
                                                        df=bootstrap_base.iloc[idx][bts_features], fillna=fillna, abs_flag=abs))
                                if len(jobs) > self.n_jobs*2:
                                    break
                            for job in futures.as_completed(jobs):
                                ginis_bootstrap.append(job.result())
                                jobs.remove(job)
                                iterations -= 1
                                break
                    gc.collect()
                else:
                    ginis_bootstrap = [self.get_features_gini(bootstrap_base.iloc[idx][self.target], bootstrap_base.iloc[idx][bts_features], fillna=fillna, abs_flag=abs)
                                       for idx in self.bootstrap]
                z = ndtri(1 - self.ci_alpha / 2)
                mean = {f: np.mean([ginis[f] for ginis in ginis_bootstrap]) for f in bts_features}
                std = {f: np.std([ginis[f] for ginis in ginis_bootstrap]) for f in bts_features}
                ginis['CI_lower'] = {f: (mean[f] - z * std[f]) for f in bts_features}
                ginis['CI_upper'] = {f: (mean[f] + z * std[f]) for f in bts_features}
        else:
            ginis['CI_lower'], ginis['CI_upper'] = self.get_ci_gini(ginis[main_sample], self.samples[main_sample])

        result = pd.DataFrame(ginis).round(2)
        if add_description:
            try:
                self.add_descriptions(features)
                tmp = self.feature_descriptions.copy()
                tmp.index += '_WOE'
                result = pd.concat([self.feature_descriptions, tmp]).merge(result, left_index=True, right_index=True, how='right')
            except:
                pass
        return result

    def calc_gini_in_time(self, samples=None, features=None, fillna=None, abs=False, prebinning=False):
        """
        Вычисление динамики джини по срезам для всех переменных
        Доступно только если задано значение self.time_column
        :param samples: список названий сэмплов для расчета. При None вычисляется на всех доступных сэмпплах
        :param features: писок переменных для расчета. При None берется self.features
        :param fillna: значение для заполнения пропусков. При None пропуски не заполняются
        :param abs: возвращать абсолютные значения джини
        :param prebinning: выполнить пребиннинг переменных перед вычислением джини

        :return: датафрейм с джини
        """
        if samples is None:
            samples = list(self.samples) + ['Bootstrap']
            main_sample = self.train_name
        else:
            main_sample = samples[0]

        if self.time_column is None:
            return pd.DataFrame()

        if features is None:
            features = self.features
        if prebinning:
            try:
                optb = optbinning.BinningProcess(variable_names=features, categorical_variables=self.cat_columns,
                                                 max_n_prebins=10, min_prebin_size=0.05)
                optb.fit(self.samples[self.train_name][features], self.samples[self.train_name][self.target])
            except Exception as e:
                self.logger.error(e)
                self.logger.warning('Error during prebinning, set prebinning=False')
                prebinning = False
        ginis_in_time = {name: self.get_time_features_gini(sample[[self.target, self.time_column] + [f for f in features if f in sample.columns]] if not prebinning
                                                                else pd.concat([sample[[self.target, self.time_column]], optb.transform(sample[features])], axis=1),
                                                           target=self.target, time_column=self.time_column, fillna=fillna, abs_flag=abs)
                         for name, sample in self.samples.items() if name in samples}
        if not self.ci_analytic and self.bootstrap_base is not None and 'Bootstrap' in samples:
            bts_features = [f for f in features if f in self.bootstrap_base.columns]
            if bts_features:
                if prebinning:
                    bootstrap_base = pd.concat([optb.transform(self.bootstrap_base[features]), self.bootstrap_base[self.target]], axis=1)
                else:
                    bootstrap_base = self.bootstrap_base.copy()
                if self.n_jobs_restr > 1:
                    with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                        ginis_bootstrap = []
                        jobs = []
                        iterations = len(self.bootstrap)
                        idx_iter = iter(self.bootstrap)
                        while iterations:
                            for idx in idx_iter:
                                jobs.append(pool.submit(self.get_time_features_gini,
                                                        df=bootstrap_base.iloc[idx][[self.target, self.time_column] + [f for f in bts_features]],
                                                        target=self.target, time_column=self.time_column, fillna=fillna, abs_flag=abs))
                                if len(jobs) > self.n_jobs * 2:
                                    break
                            for job in futures.as_completed(jobs):
                                ginis_bootstrap.append(job.result())
                                jobs.remove(job)
                                iterations -= 1
                                gc.collect()
                                break
                    gc.collect()
                else:
                    ginis_bootstrap = [self.get_time_features_gini(bootstrap_base.iloc[idx][[self.target, self.time_column] + bts_features],
                                                                   target=self.target, time_column=self.time_column, fillna=fillna, abs_flag=abs)
                                       for idx in self.bootstrap]
                time_values = sorted(bootstrap_base[self.time_column].unique())
                z = ndtri(1 - self.ci_alpha / 2)
                mean = {time: {f: np.mean([ginis[time][f] for ginis in ginis_bootstrap if time in ginis]) for f in bts_features} for time in time_values}
                std = {time: {f: np.std([ginis[time][f] for ginis in ginis_bootstrap if time in ginis]) for f in bts_features} for time in time_values}
                ginis_in_time['CI_lower'] = {time: {f: (mean[time][f] - z * std[time][f]) for f in bts_features} for time in time_values}
                ginis_in_time['CI_upper'] = {time: {f: (mean[time][f] + z * std[time][f]) for f in bts_features} for time in time_values}
        else:
            ginis_in_time['CI_lower'] = {}
            ginis_in_time['CI_upper'] = {}
            for time, group in self.samples[main_sample].groupby(self.time_column):
                ginis_in_time['CI_lower'][time], ginis_in_time['CI_upper'][time] = self.get_ci_gini(ginis_in_time[main_sample][time], group)
        time_values = sorted(list({time for name in ginis_in_time for time in ginis_in_time[name]}))
        result = pd.DataFrame([[time] + [ginis_in_time[name][time][f] if time in ginis_in_time[name] and f in ginis_in_time[name][time] else 0
                                         for f in features for name in ginis_in_time]
                               for time in time_values],
                            columns=[self.time_column] + [f'Gini {name} {f}' for f in features for name in ginis_in_time]).set_index(self.time_column).round(2)
        result.columns = pd.MultiIndex.from_product([features, list(ginis_in_time.keys())])
        if abs:
            result = result.abs()
        return result

    def corr_mat(self, sample_name=None, features=None, corr_method='pearson', corr_threshold=0.75, description_df=None, styler=False):
        """
        Вычисление матрицы корреляций
        :param sample_name: название сэмпла, из которого берутся данные. По умолчанию self.train_name
        :param features: список переменных для расчета. По умолчанию берутся из self.features
        :param corr_method: метод расчета корреляций. Доступны варианты 'pearson', 'kendall', 'spearman'
        :param corr_threshold: трэшхолд значения корреляции. Используется для выделения значений цветом
        :param description_df: Датафрейм с описанием переменных. Будет приджойнем к матрице корреляций по индексу
        :param styler: При True возвращается датафрейм styler, иначе датафрейм

        :return: датафрейм styler либо датафрейм
        """
        if features is None:
            features = self.features
        if sample_name is None:
            sample_name = self.train_name
        features = [f for f in features if pd.api.types.is_numeric_dtype(self.samples[self.train_name][f])]
        corr_df = self.samples[sample_name][features].corr(method=corr_method).rename({f: i for i, f in enumerate(features, start=1)}, axis=1)
        if description_df is not None:
            corr_df = description_df.merge(corr_df, left_index=True, right_index=True, how='right')
        corr_df.index = corr_df.index.map({f: f'{f} ({i})' for i, f in enumerate(features, start=1)})
        if styler:
            return corr_df.round(2).style.applymap(lambda x: 'color: black' if isinstance(x, str) or x > 1 or x < -1 else 'color: red'
                                                                            if abs(x) > corr_threshold else 'color: orange'
                                                                            if abs(x) > corr_threshold ** 2 else 'color: green',
                                                   subset=pd.IndexSlice[:, list(range(1, len(features) + 1))])
        else:
            return corr_df.round(2)

    def psi(self, time_column=None, sample_name=None, features=None, thres_dict=None, base_period_index=-1, n_bins=5,
            scorecard=None, plot_flag=False, out=None, sheet_name='PSI'):
        """
        Вычисление Population Stability Index
        StabilityIndex[t] = (N[i, t]/sum_i(N[i, t]) - (N[i, 0]/sum_i(N[i, 0])))* log(N[i, t]/sum_i(N[i, t])/(N[i, 0]/sum_i(N[i, 0])))
        где N[i, t]  - кол-во наблюдений со значением i в срезе t.

        :param time_column: название поля, по которому формируются срезы. При None тест проводится между сэмплами
        :param sample_name: название сэмпла, из которого берутся данные. По умолчанию self.train_name
        :param features: список переменных для расчета. По умолчанию берется self.features
        :param thres_dict: словарь с граница желтой и красной зон для значений PSI и Filled in values.
                           По-умолчанию имеет значение {'psi': {'yellow': 0.1, 'red': 0.25}, 'filled': {'yellow': 0.5, 'red': 0.25}}
        :param base_period_index: индекс основного среза в отсортированном списке значений срезов, относительного которого считается PSI остальных срезов
                                    при -1 тест для каждого среза считается относительно предыдущего
        :param n_bins: кол-во бинов на которые будут разбиты значения переменных, если кол-во уникальных значений > n_bins.
                       Не применяется к WOE-трансформированным переменным
        :param scorecard: датафрейм со скоркартой. Используется для добавления описания значений WOE в легенде графика PSI
        :param plot_flag: флаг для вывода графиков распределений
        :param out: Строка с названием эксель файла, либо объект pd.ExcelWriter для сохранения результатов. При None результат не сохраняется
        :param sheet_name: Название листа в эксель файле

        :return: датафрейм со значениями PSI
        """
        if sample_name is None:
            sample_name = self.train_name

        if features is None:
            features = self.features
        if not features:
            self.logger.error('Features are empty!')
            return
        features = list(dict.fromkeys(features))
        if thres_dict is None:
            thres_dict = {'psi': {'yellow': 0.1, 'red': 0.25},
                          'filled': {'yellow': 0.5, 'red': 0.25}
                          }
        if isinstance(out, str):
            writer = pd.ExcelWriter(add_ds_folder(self, out), engine='xlsxwriter')
        elif isinstance(out, pd.ExcelWriter):
            writer = out

        if time_column is None:
            time_column = 'tmp_sample'
            tmp_dataset = self.to_df(sample_field=time_column)
        else:
            tmp_dataset = self.samples[sample_name]
        plot = plot_flag or out is not None
        tmp_dataset = tmp_dataset[list(set(features + [rem_suffix(f) for f in features if plot and rem_suffix(f) in tmp_dataset])) + [time_column, self.target]].copy()
        for f in features:
            if plot:
                rem_suffix_f = rem_suffix(f) if f.endswith('_WOE') and rem_suffix(f) in tmp_dataset else f
                if pd.api.types.is_numeric_dtype(tmp_dataset[rem_suffix_f]):
                    tmp_dataset[f + '_filled_tmp'] = ((~tmp_dataset[rem_suffix_f].isin(list(self.special_bins.values()))) &
                                                      (tmp_dataset[rem_suffix_f].fillna(0) != 0)).astype('int8')
                else:
                    tmp_dataset[f + '_filled_tmp'] = ((~tmp_dataset[rem_suffix_f].isin(list(self.special_bins.values()))) &
                                                      (~tmp_dataset[rem_suffix_f].isna())).astype('int8')
            else:
                tmp_dataset[f + '_filled_tmp'] = 0
            if f.endswith('_WOE'):
                tmp_dataset[f] = tmp_dataset[f].astype('float64')
            elif pd.api.types.is_numeric_dtype(tmp_dataset[f]):
                no_sb = tmp_dataset[~tmp_dataset[f].isin(list(self.special_bins.values()))][f]
                if no_sb.nunique() > n_bins:
                    bins = pd.qcut(no_sb, n_bins, precision=2, duplicates='drop', retbins=True)[1]
                    tmp_dataset[f] = pd.cut(tmp_dataset[f], bins).astype('str').replace('nan', np.nan) \
                        .fillna(tmp_dataset[f].map({v: str(k) for k, v in self.special_bins.items()}))

        stats = pd.concat([tmp_dataset.groupby([f, time_column], as_index=False)\
                          .agg(size=(self.target, 'size'),
                               er=(self.target, 'mean'),
                               filled=(f + '_filled_tmp', 'sum'))\
                          .rename({f: 'value'}, axis=1)\
                          .assign(**{'feature': f})
                           for f in features])
        stats[['full_size', 'filled']] = stats.groupby(['feature', time_column])[['size', 'filled']].transform('sum')
        stats['size'] = stats['size'] / stats['full_size']
        stats['filled'] = stats['filled'] / stats['full_size']
        stat_size = stats[['feature', 'value', time_column, 'size']].groupby(['feature', 'value', time_column], sort=False)['size'].sum()\
            .unstack(time_column).reset_index().fillna(0).rename_axis(None, axis=1)
        dates = list(stat_size.drop(['feature', 'value'], axis=1).columns)
        stat_psi = stat_size[['feature', 'value']].copy()
        for i, date in enumerate(dates):
            if time_column == 'tmp_sample':
                start_date = self.train_name
            else:
                start_date = dates[i - 1 if i > 0 else 0] if base_period_index == -1 else dates[base_period_index]
            stat_psi[date] = ((stat_size[date] - stat_size[start_date]) * np.log(
                stat_size[date] / stat_size[start_date])).replace([+np.inf, -np.inf], 0).fillna(0)
        stat_psi = stat_psi.drop(['value'], axis=1).groupby(by='feature', sort=False).sum().round(2)
        if plot:
            figs = []
            if scorecard is not None:
                scorecard = scorecard[scorecard['values'] != 'all others'].copy()
                scorecard['woe_map'] = scorecard['woe'].astype('str')
                scorecard['values_map'] = scorecard['values'].astype('str') + '. WOE ' + scorecard['woe_map']
                legend_map = {add_suffix(f): f_df.set_index('woe_map')['values_map'].to_dict() for f, f_df in
                              scorecard.groupby('feature')}
            else:
                legend_map = {f: {str(float(v)): str(k) for k, v in self.special_bins.items()} for f in features}
            date_base = pd.DataFrame(stats[time_column].unique(), columns=[time_column]).sort_values(time_column)
            for f, f_stats in stats.groupby('feature', sort=False):
                fig, ax = plt.subplots(1, 1, figsize=(7 + tmp_dataset[time_column].nunique(), 5))
                ax2 = ax.twinx()
                ax.grid(False)
                ax2.grid(False)
                sorted_values = sorted(f_stats['value'].unique(), reverse=True)
                for value in sorted_values:
                    value_filter = (f_stats['value'] == value)
                    er = date_base.merge(f_stats[value_filter], on=time_column, how='left')['er']
                    height = date_base.merge(f_stats[value_filter], on=time_column, how='left')['size'].fillna(0)
                    bottom = date_base.merge(f_stats[[time_column, 'size']][f_stats['value'] > value] \
                                             .groupby(time_column, as_index=False).sum(), on=time_column, how='left')['size'].fillna(0)
                    ax.bar(range(date_base.shape[0]), height, bottom=bottom if value != sorted_values[0] else None, edgecolor='white', alpha=0.3)
                    ax2.plot(range(date_base.shape[0]), er, label=value if isinstance(value, str) else str(round(value, 3)), linewidth=2)
                plt.xticks(range(date_base.shape[0]), date_base[time_column])
                fig.autofmt_xdate()
                ax2.set_ylabel('Target Rate')
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
                ax2.annotate('Amount:', xy=(-0.5, 1), xycoords=('data', 'axes fraction'),
                             xytext=(0, 60), textcoords='offset pixels', color='black', ha='right', size=11)
                for i in range(date_base.shape[0]):
                    ax2.annotate(
                        str(int(f_stats[f_stats[time_column] == date_base[time_column][i]]['full_size'].values[0])),
                        xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, 60),
                        textcoords='offset pixels', ha='center', color='black', size=11)

                ax2.annotate('Filled in values:', xy=(-0.5, 1), xycoords=('data', 'axes fraction'), xytext=(0, 35),
                             textcoords='offset pixels', color='black', ha='right', size=11)
                for i in range(date_base.shape[0]):
                    s = f_stats[f_stats[time_column] == date_base[time_column][i]]['filled'].values[0]
                    ax2.annotate(
                        'NA' if f.endswith('_WOE') and rem_suffix(f) not in tmp_dataset else f'{round(s * 100, 1)}%',
                        xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, 35),
                        textcoords='offset pixels', ha='center',
                        color='black' if f.endswith('_WOE') and rem_suffix(f) not in tmp_dataset
                        else 'green' if s > thres_dict['filled']['yellow'] else 'orange' if s > thres_dict['filled']['red'] else 'red', size=11)

                ax2.annotate('PSI:', xy=(-0.5, 1), xycoords=('data', 'axes fraction'), xytext=(0, 10),
                             textcoords='offset pixels', color='black', ha='right', size=11)
                for i in range(date_base.shape[0]):
                    psi = round(stat_psi[stat_psi.index == f][date_base[time_column][i]].values[0], 2)
                    ax2.annotate(str(psi), xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, 10),
                                 textcoords='offset pixels', ha='center', size=11,
                                 color='green' if psi < thres_dict['psi']['yellow'] else 'orange' if psi < thres_dict['psi']['red'] else 'red')

                ax.set_ylabel('Observations')
                plt.xlabel(time_column)
                plt.suptitle(self.feature_titles[f] if f in self.feature_titles else f, fontsize=14, weight='bold')
                plt.tight_layout()
                handles, labels = ax2.get_legend_handles_labels()
                if legend_map:
                    labels = [legend_map[f][l] if f in legend_map and l in legend_map[f] else l for l in labels]
                ax2.legend(handles[::-1], labels[::-1], fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(dates))
                figs.append(fig)
                if plot_flag:
                    plt.show()
            plt.close('all')

            if out is not None:
                stat_psi.style.applymap(lambda x: 'color: red' if x > thres_dict['psi']['red'] else 'color: orange' if x > thres_dict['psi']['yellow'] else 'color: green').to_excel(writer, sheet_name=sheet_name)
                ws = writer.sheets[sheet_name]
                ws.set_column(0, 0, 40)
                ws.set_column(1, len(stat_psi.columns) + 1, 12)
                for i, fig in enumerate(figs):
                    fig_to_excel(fig, ws, row=i * (22 + 1) + len(features) + 3, col=0, scale=0.9)
                if isinstance(out, str):
                    writer.close()
                    self.logger.info(f'PSI results is saved to file {out}')
        return stat_psi

    def plot_distribution(self, features=None, bins=20, round_digits=3, plot_flag=True):
        """
        Отрисовка распределения значений переменной. Работает как с непрервными, так и дескретными переменными
        :param features: список переменных для обработки
        :param bins: кол-во бинов в распределении. Если в переменной число уникальных значений больше этого кол-ва, то она перебинивается
        :param round_digits: кол-во знаков после запятой
        :param plot_flag: флаг для вывода графиков распределений

        :return: список из графиков [plt.figure]
        """
        if features is None:
            features = self.features
        figs = []
        for feature in features:
            to_cut = self.samples[self.train_name][feature]
            if to_cut.nunique() > bins:
                if self.special_bins:
                    to_cut = to_cut[~to_cut.isin(list(self.special_bins.values()))]
                _, cuts = pd.cut(to_cut, bins=bins, right=False, precision=round_digits, retbins=True)
                cuts[0] = -np.inf
                cuts[-1] = np.inf
            fig = plt.figure(figsize=(9, 4))
            ax = fig.add_subplot(111)
            num = 0
            for name, sample in self.samples.items():
                if to_cut.nunique() > bins:
                    stats = pd.cut(sample[feature], bins=cuts, right=False, precision=round_digits).value_counts().sort_index()
                else:
                    stats = to_cut.value_counts().sort_index()
                stats = stats/stats.sum()
                plt.bar(np.array(range(stats.shape[0])) + num*0.2, stats, width=0.2, label=name)
                if name == self.train_name:
                    plt.xticks(np.array(range(stats.shape[0])) + len(self.samples) * 0.2 / 2, stats.index.astype(str))
                num += 1

            fig.autofmt_xdate()
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
            plt.legend()
            plt.suptitle(self.feature_titles[feature] if feature in self.feature_titles else feature, fontsize=14, weight='bold')
            plt.tight_layout()
            if plot_flag:
                plt.show()
            figs.append(fig)
        plt.close('all')
        return figs

    @staticmethod
    def targettrend_feature(df, special_bins, feature_titles, quantiles, plot_flag):
        magnify_trend = False
        magnify_std_number = 2
        hide_every_even_tick_from = 50
        min_size = 10
        target = df.columns[0]
        f = df.columns[1]
        tmp = df.copy()
        if special_bins:
            tmp = tmp[~tmp[f].isin(list(special_bins.values()))]
        if not pd.api.types.is_numeric_dtype(tmp[f]) or tmp[f].unique().shape[0] < quantiles:
            summarized = tmp[[f, target]].groupby([f]).agg(['mean', 'size'])
        else:
            tmp = tmp.dropna()
            if tmp[f].shape[0] < min_size * quantiles:
                current_quantiles = int(tmp[f].shape[0] / min_size)
                if current_quantiles == 0:
                    return None
            else:
                current_quantiles = quantiles
            summarized = tmp[[target]].join(pd.qcut(tmp[f], q=current_quantiles, precision=4, duplicates='drop')).groupby([f]).agg(['mean', 'size'])
            small_quantiles = summarized[target][summarized[target]['size'] < min_size]['size']
            if small_quantiles.shape[0] > 0:
                current_quantiles = int(small_quantiles.sum() / min_size) + summarized[target][summarized[target]['size'] >= min_size].shape[0]
                summarized = tmp[[target]].join(pd.qcut(tmp[f], q=current_quantiles, precision=3, duplicates='drop')).groupby([f]).agg(['mean', 'size'])

        summarized.columns = summarized.columns.droplevel()
        summarized = summarized.reset_index()
        if pd.isnull(df[f]).any():
            with_na = df[[f, target]][pd.isnull(df[f])]
            summarized.loc[-1] = [np.nan, with_na[target].mean(), with_na.shape[0]]
            summarized = summarized.sort_index().reset_index(drop=True)
        if special_bins:
            add_rows = []
            for k, v in special_bins.items():
                special_group = df[[f, target]][df[f] == v]
                if special_group.shape[0] > 0:
                    add_rows.append([k, special_group[target].mean(), special_group.shape[0]])
            if add_rows:
                summarized = pd.concat([pd.DataFrame(add_rows, columns=[f, 'mean', 'size']), summarized])
        if summarized.shape[0] == 1:
            return None

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        ax.set_ylabel('Observations')
        # blue is for the distribution
        if summarized.shape[0] > hide_every_even_tick_from:
            plt.xticks(range(summarized.shape[0]), summarized[f].astype(str), rotation=60, ha="right")
            xticks = ax.xaxis.get_major_ticks()
            for i in range(len(xticks)):
                if i % 2 == 0:
                    xticks[i].label1.set_visible(False)
        else:
            plt.xticks(range(summarized.shape[0]), summarized[f].astype(str), rotation=45, ha="right")

        ax.bar(range(summarized.shape[0]), summarized['size'], zorder=0, alpha=0.3)
        ax.grid(False)
        ax.grid(axis='y', zorder=1, alpha=0.6)
        ax2 = ax.twinx()
        ax2.set_ylabel('Target Rate')
        ax2.grid(False)

        if magnify_trend:
            ymax = np.average(summarized['mean'], weights=summarized['size']) + magnify_std_number * np.sqrt(
                np.cov(summarized['mean'], aweights=summarized['size']))
            if pd.isnull(ymax):
                ymax = summarized['mean'].mean()
            ax2.set_ylim([0, ymax])
            for i in range(len(summarized['mean'])):
                if summarized['mean'][i] > np.average(summarized['mean'], weights=summarized['size']) + magnify_std_number * np.sqrt(np.cov(summarized['mean'], aweights=summarized['size'])):
                    ax2.annotate(str(round(summarized['mean'][i], 4)),
                                 xy=(i, np.average(summarized['mean'], weights=summarized['size']) + magnify_std_number * np.sqrt(
                                     np.cov(summarized['mean'], aweights=summarized['size']))),
                                 xytext=(i, np.average(summarized['mean'], weights=summarized['size']) + (magnify_std_number + 0.05) * np.sqrt(
                                     np.cov(summarized['mean'], aweights=summarized['size']))),
                                 rotation=60, ha='left', va='bottom', color='red', size=8.5
                                 )
        # red is for the target rate values
        ax2.plot(range(summarized.shape[0]), summarized['mean'], 'ro-', linewidth=2.0, zorder=4)
        plt.suptitle(feature_titles[f] if f in feature_titles else f, fontsize=14, weight='bold')
        plt.tight_layout()
        if plot_flag:
            plt.show()
        return fig
    
    def targettrend(self, features=None, quantiles=10, plot_flag=False):
        """
        Вычисление распределения таргета по каждой переменной из заданного списка
        :param features: список переменных
        :param quantiles: кол-во квантилей для разбиения непрерыных переменных
        :param plot_flag: флаг для вывода распределения

        :return: список из графиков [plt.figure]
        """
        if features is None:
            features = self.features
        if self.n_jobs_restr > 1:
            with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                figs = list(pool.map(partial(self.targettrend_feature, special_bins=self.special_bins,
                                             feature_titles=self.feature_titles, quantiles=quantiles, plot_flag=plot_flag),
                                     [self.samples[self.train_name][[self.target, f]] for f in features]))
        else:
            figs = [self.targettrend_feature(df=self.samples[self.train_name][[self.target, f]], special_bins=self.special_bins,
                                             feature_titles=self.feature_titles, quantiles=quantiles, plot_flag=plot_flag)
                    for f in features]
        plt.close('all')
        return figs

    def CorrelationAnalyzer(self, sample_name=None, features=None, hold=None, scores=None, method='pearson',
                            threshold=0.6, drop_with_most_correlations=True, verbose=False):
        """
        Корреляционный анализ переменных на выборке, формирование словаря переменных с причиной для исключения
        :param sample_name: название сэмпла на котором проводится отбор. При None берется ds.train_sample
        :param features: исходный список переменных для анализа. При None берется self.features
        :param hold: список/сет переменных, которые не будут исключаться при корреляционном анализе
        :param scores: словарь с метриками переменных вида {переменна: метрики}, которые будут использоваться при исключении переменных.
                      При None рассчитываются однофакторные джини
        :param method: метод расчета корреляций. Доступны варианты 'pearson', 'kendall', 'spearman'
        :param threshold: граница по коэффициенту корреляции
        :param drop_with_most_correlations:  при True - итерационно исключается фактор с наибольшим кол-вом коррелирующих с ним факторов с корреляцией выше threshold
                                             при False - итерационно исключается фактор с наименьшим значением метрики из списка коррелирующих факторов
        :param verbose: флаг для вывода списка исключенных переменных

        :return: словарь переменных для исключения вида {переменная: причина исключения}
        """
        if sample_name is None:
            sample_name = self.train_name
        if features is None:
            features = self.features
        features = [f for f in features if pd.api.types.is_numeric_dtype(self.samples[sample_name][f])]
        if hold is None:
            hold = set()
        if scores is None:
            scores = self.calc_gini(features=features, samples=[sample_name])[sample_name].to_dict()
        correlations = self.samples[sample_name][features].corr(method=method).abs()
        to_check_correlation=True
        features_to_drop = {}
        while to_check_correlation:
            to_check_correlation=False
            corr_number = {}
            significantly_correlated = {}
            for var in correlations:
                var_corr = correlations[var]
                var_corr = var_corr[(var_corr.index != var) & (var_corr > threshold)].sort_values(ascending=False).copy()
                corr_number[var] = var_corr.shape[0]
                significantly_correlated[var] = str(var_corr.index.tolist())
            if drop_with_most_correlations:
                corr_number_not_in_hold = [corr_number[x] for x in corr_number if x not in hold]
                if corr_number_not_in_hold:
                    with_correlation = {x: scores[x] for x in corr_number
                                        if corr_number[x] == max(corr_number_not_in_hold)
                                        and corr_number[x] > 0 and x not in hold}
                else:
                    with_correlation = {}
            else:
                with_correlation = {x: scores[x] for x in corr_number if corr_number[x] > 0 and x not in hold}
            if with_correlation:
                feature_to_drop = min(with_correlation, key=with_correlation.get)
                features_to_drop[feature_to_drop] = f'High correlation with features: {significantly_correlated[feature_to_drop]}'
                correlations = correlations.drop(feature_to_drop, axis=1).drop(feature_to_drop, axis=0).copy()
                to_check_correlation = True
        if verbose:
            self.logger.info(f'Dropped {len(features_to_drop)} correlated features: {list(features_to_drop.keys())}')
        return features_to_drop

    def calc_metric(self, metric='gini', samples=None, features=None):
        """
        Рассчет заданной метрики для списка переменных по сэмплам
        :param metric: название метрики/теста
                          'gini'    : Gini,
                          'vif'     : расчет Variance Inflation Factor,
                          'iv'      : расчет Information Value,
                          'ks'      : тест Колмогорова-Смирнова,
                          'anderson': Anderson-Darling test,
                          'chi2'    : Chisquare test,
                          'cramer'  : Cramer-Von-mises test,
                          'en_dist' : Energy distance,m ,k,,
                          'epps'    : Epps-Singleton test,
                          'h_dist'  : Hellinger distance,
                          'jh'      : Jensen-Shannon distance,
                          'kl'      : Kullback-Leibler divergence,
                          'mann'    : Mann-Whitney U-rank test,
                          'mmd'     : Squared euclidean pairwise distance,
                          't'       : T -test,
                          'tvd'     : Total variation distance,
                          'w_dist'  : Wasserstein distance ,
                          'z'       : Z-test
        :param samples: список названий сэмплов для расчета. При None вычисляется на всех доступных сэмпплах
        :param features: список переменных для расчета. При None берется self.features

        :return: ДатаФрейм с индексом из списка переменных и полями samples со значениями посчитанных метрик
        """

        metric = metric.lower()
        if features is None:
            features = self.features
        if samples is None:
            samples = list(self.samples)
        else:
            samples = [s for s in samples if s in self.samples]

        metric_dict = {'anderson': _anderson_darling,
                        'chi2': _chi_stat_test,
                        'cramer': _cramer_von_mises,
                        'en_dist': _energy_dist,
                        'epps': _epps_singleton,
                        #'fisher': _fisher_exact_stattest,
                        #'g': _g_stat_test,
                        'h_dist': _hellinger_distance,
                        'jh': _jensenshannon,
                        'kl': _kl_div,
                        #'ks': _ks_stat_test,
                        'mann': _mannwhitneyu_rank,
                        'mmd': _mmd_stattest,
                        #'psi': _psi,
                        't': _t_test2samp,
                        'tvd': _tvd_stattest,
                        'w_dist': _wasserstein_distance_norm,
                        'z': _z_stat_test
                       }
        result = {}
        if metric == 'gini':
            return self.calc_gini(samples=samples, features=features)

        elif metric == 'vif':
            for name in samples:
                y_, X_ = dmatrices(formula_like=self.target + ' ~ ' + '+'.join(['Q("' + f + '")' for f in features]),
                                   data=self.samples[name], return_type="dataframe")
                result[name] = {features[i - 1]: variance_inflation_factor(X_.values, i) for i in range(1, X_.shape[1])}

        elif metric == 'ks':
            from scipy.stats import kstest
            result = {name: {f: kstest(self.samples[name][self.samples[name][self.target] == 0][f],
                                       self.samples[name][self.samples[name][self.target] == 1][f]).statistic
                             for f in features}
                      for name in samples}

        elif metric =='iv':
            def iv_bin(df):
                """
                Вычисляет метрику Information Value
                -----------------------------------
                Вход
                df: датафрейм из 2-х столбцов: первый - переменная, второй - флаг дефолта.
                Ограничение - второе распределение - бинарное [0,1]
                :returns значение метрики IV, ?
                """
                df = df.reset_index()
                ind, v, c = df.columns
                df[v] = df[v].astype('float64')
                out_iv = df.groupby(v).agg({c: 'sum', ind: 'count'}).reset_index()
                out_iv = out_iv[(out_iv[c] != 0) & (out_iv[c] != out_iv[ind])]

                out_iv['good'] = out_iv[ind] - out_iv[c]
                out_iv[ind] = out_iv[ind] / np.sum(out_iv[ind])
                out_iv[c] = out_iv[c] / np.sum(out_iv[c])
                out_iv['good'] = out_iv['good'] / np.sum(out_iv['good'])

                out_iv['iv'] = (out_iv['good'] - out_iv[c]) * np.log(out_iv['good'] / out_iv[c])

                return out_iv['iv'].sum()

            result = {name: {f: iv_bin(self.samples[name][[f, self.target]]) for f in features} for name in samples}

        elif metric in metric_dict:
            result = {name: {f: metric_dict[metric](self.samples[name][self.samples[name][self.target] == 0][f],
                                                      self.samples[name][self.samples[name][self.target] == 1][f])[0]
                             for f in features}
                      for name in samples}
        return pd.DataFrame.from_dict(result)

    def add_descriptions(self, features, operands=None):
        """
        Генерирует и добавляет описания для сгенерированных переменных
        :param features: список переменных
        """
        if self.feature_descriptions is None:
            return None
        if operands is None:
            operands = {}
        for f in features:
            f = rem_suffix(f)
            if f not in self.feature_titles:
                if is_cross_name(f):
                    descr = 'Кросс переменная ' + ' & '.join(['"' +  ('; '.join([l for l in self.feature_descriptions[self.feature_descriptions.index == fi].astype('str').values.flatten().tolist() if not pd.isnull(l) and l != 'nan'])
                                                                      if fi in self.feature_descriptions.index else fi) + '"' for fi in cross_split(f)])

                elif f.startswith('gen_o_'):
                    for op in operands:
                        if f'_{operands[op]}_' in f:
                            descr = 'Сгенерированная переменная ' + f' {op} '.join(['"' + ('; '.join([l for l in self.feature_descriptions[self.feature_descriptions.index == fi].astype('str').values.flatten().tolist()
                                                                                                      if not pd.isnull(l) and l != 'nan'])
                                                                                           if fi in self.feature_descriptions.index else fi) + '"'
                                                                                    for fi in f[6:].split(f'_{operands[op]}_')])
                            break
                elif f.startswith('gen_t_'):
                    name_list = f[6:].rsplit('_', maxsplit=2)
                    f0 = name_list[0]
                    descr = 'Сгенерированная переменная "' + ('; '.join([l for l in self.feature_descriptions[self.feature_descriptions.index == f0].astype('str').values.flatten().tolist()
                                                                         if not pd.isnull(l) and l != 'nan'])
                                                              if f0 in self.feature_descriptions.index else f0)
                    if name_list[1][0] == 'w':
                        descr += f'" {name_list[2]} за {name_list[1][1:]} срез'
                    elif name_list[1][0] == 'l':
                        descr += f'" относительное изменение за {name_list[1][1:]} срез'
                else:
                    continue
                self.feature_descriptions = pd.concat([self.feature_descriptions, pd.DataFrame(
                    {'feature': f, self.feature_descriptions.columns[0]: descr}, index=[0]).set_index('feature')])
                self.feature_titles[f] = f + '\n\n' + '\n'.join(wrap(descr, 75))

    @staticmethod
    def check_split_stability_1(ds, simple, i):
        if simple:
            result = ds.calc_gini(samples=[ds.train_name], prebinning=True)
        else:
            from .woe import WOE
            tmp_bin = WOE(ds=ds)
            tmp_bin.auto_fit(plot_flag=-1)
            result = tmp_bin.transform(ds).calc_gini(samples=[ds.train_name])[[ds.train_name]]
        return result.set_axis([i], axis=1)

    def check_split_stability(self, df=None, n=50, features=None, gini_deviation=0.25, pr_good=0.25, exclude_features=False,
                              simple=False, result_file='split_stability.xlsx'):
        """
        Оценивает стабильность биннинга переменных при различных сплитах выборки на сэмплы
        :param df: полная выборка. При None в качетстве выборки берется объединение всех сэмплов
        :param n: кол-во различных сплитов для теста
        :param features: список переменных для обработки
        :param gini_deviation: трэшхолд метрики std(gini)/mean(gini) до которого переменная считается стабильной
        :param pr_good: трэшхолд доли успешных биннингов
        :param exclude_features: флаг для исключения нестабильных переменных. При False будет только выведен их список
        :param simple: флаг для быстрой оценки джини. При False будет выполняться полноценный СФА на каждом сплите
        :param result_file: файл для сохранения результатов
        """
        if df is None:
            df = self.to_df()
        if features is None:
            features = self.features
        if self.n_jobs > 1:
            gini_dfs = []
            with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                jobs = []
                iterations = n
                i_iter = iter(range(n))
                while iterations:
                    for i in i_iter:
                        jobs.append(pool.submit(self.check_split_stability_1,
                                                ds=DataSamples(samples={'Train': df}, features=features, target=self.target,
                                                               time_column=self.time_column, id_column=self.id_column, result_folder=self.result_folder,
                                                               n_jobs=1, random_state=i, samples_split={}, logger=50, verbose=False),
                                                simple=simple, i=i))
                        if len(jobs) > self.n_jobs * 2:
                            break
                    for job in futures.as_completed(jobs):
                        gini_dfs.append(job.result())
                        jobs.remove(job)
                        iterations -= 1
                        break
            gc.collect()
        else:
            gini_dfs = [self.check_split_stability_1(ds=DataSamples(samples={'Train': df}, features=features, target=self.target,
                                                        time_column=self.time_column, id_column=self.id_column, result_folder=self.result_folder,
                                                        n_jobs=1, random_state=i, samples_split={}, logger=50, verbose=False),
                                                     simple=simple, i=i)
                        for i in range(n)]
        df = pd.concat(gini_dfs, axis=1)
        cols = [i for i in range(n) if i in df.columns]
        df = df[cols]
        df.index = df.index.map(rem_suffix)
        df['gini_deviation'] = df[cols].std(axis=1) / df[cols].mean(axis=1)
        df['pr_good'] = (~df[cols].isnull()).sum(axis=1) / n
        df['to_exclude'] = (df['gini_deviation'] > gini_deviation) | (df['pr_good'] < pr_good)
        features_to_exclude = df[df['to_exclude']].index.to_list()
        self.logger.setLevel(self.logger_level)
        if features_to_exclude:
            self.logger.info(f'{len(features_to_exclude)} unstable features: {features_to_exclude}')
            if exclude_features:
                self.features = [f for f in self.features if f not in set(features_to_exclude)]
                self.logger.info('Unstable features are excluded')
        else:
            self.logger.info('All features are stable!')
        df.to_excel(self.result_folder + result_file)

    def prebinning(self, X=None, y=None):
        """
        Выполняет биннинг переменных без проверок на стабильность и переобучение
        :param X: Вектор переменных. Может иметь тип Датафрейм/Серия/массив
        :param y: Вектор целевой переменной

        :return: трансформированный вектор
        """
        if X is None and y is None:
            samples = copy.deepcopy(self.samples)
            for f in self.features:
                try:
                    optb = optbinning.OptimalBinning(dtype='categorical' if f in self.cat_columns else 'numerical',
                                                     prebinning_method='cart', max_n_prebins=10,
                                                     min_prebin_size=0.05, monotonic_trend='auto')
                    optb.fit(self.samples[self.train_name][f], self.samples[self.train_name][self.target])
                    for name in samples:
                        samples[name][f] = optb.transform(samples[name][f])
                except:
                    pass
            return samples
        if isinstance(X, str):
            X_tr = optbinning.OptimalBinning(dtype='categorical' if X in self.cat_columns else 'numerical',
                                             prebinning_method='cart', max_n_prebins=10,
                                             min_prebin_size=0.05, monotonic_trend='auto')\
                .fit_transform(self.samples[self.train_name][X], self.samples[self.train_name][self.target])
        elif isinstance(X, pd.DataFrame):
            X_tr = optbinning.BinningProcess(variable_names=list(X.columns), categorical_variables=[], max_n_prebins=10,
                                             min_prebin_size=0.05, n_jobs=1).fit_transform(X, y)
            X_tr.index = X.index
        else:
            X_tr = optbinning.OptimalBinning(dtype='numerical', prebinning_method='cart', max_n_prebins=10,
                                             min_prebin_size=0.05, monotonic_trend='auto').fit_transform(X, y)
        return X_tr

    def feature_generator(self, features=None, prebinning=True, chunk_size=10000, gini_threshold=10, operands=True,
                          lags=True, aggs=None, corr_method='spearman', corr_thres=0.80):
        """
        Генератор переменных
        Генерирует новые переменные как парные комбинации переменных из списка features используя операнды из словаря operands
        Для тайм-серий генерирует переменные, используя оконные функции с размером окон из списка lags с агрегирующими функциями aggs,
                и переменные с относительным изменением с лагом из списка lags
        После генерации переменная добавляется в список переменных и для нее генерируется описание только в случае прохождения проверок на джини, корреляцию и стат значимость
        Вся выборка должна находиться в одной сэмпле
        :param features: список исходных переменных. При None берется self.features
        :param prebinning: флаг для предварительного биннинга переменных перед выполнением тестов
        :param chunk_size: максимальный размер пачек переменных для обработки
        :param gini_threshold: граница джини, ниже которой переменные отсекаются
        :param operands: словарь операндов для парных комбинаций. При True используется {'+': 'plus', '-': 'minus', '/': 'div'}
        :param lags: список лагов для тайм-серий. При True используется [1, 3, 6, 9, 12]
        :param aggs: список агрегатов для оконных функций. При None используется ['min', 'max', 'mean', 'std']
        :param corr_method: метод расчета корреляций. Доступны варианты 'pearson', 'kendall', 'spearman'
        :param corr_thres: граница по коэффициенту корреляции
        """
        from tsfresh.feature_selection.selection import select_features

        def check_feature(s, x, y):
            def get_f_ini(f):
                if f.startswith('gen_o_'):
                    for op in operands:
                        if f'_{op}_' in f:
                            return f[6:].split(f'_{op}_')
                elif f.startswith('gen_t_'):
                    return f[6:].rsplit('_', maxsplit=2)[0:1]
                return []

            for f in get_f_ini(s):
                if abs(df_new[f].corr(pd.Series(x), method=corr_method)) > corr_thres:
                    return False
            return self.get_f_gini(y, x, abs_flag=True) > gini_threshold

        def selection(tmp):
            if not tmp.empty:
                tmp_tr = self.prebinning(tmp, y) if prebinning else tmp.fillna(fillna)
                tmp_features = [f for f in tmp_tr.columns if check_feature(f, tmp_tr[f], y)]
                if tmp_features:
                    to_add = list(set(select_features(pd.concat([df_new, tmp_tr[tmp_features]], axis=1),
                                                      y, n_jobs=self.n_jobs).columns) - set(df_new.columns))
                    if to_add:
                        df[to_add] = tmp[to_add]

        if len(self.samples) > 1:
            self.logger.error('DataSample must contain only one sample!')
            return
        if self.bootstrap_base is not None:
            self.logger.warning('self.bootstrap_base is set to None')
        if features is None:
            features = [f for f in self.features if f not in set(self.cat_columns)]
        fillna = -999999
        df = self.samples[self.train_name]
        if self.time_column and self.id_column:
            df.sort_values(by=[self.id_column, self.time_column], inplace=True)
        y = df[self.target]
        df_new = self.prebinning(df[features], y) if prebinning else df[features].copy().fillna(fillna)
        ginis = {f: self.get_f_gini(y, df_new[f], abs_flag=True) for f in features}
        features = [f for f in features if ginis[f] >= gini_threshold]
        if operands is True:
            operands = {'+': 'plus', '-': 'minus', '/': 'div'}
        elif not operands:
            operands = {}
        elif not isinstance(operands, dict):
            operands = {x: x for x in operands}
        if lags is True:
            lags = [1, 3, 6, 9, 12]
        elif lags is False:
            lags = []
        if aggs is None:
            aggs = ['min', 'max', 'mean', 'std']
        features_ini = set(df.columns)
        tmp = pd.DataFrame(index=df.index)
        c1 = int(len(features) * (len(features) - 1) / 2 * len(operands))
        c2 = len(features) * len(lags) * (len(aggs) + 1)
        self.logger.info(f'Creating combinations...')
        with tqdm(total=c1) as pbar:
            for f1, f2 in combinations(features, 2):
                if abs(df_new[f1].corr(df_new[f2], method=corr_method)) < corr_thres:
                    for op in operands:
                        f = f'gen_o_{f1}_{operands[op]}_{f2}'
                        pbar.update()
                        if f not in features_ini:
                            if self.special_bins:
                                tmp[f] = np.where(df[f1].isin(list(self.special_bins.values())), df[f1],
                                                  np.where(df[f2].isin(list(self.special_bins.values())), df[f2],
                                                           eval(f'df[f1] {op} df[f2]').replace([-np.inf, np.inf], np.nan)))
                            else:
                                tmp[f] = eval(f'df[f1] {op} df[f2]').replace([-np.inf, np.inf], np.nan)
                            if len(tmp.columns) > chunk_size:
                                selection(tmp)
                                tmp = pd.DataFrame(index=df.index)
                else:
                    pbar.update(len(operands))
            selection(tmp)
        if self.time_column and self.id_column:
            if not df[[self.time_column, self.id_column]].duplicated().any():
                dates = sorted(list(df[self.time_column].unique()))
                self.logger.info(f'Creating window features...')
                with tqdm(total=c2) as pbar:
                    for lag in lags:
                        tmp = df.rolling(lag).agg({f: aggs for f in features})
                        tmp.columns = tmp.columns.map(lambda x: f'gen_t_{x[0]}_w{lag}_{x[1]}')
                        tmp[[f'gen_t_{f}_l{lag}_reldelta' for f in features]] = (df[features] - df[features].shift(lag)) / df[features]
                        tmp[~((df[self.id_column] == df[self.id_column].shift(lag)) & (df[self.time_column].map({d: dates[i - lag] for i, d in enumerate(dates)}) == df[self.time_column].shift(lag)))] = np.nan
                        if self.special_bins:
                            for f in features:
                                tmp.loc[df[f].isin(list(self.special_bins.values())), [f'gen_t_{f}_l{lag}_reldelta'] + [f'gen_t_{f}_w{lag}_{agg}' for agg in aggs]] = df[f]
                        tmp = tmp.replace([-np.inf, np.inf], np.nan)
                        tmp = tmp[[f for f in tmp.columns if f not in features_ini and ~tmp[f].isna().all()]]
                        selection(tmp)
                        pbar.update(len(features) * (len(aggs) + 1))
                c1 += c2
            else:
                self.logger.warning(f'DataSample has duplicates by fields {[self.time_column, self.id_column]}. Calculation of window features is not possible')
        new_features = list(set(df.columns) - features_ini)
        self.add_descriptions(new_features, operands)
        self.features += new_features
        self.logger.info(f'{c1} features were generated, {len(new_features)} of them passed the tests and were added to sample {self.train_name}')

    def copy(self, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)

    def get_ci_gini(self, ginis, df=None):
        if df is None:
            df = self.samples[self.train_name]
        vc = df[self.target].value_counts()
        try:
            G, B = vc[0], vc[1]
        except:
            return ginis, ginis
        ci_lower = {}
        ci_upper = {}
        z = ndtri(1 - self.ci_alpha / 2)
        for f in ginis:
            gini = ginis[f] / 100
            try:
                std = ((1 - gini ** 2 + (B - 1) * (4 * (gini + 1) / (3 - gini) - (gini + 1) ** 2) + (G - 1) * (4 * (gini + 1) ** 2 / (3 + gini) - (gini + 1) ** 2)) / (B * G)) ** 0.5 if gini != 0 else 0
            except:
                std = 0
            ci_lower[f] = (gini - z * std) * 100
            ci_upper[f] = (gini + z * std) * 100
        return ci_lower, ci_upper

    def reduce_mem_usage(self, df):
        try:
            df[self.target] = df[self.target].astype('int8')
        except:
            pass
        for f in self.features:
            if pd.api.types.is_numeric_dtype(df[f]):
                c_max = df[f].abs().max()
                if pd.api.types.is_integer_dtype(df[f]):
                    if c_max < np.iinfo('int8').max:
                        df[f] = df[f].astype('int8')
                    elif c_max < np.iinfo('int16').max:
                        df[f] = df[f].astype('int16')
                    elif c_max < np.iinfo('int32').max:
                        df[f] = df[f].astype('int32')
                else:
                    if c_max < 10**7:
                        df[f] = df[f].astype('float32')
            elif f in self.cat_columns:
                df[f] = df[f].astype('category')

    def data_check(self, out="data_check.xlsx", cutoffs: dict = None, drop_bad_features: bool = False):
        """
        Проверяет качество данных и создает xlsx отчет с результатами проверок

        HAS_NO_FILLED - нет заполненных значений
        HAS_CATEGORY_WITH_GTE130 - категориальные слишком гранулированные, слишком много уникальных значений
        HAS_GTE_97_5_MISSED - более 97,5% пропущено
        HAS_UNIQUE_GTE_95_PERC - слишком много уникальных значений, подозрение на ID

        Для всех статистик используется заранее заданный набор признаков self.features.
        Флаг фактора к исключению проставляется на основании соответствующего cutoff из словаря cutoofs для:
            - PSI, если значения больше или равно 'red';
            - GINI, если значение меньше 'red';
            - IV, если значение меньше 'red';
        Также флаг к исключению проставляется, если:
            - нет заполненных значений;
            - слишком много уникальных значений (для категориальных факторов);
            - в факторе более 97,5% пропусков;
            - слишком много уникальных значений (для числовых факторов);


        :param out: название эксель файла для сохранения статистики
        :param cutoffs: словарь для задания отсечек со следующей структурой:
                    cutoffs = {
                        'psi': {'red':0.2, 'yellow':0.1}
                        'gini': {'red':0.05, 'yellow':0.1},
                        'iv': {'red':0.02, 'yellow':0.1}
                    }
        :param drop_bad_features: флаг для исключения из self.features переменных, не прошедших проверки

        :return: Список признаков к исключению


        """

        out = add_ds_folder(self, out if out else "data_check.xlsx" )

        if not cutoffs:
            cutoffs = {
                'psi': {'red': 0.5, 'yellow': 0.25},
                'gini': {'red': 5, 'yellow': 10},
                'iv': {'red': 0.02, 'yellow': 0.1},
                'moda': {'red': 0.95, 'yellow': 0.8},
                'ci_gini': {'red': 0.8, 'yellow': 0.5}
            }
        features = self.features

        # первичие статистики на данных из набора обучения
        df = self.samples[self.train_name]
        len_df = len(df)
        attribute_list = []

        for f in features:
            x = df[~df[f].isin(list(self.special_bins.values()))][f]
            nunique = df[f].nunique()
            row = {'feature': f,
                   'type_val': df[f].dtype,
                   'count_val': len_df,
                   'count_dist': nunique,
                   'count_miss': df[f].isna().sum(),
                   'moda_val': x.mode()[0] if nunique > 0 and len(x) > 0 else np.nan,
                   'count_value_moda': (x == x.mode()[0]).sum() if nunique > 0 and len(x) > 0 else len_df,
                   }
            try:
                add_stat = {'min_val': x.min(),
                            'max_val': x.max(),
                            'val_mediana': x.median(skipna=True),
                            'stand_d_val': np.nanstd(x),
                            'percentile_1': np.nanpercentile(x, 1),
                            'percentile_2': np.nanpercentile(x, 2),
                            'percentile_5': np.nanpercentile(x, 5),
                            'percentile_98': np.nanpercentile(x, 98),
                            'percentile_99': np.nanpercentile(x, 99),

                            }
            except:
                add_stat = {}
            row.update(add_stat)
            attribute_list.append(row)
        primary_stats_df = pd.DataFrame(attribute_list)

        # доля моды в семпле
        primary_stats_df['frac_moda'] = primary_stats_df['count_value_moda'] / len_df

        # - нет заполненных значений
        primary_stats_df['HAS_NO_FILLED'] = (primary_stats_df["count_miss"] == len_df).astype('int8')

        # - слишком много уникальных значений, категориальные слишком гранулированные
        primary_stats_df['HAS_CATEGORY_WITH_GTE130'] = ((primary_stats_df["count_dist"] >= 130) & (
            primary_stats_df["count_dist"].isin(['category', 'object']))).astype('int8')
        # - одно значение и менее половины пропусков
        primary_stats_df['HAS_ONE_VALUE_AND_LESS_HALF_MISSED'] = (
                    (primary_stats_df["count_dist"] == 1) & (primary_stats_df["count_miss"] < len_df * 0.5)).astype(
            'int8')

        # - более 97,5% пропущено
        primary_stats_df['HAS_GTE_97_5_MISSED'] = ((primary_stats_df["count_miss"] >= len_df * 0.975)).astype('int8')

        # - слишком много уникальных значений, подозрение на ID
        primary_stats_df['HAS_UNIQUE_GTE_95_PERC'] = ((primary_stats_df["count_dist"] >= len_df * 0.95)).astype('int8')

        primary_stats_df['TOTAL_FL_PRIM_STATS'] = primary_stats_df[
            ['HAS_NO_FILLED', 'HAS_CATEGORY_WITH_GTE130', 'HAS_ONE_VALUE_AND_LESS_HALF_MISSED', 'HAS_GTE_97_5_MISSED',
             'HAS_UNIQUE_GTE_95_PERC']].max(axis=1)

        # Создаем экземпляр класса с преобразованными признаками
        ds_transformed = DataSamples(
            samples=self.prebinning(None, None),
            target=self.target,
            result_folder=self.result_folder,
            features=features,
            time_column=self.time_column,
            verbose=False,
            logger=50,
            random_state=0
        )

        # расчет PSI, если есть срез по датам, или нет
        if self.time_column and len(self.samples) > 1:
            psi_stats_df = pd.concat(
                [
                    ds_transformed.psi(),
                    ds_transformed.psi(time_column=self.time_column)
                ], axis=1
            ).drop(columns=[self.train_name]).add_prefix('PSI_').reset_index()
        elif self.time_column:
            psi_stats_df = ds_transformed.psi(time_column=self.time_column).add_prefix('PSI_').reset_index()
        elif len(self.samples) > 1:
            psi_stats_df = ds_transformed.psi().drop(columns=[self.train_name]).add_prefix('PSI_').reset_index()
        else:
            self.logger.info('No PSI calculation. Check samples or time_column')
            psi_stats_df = pd.DataFrame(columns=['feature'])

        # записываем максимальное значение PSI в отдельный столбец
        psi_stats_df['PSI_Max'] = psi_stats_df.max(axis=1, numeric_only=True)

        # расчет Gini
        gini_stats_df = ds_transformed.calc_gini(
            samples=[self.train_name],
            abs=True, prebinning=False
        ).add_prefix('GINI_').rename_axis('feature').reset_index()
        # расчет IV
        iv_stats_df = ds_transformed.calc_metric(
            metric='iv', samples=[self.train_name]
        ).add_prefix('IV_').rename_axis('feature').reset_index()

        # задаем названия столбцов
        gini_col_name = 'GINI_' + self.train_name
        iv_col_name = 'IV_' + self.train_name

        # объединение данных
        res = pd.merge(psi_stats_df, primary_stats_df, on="feature", how='right')
        res = pd.merge(gini_stats_df, res, on="feature", how='right')
        res = pd.merge(iv_stats_df, res, on="feature", how='right')

        res['TOTAL_FL_TO_DROP'] = ((res['TOTAL_FL_PRIM_STATS'] > 0) |
                                   (res[iv_col_name] < cutoffs['iv']['red']) |
                                   (res[gini_col_name] < cutoffs['gini']['red']) |
                                   (res['PSI_Max'] >= cutoffs['psi']['red']) |
                                   (res['frac_moda'] >= cutoffs['moda']['red']) |
                                   (res['GINI_CI_upper'] > res[gini_col_name] + cutoffs['ci_gini']['red'] * res[gini_col_name])).astype('int')

        res = res.sort_values(by='TOTAL_FL_TO_DROP', ascending=False)

        # записываем факторы к исключению в список
        features_to_exclude = res[res['TOTAL_FL_TO_DROP'] == 1]['feature'].tolist()

        self.logger.setLevel(self.logger_level)
        self.logger.info(f"{len(features_to_exclude)}/{len(features)} features are of low quality: {features_to_exclude}")
        if drop_bad_features:
            self.features = [f for f in self.features if f not in features_to_exclude]
            self.cat_columns = [f for f in self.cat_columns if f not in features_to_exclude]

        # изменяем названия столбцов на более понятные
        res = res.rename(columns={
            'frac_moda': 'Доля моды',
            'HAS_NO_FILLED': '100% пропусков',
            'HAS_CATEGORY_WITH_GTE130': 'Сильная грануляция (категория)',
            'HAS_ONE_VALUE_AND_LESS_HALF_MISSED': 'Одно значение и менее половины пропусков',
            'HAS_GTE_97_5_MISSED': 'Более 97,5% пропусков',
            'HAS_UNIQUE_GTE_95_PERC': 'Уникальных значений больше 95%',
            'TOTAL_FL_TO_DROP': 'Фактор к исключению',
            'type_val': 'Тип',
            'count_val': 'Кол-во значений',
            'count_dist': 'Кол-во уникальных значений',
            'count_miss': 'Кол-во пропусков',
            'min_val': 'Минимальное значение',
            'max_val': 'Максимальное значение',
            'val_mediana': 'Медиана',
            'moda_val': 'Мода',
            'count_value_moda': 'Кол-во значений моды',
            'stand_d_val': 'Стандартное отклонение',
        }).drop(columns=['TOTAL_FL_PRIM_STATS'])

        # проставляем сигналы в соответствии с заданными cutoff
        def highlight_features_to_drop(row):
            ret = ["" for _ in row.index]
            if row['Фактор к исключению'] == 1:
                ret[row.index.get_loc("feature")] = "color: red"
            return ret

        res = res.style.apply(highlight_features_to_drop, axis=1). \
            applymap(
            lambda x: "color: red"
            if x < cutoffs['iv']['red']
            else "color: orange"
            if x < cutoffs['iv']['yellow']
            else "color: green",
            subset=['IV_' + self.train_name]
        ). \
            applymap(
            lambda x: "color: red"
            if x >= cutoffs['psi']['red']
            else "color: orange"
            if x >= cutoffs['psi']['yellow']
            else "color: green",
            subset=[x for x in res.columns if x.startswith('PSI_')]
        ). \
            applymap(
            lambda x: "color: red"
            if x == 1
            else "color: green",
            subset=res.columns[-6:]
        ). \
            applymap(
            lambda x: "color: red"
            if x >= cutoffs['moda']['red']
            else "color: orange"
            if x >= cutoffs['moda']['yellow']
            else "color: green",
            subset=['Доля моды']
        ). \
            applymap(
            lambda x: "color: red"
            if x < cutoffs['gini']['red']
            else "color: orange"
            if x < cutoffs['gini']['yellow']
            else "color: green",
            subset=[gini_col_name]
        )

        # записываем данные в файл
        with pd.ExcelWriter(out) as writer:
            res.to_excel(writer, sheet_name="features", index=False)
        self.logger.info(f"Data checks saved to file: {out}")

    def add_aux(self, ds_aux):
        if ds_aux.time_column is not None:
            self.time_column = ds_aux.time_column
            for name in self.samples:
                self.samples[name][self.time_column] = ds_aux.samples[name][self.time_column]
            if self.bootstrap_base is not None:
                self.bootstrap_base[self.time_column] = ds_aux.bootstrap_base[self.time_column]

    @staticmethod
    def get_f_gini(y, f, fillna=None, abs_flag=False):
        if fillna is not None:
            f = f.copy().fillna(fillna)
        try:
            gini = (2 * roc_auc_score(y, -f) - 1) * 100
            if abs_flag:
                gini = abs(gini)
        except:
            gini = 0
        return gini

    @staticmethod
    def get_features_gini(y, df, fillna=None, abs_flag=False):
        return {f: DataSamples.get_f_gini(y, df[f], fillna=fillna, abs_flag=abs_flag) for f in df}

    @staticmethod
    def get_time_features_gini(df, target, time_column, fillna=None, abs_flag=False):
        return {time: DataSamples.get_features_gini(group[target], group.drop([target, time_column], axis=1), fillna=fillna, abs_flag=abs_flag)
                for time, group in df.groupby(time_column)}
