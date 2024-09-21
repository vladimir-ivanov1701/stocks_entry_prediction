# -*- coding: utf-8 -*-

from .data import DataSamples
from ._utils import add_suffix, rem_suffix, is_cross_name, cross_name, cross_split, make_header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.proportion import proportion_confint
from scipy.special import ndtri
import re
import os
import openpyxl
import gc
import copy
import itertools
import warnings
from tqdm import tqdm
from textwrap import wrap
from concurrent import futures
from functools import partial
import json
try:
    import optbinning
except:
    pass

warnings.simplefilter('ignore')
plt.rc('font', family='Verdana', size=12)
try:
    plt.style.use([s for s in plt.style.available if 'darkgrid' in s][0])
except:
    pass
pd.set_option('display.precision', 3)
gc.enable()


class WOE:
    """
    Класс для ВОЕ-трансформации переменных
    """
    def __init__(self, ds=None, features=None, scorecard=None, round_digits=3, round_woe=2, rounding_migration_coef=0.001,
                 simple=True, n_folds=5, woe_adjust=0.5, alpha=0, alpha_range=None, alpha_scoring='neg_log_loss',
                 alpha_best_criterion='min', sb_process='nearest_or_separate', sb_min_part=0.02,
                 others='missing_or_min', opposite_sign_to_others=False):
        """
        :param ds: ДатаСэмпл, для которого будут рассчитываться биннинги
        :param features: список переменных. При None берется ds.features
        :param scorecard: путь к эксель файлу или датафрейм с готовыми биннингами для импорта

        ---Параметры для расчета WOE---
        :param round_digits: число знаков после запятой для округления значений границ бинов.
                 При округлении границ бинов происходит проверка на долю мигрирующих наблюдений. Если округление приедет к миграции большой доли наблюдений,
                 то round_digits увеличивается до тех пор, пока доля не упадет ниже rounding_migration_coef
        :param round_woe: число знаков после запятой для округление значение WOE
        :param rounding_migration_coef: максимально допустимая доля наблюдений для миграции между бинами при округлении
        :param simple: если True, то расчет WOE происходит на трэйн сэмпле, иначе берется среднее значение по фолдам
        :param n_folds: кол-во фолдов для расчета WOE при simple=False
        :param woe_adjust: корректировочный параметр для расчета EventRate_i в бине i:
                           EventRate_i = (n1_i+woe_adjust)/(n0_i+woe_adjust),
                           где n1_i - кол-во наблюдений с target=1 в бине i, n0_i - кол-во наблюдений с target=0 в бине i
        :param alpha: коэффициент регуляризации для расчета WOE. Если задан, то WOE для бина i вычисляется по формуле:
                      SmoothedWOE_i = ln((n + alpha)*EventRate/(n*EventRate_i + alpha)),
                      где n - число наблюдений, EventRate = n1/n0,
                      n1 - общее кол-во наблюдений с target=1, n0 - общее кол-во наблюдений с target=0
        :param alpha_range: если alpha=None, то подбирается оптимальное значение alpha из диапазона alpha_range. При None берется диапазон range(10, 100, 10)
        :param alpha_scoring: метрика, используемая для оптимизации alpha
        :param alpha_best_criterion: 'min' - минимизация метрики alpha_scoring, 'max' - максимизация метрики

        ---Обработка специальных бинов ---
        :param sb_process: способ обработки специальных бинов
                'separate' - всегда помещать в отдельный бин
                'min_or_separate' - если доля значений в бине меньше sb_min_part, то объединять с минимальным по WOE бином, иначе помещать в отдельный бин
                'max_or_separate' - если доля значений в бин меньше sb_min_part, то объединять с максимальным по WOE бином, иначе помещать в отдельный бин
                'nearest_or_separate' - если доля значений в бин меньше sb_min_part, то объединять с ближайшим по WOE бином, иначе помещать в отдельный бин
        :param sb_min_part: минимальная доля значений в бин для выделения отдельного бина при sb_process 'min_or_separate', 'max_or_separate' или 'nearest_or_separate'

        ---Обработка остальных значений, не попавших в биннинг---
        :param others: Способ обработки значений, не попавших в биннинг:
                'min': остальным значениям присваивается минимальный WOE
                'max': остальным значениям присваивается максимальный WOE
                'missing_or_min': если есть бакет с пустыми значениями, то остальным значениям присваивается его WOE, иначе минимальный WOE
                'missing_or_max': если есть бакет с пустыми значениями, то остальным значениям присваивается его WOE, иначе максимальный WOE
                float: отсутствующим значениям присваивается заданный фиксированный WOE
        :param opposite_sign_to_others: В случае, когда непрерывная переменная на выборке для разработки имеет только один знак,
                то все значения с противоположным знаком относить к others
        """
        self.round_digits = round_digits
        self.round_woe = round_woe
        self.rounding_migration_coef = rounding_migration_coef
        self.simple = simple
        self.n_folds = n_folds
        self.woe_adjust = woe_adjust
        self.alpha = alpha
        self.alpha_range = alpha_range if self.alpha is None else None
        self.alpha_scoring = alpha_scoring
        self.alpha_best_criterion = alpha_best_criterion
        self.sb_process = sb_process
        self.sb_min_part = sb_min_part
        self.opposite_sign_to_others = opposite_sign_to_others

        if ds is None:
            ds = DataSamples(logger=None, verbose=False)
        if isinstance(scorecard, str):
            scorecard = pd.read_excel(scorecard)
        if features is None:
            if ds.features:
                features = ds.features
            elif scorecard is not None:
                features = list(scorecard['feature'].unique())

        if ds.samples is not None and (ds.time_column is not None or ds.id_column is not None):
            samples = {name: ds.samples[name][[f for f in [ds.time_column, ds.id_column] if f is not None]] for name in ds.samples}
        else:
            samples = None
        self.ds_aux = DataSamples(samples=samples, features=[], cat_columns=[], target=ds.target, time_column=ds.time_column, id_column=ds.id_column,
                                  feature_descriptions=ds.feature_descriptions, train_name=ds.train_name,
                                  result_folder=ds.result_folder, n_jobs=ds.n_jobs, logger=ds.logger, verbose=False)
        if ds.bootstrap_base is not None and ds.time_column is not None:
            self.ds_aux.bootstrap_base = ds.bootstrap_base[[ds.time_column]]
        if others in ['min', 'max', 'missing_or_min', 'missing_or_max'] or isinstance(others, (int, float)):
            self.others = others
        else:
            self.ds_aux.logger.warning('Parameter others is incorrect. Set others = "missing_or_min".')
            self.others = 'missing_or_min'
        self.feature_woes = {rem_suffix(f): self.create_feature_woe(ds, rem_suffix(f)) for f in features if not is_cross_name(f)} if features else {}
        self.feature_crosses = {}
        if scorecard is not None:
            self.import_scorecard(scorecard, verbose=False, fit_flag=False)

    def create_feature_woe(self, ds, f):
        """
        Создание объекта класса FeatureWOE
        :param ds: ДатаСэмпл для обработки
        :param f: переменная

        :return: FeatureWOE
        """
        return FeatureWOE(ds, f, round_digits=self.round_digits, round_woe=self.round_woe,
                          rounding_migration_coef=self.rounding_migration_coef, simple=self.simple, n_folds=self.n_folds,
                          woe_adjust=self.woe_adjust, alpha=self.alpha, alpha_range=self.alpha_range,
                          alpha_scoring=self.alpha_scoring, alpha_best_criterion=self.alpha_best_criterion,
                          sb_process=self.sb_process, sb_min_part=self.sb_min_part,
                          others_process=self.others, opposite_sign_to_others=self.opposite_sign_to_others)

    def fit(self, features=None, new_groups=True, plot_flag=True, method='tree', max_n_bins=10, min_bin_size=0.05,
            criterion='entropy', scoring='neg_log_loss', max_depth=None, monotonic=False, solver='cp', divergence='iv'):
        """
        Пересчет биннинга для списка переменных
        :param features: список переменных для обработки. При None обрабатываются list(self.feature_woes) + list(self.cross_features)
        :param new_groups: False - пересчитаются только WOE в биних,
                           True - также пересчитываются и границы бинов
        :param plot_flag: флаг для вывода графиков с биннингом

        --- Метод биннинга ---
        :param method: 'tree' - биннинг деревом, 'opt' - биннинг деревом с последующей оптимизацией границ бинов библиотекой optbinning
        :param max_n_bins: максимальное кол-во бинов
        :param min_bin_size: минимальное число (доля) наблюдений в каждом листе дерева.
                                Если min_bin_size < 1, то трактуется как доля наблюдений от обучающей выборки

        --- Параметры биннинга для метода 'tree' ---
        :param criterion: критерий расщепления. Варианты значений: 'entropy', 'gini'
        :param scoring: метрика для оптимизации
        :param max_depth: максимальная глубина дерева

        --- Параметры биннинга для метода 'opt' ---
        :param monotonic: флаг для оптимизации биннинга к монотонному тренду
        :param solver: солвер для оптимизации биннинга:
                        'cp' - constrained programming
                        'mip' - mixed-integer programming
                        'ls' - LocalSorver (www.localsorver.com)
        :param divergence: метрика для максимизации:
                        'iv' - Information Value,
                        'js' - Jensen-Shannon,
                        'hellinger' - Hellinger divergence,
                        'triangular' - triangular discrimination
        """
        if features is None:
            features = list(self.feature_woes) + list(self.cross_features)
        features = [rem_suffix(f) for f in features]
        for f in features:
            if f in self.feature_woes:
                self.feature_woes[f].fit(new_groups=new_groups, method=method, max_n_bins=max_n_bins,
                                         min_bin_size=min_bin_size, criterion=criterion, scoring=scoring,
                                         max_depth=max_depth, monotonic=monotonic, solver=solver,
                                         divergence=divergence, verbose=True)
            elif self.is_cross(f):
                f1, f2 = cross_split(f)
                self.feature_crosses[f1].fit(cross_features=[f2], new_groups=new_groups, method=method, max_n_bins=max_n_bins,
                                             min_bin_size=min_bin_size, criterion=criterion, scoring=scoring,
                                             max_depth=max_depth, monotonic=monotonic, solver=solver,
                                             divergence=divergence, verbose=True)
        if plot_flag:
            self.plot_bins(features=features, folder=None, plot_flag=True, show_groups=True)

    @staticmethod
    def get_f2_impurity(feature_woe, df1_group, criterion='entropy', max_depth=5, min_bin_size=0.05, max_n_bins=10):
        impurity = 0
        min_samples_leaf = min_bin_size if min_bin_size > 1 else int(round(len(df1_group) * min_bin_size, 0))
        for group, df in feature_woe.ds.samples[feature_woe.ds.train_name].drop(['group'], axis=1) \
                .merge(df1_group, how='left', left_index=True, right_index=True).groupby('group'):
            df = df[~df[feature_woe.feature].isin([feature_woe.special_bins.values()])]
            if len(df) > min_samples_leaf* 2:
                x_train = df[feature_woe.feature]
                if feature_woe.categorical:
                    tmp_groups = feature_woe.groups
                    feature_woe.groups = {group: [value] for group, value in enumerate(x_train.unique())}
                    df['group'] = feature_woe.set_groups(data=x_train, inplace=False)
                    x_train = df['group'].map(feature_woe.calc_simple_woes(df=df))
                    feature_woe.groups = tmp_groups
                y_tran = df[feature_woe.ds.target]
                try:
                    tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                  max_leaf_nodes=max_n_bins,
                                                  random_state=feature_woe.ds.random_state)
                    tree.fit(x_train[:, None], y_tran)
                    impurity += sum(tree.tree_.impurity)
                except:
                    pass
        return impurity

    @staticmethod
    def auto_fit_feature(feature_obj, ds_aux, auto_fit_parameters):
        """
        Автобиннинг одной переменной
        :param feature_obj: объект класса FeatureWOE или FeatureCross
        :param auto_fit_parameters: словарь с параметрами атвобиннинга

        :return: флаг успешного завершения, список из датафреймов с логами
        """
        return feature_obj.auto_fit(ds_aux=ds_aux, **auto_fit_parameters)

    def auto_fit(self, features=None, autofit_folder='auto_fit', plot_flag=-1, verbose=False,
                 woe_best_samples=None, method='opt', max_n_bins=10, min_bin_size=0.05,
                 criterion='entropy', scoring='neg_log_loss', max_depth=5, solver='cp', divergence='iv',
                 WOEM_on=True, WOEM_woe_threshold=0.1, WOEM_sb=False,
                 SM_on=True, SM_target_threshold=5, SM_size_threshold=100,
                 BL_on=True, BL_allow_Vlogic_to_increase_gini=10, BL_spec_form=False,
                 G_on=True, G_gini_threshold=5, G_with_test=True, G_gini_decrease_threshold=0.4, G_min_gini_in_time=-100,
                 WOEO_on=True, WOEO_all_samples=True,
                 PSI_on=True, PSI_base_period_index=-1, PSI_threshold=0.5,
                 cross_1l=None, cross_2l=0, cross_config=None):
        """
        Поиск оптимального биннинга, удовлетворяющего набору проверок. Итерационно для каждой переменной выполняются следующие шаги:
            1) Исходное разбиение на n бинов, где n на первой итерации равно max_n_bins
            2) Выполняются слияния соседних бинов с близким WOE (при выставленном флаге WOEM_on=True) и малых бинов (SM_on=True)
            3) Проводятся проверки на бизнес-логику (BL_on=True), стабильность и величину джини (G_on=True) и стабильность тренда (WOEO_on=True)
            4) Если любая из проверок проваливается, то уменьшаем n на 1 и возвращаемся на шаг 1
        Если после перебора всех n проверки так и не пройдены, то переменная исключается из списка.

        Значение любого из параметров биннинга может быть задано списком, например
            method=['opt', 'tree']
        В этом случае вся процедура повторяется для каждого набора параметров и затем выбирается биннинг с
        максимальным джини на сэмплах заданных параметром woe_best_samples.

        :param features: список переменных для обработки. При None обрабатываются все self.feature_woes
        :param autofit_folder: название папки, в которую будут сохранены результаты автобиннинга
        :param plot_flag: флаг для вывода графиков с биннингом:
                            -1 - графики не строить
                            0, False - графики сохранить в папку autofit_folder/Figs_binning, но не выводить в аутпут
                            1, True - графики сохранить в папку autofit_folder/Figs_binning и вывести в аутпут
        :param verbose: флаг для вывода подробных комментариев в процессе работы
        :param woe_best_samples: список сэмплов, джини которых будет учитываться при выборе лучшего биннинга.  При None берется джини на трэйне

        --- Метод биннинга ---
        :param method: 'tree' - биннинг деревом, 'opt' - биннинг деревом с последующей оптимизацией границ бинов библиотекой optbinning
        :param max_n_bins: максимальное кол-во бинов
        :param min_bin_size: минимальное число (доля) наблюдений в каждом листе дерева.
                                Если min_bin_size < 1, то трактуется как доля наблюдений от обучающей выборки

        --- Параметры биннинга для метода 'tree' ---
        :param criterion: критерий расщепления. Варианты значений: 'entropy', 'gini', 'log_loss'
        :param scoring: метрика для оптимизации
        :param max_depth: максимальная глубина дерева

        --- Параметры биннинга для метода 'opt' ---
        :param solver: солвер для оптимизации биннинга:
                        'cp' - constrained programming
                        'mip' - mixed-integer programming
                        'ls' - LocalSorver (www.localsorver.com)
        :param divergence: метрика для максимизации:
                        'iv' - Information Value,
                        'js' - Jensen-Shannon,
                        'hellinger' - Hellinger divergence,
                        'triangular' - triangular discrimination

        --- Параметры проверок ---
        :param WOEM_on: флаг проверки на разницу WOE между соседними бинами
        :param WOEM_woe_threshold: минимальная разрешенная дельта WOE между соседними бинами
        :param WOEM_sb: должна ли выполняться проверка для отдельных спец бинов
        :param SM_on: флаг проверки на размер бина
        :param SM_target_threshold: минимальное кол-во (доля) наблюдений с целевым событием в бине
        :param SM_size_threshold: минимальное кол-во (доля) наблюдений в бине
        :param BL_on: флаг проверки на бизнес-логику
        :param BL_allow_Vlogic_to_increase_gini: разрешить V-образную бизнес-логику, если она приводит к увеличению джини переменной на эту величину относительного монотонного тренда.
                                                 При значении 100 V-образная бизнес-логика запрещена
        :param BL_spec_form: бизнес-логика должна иметь заданное в ds.features_bl направление
        :param G_on: флаг проверки на джини
        :param G_gini_threshold: минимальное допустимое джини переменной. Нижняя граница доверительного интервала на трейне/бутстрепе должна быть выше этой величины
        :param G_with_test: так же проверяется джини на всех доступных сэмплах
        :param G_gini_decrease_threshold: допустимое уменьшение джини на всех сэмплах относительно трэйна.
                    В случае, если значение > 1, то проверяется условие gini(train) - gini(sample) <= G_gini_decrease_threshold
                              если значение <= 1, то проверяется условие 1 - gini(sample)/gini(train) <= G_gini_decrease_threshold
        :param G_min_gini_in_time: минимальное допустимое джини в разрезе срезов.
                    В случае, если значение > 1, то все значения джини на срезах должны удовлетворять условию gini(time) >= G_min_gini_in_time
                              если значение < 0, то все значения джини на срезах должны удовлетворять условию gini(time) >= gini(train) + G_min_gini_in_time
        :param WOEO_on: флаг проверки на сохранение тренда WOE
        :param WOEO_all_samples: проверять сохранение тренда WOE на всех сэмплах относительно трейна
        :param PSI_on: флаг проверки на стабильность популяции
        :param PSI_base_period_index: индекс основного среза в отсортированном списке значений срезов, относительного которого считается PSI остальных срезов
                                    при -1 тест для каждого среза считается относительно предыдущего
        :param PSI_threshold: максимально допустимое значение PSI на любом из срезов
        --- Кросс переменные ---
        :param cross_1l: переменные первого уровня для которых будут искаться лучшие кросс пары. Может принимать значения:
                           list - берется заданный список переменных
                           int - список формируется автоматически из заданного кол-ва переменных с лучшим джини
                           None  - берется список features
        :param cross_2l: переменные второго уровня, которые будут составлять пары с каждой переменной первого уровня. Может принимать значения:
                           list - берется заданный список переменных
                           int - для каждой переменной первого уровня отбираются свое заданное кол-во лучших переменных с максимальной метрикой criterion:
                                    0 - поиск не производится
                                    -1 - рассматриваются все возможные кросс пары
        :param cross_config: словарь с параметрами для перебининга переменных первого уровня
        """
        def save_check_dfs(dfs, woe_best_samples, check_file):
            def save_res_file(df, c, writer):
                if df.empty:
                    return None
                width = {'Log': {'A': 40, 'B': 12, 'C': 12, 'D': 12, 'E': 60},
                         'Business Logic': {'A': 40, 'others': 15, 'freeze': ['C2']},
                         'Gini': {'A': 40, 'B': 12, 'others': 15, 'freeze': ['C2']},
                         'WOE': {'A': 40, 'B': 12, 'C': 12, 'others': 15, 'freeze': ['D2']},
                         'PSI': {'A': 40, 'B': 12, 'others': 15, 'freeze': ['C2']},
                         'Params': {'A': 40, 'B': 12}
                         }
                df.to_excel(writer, sheet_name=c, index=False)
                worksheet = writer.sheets[c]
                for k in width[c]:
                    if k not in ['others', 'freeze']:
                        worksheet.column_dimensions[k].width = width[c][k]
                    elif k == 'others':
                        for cn in range(len(width[c]) - 2, worksheet.max_column + 1):
                            worksheet.column_dimensions[openpyxl.utils.get_column_letter(cn)].width = width[c]['others']
                    elif k == 'freeze':
                        for l in width[c]['freeze']:
                            worksheet.freeze_panes = worksheet[l]

            f_gini = {}
            Vlogic_features = set()
            if dfs:
                with pd.ExcelWriter(check_file, engine="openpyxl") as writer, \
                     pd.ExcelWriter(check_file[:-5] + '_all.xlsx', engine="openpyxl") as writer_all:
                    for c in ['Log', 'Business Logic', 'Gini', 'WOE', 'PSI']:
                        res_dfs = [dfs[f][c] for f in dfs if dfs[f]]
                        if not res_dfs:
                            continue
                        res_df = pd.concat(res_dfs).reset_index(drop=True)
                        if 'feature1' in res_df.columns:
                            group_columns = ['feature1', 'group1', 'feature']
                        else:
                            group_columns = ['feature']
                        save_res_file(res_df, c, writer_all)
                        if not res_df.empty:
                            if 'iteration' in res_df.columns:
                                res_df = res_df[res_df.groupby(group_columns)['iteration'].transform('max') == res_df['iteration']]
                            save_res_file(res_df, c, writer)
                            if c in ['Gini', 'Business Logic']:
                                res_df = res_df[res_df['feature'].isin([f for f in features if self.feature_woes[f].is_active])]
                                if c == 'Business Logic':
                                    Vlogic_features = set(res_df[res_df['trend_type'] == 'V-shape']['feature'].values)
                                if c == 'Gini':
                                    f_gini = res_df.set_index(group_columns).round(2)[[f for f in woe_best_samples if f in res_df.columns]].mean(axis=1).to_dict()
                    df_params = pd.DataFrame().from_dict(auto_fit_parameters, orient='index', columns=['value']).rename_axis('Parameter', axis=1).reset_index()
                    save_res_file(df_params, 'Params', writer)
                    save_res_file(df_params, 'Params', writer_all)
            return f_gini, Vlogic_features

        loc_args = locals()
        self.ds_aux.logger.info(make_header('SFA', 150))
        autofit_folder = autofit_folder.rstrip('/')
        folder = self.ds_aux.result_folder + autofit_folder + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        if features is None:
            features = list(self.feature_woes)
        cross_features = [rem_suffix(f) for f in features if is_cross_name(f)]
        features = [rem_suffix(f) for f in features if not is_cross_name(f)]
        params = ['method', 'max_n_bins', 'min_bin_size', 'criterion', 'scoring', 'max_depth', 'solver', 'divergence',
                  'WOEM_on',  'WOEM_woe_threshold', 'WOEM_sb', 'SM_on', 'SM_target_threshold', 'SM_size_threshold',
                  'G_on', 'G_gini_threshold', 'G_gini_decrease_threshold', 'G_with_test', 'G_min_gini_in_time',
                  'BL_on', 'BL_allow_Vlogic_to_increase_gini', 'BL_spec_form',
                  'WOEO_on', 'WOEO_all_samples',
                  'PSI_on', 'PSI_base_period_index', 'PSI_threshold']
        params_space = {f: loc_args[f] if isinstance(loc_args[f], list) else [loc_args[f]] for f in params}
        params_space['verbose'] = [verbose] if self.ds_aux.n_jobs == 1 else [False]
        params_list = []
        for param in itertools.product(*params_space.values()):
            auto_fit_parameters = dict(zip(params_space, param))
            if auto_fit_parameters['method'] == 'opt':
                for p in ['criterion', 'scoring', 'max_depth']:
                    auto_fit_parameters.pop(p)
            elif auto_fit_parameters['method'] == 'tree':
                for p in ['solver', 'divergence']:
                    auto_fit_parameters.pop(p)
            if auto_fit_parameters not in params_list:
                params_list.append(auto_fit_parameters)
        file_scorecard = f'{self.ds_aux.result_folder}{autofit_folder}_scorecard.xlsx'
        if woe_best_samples:
            woe_best_samples = [f for f in woe_best_samples if f in self.ds_aux.samples]
        if not woe_best_samples:
            woe_best_samples = [self.ds_aux.train_name]
        scorecard = pd.DataFrame()

        # ------------------------------------------ autobinning for features ------------------------------------------
        if features:
            self.ds_aux.logger.info(f'Performing autobinning with parameters space of size {len(params_list)}...')
            f_gini = {}
            Vlogic_features = {}
            check_dfs = {num_p: {} for num_p in range(1, len(params_list) + 1)}
            for num_p, auto_fit_parameters in enumerate(params_list, start=1):
                self.ds_aux.logger.info(f'Using parameters set {num_p}/{len(params_list)}: {auto_fit_parameters}')
                for p in ['SM_target_threshold', 'SM_size_threshold']:
                    if auto_fit_parameters[p] and auto_fit_parameters[p] < 1:
                        auto_fit_parameters[p] = int(round(len(self.ds_aux.samples[self.ds_aux.train_name]) * auto_fit_parameters[p], 0))
                self.ds_aux.logger.info(f'Processing {len(features)} features on {self.ds_aux.n_jobs} CPU{"s" if self.ds_aux.n_jobs > 1 else ""}...')
                if self.ds_aux.n_jobs > 1 and len(features) > self.ds_aux.n_jobs:
                    with futures.ProcessPoolExecutor(max_workers=self.ds_aux.n_jobs) as pool:
                        pool_map = pool.map(partial(self.auto_fit_feature, ds_aux=self.ds_aux, auto_fit_parameters=auto_fit_parameters),
                                             [self.feature_woes[feature] for feature in features])
                        for i, result in enumerate(tqdm(pool_map, total=len(features)) if self.ds_aux.logger.level <= 20 else pool_map):
                            self.feature_woes[features[i]].is_active, check_dfs[num_p][features[i]] = result
                        gc.collect()
                else:
                    for feature in tqdm(features) if self.ds_aux.logger.level <= 20 else features:
                        self.feature_woes[feature].is_active, check_dfs[num_p][feature] = \
                            self.feature_woes[feature].auto_fit(ds_aux=self.ds_aux, **auto_fit_parameters)
                f_gini[num_p], Vlogic_features[num_p] = save_check_dfs(dfs=check_dfs[num_p], woe_best_samples=woe_best_samples,
                                                                       check_file=f'{folder}checks{f"_{num_p}" if len(params_list) > 1 else ""}.xlsx')
                woe_dfs = [check_dfs[num_p][f]['scorecard'] for f in check_dfs[num_p] if self.feature_woes[f].is_active]
                if woe_dfs:
                    scorecard = pd.concat(woe_dfs)
                    scorecard.to_excel(f'{folder}scorecard{f"_{num_p}" if len(params_list) > 1 else ""}_all.xlsx', index=False)
                    scorecard = scorecard[scorecard.groupby(['feature'])['iteration'].transform('max') == scorecard['iteration']]
                    scorecard.to_excel(f'{folder}scorecard{f"_{num_p}" if len(params_list) > 1 else ""}.xlsx', index=False)
                else:
                    self.ds_aux.logger.info(f'There are no successfully binned features with parameters set {num_p}!')
    
            if len(params_list) > 1:
                WOE_best_dfs = []
                for f in features:
                    ginis = {num_p: f_gini[num_p][f] if f not in Vlogic_features[num_p] else f_gini[num_p][f] - params_list[num_p - 1]['BL_allow_Vlogic_to_increase_gini']
                             for num_p in f_gini if f in f_gini[num_p]}
                    if ginis:
                        num_p = max(ginis, key=ginis.get)
                        tmp = check_dfs[num_p][f]['scorecard']
                        tmp['params_set'] = num_p
                        tmp[f'Gini {woe_best_samples[0] if len(woe_best_samples) == 1 else f"avg {woe_best_samples}"}'] = round(f_gini[num_p][f], 2)
                        WOE_best_dfs.append(tmp[tmp.groupby(['feature'])['iteration'].transform('max') == tmp['iteration']])
                if WOE_best_dfs:
                    scorecard = pd.concat(WOE_best_dfs)
                else:
                    self.ds_aux.logger.info(f'There are no successfully binned features with any parameters set!')
            if not scorecard.empty:
                scorecard.to_excel(file_scorecard, index=False)
                self.import_scorecard(scorecard, verbose=False, fit_flag=False)
            excluded = [f for f in features if not self.feature_woes[f].is_active]
            if excluded:
                self.ds_aux.logger.info(f'Excluded {len(excluded)} features {excluded} because no suitable binning was found for them')
                exclude_df = pd.concat([check_dfs[1][f]['Log'] for f in excluded]).drop_duplicates(subset=['feature'], keep='last')
                exclude_df.loc[exclude_df['result'] == 'Exception', 'reason'] = 'Exception'
                self.ds_aux.logger.info(f"Statistics on reasons for exclusion:\n{exclude_df['reason'].value_counts().to_string()}")
        else:
            excluded = []

        # --------------------------------------- autobinning for cross features ---------------------------------------
        if cross_features or cross_2l != 0:
            self.ds_aux.logger.info(make_header('CROSS-FEATURES ANALYSIS', 150))
            autofit_folder += '_cross'
            folder = self.ds_aux.result_folder + autofit_folder + '/'

            f_gini = {}
            features_active = set([f for f in self.feature_woes if self.feature_woes[f].is_active])
            if scorecard.empty:
                scorecard = self.export_scorecard()
            new_sc = pd.DataFrame()
            if cross_features:
                crosses = {}
                for f in cross_features:
                    f1, f2 = cross_split(f)
                    if f1 in features_active:
                        try:
                            crosses[f1].append(f2)
                        except:
                            crosses[f1] = [f2]
                    else:
                        self.ds_aux.logger.info(f'No active binning for feature {f1} found. Cross feature {f} skipped.')
            else:
                if cross_1l is None:
                    cross_1l = features_active
                elif isinstance(cross_1l, int):
                    cross_1l = {rem_suffix(f) for f in pd.concat(
                        [self.feature_woes[f].transform().calc_gini(samples=[self.feature_woes[f].ds.train_name])[
                             self.feature_woes[f].ds.train_name] for f in features_active]
                    ).sort_values(ascending=False)[:cross_1l].index}
                cross_1l = set(cross_1l) & features_active
                crosses = {}
                if isinstance(cross_2l, int):
                    if cross_2l == -1 or cross_2l >= len(features_active) - 1:
                        crosses = {f1: [f2 for f2 in features_active if f2 != f1] for f1 in cross_1l}
                    else:
                        self.ds_aux.logger.info('Finding the best pairs to first-level features...')
                        for f1 in tqdm(cross_1l) if self.ds_aux.logger.level <= 20 else cross_1l:
                            df1_group = self.feature_woes[f1].ds.samples[self.feature_woes[f1].ds.train_name][['group']]
                            impurity = {}
                            if self.ds_aux.n_jobs > 1 and len(features_active) > self.ds_aux.n_jobs:
                                f2_features = [f2 for f2 in features_active if f2 != f1]
                                with futures.ProcessPoolExecutor(max_workers=self.ds_aux.n_jobs) as pool:
                                    for i, result in enumerate(pool.map(partial(self.get_f2_impurity, df1_group=df1_group),
                                                                        [self.feature_woes[f2] for f2 in f2_features])):
                                        impurity[f2_features[i]] = result
                                    gc.collect()
                            else:
                                impurity = {f2: self.get_f2_impurity(self.feature_woes[f2], df1_group) for f2 in
                                            features_active if f2 != f1}
                            crosses[f1] = sorted(impurity, key=impurity.get, reverse=True)[:cross_2l]
                elif isinstance(cross_2l, list):
                    crosses = {f1: cross_2l for f1 in cross_1l}
                cross_features = [cross_name(f1, f2) for f1 in crosses for f2 in crosses[f1]]

            if cross_config:
                self.ds_aux.logger.info('Rebinning of first level features with different set of parameters...')
                if 'woe_best_samples' in cross_config:
                    woe_best_samples = [f for f in cross_config['woe_best_samples'] if f in self.ds_aux.samples or 'Bootstrap' if f]
                    if not woe_best_samples:
                        woe_best_samples = [self.ds_aux.train_name]
                cross_config = {f: v if isinstance(v, list) else [v] for f,v in cross_config.items() if f in params}
                params_space.update(cross_config)
                fl_auto_fit_parameters = params_space.copy()
                fl_auto_fit_parameters.update({'features': list(crosses), 'G_gini_threshold': G_gini_threshold,
                                               'autofit_folder': autofit_folder, 'plot_flag': -1,
                                               'verbose': False, 'cross_2l': 0, 'cross_config': None})
                new_binning = copy.deepcopy(self)
                new_binning.ds_aux.logger.setLevel(50)
                ini_logger_level = new_binning.ds_aux.logger_level
                new_binning.ds_aux.logger_level = 50
                for f in new_binning.feature_woes:
                    new_binning.feature_woes[f].ds.logger_level = 50
                new_binning.auto_fit(**fl_auto_fit_parameters)
                new_binning.ds_aux.logger.setLevel(ini_logger_level)
                for f in new_binning.feature_woes:
                    new_binning.feature_woes[f].ds.logger_level = ini_logger_level
                crosses = {f1: v for f1, v in crosses.items() if new_binning.feature_woes[f1].is_active}
            else:
                new_binning = self
                if not os.path.exists(folder):
                    os.makedirs(folder)
            self.ds_aux.logger.info('Creating feature_crosses...')
            for f1 in tqdm(crosses) if self.ds_aux.logger.level <= 20 else crosses:
                self.feature_crosses[f1] = FeatureCross(new_binning.feature_woes[f1])
                self.feature_crosses[f1].add_f2_features([new_binning.feature_woes[f2] for f2 in crosses[f1]])

            file_scorecard = f'{self.ds_aux.result_folder}{autofit_folder}_scorecard.xlsx'
            params_space['BL_allow_Vlogic_to_increase_gini'] = [100]
            params_space['max_n_bins'] = [min(max(params_space['max_n_bins']), 5)]
            params_space['G_on'] = [False]
            params_list = []
            for param in itertools.product(*params_space.values()):
                auto_fit_parameters = dict(zip(params_space, param))
                if auto_fit_parameters['method'] == 'opt':
                    for p in ['criterion', 'scoring', 'max_depth']:
                        auto_fit_parameters.pop(p)
                elif auto_fit_parameters['method'] == 'tree':
                    for p in ['solver', 'divergence']:
                        auto_fit_parameters.pop(p)
                if auto_fit_parameters not in params_list:
                    params_list.append(auto_fit_parameters)
            self.ds_aux.logger.info(f'Performing autobinning for cross features with parameters space of size {len(params_list)}...')

            check_dfs = {num_p: {} for num_p in range(1, len(params_list) + 1)}
            checks_result = {num_p: {} for num_p in range(1, len(params_list) + 1)}

            for num_p, auto_fit_parameters in enumerate(params_list, start=1):
                self.ds_aux.logger.info(f'Using parameters set {num_p}/{len(params_list)}: {auto_fit_parameters}')
                for p in ['SM_target_threshold', 'SM_size_threshold']:
                    if auto_fit_parameters[p] and auto_fit_parameters[p] < 1:
                        auto_fit_parameters[p] = int(round(len(self.ds_aux.samples[self.ds_aux.train_name]) * auto_fit_parameters[p], 0))
                self.ds_aux.logger.info(f'Processing {len(crosses)} first level features on {self.ds_aux.n_jobs} CPU{"s" if self.ds_aux.n_jobs > 1 else ""}...')

                if self.ds_aux.n_jobs > 1 and len(crosses) > 3:
                    with futures.ProcessPoolExecutor(max_workers=self.ds_aux.n_jobs) as pool:
                        crosses_list = list(crosses)
                        pool_map = pool.map(partial(self.auto_fit_feature, ds_aux=self.ds_aux, auto_fit_parameters=auto_fit_parameters),
                                                    [self.feature_crosses[f1] for f1 in crosses_list])
                        for i, result in enumerate(tqdm(pool_map, total=len(crosses_list)) if self.ds_aux.logger.level <= 20 else pool_map):
                            checks_result[num_p][crosses_list[i]], check_dfs[num_p][crosses_list[i]] = result
                        gc.collect()
                else:
                    for f1 in tqdm(crosses) if self.ds_aux.logger.level <= 20 else crosses:
                        checks_result[num_p][f1], check_dfs[num_p][f1] = self.feature_crosses[f1].auto_fit(ds_aux=self.ds_aux, **auto_fit_parameters)

                for f1 in crosses:
                    if check_dfs[num_p][f1]:
                        for c in ['Log', 'Business Logic', 'Gini', 'WOE', 'PSI']:
                            check_dfs[num_p][f1][c].insert(loc=0, column='feature1', value=f1)
                f_gini[num_p], _ = save_check_dfs(dfs=check_dfs[num_p], woe_best_samples=woe_best_samples, check_file=f'{folder}checks_crosses{f"_{num_p}" if len(params_list) > 1 else ""}.xlsx')
                woe_dfs = [check_dfs[num_p][f]['scorecard'] for f in check_dfs[num_p] if check_dfs[num_p][f]]
                if woe_dfs and not pd.concat(woe_dfs).empty:
                    new_sc = pd.concat(woe_dfs)
                    new_sc = new_sc[~new_sc['feature'].isin([cross_name(f1, f2) for f1 in checks_result[num_p] for f2 in checks_result[num_p][f1] if not checks_result[num_p][f1][f2]])]
                    new_sc.to_excel(f'{folder}scorecard_crosses{f"_{num_p}" if len(params_list) > 1 else ""}.xlsx', index=False)
                else:
                    self.ds_aux.logger.info(f'There are no successfully binned cross features with parameters set {num_p}!')
            if len(params_list) > 1:
                woe_dfs = []
                others_dfs = {}
                for k in {k for i in f_gini for k in f_gini[i]}:
                    f1, group, f2 = k
                    g = {i: f_gini[i][k] for i in f_gini if k in f_gini[i] and checks_result[i][f1][f2]}
                    if g:
                        num_p = max(g, key=g.get)
                        tmp = check_dfs[num_p][f1]['scorecard']
                        tmp = tmp[tmp['feature'] == cross_name(f1, f2)]
                        tmp['params_set'] = num_p
                        others_dfs[cross_name(f1, f2)] = tmp[tmp['group'] == -100].copy()
                        woe_dfs.append(tmp[tmp['group'].str[1:-1].str.split(', ').str[0] == str(group)])
                woe_dfs += list(others_dfs.values())
                if woe_dfs:
                    new_sc = pd.concat(woe_dfs).sort_values(by=['feature', 'group'])
                else:
                    self.ds_aux.logger.info(f'There are no successfully binned cross features with any parameters set!')
            self.feature_crosses.clear()
            if not new_sc.empty:
                scorecard = pd.concat([scorecard, new_sc])
                scorecard.to_excel(file_scorecard, index=False)
                self.import_scorecard(scorecard, verbose=False, fit_flag=False)
        self.ds_aux.logger.info(f'Scorecard saved to the file {file_scorecard}')
        if plot_flag != -1:
            self.ds_aux.logger.info('Plotting binnings...')
            self.plot_bins(folder=folder + 'Figs_binning', features=features + cross_features, plot_flag=plot_flag, verbose=True)
        self.ds_aux.logger.info(f'All done!{f" {len(features) - len(excluded)}/{len(features)} features successfully binned." if features else ""}'
              f'{f" Found {len([f for f in self.cross_features if rem_suffix(f) in cross_features])} cross features." if cross_features else ""}')

    @staticmethod
    def plot_bins_feature(feature_woe, ds_aux, folder=None, plot_flag=True, show_groups=False, all_samples=False, stat_size=None):
        """
        Отрисовывает биннингодной переменной
        :param feature_woe: объект класса FeatureWOE
        :param ds_aux: вспомогательный ДатаСэмпл с полем среза
        :param folder: папка, в которую должны быть сохранены рисунки. По умолчанию не сохраняются
        :param plot_flag: флаг для вывода рисунка
        :param show_groups: флаг для отображения номер групп на рисунке
        :param all_samples: отрисовка бинов по всем сэмплам, может принимать значения:
                            0, False - строятся бины только на трэйне
                            1, True – строятся бины по всем сэмплам, таргет рейт указывается только для трэйна
                            >1  – строятся бины и таргет рейт указывается по всем сэмплам
        :param stat_size: размер шрифта для вывода статистики по бинам на графике
        """
        f_WOE = add_suffix(feature_woe.feature)
        ds = feature_woe.transform()
        if stat_size is None:
            stat_size = 12 if len(feature_woe.groups) < 8 else 10
        if ds.samples is None:
            return None
        if not feature_woe.categorical:
            if ds_aux.time_column is None:
                fig, (ax_2, ax_1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3.5]}, figsize=(13.5, 6))
            else:
                fig, axd = plt.subplot_mosaic([['ul', 'ur'], ['l', 'l']], figsize=(13.5, 9),
                                              gridspec_kw={'width_ratios': [1, 3.5], 'height_ratios': [2, 1]})
                ax_1 = axd['ur']
                ax_2 = axd['ul']
                ax_3 = axd['l']
            x = ds.samples[ds.train_name][feature_woe.feature]
            if feature_woe.special_bins:
                x = x.replace(list(feature_woe.special_bins.values()), np.nan).dropna()
            x_stat = {'Max': x.max(), 'Median': x.median(), 'Min': x.min()}
            xlim = [x.quantile(0.05), x.quantile(0.95)]
            if x.min() >= 0:
                xlim[0] = x.min()
            if x.max() <= 0:
                xlim[1] = x.max()
            x = x[x.between(*xlim)]
            x.plot.hist(ax=ax_2, bins=20, rwidth=0.7, xlim=xlim)
            for name, sample in ds.samples.items():
                try:
                    sample[feature_woe.feature][sample[feature_woe.feature].between(*xlim)].plot(kind='kde', ax=ax_2, secondary_y=True, xlim=xlim, label=name)
                except:
                    pass
            ax_2.set_ylabel('Amount')
            ax_2.grid(True)
            ax_2.right_ax.set_yticklabels([])
            ax_2.right_ax.set_ylim(0, ax_2.right_ax.get_ylim()[1])
            ax_2.right_ax.grid(False)
            for tick in ax_2.get_xticklabels():
                tick.set_rotation(30)
            ax_2.tick_params(axis='both', which='both', length=5, labelbottom=True)
            ax_2.xaxis.get_label().set_visible(True)
            h1, l1 = ax_2.right_ax.get_legend_handles_labels()
            ax_2.legend(h1, [l.replace(' (right)', '') for l in l1], fontsize=10, loc='upper right')
            for i, (k, v) in enumerate(x_stat.items()):
                ax_2.annotate(f'{k}: %.{feature_woe.round_digits}f' % v, xy=(0, 1), xycoords=('axes fraction', 'axes fraction'),
                              xytext=(0, 15 + 22*i), textcoords='offset pixels', color='black', ha='left')
        else:
            if ds_aux.time_column is None:
                fig, ax_1 = plt.subplots(1, 1, figsize=(13.5, 6))
            else:
                fig, (ax_1, ax_3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(13.5, 9))
        if ds_aux.time_column is not None:
            ds.add_aux(ds_aux)
            gini_in_time = ds.calc_gini_in_time()
            gini_in_time.columns = [x[1] for x in gini_in_time.columns]
            gini_in_time[list(ds.samples)].plot(ax=ax_3, marker='o')
            ax_3.fill_between(gini_in_time.index,
                              gini_in_time['CI_lower'],
                              gini_in_time['CI_upper'],
                              alpha=0.1, color='blue', label=f'{int(round((1 - ds.ci_alpha) * 100))}% CI')
            ax_3.set_ylim(min(0, gini_in_time[list(ds.samples)].min(axis=1).min()), ax_3.get_ylim()[1])
            ax_3.set_ylabel('Gini')
            for tick in ax_3.get_xticklabels():
                tick.set_rotation(30)
            ax_3.tick_params(axis='both', which='both', length=5, labelbottom=True)
            ax_3.xaxis.get_label().set_visible(False)
            ax_3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            df_stat = ds.samples[ds.train_name].groupby(ds.time_column).agg({ds.target: ['count', 'mean']}).set_axis(['Amount', 'Target rate'], axis=1)\
                .merge(ds.psi(time_column=ds.time_column, features=[f_WOE]).T.set_axis(['PSI'], axis=1), left_index=True, right_index=True, how='left')
            df_stat['Target rate'] = (df_stat['Target rate'] * 100).round(2).astype('str') + '%'
            for stat, shift in {'PSI': 10, 'Target rate': 30,  'Amount': 50}.items():
                ax_3.annotate(stat, xy=(-0.5, 1), xycoords=('data', 'axes fraction'),
                              xytext=(0, shift), textcoords='offset pixels', color='black', ha='right', size=stat_size-2)
                for i, val in enumerate(df_stat[stat].values):
                    ax_3.annotate(str(val), xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, shift),
                                  textcoords='offset pixels', ha='center',size=stat_size-2,
                                  color='black' if stat != 'PSI' else 'green' if val < 0.1 else 'orange' if val < 0.25 else 'red')
        woe_df = feature_woe.calc_woe_confint(all_samples=all_samples).rename({'n': 'Amount_0', 'n1': 'target_0'}, axis=1)
        if ds_aux.id_column is not None:
            to_calc = feature_woe.ds.samples[ds.train_name].copy()
            to_calc[ds_aux.id_column] = ds_aux.samples[ds_aux.train_name][ds_aux.id_column]
            to_calc[f'{ds_aux.id_column}_1'] = to_calc.apply(lambda row: row[ds_aux.id_column] if row[ds.target] == 1 else np.nan, axis=1)
            woe_df = woe_df.merge(to_calc.groupby('group').agg({ds_aux.id_column: 'nunique',
                                                             f'{ds_aux.id_column}_1': 'nunique'}) \
                                  .set_axis(['Unique ids', 'Class 1 unique ids'], axis=1), left_index=True, right_index=True)
        else:
            woe_df['Unique ids'] = woe_df['Amount_0']
            woe_df['Class 1 unique ids'] = woe_df['target_0']
        woe_df = woe_df.reset_index()
        samples_to_show = [ds.train_name]
        if all_samples:
            samples_to_show += [s for s in feature_woe.ds.samples if s != ds.train_name]
            for i, sample in enumerate(samples_to_show[1:], start=1):
                tmp = feature_woe.ds.samples[sample].copy()
                tmp['group'] = feature_woe.set_groups(data=tmp[feature_woe.feature])
                woe_df = woe_df.merge(tmp.groupby('group').agg({ds.target: ['sum', 'size']}).astype('int64').set_axis([f'target_{i}', f'Amount_{i}'], axis=1).reset_index(), on='group', how='outer')
        woe_df[f'woe_{ds.train_name}'] = woe_df['woe']
        woe_df['down_trend'] = ((woe_df['woe'] < woe_df['woe'].shift(1)) & (woe_df['group'].shift(1) >= 0)) if not feature_woe.categorical else False
        woe_df['label'] = woe_df['group'].map(feature_woe.groups).astype('str').apply(lambda x: '\n'.join(wrap(str(x), 25)))
        woe_df['Target rate'] = (woe_df['target_0'] * 100 / woe_df['Amount_0']).round(2).astype('str') + '%'
        woe_df = woe_df.fillna(0)
        if feature_woe.categorical:
            woe_df = woe_df[pd.isnull(woe_df['woe']) == False].sort_values('woe').reset_index()

        label_sb = {}
        for sb, g in feature_woe.special_bins_groups.items():
            try:
                label_sb[g].append(str(sb))
            except:
                label_sb[g] = [str(sb)]
        for g, label_l in label_sb.items():
            label = ',\n'.join(label_l)
            if g < 0:
                woe_df.loc[woe_df['group'] == g, 'label'] = label
            else:
                woe_df.loc[woe_df['group'] == g, 'label'] += ',\n' + label

        if not feature_woe.categorical:
            woe_df['label'] = woe_df['label'].replace({']': ')', '\[-inf': '(-inf'}, regex=True)
        if show_groups:
            woe_df['label'] += '\n' + 'group ' + woe_df['group'].astype('str')
        ax_1.set_ylabel('Sample fraction')
        ax_1.set_xticks(range(woe_df.shape[0]))
        ax_1.set_xticklabels(woe_df['label'], rotation=30, ha='right', fontsize=10 if show_groups else 11)
        ax2 = ax_1.twinx()
        ax2.set_ylabel('WOE', loc='bottom')
        ax2.grid(axis='y', zorder=1, alpha=0.6)
        for i, sample in enumerate(samples_to_show):
            w = 0.8/len(samples_to_show)
            shift = (i - (len(samples_to_show) - 1)/2)*(w + 0.03)
            amt = woe_df[f'Amount_{i}'].sum()
            ax_1.bar(woe_df.index + shift, (woe_df[f'Amount_{i}'] - woe_df[f'target_{i}']) / amt, width=w, zorder=0, alpha=1 - i * 0.25, color='forestgreen', label=f'Class 0{" (" + sample + ")" if len(samples_to_show) > 1 else ""}')
            ax_1.bar(woe_df.index + shift, woe_df[f'target_{i}'] / amt, bottom=(woe_df[f'Amount_{i}'] - woe_df[f'target_{i}']) / amt, width=w, zorder=0, alpha=1 - i * 0.25, color='indianred', label=f'Class 1{" (" + sample + ")" if len(samples_to_show) > 1 else ""}')
            if all_samples > 1 or i == 0:
                if all_samples <= 1:
                    shift = 0
                for x, y in woe_df[f'woe_{sample}'].items():
                    ax2.annotate(str(y), xy=(x + shift, y), xytext=(30 if woe_df['down_trend'][x] else -30, 10), textcoords='offset pixels',
                                 color='black', ha='center', size=11 if (all_samples <= 1 or len(samples_to_show) < 3) else 10)
                if i == 0:
                    if feature_woe.categorical:
                        ax2.plot(woe_df.index + shift, woe_df['woe'], 'bo', linewidth=2.0, zorder=4, label='WOE')
                    else:
                        woe_df['line'] = woe_df['group'].apply(lambda x: not isinstance(x, str) and x >= 0).astype('int')
                        ax2.plot(woe_df[woe_df['line'] == 1].index + shift, woe_df[woe_df['line'] == 1]['woe'], 'bo-', linewidth=2.0, zorder=4, label='WOE')
                        ax2.plot(woe_df[woe_df['line'] == 0].index + shift, woe_df[woe_df['line'] == 0]['woe'], 'bo', linewidth=2.0, zorder=4)
                    ax2.errorbar(woe_df.index + shift, (woe_df['woe_lower'] + woe_df['woe_upper']) / 2,
                                 xerr=0,  yerr=(woe_df['woe_upper'] - woe_df['woe_lower']) / 2,
                                 color='b', capsize=5, fmt='none', alpha=0.4)
                else:
                    ax2.plot(woe_df.index + shift, woe_df[f'woe_{sample}'], 'bo', linewidth=2.0, zorder=4, alpha=1 - i * 0.25)

        for stat, shift in {'Target rate': 15, 'Class 1 unique ids': 37, 'Unique ids': 59, 'Amount_0': 81}.items():
            ax_1.annotate(stat.replace('_0', ''), xy=(-0.5, 1), xycoords=('data', 'axes fraction'),
                          xytext=(0, shift), textcoords='offset pixels', color='black', ha='right', size=stat_size)
            for i, val in enumerate(woe_df[stat].values):
                ax_1.annotate(str(val), xy=(i, 1), xycoords=('data', 'axes fraction'),
                              xytext=(0, shift), textcoords='offset pixels', color='black', ha='center', size=stat_size)
        ginis = ds.calc_gini(features=[f_WOE]).T[f_WOE].to_dict()
        ginis[f'{"Analytic" if ds.ci_analytic or ds.bootstrap_base is None else "Bootstrap"} {int(round((1 - ds.ci_alpha) * 100))}% CI'] = f"({ginis['CI_lower']}; {ginis['CI_upper']})"
        ax_1.annotate('Gini: %s' % ('; '.join([f'{name} {gini}' for name, gini in ginis.items() if name not in ['CI_lower', 'CI_upper']])),
                      xy=(0.5, 1), xycoords=('figure fraction', 'axes fraction'), xytext=(0, 115), textcoords='offset pixels', color='blue', ha='center')
        ax_1.grid(False)
        ax_1.tick_params(axis='y', which='both', length=5)
        h1, l1 = ax_1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.27 if len(samples_to_show) > 1 else 1.23, 1), fontsize=9 if len(samples_to_show) > 1 else 11)
        plt.suptitle(ds_aux.feature_titles[feature_woe.feature] if feature_woe.feature in ds_aux.feature_titles
                     else feature_woe.feature, fontsize=13, weight='bold')
        fig.tight_layout()
        if folder is not None:
            fig.savefig(f'{folder}/{f_WOE}.png', bbox_inches="tight")
        if plot_flag:
            plt.show()
        return fig

    @staticmethod
    def plot_bins_feature_cross(params, ds_aux, folder=None, plot_flag=True, show_groups=False, all_samples=False, stat_size=None):
        """
        Отрисовывает биннингодной переменной
        :param feature_woe: объект класса FeatureWOE
        :param ds_aux: вспомогательный ДатаСэмпл с полем среза
        :param folder: папка, в которую должны быть сохранены рисунки. По умолчанию не сохраняются
        :param plot_flag: флаг для вывода рисунка
        :param show_groups: флаг для отображения номер групп на рисунке
        :param all_samples: отрисовка бинов по всем сэмплам, может принимать значения:
                            0, False - строятся бины только на трэйне
                            1, True – строятся бины по всем сэмплам, таргет рейт указывается только для трэйна
                            >1  – строятся бины и таргет рейт указывается по всем сэмплам
        :param stat_size: размер шрифта для вывода статистики по бинам на графике
        """
        feature_cross, f2 = params
        feature_woe = feature_cross.feature_woe
        feature_woe2 = list(feature_cross.cross[f2].values())[0]
        f = cross_name(feature_woe.feature, f2)
        f_WOE = add_suffix(f)
        ds = feature_cross.transform(features=[f2])
        if stat_size is None:
            stat_size = 12 - len(feature_cross.cross[f2])*max([len(fw.woes) for fw in feature_cross.cross[f2].values()]) // 6
        group_coord = {}
        group_w = {}
        group_d = {}
        for i, (g, fw) in enumerate(feature_cross.cross[f2].items()):
            for j, g2 in enumerate(fw.groups):
                w = 0.8 / len(fw.groups)
                shift = (j - (len(fw.groups) - 1) / 2) * (w + 0.03)
                group_coord[str([g, g2])] = i + shift
                group_w[str([g, g2])] = w
                if len(fw.groups) == 1:
                    group_d[g] = [i - w / 2, i + w / 2]
                elif j == 0:
                    group_d[g] = [i + shift - w / 2, ]
                elif j == len(fw.groups) - 1:
                    group_d[g].append(i + shift + w / 2)
        if ds.samples is None:
            return None

        if ds_aux.time_column is None:
            fig, (ax_1, ax_x) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 0.5]}, figsize=(13.5, 6.5))
        else:
            fig, (ax_1, ax_x, ax_3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [6, 0.5, 3]},
                                                   figsize=(13.5, 9.5))
        if ds_aux.time_column is not None:
            ds.add_aux(ds_aux)
            gini_in_time = ds.calc_gini_in_time()
            gini_in_time.columns = [x[1] for x in gini_in_time.columns]
            gini_in_time[list(ds.samples)].plot(ax=ax_3, marker='o')
            ax_3.fill_between(gini_in_time.index,
                              gini_in_time['CI_lower'],
                              gini_in_time['CI_upper'],
                              alpha=0.1, color='blue', label=f'{int(round((1 - ds.ci_alpha) * 100))}% CI')
            ax_3.set_ylim(min(0, gini_in_time[list(ds.samples)].min(axis=1).min()), ax_3.get_ylim()[1])
            ax_3.set_ylabel('Gini')
            for tick in ax_3.get_xticklabels():
                tick.set_rotation(30)
            ax_3.tick_params(axis='both', which='both', length=5, labelbottom=True)
            ax_3.xaxis.get_label().set_visible(False)
            ax_3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            df_stat = ds.samples[ds.train_name].groupby(ds.time_column).agg({ds.target: ['count', 'mean']}).set_axis(['Amount', 'Target rate'], axis=1) \
                .merge(ds.psi(time_column=ds.time_column, features=[f_WOE]).T.set_axis(['PSI'], axis=1),  left_index=True, right_index=True, how='left')
            df_stat['Target rate'] = (df_stat['Target rate'] * 100).round(2).astype('str') + '%'
            for stat, shift in {'PSI': 10, 'Target rate': 30, 'Amount': 50}.items():
                ax_3.annotate(stat, xy=(-0.5, 1), xycoords=('data', 'axes fraction'), xytext=(0, shift),
                              textcoords='offset pixels', color='black', ha='right', size=stat_size-2)
                for i, val in enumerate(df_stat[stat].values):
                    ax_3.annotate(str(val), xy=(i, 1), xycoords=('data', 'axes fraction'), xytext=(0, shift),
                                  textcoords='offset pixels', ha='center', size=stat_size-2,
                                  color='black' if stat != 'PSI' else 'green' if val < 0.1 else 'orange' if val < 0.25 else 'red')

        to_calc = feature_cross.get_data(f2)
        to_calc['group'] = feature_cross.set_groups(data=to_calc, f2=f2)
        if ds_aux.id_column is not None:
            to_calc[ds_aux.id_column] = ds_aux.samples[ds_aux.train_name][ds_aux.id_column]
            to_calc[f'{ds_aux.id_column}_1'] = to_calc.apply(
                lambda row: row[ds_aux.id_column] if row[ds.target] == 1 else np.nan, axis=1)
            woe_df = to_calc.groupby('group').agg({ds.target: ['sum', 'size'],
                                                   ds_aux.id_column: 'nunique',
                                                  f'{ds_aux.id_column}_1': 'nunique'}).astype('int64').set_axis(
                ['target_0', 'Amount_0', 'Unique ids', 'Class 1 unique ids'], axis=1)
        else:
            woe_df = to_calc.groupby('group').agg({ds.target: ['sum', 'size']}).astype('int64').set_axis(['target_0', 'Amount_0'], axis=1)
            woe_df['Unique ids'] = woe_df['Amount_0']
            woe_df['Class 1 unique ids'] = woe_df['target_0']
        woe_df = woe_df.reset_index()
        samples_to_show = [ds.train_name]
        if all_samples:
            samples_to_show += [s for s in feature_woe.ds.samples if s != ds.train_name]
            for i, sample in enumerate(samples_to_show[1:], start=1):
                tmp = feature_cross.get_data(f2, sample=sample)
                tmp['group'] = feature_cross.set_groups(data=tmp, f2=f2)
                woe_df = woe_df.merge(tmp.groupby('group').agg({ds.target: ['sum', 'size']}).set_axis(
                    [f'target_{i}', f'Amount_{i}'], axis=1).reset_index(), on='group', how='outer')
                if all_samples > 1 and i > 0:
                    woe_df[f'woe_{i}'] = woe_df['group'].map(feature_woe.calc_simple_woes(df=tmp))
        woe_df['woe_0'] = woe_df['group'].map(feature_cross.get_woes(f2))
        woe_df['label'] = woe_df['group'].map(feature_cross.get_values(f2)).astype('str').apply(lambda x: '\n'.join(wrap(str(x), 25)))
        woe_df['Target rate'] = (woe_df['target_0'] * 100 / woe_df['Amount_0']).round(1).astype('str') + '%'
        woe_df['coord'] = woe_df['group'].map(group_coord)
        woe_df['w'] = woe_df['group'].map(group_w)
        woe_df['group1'] = woe_df['group'].str[1:-1].str.split(', ').str[0].astype('int')
        woe_df['group2'] = woe_df['group'].str[1:-1].str.split(', ').str[1].astype('int')
        woe_df['down_trend'] = ((woe_df['woe_0'] < woe_df['woe_0'].shift(1)) & (
                woe_df['group1'] == woe_df['group1'].shift(1)) & (woe_df['group2'].shift(1) >= 0)) if not feature_woe2.categorical else False
        woe_df = woe_df.fillna(0)
        for g1 in feature_woe.groups:
            label_sb = {}
            fw2 = feature_cross.cross[f2][g1]
            for sb, g2 in fw2.special_bins_groups.items():
                if fw2.ds.samples[fw2.ds.train_name][fw2.feature].isin([fw2.special_bins[sb]]).any():
                    try:
                        label_sb[g2].append(str(sb))
                    except:
                        label_sb[g2] = [str(sb)]
            for g2, label_l in label_sb.items():
                label = ',\n'.join(label_l)
                if g2 < 0 or fw2.ds.samples[fw2.ds.train_name][fw2.feature].isin(list(fw2.special_bins.values())).all():
                    woe_df.loc[(woe_df['group1'] == g1) & (woe_df['group2'] == g2), 'label'] = label
                else:
                    woe_df.loc[(woe_df['group1'] == g1) & (woe_df['group2'] == g2), 'label'] += ', ' + label

        if not feature_woe2.categorical:
            woe_df['label'] = woe_df['label'].replace({']': ')', '\[-inf': '(-inf'}, regex=True)
        if show_groups:
            woe_df['label'] += '\n' + 'group ' + woe_df['group'].astype('str')
        ax_1.set_ylabel('Sample fraction')
        ax_1.set_xticks(woe_df['coord'])
        ax_1.set_xticklabels(woe_df['label'], rotation=30, ha='right', fontsize=stat_size - 2 if show_groups else stat_size)

        ax2 = ax_1.twinx()
        ax2.set_ylabel('WOE', loc='bottom')
        ax2.grid(axis='y', zorder=1, alpha=0.6)
        for i, sample in enumerate(samples_to_show):
            w = woe_df['w'] * 0.9 / len(samples_to_show)
            shift = (i - (len(samples_to_show) - 1) / 2) * (w + 0.03)
            amt = woe_df[f'Amount_{i}'].sum()
            ax_1.bar(woe_df['coord'] + shift, (woe_df[f'Amount_{i}'] - woe_df[f'target_{i}']) / amt, width=w,
                     zorder=0, alpha=1 - i * 0.25, color='forestgreen',
                     label=f'Class 0{" (" + sample + ")" if len(samples_to_show) > 1 else ""}')
            ax_1.bar(woe_df['coord'] + shift, woe_df[f'target_{i}'] / amt,
                     bottom=(woe_df[f'Amount_{i}'] - woe_df[f'target_{i}']) / amt, width=w, zorder=0,
                     alpha=1 - i * 0.25, color='indianred',
                     label=f'Class 1{" (" + sample + ")" if len(samples_to_show) > 1 else ""}')
            if all_samples > 1 or i == 0:
                if all_samples <= 1:
                    shift = 0
                for x, y in woe_df[f'woe_{i}'].items():
                    ax2.annotate(str(y), xy=(woe_df['coord'][x] + shift, y),
                                 xytext=(30 if woe_df['down_trend'][x] else -30, 10), textcoords='offset pixels',
                                 color='black', ha='center',
                                 size=stat_size if (all_samples <= 1 or len(samples_to_show) < 3) else stat_size - 1)
                if i == 0:
                    if feature_woe2.categorical:
                        ax2.plot(woe_df['coord'] + shift, woe_df['woe_0'], 'bo', linewidth=2.0, zorder=4, label='WOE')
                    else:
                        for j, (g, group) in enumerate(woe_df.groupby('group1')):
                            group['line'] = group['group2'].apply(lambda x: not isinstance(x, str) and x >= 0).astype('int')
                            ax2.plot(group[group['line'] == 1]['coord'] + shift, group[group['line'] == 1]['woe_0'],
                                     'bo-', linewidth=2.0, zorder=4, label='WOE' if j == 0 else None)
                            ax2.plot(group[group['line'] == 0]['coord'] + shift, group[group['line'] == 0]['woe_0'],
                                     'bo', linewidth=2.0, zorder=4)
                else:
                    ax2.plot(woe_df['coord'] + shift, woe_df[f'woe_{i}'], 'bo', linewidth=2.0, zorder=4, alpha=1 - i * 0.25)
            if i == 0:
                if not feature_woe.ds.ci_analytic and feature_woe.ds.bootstrap_base is not None:
                    tmp = feature_woe.ds.bootstrap_base.merge(feature_woe2.ds.bootstrap_base[f2], left_index=True,
                                                              right_index=True)
                    feature_woe.ds.bootstrap_base['group'] = feature_cross.set_groups(data=tmp, f2=f2)
                woe_df = woe_df.merge(feature_woe.calc_woe_confint(df=to_calc, set_groups=False).reset_index()[['group', 'woe_lower', 'woe_upper']], on=['group'], how='left')
                ax2.errorbar(woe_df['coord'] + shift, (woe_df['woe_upper'] + woe_df['woe_lower'])/2, xerr=0, yerr=(woe_df['woe_upper'] - woe_df['woe_lower'])/2,
                             color='b', capsize=5, fmt=',', alpha=0.4)

        for stat, shift in {'Target rate': 15, 'Class 1 unique ids': 37, 'Unique ids': 59, 'Amount_0': 81}.items():
            ax_1.annotate(stat.replace('_0', ''), xy=(-0.5, 1), xycoords=('data', 'axes fraction'),
                          xytext=(0, shift), textcoords='offset pixels', color='black', ha='right', size=stat_size)
            for i, val in enumerate(woe_df[stat].values):
                ax_1.annotate(str(val), xy=(woe_df['coord'][i], 1), xycoords=('data', 'axes fraction'),
                              xytext=(0, shift), textcoords='offset pixels', color='black', ha='center', size=stat_size)
        ginis = ds.calc_gini(features=[f_WOE]).T[f_WOE].to_dict()
        ginis[f'{"Analytic" if ds.ci_analytic or ds.bootstrap_base is None else "Bootstrap"} {int(round((1 - ds.ci_alpha) * 100))}% CI'] = f"({ginis['CI_lower']}; {ginis['CI_upper']})"
        ax_1.annotate('Gini: %s' % (
            '; '.join([f'{name} {gini}' for name, gini in ginis.items() if name not in ['CI_lower', 'CI_upper']])),
                      xy=(0.5, 1), xycoords=('figure fraction', 'axes fraction'), xytext=(0, 115),
                      textcoords='offset pixels', color='blue', ha='center')

        ax_1.grid(False)
        ax_1.tick_params(axis='y', which='both', length=5)
        h1, l1 = ax_1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.27 if len(samples_to_show) > 1 else 1.15, 1),
                    fontsize=9 if len(samples_to_show) > 1 else 11)
        plt.suptitle(ds_aux.feature_titles[f] if f in ds_aux.feature_titles else f, fontsize=13, weight='bold')
        fig.tight_layout()
        ax_x.set_xlim(ax_1.get_xlim())
        ax_x.set_yticks([])
        ax_x.axis('off')
        ax_1.annotate(f2, xy=(1, 0), xytext=(10, -30), textcoords='offset pixels', ha='left', va='top',
                      xycoords='axes fraction')
        ax_x.annotate(feature_woe.feature, xy=(1, 1), xytext=(10, 10), xycoords='axes fraction',
                      textcoords='offset pixels', ha='left', va='top')
        for g, d in group_d.items():
            ax_x.annotate('', xy=(d[0], 1), xytext=(d[1], 1), xycoords=('data', 'axes fraction'),
                          arrowprops={'arrowstyle': '|-|', 'facecolor': 'red'}, annotation_clip=False)
            if g >= 0:
                label = '\n'.join(wrap(str(feature_woe.groups[g]), 100 // len(feature_woe.groups)))
                if not feature_woe.categorical:
                    label = label.replace(']', ')').replace('[-inf', '(-inf')
                label_sb = ',\n'.join([sb for sb, g2 in feature_woe.special_bins_groups.items() if g == g2])
                if label_sb:
                    label += ',\n' + label_sb
            else:
                label = ',\n'.join([sb for sb, g2 in feature_woe.special_bins_groups.items() if g == g2])
            ax_x.annotate(label, xy=((d[0] + d[1]) / 2, 1), xytext=(0, -10), xycoords=('data', 'axes fraction'),
                          textcoords='offset pixels', ha='center', va='top',
                          size=stat_size if len(label) < 100 else stat_size - 1 if len(label) < 200 else stat_size - 2)
        if folder is not None:
            fig.savefig(f'{folder}/{f_WOE}.png', bbox_inches="tight")
        if plot_flag:
            plt.show()
        return fig

    def plot_bins(self, features=None, folder=None, plot_flag=True, show_groups=False, verbose=False, all_samples=False,
                  stat_size=None):
        """
        Отрисовка биннинга
        :param features: список переменных для обработки. При None отрисоываются все активные переменные
        :param folder: папка, в которую должны быть сохранены рисунки. При None не сохраняются
        :param plot_flag: флаг для вывода рисунка
        :param show_groups: флаг для отображения номер групп на рисунке
        :param verbose: флаг для отображения счетчика обработанных рисунков
        :param all_samples: отрисовка бинов по всем сэмплам, может принимать значения:
                            0, False - строятся бины только на трэйне
                            1, True – строятся бины по всем сэмплам, таргет рейт указывается только для трэйна
                            >1  – строятся бины и таргет рейт указывается по всем сэмплам
        :param stat_size: размер шрифта для вывода статистики по бинам на графике

        :return: список из графиков [plt.figure]
        """
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        if features is None:
            features = list(self.feature_woes) + list(self.cross_features)
        features = [rem_suffix(f) for f in features]
        if plot_flag or self.ds_aux.logger.level > 20:
            verbose = False
        cross_features = [f for f in features if self.is_cross(f)]
        features = [f for f in features if f in self.feature_woes and self.feature_woes[f].is_active]
        if self.ds_aux.n_jobs > 1 and len(features) > self.ds_aux.n_jobs:
            with futures.ProcessPoolExecutor(max_workers=self.ds_aux.n_jobs) as pool:
                pool_iter = pool.map(partial(self.plot_bins_feature, ds_aux=self.ds_aux, folder=folder, plot_flag=plot_flag,
                                             show_groups=show_groups, all_samples=all_samples, stat_size=stat_size),
                                     [self.feature_woes[f] for f in features])
                pool_iter_cross = pool.map(partial(self.plot_bins_feature_cross, ds_aux=self.ds_aux, folder=folder,
                                             plot_flag=plot_flag, show_groups=show_groups, all_samples=all_samples, stat_size=stat_size),
                                     [(self.feature_crosses[cross_split(f)[0]], cross_split(f)[1]) for f in cross_features])
                figs = list(tqdm(pool_iter, total=len(features)) if verbose else pool_iter) + list(tqdm(pool_iter_cross, total=len(cross_features)) if verbose else pool_iter_cross)
            gc.collect()
        else:
            figs = [self.plot_bins_feature(feature_woe=self.feature_woes[f], ds_aux=self.ds_aux, folder=folder, plot_flag=plot_flag,
                                           show_groups=show_groups, all_samples=all_samples, stat_size=stat_size)
                    for f in (tqdm(features) if verbose else features)] + \
                   [self.plot_bins_feature_cross(params=(self.feature_crosses[cross_split(f)[0]], cross_split(f)[1]),
                                                 ds_aux=self.ds_aux, folder=folder, plot_flag=plot_flag, show_groups=show_groups,
                                                 all_samples=all_samples, stat_size=stat_size)
                    for f in (tqdm(cross_features) if verbose else cross_features)]
        plt.close('all')
        return figs

    def merge(self, feature, groups_list, plot_flag=True):
        """
        Объединение двух бинов
        :param feature: переменная
        :param groups_list: [group1, group2] - список из двух бинов для объединения
        :param plot_flag: флаг вывода графика после разделения
        """

        self.ds_aux.logger.info(f'{feature}: merging groups {groups_list}')
        feature = rem_suffix(feature)
        if feature in self.feature_woes:
            self.feature_woes[feature].merge(groups_list)
        elif self.is_cross(feature):
            f1, f2 = cross_split(feature)
            if isinstance(groups_list[0], list) and isinstance(groups_list[1], list):
                if groups_list[0][0] != groups_list[1][0]:
                    self.ds_aux.logger.error(f'For cross features first level groups must match! {groups_list[0][0]} != {groups_list[1][0]}')
                    return None
                self.feature_crosses[f1].cross[f2][groups_list[0][0]].merge([groups_list[0][1], groups_list[1][1]])
                self.feature_crosses[f1].fit([f2], new_groups=False)
            else:
                self.ds_aux.logger.error('For cross features groups_list must consist of two lists, for example [[0, 1], [0, 2]]')
                return None
        else:
            self.ds_aux.logger.error(f'{feature} does not exist! Skipping...')
            return None
        if plot_flag:
            self.plot_bins(features=[feature], folder=None, plot_flag=True, show_groups=True)

    def split(self, feature, group=None, to_add=None, min_bin_size=0.05, criterion='entropy', scoring='neg_log_loss',
              plot_flag=True):
        """
        Разделение выбранного бина на две части
        :param feature: переменная
        :param group: номер бина для разделения, начиная с 0
        :param to_add: для числовых - граница между бинами, для категориальных - список значений для выделения в новый бин
        :param min_bin_size: минимальное число (доля) наблюдений в каждом бине
        :param criterion: критерий расщепления. Варианты значений: 'entropy', 'gini'
        :param scoring: метрика для оптимизации
        :param plot_flag: флаг вывода графика после разделения
        """

        self.ds_aux.logger.info(f'{feature}: splitting group {group}')
        feature = rem_suffix(feature)
        if feature in self.feature_woes:
            self.feature_woes[feature].split(group=group, to_add=to_add, min_bin_size=min_bin_size, criterion=criterion, scoring=scoring)
        elif self.is_cross(feature):
            f1, f2 = cross_split(feature)
            if isinstance(group, list):
                self.feature_crosses[f1].cross[f2][group[0]].split(group=group[1], to_add=to_add, min_bin_size=min_bin_size, criterion=criterion, scoring=scoring)
                self.feature_crosses[f1].fit([f2], new_groups=False)
            else:
                self.ds_aux.logger.error('For cross features group must be a list!')
                return None
        else:
            self.ds_aux.logger.error(f'{feature} does not exist! Skipping...')
            return None
        if plot_flag:
            self.plot_bins(features=[feature], folder=None, plot_flag=True, show_groups=True)

    def show_history(self, feature):
        """
        Вывод истории биннинга одной переменной
        :param feature: название переменной
        """
        if feature not in self.feature_woes:
            self.ds_aux.logger.error(f'{feature} not in included features!')
        else:
            curr_iteration = self.feature_woes[feature].curr_iteration()
            for i, h in enumerate(self.feature_woes[feature].history):
                self.ds_aux.logger.info(f'Iteration {i}{" (current)" if i == curr_iteration else ""}')
                self.rollback(feature, iteration=i, plot_flag=True)
            if curr_iteration != len(self.feature_woes[feature].history) - 1:
                self.rollback(feature, iteration=curr_iteration, plot_flag=False)

    def rollback(self, feature, iteration=None, plot_flag=True):
        """
        Откат биннинга на одну из предыдущих итераций
        :param feature: пемеренная
        :param iteration: номер итерации. Возможны отрицательные значения: -1 - последняя итерация, -2 - предпоследняя и т.д.
        :param plot_flag: флаг вывода графика после отката
        """
        if feature not in self.feature_woes:
            self.ds_aux.logger.error(f'{feature} does not exist! Skipping...')
        else:
            self.feature_woes[feature].rollback(iteration=iteration)
            if plot_flag:
                self.plot_bins(features=[feature], folder=None, plot_flag=True, show_groups=True)

    def transform(self, ds, features=None, verbose=False):
        """
        Трансформация ДатаСэмпла
        :param ds: ДатаСэмпл
        :param features: список переменных для трансформации. При None берутся ds.features для которых есть активный биннинг
        :param verbose: флаг для вывода комментариев в процессе работы

        :return: трансформированный ДатаСэмпл
        """
        if ds.samples is None:
            return ds
        if features is None:
            features = set(ds.features) | self.cross_features
        features = [rem_suffix(f) for f in features]
        new_features = []
        skipped_features = []
        if verbose:
            self.ds_aux.logger.info('Transforming features...')
        for f in features:
            f_WOE = add_suffix(f)
            if f in self.feature_woes and self.feature_woes[f].is_active:
                for name in ds.samples:
                    ds.samples[name][f_WOE] = self.feature_woes[f].set_avg_woes(data=ds.samples[name][f])
                if ds.bootstrap_base is not None:
                    ds.bootstrap_base[f_WOE] = self.feature_woes[f].set_avg_woes(data=ds.bootstrap_base[f])
            elif self.is_cross(f):
                f1, f2 = cross_split(f)
                for name in ds.samples:
                    ds.samples[name][f_WOE] = self.feature_crosses[f1].set_avg_woes(data=ds.samples[name][[f1, f2]], f2=f2)
                if ds.bootstrap_base is not None:
                    ds.bootstrap_base[f_WOE] = self.feature_crosses[f1].set_avg_woes(data=ds.bootstrap_base[[f1, f2]], f2=f2)
            else:
                skipped_features.append(f)
                continue
            new_features.append(f_WOE)
        if skipped_features and verbose:
            self.ds_aux.logger.warning(f"Can't transform {len(skipped_features)} features: {skipped_features}")
        if new_features:
            ds.features = [f for f in ds.features if f.endswith('_WOE') and f not in new_features] + new_features
        return ds

    def export_scorecard(self, out=None, features=None, full=True, history=False):
        """
        Сохранение биннинга в файл
        :param out: название файла
        :param features: список переменных для сохранения. При None сохраняются все, имеющие активный биннинг
        :param full: если True, то добавляет в файл поля с доп статистикой по бинам
        :param history: если True, то сохраняется вся история биннингов. Дубли биннингов удаляются, текущий биннинг записывается последней итерацией

        :return: датафрейм со скоркартой
        """
        if features is None:
            features = [f for f in self.feature_woes if self.feature_woes[f].is_active] + list(self.cross_features)
        features = [rem_suffix(f) for f in features]
        dfs = []
        for f in features:
            if f in self.feature_woes and self.feature_woes[f].is_active:
                if history:
                    curr_iteration = self.feature_woes[f].curr_iteration()
                    processed = []
                    i2 = 0
                    for i, h in enumerate(self.feature_woes[f].history):
                        if h not in processed and h != self.feature_woes[f].history[curr_iteration]:
                            self.feature_woes[f].rollback(iteration=i)
                            dfs.append(self.feature_woes[f].export_scorecard(full=full, iteration=i2))
                            processed.append(h)
                            i2 += 1
                    self.feature_woes[f].rollback(iteration=curr_iteration)
                    dfs.append(self.feature_woes[f].export_scorecard(full=full, iteration=i2))
                else:
                    dfs.append(self.feature_woes[f].export_scorecard(full=full))
            elif self.is_cross(f):
                f1, f2 = cross_split(f)
                dfs.append(self.feature_crosses[f1].export_scorecard(features=[f2], full=full))
            else:
                self.ds_aux.logger.warning(f'No active binning for feature {f} found. Skipped.')
        if dfs:
            df = pd.concat(dfs).reset_index(drop=True)
        else:
            df = pd.DataFrame()
        if out is not None:
            df.to_excel(out, index=False)
        return df

    def import_scorecard(self, scorecard, features=None, verbose=False, fit_flag=False):
        """
        Импорт биннинга из файла
        :param scorecard: путь к эксель файлу или датафрейм с готовыми биннингами для импорта
        :param features: список переменных для импорта биннинга. При None биннинг импортируется для всех, которые есть в файле
        :param verbose: флаг для вывода комментариев в процессе работы
        :param fit_flag: при True - WOE всех бинов перерасчитываются на текущей выборке
                         при False - WOE берутся из скоркарты. Если поле 'woe' в скоркарте отсутствует, то автоматически ставится fit_flag=True
        """
        if isinstance(scorecard, str):
            if scorecard[-5:] == '.xlsx' or scorecard[-4:] == '.xls':
                scorecard = pd.read_excel(scorecard)
            else:
                self.ds_aux.logger.error('Unknown format of import file. Abort.')
                return None
        if features is None:
            features = list(scorecard['feature'].unique())
        replaced = []
        for feature in features:
            if not is_cross_name(feature):
                if feature in self.feature_woes:
                    replaced.append(feature)
                    self.feature_woes[feature].import_scorecard(scorecard[scorecard['feature'] == feature], fit_flag=fit_flag)
        scorecard = scorecard[(scorecard['feature'].str.startswith('cross_')) & (scorecard['feature'].str.contains('&'))].copy()
        if not scorecard.empty:
            scorecard['f1'] = scorecard['feature'].str[6:].str.split('&').str[0]
            scorecard['f2'] = scorecard['feature'].str[6:].str.split('&').str[1]
            scorecard['values1'] = scorecard['values'].str.split(' & ').str[0]
            scorecard['values2'] = scorecard['values'].str.split(' & ').str[1]
            if 'missing' in scorecard:
                scorecard['categorical_type1'] = scorecard['categorical_type'].astype('str').str[1:-1].str.split(', ').str[0].fillna('')
                scorecard['categorical_type2'] = scorecard['categorical_type'].astype('str').str[1:-1].str.split(', ').str[1].fillna('')
                scorecard['group1'] = np.where(scorecard['group'] != 'others',
                                               scorecard['group'].str[1:-1].str.split(', ').str[0].fillna(0), 'others')
                scorecard['group2'] = np.where(scorecard['group'] != 'others',
                                               scorecard['group'].str[1:-1].str.split(', ').str[1].fillna(0), 'others')
                scorecard['missing1'] = np.where(scorecard['missing'].isin([0, 1, '0', '1']), 0,
                                                 scorecard['missing'].astype('str').str[1:-1].str.split(', ').str[0].fillna('0')
                                                 )
                scorecard['missing2'] = np.where(scorecard['missing'].isin([0, 1, '0', '1']), scorecard['missing'],
                                                 scorecard['missing'].astype('str').str[1:-1].str.split(', ').str[1].fillna('0')
                                                 )
                scorecard[['group1', 'group2', 'missing1', 'missing2']] = scorecard[['group1', 'group2', 'missing1', 'missing2']].applymap(lambda x: pd.to_numeric(x, errors='ignore'))
            else:
                scorecard['special_bins']  = scorecard['special_bins'].replace('', ' & ')
                scorecard['special_bins1'] = scorecard['special_bins'].str.split(' & ').str[0]
                scorecard['special_bins2'] = scorecard['special_bins'].str.split(' & ').str[1]
                scorecard['group1'] = np.where(scorecard['group'] != -100,
                                               scorecard['group'].str[1:-1].str.split(', ').str[0].fillna(0), -100)
                scorecard['group2'] = np.where(scorecard['group'] != -100,
                                               scorecard['group'].str[1:-1].str.split(', ').str[1].fillna(0), -100)
                scorecard[['group1', 'group2']] = scorecard[['group1', 'group2']].applymap(lambda x: pd.to_numeric(x, errors='ignore'))
            for f1, scorecard_f1 in scorecard.groupby('f1'):
                fw = copy.deepcopy(self.feature_woes[f1]) if f1 in self.feature_woes else self.create_feature_woe(DataSamples(logger=None, verbose=False), f1)
                if 'missing' in scorecard:
                    sc = scorecard_f1[['f1', 'categorical_type1', 'group1', 'values1', 'woe', 'missing1']]\
                                    .set_axis(['feature', 'categorical_type', 'group', 'values', 'woe', 'missing'], axis=1)\
                                    .drop_duplicates(subset=['group'])
                else:
                    if 'woe' not in scorecard_f1.columns:
                        scorecard_f1['woe'] = 0
                    sc = scorecard_f1[['f1', 'group1', 'values1', 'woe', 'special_bins1']] \
                        .set_axis(['feature', 'group', 'values', 'woe', 'special_bins'], axis=1) \
                        .drop_duplicates(subset=['group'])
                fw.import_scorecard(scorecard=sc, verbose=False, fit_flag=False)
                self.feature_crosses[f1] = FeatureCross(fw)
                self.feature_crosses[f1].add_f2_features([copy.deepcopy(self.feature_woes[f2]) if f2 in self.feature_woes else self.create_feature_woe(DataSamples(logger=None, verbose=False), f2)
                                                          for f2 in scorecard_f1['f2'].unique()])
                self.feature_crosses[f1].import_scorecard(scorecard_f1, verbose=verbose, fit_flag=fit_flag)

            self.ds_aux.add_descriptions(features=self.cross_features)
        if replaced and verbose:
            self.ds_aux.logger.info(f'Replaced binning for {len(replaced)} features: {replaced}')

    def calc_woe_confint(self, features=None, all_samples=False):
        """
        Вычисляет доверительные интервалы WOE.
        Если заданы бутстреп сэмплы, то доверительные интервалы вычисляются бутсрепом, иначе  - аналитически
        :param features: список переменных для рачета. При None обрабатываются list(self.feature_woes)
        :param all_samples: посчитать WOE на всех остальных сэмплах

        :return: Датафрейм с полями
            n                  - кол-во наблюдений в бине на трейне
        	n1	               - кол-во наблюдений класса 1 в бине на трейне
        	n0                 - кол-во наблюдение класса 0 в бине на трейне
        	woe	               - WOE на трейне
        	correct_trend      - флаг сохранения тренда WOE на всех сэмплах относительно трейна (только при all_samples=True)
        	woe_{name}         - WOE посчитанное на сэмпле name (только при all_samples=True)
        	woe_lower          - нижняя граница доверительного интервала
        	woe_upper          - верхняя граница доверительного интервала
        	overlap            - флаг перекрытия доверительных интервалов с предыдущим бином
        	feature            - переменная
        """
        if features is None:
            features = list(self.feature_woes)
        return pd.concat([self.feature_woes[f].calc_woe_confint(all_samples=all_samples).assign(**{'feature': f})
                          for f in features if f in self.feature_woes and self.feature_woes[f].is_active])

    @property
    def cross_features(self):
        return {cross_name(f1, f2, add_woe=True) for f1 in self.feature_crosses for f2 in self.feature_crosses[f1].cross}

    def is_cross(self, f):
        f1, f2 = cross_split(f)
        return f2 and f1 in self.feature_crosses and f2 in self.feature_crosses[f1].cross
#----------------------------------------------------------------------------------------------------------


class FeatureWOE:
    '''
    ВОЕ-трансформация для отдельного фактора. Для каждого фактора должен быть создан свой экземпляр
    '''
    def __init__(self, ds, feature, round_digits=3, round_woe=2, rounding_migration_coef=0.001,
                 simple=False, n_folds=5, woe_adjust=0.5, alpha=0, alpha_range=None, alpha_scoring='neg_log_loss',
                 alpha_best_criterion='min', sb_process='min_or_separate', sb_min_part=0.05,
                 others_process='missing', opposite_sign_to_others=False):
        """
        :param ds: ДатаСэмпл, для которого будут рассчитываться биннинги
        :param feature: переменная

        ---Параметры для расчета WOE---
        :param round_digits: число знаков после запятой для округление границ бинов и WOE.
                 При округлении границ бинов происходит проверка на долю мигрирующих наблюдений. Если округление приедет к миграции большой доли наблюдений,
                 то round_digits увеличивается до тех пор, пока доля не упадет ниже rounding_migration_coef
        :param round_woe: число знаков после запятой для округление значение WOE
        :param rounding_migration_coef: максимально допустимая доля наблюдений для миграции между бинами при округлении
        :param simple: если True, то расчет WOE происходит на трэйн сэмпле, иначе берется среднее значение по фолдам
        :param n_folds: кол-во фолдов для расчета WOE при simple=False
        :param woe_adjust: корректировочный параметр для расчета EventRate_i в бине i:
                           EventRate_i = (n1_i + woe_adjust)/(n0_i + woe_adjust),
                           где n1_i - кол-во наблюдений с target=1 в бине i, n0_i - кол-во наблюдений с target=0 в бине i
        :param alpha: коэффициент регуляризации для расчета WOE. Если задан, то WOE для бина i вычисляется по формуле:
                      SmoothedWOE_i = log((n + alpha)*EventRate/(n*EventRate_i + alpha)),
                      где n - число наблюдений, EventRate = n1/n0,
                      n1 - общее кол-во наблюдений с target=1, n0 - общее кол-во наблюдений с target=0
        :param alpha_range: если alpha=None, то подбирается оптимальное значение alpha из диапазона alpha_range
        :param alpha_scoring: метрика, используемая для оптимизации alpha
        :param alpha_best_criterion: 'min' - минимизация метрики alpha_scoring, 'max' - максимизация метрики

        ---Обработка пустых значений---
        :param sb_process: способ обработки специальных бинов
                'separate' - всегда помещать в отдельный бин
                'min_or_separate' - если доля значений в бине меньше sb_min_part, то объединять с минимальным по WOE бином, иначе помещать в отдельный бин
                'max_or_separate' - если доля значений в бин меньше sb_min_part, то объединять с максимальным по WOE бином, иначе помещать в отдельный бин
                'nearest_or_separate' - если доля значений в бин меньше sb_min_part, то объединять с ближайшим по WOE бином, иначе помещать в отдельный бин
        :param sb_min_part: минимальная доля значений в бин для выделения отдельного бина при sb_process 'min_or_separate', 'max_or_separate' или 'nearest_or_separate'

        ---Обработка значений, отсутствующих в биннинге---
        :param others_process: Способ обработки:
                'missing': отсутствующим значениям присваивается WOE, соответствующий пустым значениям
                'min': отсутствующим значениям присваивается минимальный WOE
                'max': отсутствующим значениям присваивается максимальный WOE
                float: отсутствующим значениям присваивается заданный фиксированный WOE
        :param opposite_sign_to_others: В случае, когда непрерывная переменная на выборке для разработки имеет только один знак,
                то все значения с противоположным знаком относить к others
        """
        self.feature = feature
        self.round_digits = round_digits
        self.round_woe = round_woe
        self.rounding_migration_coef = rounding_migration_coef
        self.simple = simple
        self.n_folds = n_folds
        self.woe_adjust = woe_adjust
        self.alpha = alpha
        self.alpha_recalc = self.alpha is None
        self.alpha_range = alpha_range
        self.alpha_scoring = alpha_scoring
        self.alpha_best_criterion = alpha_best_criterion
        self.others_process = others_process
        self.opposite_sign_to_others = opposite_sign_to_others
        self.special_bins = ds.special_bins
        self.feature_bl = ds.features_bl[feature] if ds.features_bl is not None and feature in ds.features_bl else None
        self.special_bins_groups = {v: -i for i, v in enumerate(self.special_bins, start=1)}
        if sb_process.endswith('_or_separate'):
            self.special_bins_process = {}
            for sb, v in self.special_bins.items():
                if ds.samples and ds.samples[ds.train_name][self.feature].isin([v]).mean() < sb_min_part:
                    self.special_bins_process[sb] = sb_process.rsplit('_or_separate', maxsplit=1)[0]
                else:
                    self.special_bins_process[sb] = 'separate'
        else:
            self.special_bins_process = {sb: sb_process for sb in self.special_bins_groups}

        self.woe_thres_sb = 0
        self.others_woe = np.nan
        # for categorical - {group_number: [list of values]}, for ordered -{group_number: [left_bound, right_bound]}
        if feature in ds.cat_columns and ds.samples:
            self.categorical = True
        else:
            self.categorical = False
        self.woes = {0: 0} # group_number:woe
        self.history = []
        if ds.samples:
            samples = {name: ds.samples[name][[ds.target, feature]] for name in ds.samples}
        else:
            samples = None        
        self.ds = DataSamples(samples=samples, target=ds.target, features=[feature], cat_columns=[feature] if feature in ds.cat_columns else [],
                              n_jobs=1, random_state=ds.random_state, ci_analytic=ds.ci_analytic, ci_alpha=ds.ci_alpha,
                              result_folder=ds.result_folder, logger=ds.logger, verbose=False, special_bins=ds.special_bins)
        if ds.bootstrap_base is not None:
            self.ds.bootstrap_base = ds.bootstrap_base[[ds.target, feature]]
            self.ds.bootstrap = ds.bootstrap
        self.reset_groups()
        self.is_active = False

    def auto_fit(self, ds_aux=None, verbose=True, method='tree', max_n_bins=10, min_bin_size=0.05,
                 criterion='entropy', scoring='neg_log_loss', max_depth=None, solver='cp', divergence='iv',
                 WOEM_on=True, WOEM_woe_threshold=0.05, WOEM_sb=False,
                 SM_on=True, SM_target_threshold=5, SM_size_threshold=100,
                 BL_on=True, BL_allow_Vlogic_to_increase_gini=10, BL_spec_form=True,
                 G_on=True, G_gini_threshold=5, G_with_test=True, G_gini_decrease_threshold=0.4, G_min_gini_in_time=-100,
                 WOEO_on=True, WOEO_all_samples=True,
                 PSI_on=True, PSI_base_period_index=-1, PSI_threshold=0.5):
        '''
        Attempts to find suitable binning satisfying all conditions and passing all checks, adjusted by user parameters
        Values for groups amount are taken from groups_range in the specified order, feature is being fitted,
        then gini, business logic and WoE order checks are ran. If all checks are passed, then current binning stays,
        otherwise the feature is fitted with the next value of groups amount and checks are carried out again.

        Parameters
        -----------
        max_n_bins: list of integers that will be used as max_leaf_nodes value in FeatureWOE.fit method in the specified order
        verbose: if comments and graphs from checks should be printed

        Fit options:
        -----------
        scoring: a measure for cross-validation used used for optimal WOE splitting
        max_depth: the maximum of the DecisionTree depth used for optimal WOE splitting
        min_samples_leaf: the minimum of the DecisionTree leaf size used for optimal WOE splitting (one value)

        Size merge options (SM):
        -----------------------
        SM_on: flag to turn on/off WoE merge process
        SM_target_threshold: min number of targets for group to not be considered small
        SM_size_threshold: min number of observations for group to not be considered small

        WoE merge options (WOEM):
        -----------------------
        WOEM_on: flag to turn on/off merging by WoE process
        WOEM_woe_threshold: if woe difference between groups (neighboring groups for interval) is less then this threshold, then they are to be merged
        WOEM_sb: should woe difference with special bins also be checked

        Business logic check options (BL):
        ---------------------------------
        BL_on: flag to turn on/off business logic check
        BL_allow_Vlogic_to_increase_gini:

        WoE order check options (WOEO):
        ------------------------------
        WOEO_on: flag to turn on/off WoE order check
        WOEO_all_samples: проверять сохранение тренда WOE на всех сэмплах относительно трейна

        Gini check options (G):
        ----------------------
        G_on: flag to turn on/off Gini check
        G_gini_threshold: gini on train and validate/95% bootstrap should be greater then this
        G_gini_decrease_threshold: gini decrease from train to validate/95% bootstrap deviation from mean to mean should be greater then this
        G_gini_increase_restrict: if gini increase should also be restricted
        G_with_test: should features be also checked on test sample (and be excluded in case of failure)

        Returns
        ----------
        A boolean value: True, if successful binning was found, else False
        and dataframes with log, gini, business logic, WoE and ER information for export
        '''
        def checks_to_dfs(checks):
            return {'Log': pd.DataFrame(checks['Log'], columns=['feature', 'iteration', 'n_bins', 'result', 'reason']),
                    'Business Logic': pd.DataFrame(checks['Business Logic'], columns=['feature', 'iteration', 'trend', 'trend_type']),
                    'Gini': pd.DataFrame(checks['Gini']),
                    'WOE': pd.concat(checks['WOE']) if checks['WOE'] else pd.DataFrame(),
                    'PSI':  pd.concat(checks['PSI']) if checks['PSI'] else pd.DataFrame(),
                    'scorecard': pd.concat(checks['scorecard']) if checks['scorecard'] else pd.DataFrame(),
                   }
        f_WOE = add_suffix(self.feature)
        checks = {c: [] for c in ['Log', 'Business Logic', 'Gini', 'WOE', 'PSI', 'scorecard']}        
        n_bins_fact = max_n_bins + 1
        iteration = 0
        V_shape = {'gini': 0, 'fact_number_groups': 0, 'iteration': 0}
        if not verbose:
            self.ds.logger.setLevel(50)
        self.ds.logger.info(make_header(f'Auto binning for {self.feature}', 100))
        self.woe_thres_sb = WOEM_woe_threshold if WOEM_on and WOEM_sb else 0
        for n_bins in range(max_n_bins, 1, -1):
            if n_bins >= n_bins_fact:
                continue
            iteration +=1
            self.ds.logger.info(make_header(f'Searching for the best split into {n_bins} groups', 75))
            try:
                self.fit(new_groups=True, to_history=False, method=method, max_n_bins=n_bins, min_bin_size=min_bin_size,
                         monotonic=BL_on and (BL_allow_Vlogic_to_increase_gini == 100) or (V_shape['gini'] > 0),
                         criterion=criterion, scoring=scoring, max_depth=max_depth, solver=solver, divergence=divergence,
                         verbose=verbose)
            except Exception as e:
                self.reset_groups()
                self.ds.logger.error(f'Exception! Feature {self.feature}, fit: {e}')
                checks['Log'].append([self.feature, iteration, n_bins, 'Exception', f'Fit: {e}'])
                checks['Gini'].append({'feature': self.feature, 'iteration': iteration, self.ds.train_name: 0})
                continue
            n_bins_fact = len([x for x in self.woes if not pd.isnull(self.woes[x]) and x != -1 and not isinstance(x, str)])
            if SM_on:
                try:
                    self.merge_by_size(target_threshold=SM_target_threshold, size_threshold=SM_size_threshold)
                except Exception as e:
                    self.ds.logger.error(f'Exception! Feature {self.feature}, Merging by size: {e}')

            if WOEM_on:
                try:
                    self.merge_by_woe(woe_threshold=WOEM_woe_threshold)
                except Exception as e:
                    self.ds.logger.error(f'Exception! Feature {self.feature}, Merging by WOE: {e}')

            checks['scorecard'].append(self.export_scorecard(iteration=iteration))
            if len([x for x in self.woes if not pd.isnull(self.woes[x])]) == 1:
                self.ds.logger.info(f'After the attempt with {n_bins} groups only one group remains. Continue cycle...')
                checks['Log'].append([self.feature, iteration, n_bins_fact, 'Failure', 'After merging only one group remains'])
                checks['Gini'].append({'feature': self.feature, 'iteration': iteration, self.ds.train_name: self.ds.calc_gini(features=[f_WOE], samples=[self.ds.train_name]).values[0][0]})
                continue
            self.to_history(dupl=False)

            if BL_on:
                try:
                    BL_trend_type, BL_check = self.BusinessLogicChecker(allow_Vlogic_to_increase_gini=BL_allow_Vlogic_to_increase_gini if V_shape['gini'] == 0 else 100, spec_form=BL_spec_form)
                except Exception as e:
                    self.ds.logger.error(f'Exception! Feature {self.feature}, Business logic checks: {e}')
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Exception', f'Business logic checks: {e}'])
                    checks['Gini'].append({'feature': self.feature, 'iteration': iteration, self.ds.train_name: self.ds.calc_gini(features=[f_WOE], samples=[self.ds.train_name]).values[0][0]})
                    continue
                BL_check[1] = iteration
                checks['Business Logic'].append(BL_check)
                if BL_trend_type == 'no trend':
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Failure', 'Business logic check failed'])
                    checks['Gini'].append({'feature': self.feature, 'iteration': iteration, self.ds.train_name: self.ds.calc_gini(features=[f_WOE], samples=[self.ds.train_name]).values[0][0]})
                    continue

            if PSI_on or G_on:
                f_WOE = add_suffix(self.feature)
                for name in self.ds.samples:
                    if name != self.ds.train_name:
                        self.ds.samples[name][f_WOE] = self.set_avg_woes(data=self.ds.samples[name][self.feature])
                if self.ds.bootstrap_base is not None:
                    self.ds.bootstrap_base[f_WOE] = self.set_avg_woes(data=self.ds.bootstrap_base[self.feature])
                self.ds.add_aux(ds_aux)

            if PSI_on:
                try:
                    correct, psi = self.PsiChecker(base_period_index=PSI_base_period_index, threshold=PSI_threshold)
                except Exception as e:
                    self.ds.logger.error(f'Exception! Feature {self.feature}, PSI checks: {e}')
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Exception', f'PSI checks: {e}'])
                    continue
                psi.insert(loc=0, column='feature', value=self.feature)
                psi.insert(loc=1, column='iteration', value=iteration)
                checks['PSI'].append(psi)
                if not correct:
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Failure', 'PSI check failed'])
                    continue

            if G_on:
                try:
                    correct, gini_values = self.GiniChecker(gini_threshold=G_gini_threshold,
                                                            gini_decrease_threshold=G_gini_decrease_threshold,
                                                            with_test=G_with_test,
                                                            min_gini_in_time=G_min_gini_in_time)
                except Exception as e:
                    self.ds.logger.error(f'Exception! Feature {self.feature}, Gini checks: {e}')
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Exception', f'Gini checks: {e}'])
                    continue
                checks['Gini'].append({**{'feature': self.feature, 'iteration': iteration}, **gini_values})
                if not correct:
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Failure', 'Gini check failed'])
                    continue
            else:
                gini_values = self.ds.calc_gini(features=[f_WOE], samples=[self.ds.train_name]).T[f_WOE].to_dict()
                checks['Gini'].append({'feature': self.feature, 'iteration': iteration, self.ds.train_name: gini_values[self.ds.train_name]})

            if WOEO_on:
                try:
                    correct, woe_check = self.WOEOrderChecker(all_samples=WOEO_all_samples)
                except Exception as e:
                    self.ds.logger.error(f'Exception! Feature {self.feature}, WOE order checks: {e}')
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Exception', f'WOE order checks: {e}'])
                    continue
                woe_check.insert(loc=0, column='feature', value=self.feature)
                woe_check.insert(loc=1, column='iteration', value=iteration)
                checks['WOE'].append(woe_check)
                if not correct:
                    checks['Log'].append([self.feature, iteration, n_bins_fact, 'Failure', 'WOE order check failed'])
                    continue

            if BL_on and BL_trend_type == 'V-shape':
                checks['Log'].append([self.feature, iteration, n_bins_fact, 'Success', 'Business logic is V-shaped, keep iterating to compare with the monotonic trend'])
                if gini_values[self.ds.train_name] > V_shape['gini']:
                    V_shape = {'gini': gini_values[self.ds.train_name],
                               'fact_number_groups': n_bins_fact,
                               'iteration': iteration}
                self.ds.logger.info('Business logic is V-shaped, keep iterating to compare with the monotonic trend.')
                continue

            checks['Log'].append([self.feature, iteration, n_bins_fact, 'Success', ''])
            checks_dfs = checks_to_dfs(checks)
            if V_shape['gini'] > 0:
                gini_delta = V_shape['gini'] - gini_values[self.ds.train_name]
                if gini_delta > BL_allow_Vlogic_to_increase_gini:
                    for c in checks_dfs:
                        if c != 'Log':
                            check_V = checks_dfs[c][checks_dfs[c]['iteration'] == V_shape['iteration']]
                            check_V['iteration'] = iteration + 1
                        else:
                            check_V = pd.DataFrame([[self.feature, iteration + 1, V_shape['fact_number_groups'], 'Success', f'V-shaped binning is chosen, because it gives an increase gini by {round(gini_delta, 2)}']],
                                                   columns=['feature', 'iteration', 'n_bins', 'result', 'reason'])
                        checks_dfs[c] = pd.concat([checks_dfs[c], check_V])
                    self.ds.logger.info(f'Finally V-shaped binning with {V_shape["fact_number_groups"]} groups is chosen, because it gives an increase gini by {round(gini_delta, 2)}')
            self.ds.logger.setLevel(self.ds.logger_level)
            return True, checks_dfs
        else:
            checks_dfs = checks_to_dfs(checks)
            if V_shape['gini'] > 0:
                for c in checks_dfs:
                    if c != 'Log':
                        check_V = checks_dfs[c][checks_dfs[c]['iteration'] == V_shape['iteration']]
                        check_V['iteration'] = iteration + 1
                    else:
                        check_V = pd.DataFrame([[self.feature, iteration + 1, V_shape['fact_number_groups'], 'Success',
                                                'V-shaped binning is chosen, because other binnings do not pass checks']],
                                               columns=['feature', 'iteration', 'n_bins', 'result', 'reason'])
                    checks_dfs[c] = pd.concat([checks_dfs[c], check_V])
                self.ds.logger.info(f'Finally V-shaped binning with {V_shape["fact_number_groups"]} groups is chosen, because other binnings do not pass checks')
                self.ds.logger.setLevel(self.ds.logger_level)
                return True, checks_dfs
            else:
                if verbose:
                    self.ds.logger.info('After all attempts no suitable binning was found.')
                self.ds.logger.setLevel(self.ds.logger_level)
                return False, checks_dfs

    def fit(self, new_groups=True, woe_fit=True, to_history=True, method='tree', max_n_bins=10, min_bin_size=0.05,
            criterion='entropy', scoring='neg_log_loss', max_depth=5, monotonic=False, solver='cp', divergence='iv',
            verbose=False):
        '''
        Optimizes alpha, determines optimal split into WOE intervals and calculates WOE. After that, the class is ready to
        transform unknown datasets containing the feature.

        Parameters
        -----------
        new_groups: if True, the bounds of the feature woe groups are deleted; usefull for refitting a Feature_woe object
        scoring: a measure for cross-validation used used for optimal WOE splitting
        max_leaf_nodes: the maximum of the DecisionTree leaves number used for optimal WOE splitting
        max_depth: the maximum of the DecisionTree depth used for optimal WOE splitting
        min_samples_leaf: the minimum of the DecisionTree leaf size used for optimal WOE splitting (one value)
        '''
        self.is_active = True
        #optimal bounds calculation
        if new_groups:
            # For categorical features:
            # intermediate WOEs are calculated for each category => the features turns from categorical to numerical => for the further transformations the feature is considered numerical
            df = self.ds.samples[self.ds.train_name].copy()
            N = self.ds.samples[self.ds.train_name].shape[0]
            df = df[~df[self.feature].isin(self.special_bins.values())]
            if not df.empty:
                x_train = df[self.feature]
                y_train = df[self.ds.target]
                if method == 'tree':
                    if self.categorical:
                        x_train = self.categorical_to_interval(data=x_train)
                        pre_groups = self.groups.copy()
                    # GridSearchCV parameters
                    parameters_grid = {'max_leaf_nodes': [max_n_bins],
                                       'min_samples_leaf': [min_bin_size if min_bin_size > 1 else int(round(N * min_bin_size, 0))],
                                       'max_depth': [max_depth]}
                    # search for the best split
                    # decision tree of split with pre-processed missings
                    try:
                        final_tree = self.fit_gridsearch(x_train, y_train, parameters_grid, criterion, scoring)
                        final_tree.fit(pd.DataFrame(x_train), y_train)
                        boundaries = sorted(self.get_tree_splits(final_tree))
                    except:
                        boundaries = [-np.inf, np.inf]
                    self.groups = {g: boundaries[g: g + 2] for g in range(len(boundaries) - 1)}
                    self.set_round_groups(with_acceptable_migration=True)
                    if self.categorical:
                        self.categorical_recover(pre_groups)
                else:
                    if min_bin_size > 1:
                        min_bin_size = min_bin_size / N
                    min_bin_size = min(min_bin_size * N / df.shape[0], 0.5)
                    optb = optbinning.OptimalBinning(dtype='categorical' if self.categorical else 'numerical',
                                                     prebinning_method='cart', max_n_prebins=max_n_bins, min_prebin_size=min_bin_size,
                                                     monotonic_trend='auto_asc_desc' if monotonic else 'auto',
                                                     solver=solver, divergence=divergence)
                    optb.fit(x_train, y_train)
                    if self.categorical:
                        try:
                            self.groups = {g: sorted(list(val)) for g, val in enumerate(optb.splits)}
                        except:
                            self.groups = {g: list(val) for g, val in enumerate(optb.splits)}
                    else:
                        boundaries = [-np.inf] + list(optb.splits) + [np.inf]
                        self.groups = {g: boundaries[g: g + 2] for g in range(len(boundaries) - 1)}
                        self.set_round_groups(with_acceptable_migration=True)

                if self.opposite_sign_to_others and not self.categorical:
                    if df[df[self.feature] < 0].empty:
                        self.groups[0][0] = 0
                    if df[df[self.feature] >= 0].empty:
                        self.groups[max(self.groups.keys())][1] = 0
            else:
                self.groups = {}
            self.special_bins_groups = {v: -i for i, v in enumerate(self.special_bins, start=1)}
            self.groups.update({g: [self.special_bins[sb]] for sb, g in self.special_bins_groups.items()})
        self.groups = dict(sorted(self.groups.items()))
        self.set_groups(inplace=True)
        if woe_fit:
            self.woe_fit()
        self.ds.samples[self.ds.train_name][add_suffix(self.feature)] = self.ds.samples[self.ds.train_name]['group'].map(self.woes).astype('float32')
        if to_history:
            self.to_history()
        if verbose:
            self.print_woe()

    def reset_groups(self, fit=False):
        if self.categorical:
            tmp = self.ds.samples[self.ds.train_name][self.feature]
            values = tmp[~tmp.isin(self.special_bins.values())].unique().tolist()
            try:
                self.groups = {0: sorted(values)}
            except:
                self.groups = {0: values}
        else:
            self.groups = {0: [-np.inf, np.inf]}
        self.special_bins_groups = {sb: 0 for sb in self.special_bins}
        if fit and self.ds.samples:
            self.fit(new_groups=False)

    def woe_fit(self):
        '''
        Calculates WOE for FeatureWOE due to simple parameter
        '''
        # optimal alpha calculation
        if self.alpha_recalc:
            self.alpha = self.optimize_alpha()
            self.ds.logger.info(f'Optimal alpha: {self.alpha}')
        woes = self.calc_woes()
        woes_no_sb = {g: w for g, w in woes.items() if g >= 0}
        refit = False
        if woes_no_sb:
            for sb, g in self.special_bins_groups.items():
                if g < 0:
                    if g in woes and sb in self.special_bins_process:
                        if self.special_bins_process[sb] == 'min':
                            self.special_bins_groups[sb] = min(woes_no_sb, key=woes_no_sb.get)
                        elif self.special_bins_process[sb] == 'max':
                            self.special_bins_groups[sb] = max(woes_no_sb, key=woes_no_sb.get)
                        elif self.special_bins_process[sb] == 'nearest' or any([abs(w - woes[g]) < self.woe_thres_sb for w in woes_no_sb.values()]):
                            self.special_bins_groups[sb] = self.find_nearest(woes[g], woes_no_sb)
                        if self.special_bins_groups[sb] >= 0:
                            del self.groups[g]
                            refit = True
                    elif g in self.groups:
                        del self.groups[g]
        if refit:
            self.set_groups(inplace=True)
            woes = self.calc_woes()
        if self.others_process in ['missing_or_min', 'missing_or_max']:
            if 'nan' in self.special_bins_groups and self.special_bins_groups['nan'] in woes:
                self.others_woe = woes[self.special_bins_groups['nan']]
            elif self.others_process == 'missing_or_min':
                self.others_woe = min(woes.values())
            elif self.others_process == 'missing_or_max':
                self.others_woe = max(woes.values())
        elif self.others_process == 'min':
            self.others_woe = min(woes.values())
        elif self.others_process == 'max':
            self.others_woe = max(woes.values())
        else:
            self.others_woe = self.others_process
        self.woes = woes
        return woes

    def calc_woes(self, simple=False):
        if self.simple or simple:
            return self.calc_simple_woes()
        else:
            try:
                woe_folds = self.calc_woe_folds()[0]
                return {group: round(np.mean(np.array([woe_folds[group][fold] for fold in woe_folds[group]])), self.round_woe) for group in woe_folds}
            except ValueError:
                self.ds.logger.error('ValueError for WOE calculation. Please check n_folds parameter and group sizes. Turning to simple WOE calculation... Hope it works.')
                return self.calc_simple_woes()

    def calc_simple_woes(self, df=None):
        '''
        Simply calculates regularized WOE for each interval.
        Formula for regularized WOE of the i-th interval (value group) of feature:
        SnoothedWOE_i = log((n + alpha)*DefaultRate/(n*DefaultRate_i + alpha)),
        where n is number of samples, DefaultRate = N_bad/N_good, DefaultRate_i = (N_bad_i+woe_adjust)/(N_good_i+woe_adjust).

        Returns
        ----------
        woes: {group: woe}
        '''
        groups_stat = self.calc_groups_stat(df)
        alpha = self.alpha if self.alpha is not None else 0
        groups_stat['woe'] = (np.log(groups_stat['n1'].sum() / groups_stat['n0'].sum() * (alpha + groups_stat['n']) /
                                       (groups_stat['n'] * (groups_stat['n1'] + self.woe_adjust) / (
                                               groups_stat['n0'] + self.woe_adjust) + alpha))).round(self.round_woe)
        return groups_stat['woe'].to_dict()

    def woe_folds(self):
        '''
        Breaks the feature into folds for each value group (interval) and calculates regularized WOE for each fold.
        Formula for regularized WOE of the i-th interval (value group) of feature:
        SmoothedWOE_i = log((n + alpha)*DefaultRate/(n*DefaultRate_i + alpha)),
        where n is number of samples, DefaultRate = N_bad/N_good, DefaultRate_i = N_bad_i/N_good_i.

        WOE for folds is simular to cross-validation:
        1. Samples of an interval (value group) are divided into n_folds folds
        2. For each (n_folds - 1) intervals (value groups) SmoothedWOE is calculated
        3. For each fold its SmoothedWOE is the SmoothedWOE calculated on the other folds

        Example:
        Feature 'AGE', interval '25-35 years old', n_folds = 5.
        All the clients with AGE between 25 and 35 years are divided into 5 folds. For clients of the 1st fold SmoothedWOE value is SmoothedWOE calculated on 2-5 folds.

        Returns
        ----------
        woes: {left_bound : {fold_num : woe}} where left_bound is a lower bound of an interval, fold_num is a number of a fold (from 0 to n_folds-1), woe is a WOE value for the fold
        folds: {left_bound : {fold_num : fold_indexes}} where left_bound is a lower bound of an interval, fold_num is a number of a fold (from 0 to n_folds-1),  fold_indexes is indexes of samples in the fold
        '''
        groups_stat = self.calc_groups_stat()
        DR = groups_stat['n1'].sum() / groups_stat['n0'].sum()

        folds = {}
        # calculation of folds
        for group, data_i in self.ds.samples[self.ds.train_name].groupby('group'):
            folds[group] = {}
            if data_i.shape[0] > self.n_folds:
                skf = StratifiedKFold(self.n_folds)
                tmp = 0
                for train_index, test_index in skf.split(X = data_i, y = data_i[self.ds.target]):
                    # indexes addition
                    folds[group][tmp] = [data_i.iloc[train_index].index, data_i.iloc[test_index].index]
                    tmp = tmp + 1
            else:
                folds[group][0] = [data_i.index, data_i.index]
        # WOE for each fold and interval
        woes = {}

        # for each interval
        for group, data_i in self.ds.samples[self.ds.train_name].groupby('group'):
            woes[group] = {}
            #for each fold
            for fold in folds[group]:
                indexes_fold = folds[group][fold][0]
                data_fold = data_i.loc[indexes_fold]
                N_b_i = data_fold[self.ds.target].sum()
                N_g_i = data_fold.shape[0] - N_b_i
                n = N_g_i + N_b_i
                if n != 0:
                    DR_i = (N_b_i + self.woe_adjust)/(N_g_i + self.woe_adjust)
                    n = N_g_i + N_b_i
                    smoothed_woe_i = round(np.log(DR*(self.alpha + n)/(n*DR_i + self.alpha)), self.round_woe)
                    woes[group][fold] = smoothed_woe_i

        #removing bounds with no data cooresponding to them (in case of empty dictionary for folds)
        woes={x:woes[x] for x in woes if woes[x]!={}}
        return woes, folds
        # woes: {group : {fold_num : woe}}
        # folds: {group : {fold_num : fold_indexes}}

    def calc_woe_folds(self):
        '''
        Calculates WOE for each sample according to folds

        Returns
        ----------
        woes - a dictionary {group: {fold_number: woe}},
        result - a list of values transformed to woe by folds
        '''
        if self.alpha is None:
            self.alpha = 0
        woes, folds = self.woe_folds()
        # for each sample finds its interval (values group), fold and, consequently, WOE
        result = [woes[group][fold] for index, row in self.ds.samples[self.ds.train_name][[self.feature, self.ds.target]].iterrows()
                                    for group in folds
                                    for fold in folds[group]
                                    if index in folds[group][fold][1]]
        return woes, result

    def optimize_alpha(self):
        '''
        Optimal alpha selection for WoE-transformed data

        Returns
        --------
        optimal alpha

        '''
        if self.alpha_range is None:
            self.alpha_range = range(10, 100, 10)
        classifier = LogisticRegression(random_state=self.ds.random_state)
        scores = {}
        for alpha in self.alpha_range:
            self.alpha = alpha
            if self.simple:
                x = self.set_avg_woes(woes=self.calc_woes())
            else:
                x = self.calc_woe_folds()[1]
            scores[alpha] = np.mean(cross_val_score(classifier, x, self.ds.samples[self.ds.train_name][self.ds.target],
                                                    cv=5, scoring=self.alpha_scoring))
        if self.alpha_best_criterion == 'min':
            self.alpha = min(scores, key=scores.get)
        elif self.alpha_best_criterion == 'max':
            self.alpha = max(scores, key=scores.get)
        else:
            self.alpha = 0
        return self.alpha

    def get_condlist(self, data=None, groups_map=None, lang=None):
        condlist = []
        condgroups = []
        data_s = '' if lang is None else f"df['{self.feature}']" if lang == 'py' else self.feature
        for sb, v in self.special_bins.items():
            g = self.special_bins_groups[sb]
            if g in self.groups:
                if v != v:
                    cond = data.isnull() if lang is None else f"{data_s}.isnull()" if lang == 'py' else f"{data_s} is NULL"
                else:
                    cond = data == v if lang is None else f"{data_s} == {v}" if lang == 'py' else f"{data_s} = {v}"
                condlist.append(cond)
                condgroups.append(g if groups_map is None or g not in groups_map else groups_map[g])
        for g, v in self.groups.items():
            if g >= 0:
                if self.categorical:
                    cond = data.isin(v) if lang is None else f'{data_s}.isin({v})' if lang == 'py' else f'{data_s} IN ({str(v)[1:-1]})'
                else:
                    if isinstance(v, list):
                        if len(v) == 2:
                            if lang is None:
                                cond = (data >= v[0]) & (data < v[1])
                            else:
                                first = f'({data_s} >= {v[0]})' if v[0] != -np.inf else ''
                                second = f'({data_s} < {v[1]})' if v[1] != np.inf else ''
                                if not first and not second:
                                    first = f"~{data_s}.isnull()" if lang == 'py' else f"{data_s} is not NULL"
                                cond = f"{first}{'' if (not first or not second) else ' & ' if lang == 'py' else ' and '}{second}"
                        else:
                            cond = data == v[0]  if lang is None else f"{data_s} == {v[0]}" if lang == 'py' else f"{data_s} = {v[0]}"
                    else:
                        cond = data == v  if lang is None else f"{data_s} == {v}" if lang == 'py' else f"{data_s} = {v}"
                condlist.append(cond)
                condgroups.append(g if groups_map is None or g not in groups_map else groups_map[g])
        return condlist, condgroups

    def get_transform_func(self, start='', lang='py'):
        s = start
        i = 0
        for cond, woe in zip(*self.get_condlist(groups_map=self.woes, lang=lang)):
            if lang == 'py':
                s += f"np.where({cond}, {woe}, \n" + " " * (len(start) + 9 * (i + 1))
            else:
                s += f"        WHEN {cond} THEN {woe}\n"
            i += 1
        if lang == 'py':
            s += f'{self.others_woe}{")" * i}'.replace('nan', 'np.nan')
        else:
            s += f'        ELSE {self.others_woe}\n    END'
        return s

    def set_avg_woes(self, data=None, woes=None):
        '''
        Replaces all values of a feature to related WOE

        Parameters
        -----------
        data: a Series, containing initial values of feature
        woes: a dictionary with WOEs for groups

        Returns
        -----------
        a Series of WOE-transformed feature values
        '''
        if data is None:
            data = self.ds.samples[self.ds.train_name][self.feature]
        if woes is None:
            woes = self.woes
        return np.select(*self.get_condlist(data, groups_map=woes), self.others_woe).astype('float32')

    def calc_groups_stat(self, df=None):
        groups_stat = (df if df is not None else self.ds.samples[self.ds.train_name]).groupby('group')\
                      .agg(n=(self.ds.target, 'count'), n1=(self.ds.target, 'sum'))
        groups_stat['n0'] = groups_stat['n'] - groups_stat['n1']
        groups_stat['woe'] = groups_stat.index.map(self.woes)
        return groups_stat

    def set_groups(self, data=None, inplace=False):
        '''
        Replaces all values of a feature to related group

        Parameters
        -----------
        data: a Series, containing initial values of feature
        inplace: set groups in self.ds.samples[self.ds.train_name]

        Returns
        -----------
        a Series of corresponding groups for input values
        '''
        if data is None:
            data = self.ds.samples[self.ds.train_name][self.feature]
        result = np.select(*self.get_condlist(data), -100).astype('int8')
        if inplace:
            self.ds.samples[self.ds.train_name]['group'] = result
        return result

    def print_woe(self):
        '''
        Prints WOE parameters in a standard and convenient way
        '''
        if self.ds.logger.level <= 20:
            groups_stat = self.calc_groups_stat().reset_index()
            groups_stat[['n', 'n1']] = groups_stat[['n', 'n1']].round(0).astype('int')
            groups_stat['values'] = groups_stat['group'].map(self.groups).astype('str')
            for sb, g in self.special_bins_groups.items():
                if g < 0:
                    groups_stat.loc[groups_stat['group'] == g, 'values'] = f'{sb}'
                elif not self.categorical:
                    groups_stat.loc[groups_stat['group'] == g, 'values'] += f', {sb}'
            self.ds.logger.info(f"Current binning for {self.feature}:\n{groups_stat[['group', 'values', 'woe', 'n', 'n1']].to_string(index=False)}")

    def BusinessLogicChecker(self, allow_Vlogic_to_increase_gini=100, spec_form=False):
        '''
        Checks if the business logic condition is True

        Parameters
        -----------
        allow_Vlogic_to_increase_gini:

        Returns
        ----------
        Boolean - whether the check was successful and dataframe of check log
        '''
        self.ds.logger.info(make_header('Business logic checks', 50))
        if not self.categorical:
            woes_dropna = {self.groups[x][0]: self.woes[x] for x in self.woes if
                           isinstance(self.groups[x], list) and len(self.groups[x]) == 2}
            groups_info = pd.DataFrame(woes_dropna, index=['woe']).transpose().reset_index().rename({'index': 'lower'},
                                                                                                    axis=1)
            groups_info['upper'] = groups_info['lower'].shift(-1).fillna(np.inf)
            if groups_info.shape[0] == 1:
                self.ds.logger.info('Only one group with non-missing values is present. Skipping trend check...')
                trend = ''
                trend_type = 'one group'
            else:
                gi_check = groups_info.dropna(how='all', subset=['lower', 'upper'])[['woe', 'lower', 'upper']].copy()
                gi_check['risk_trend'] = np.sign((gi_check['woe'] - gi_check['woe'].shift(1)).dropna()).apply(lambda x: '+' if (x > 0) else '-' if (x < 0) else '0')
                trend = gi_check['risk_trend'].str.cat()
                if spec_form and self.feature_bl:
                    if re.fullmatch(''.join([x.replace('+', '\+') + '*' for x in self.feature_bl]), trend):
                        trend_type = 'specific form'
                        self.ds.logger.info(f'WOE trend is specific form')
                    else:
                        trend_type = 'no trend'
                        if re.fullmatch('-*|\+*', trend):
                            self.ds.logger.info(f'WOE trend is monotonic but not match specific form')
                        else:
                            self.ds.logger.info(f'WOE trend does not have right shape')
                else:
                    if re.fullmatch('-*|\+*', trend):
                        trend_type = 'monotonic'
                        self.ds.logger.info(f'WOE trend is monotonic')
                    elif allow_Vlogic_to_increase_gini < 100 and re.fullmatch('-*\+*|\+*-*', trend):
                        gini = self.ds.calc_gini(features=[add_suffix(self.feature)], samples=[self.ds.train_name]).at[add_suffix(self.feature), self.ds.train_name]
                        if gini < allow_Vlogic_to_increase_gini:
                            trend_type = 'no trend'
                            self.ds.logger.info(f'WOE trend is V-shaped, but gini is too low ({gini} < {allow_Vlogic_to_increase_gini})')
                        else:
                            trend_type = 'V-shape'
                            self.ds.logger.info(f'WOE trend is V-shaped')
                    else:
                        trend_type = 'no trend'
                        self.ds.logger.info(f'WOE trend does not have right shape')
            if self.ds.logger.level <= 20:
                fig = plt.figure(figsize=(5, 0.5))
                plt.plot(range(len(groups_info.dropna(how='all', subset=['lower', 'upper'])['lower'])),
                         groups_info.dropna(how='all', subset=['lower', 'upper'])['woe'], color='red')
                plt.xticks(range(len(groups_info.dropna(how='all', subset=['lower', 'upper'])['lower'])),
                           round(groups_info.dropna(how='all', subset=['lower', 'upper'])['lower'], 3), fontsize=8)
                plt.ylabel('WOE', fontsize=8)
                plt.yticks(fontsize=8)
                fig.autofmt_xdate()
                plt.show()
        else:
            trend = ''
            trend_type = 'categorical'
            self.ds.logger.info('Categorical feature. Skipping trend check...')
        self.ds.logger.info('...Passed!' if trend_type != 'no trend' else '...Failed!')
        return trend_type, [self.feature, 0, trend, trend_type]

    def PsiChecker(self, base_period_index=-1, threshold=0.5):
        self.ds.logger.info(make_header('PSI checks', 50))
        psi = self.ds.psi(features=[add_suffix(self.feature)], time_column=self.ds.time_column, base_period_index=base_period_index)
        psi_max = psi.max(axis=1).max()
        result = psi_max < threshold
        self.ds.logger.info(f'Max PSI is {psi_max} {"<" if result else ">"} {threshold}')
        self.ds.logger.info('...Passed!' if result else '...Failed!')
        return result, psi

    def GiniChecker(self, gini_threshold=5, gini_decrease_threshold=0.2, with_test=False, min_gini_in_time=-100):
        '''
        Checks if gini of the feature is significant and stable enough

        Parameters
        -----------
        gini_threshold: gini on train and validate/95% bootstrap should be greater then this
        gini_decrease_threshold: gini decrease from train to validate/95% bootstrap deviation from mean to mean should be greater then this
        with_test: add checking of gini values on test (calculation is always on)

        Returns
        ----------
        Boolean - whether the check was successful and dictionary of gini values for all available samples
        '''
        self.ds.logger.info(make_header('Gini checks', 50))
        gini_correct = True
        f_WOE = add_suffix(self.feature)
        samples = list(self.ds.samples) if with_test else [self.ds.train_name]
        gini_values = self.ds.calc_gini(features=[f_WOE], samples=samples).T[f_WOE].to_dict()
        for name in samples:
            gini = gini_values[name]
            if gini < gini_threshold:
                gini_correct = False
                self.ds.logger.info(f'Gini {name} is less then threshold {gini_threshold}')
                break
            if name != self.ds.train_name:
                if gini_decrease_threshold <= 1:
                    decrease = 1 - gini / gini_values[self.ds.train_name]
                else:
                    decrease = gini_values[self.ds.train_name] - gini
                if decrease > gini_decrease_threshold:
                    gini_correct = False
                    self.ds.logger.info(f'Gini change from {self.ds.train_name} to {name} is greater then threshold: {round(decrease, 2)} > {gini_decrease_threshold}')
                    break
        if gini_correct and self.ds.time_column and min_gini_in_time > -100:
            gini = self.ds.calc_gini_in_time(features=[f_WOE], samples=samples)[f_WOE][samples].min().min()
            if min_gini_in_time >= 0:
                if gini < min_gini_in_time:
                    gini_correct = False
                    self.ds.logger.info(f'Min Gini in time is below the threshold: {gini} < {min_gini_in_time}')
            else:
                if gini < gini_values[self.ds.train_name] + min_gini_in_time:
                    gini_correct = False
                    self.ds.logger.info(f'Min Gini in time is below then Gini {self.ds.train_name} {min_gini_in_time}: {gini} < {gini_values[self.ds.train_name] + min_gini_in_time}')
        self.ds.logger.info('\n' + pd.DataFrame(gini_values, index=['Gini']).round(2).to_string())
        self.ds.logger.info('...Passed!' if gini_correct else '...Failed!')
        return gini_correct, gini_values

    def calc_woe_confint(self, all_samples=False, df=None, set_groups=True):
        """
        Вычисляет доверительные интервалы WOE.
        :param all_samples: посчитать WOE на всех остальных сэмплах

        :return: Датафрейм с полями
            n                  - кол-во наблюдений в бине на трейне
        	n1	               - кол-во наблюдений класса 1 в бине на трейне
        	n0                 - кол-во наблюдение класса 0 в бине на трейне
        	woe	               - WOE на трейне
        	correct_trend      - флаг сохранения тренда WOE на всех сэмплах относительно трейна (только при all_samples=True)
        	woe_{name}         - WOE посчитанное на сэмпле name (только при all_samples=True)
        	woe_lower          - нижняя граница доверительного интервала
        	woe_upper          - верхняя граница доверительного интервала
        	overlap            - флаг перекрытия доверительных интервалов с предыдущим бином
        """
        df = self.calc_groups_stat(df=df)
        if all_samples:
            df['correct_trend'] = True
            for name, sample in self.ds.samples.items():
                if name != self.ds.train_name:
                    sample['group'] = self.set_groups(data=sample[self.feature])
                    df[f'woe_{name}'] = df.index.map(self.calc_simple_woes(df=sample))
                    df['correct_trend'] = (df['correct_trend']) & (np.sign(df['woe'] - df['woe'].shift(1)) == np.sign(df[f'woe_{name}'] - df[f'woe_{name}'].shift(1)))
            df.loc[df.index <= 0, 'correct_trend'] = np.nan
        if self.ds.ci_analytic or self.ds.bootstrap_base is None:
            pu, pl = zip(*df.apply(lambda row: proportion_confint(row['n1'], row['n'], alpha=self.ds.ci_alpha, method="wilson"), axis=1))
            alpha = self.alpha if self.alpha is not None else 0
            df['woe_lower'] = np.log(df['n1'].sum() / df['n0'].sum() * (alpha + df['n']) /
                                      (df['n'] * (pl * df['n'] + self.woe_adjust) / (
                                              (1 - np.array(pl)) * df['n'] + self.woe_adjust) + alpha))
            df['woe_upper'] = np.log(df['n1'].sum() / df['n0'].sum() * (alpha + df['n']) /
                                      (df['n'] * (pu * df['n'] + self.woe_adjust) / (
                                              (1 - np.array(pu)) * df['n'] + self.woe_adjust) + alpha))
        else:
            if set_groups:
                self.ds.bootstrap_base['group'] = self.set_groups(data=self.ds.bootstrap_base[self.feature])
            bootstrap_woes_l = [self.calc_simple_woes(df=self.ds.bootstrap_base.iloc[idx]) for idx in self.ds.bootstrap]
            bootstrap_woes = {g: np.array([w[g] for w in bootstrap_woes_l if g in w]) for g in self.ds.bootstrap_base['group'].unique()}
            bootstrap_mean = df.index.map({g: np.mean(v) for g, v in bootstrap_woes.items()})
            woe_std = df.index.map({g: np.std(v) for g, v in bootstrap_woes.items()})
            z = ndtri(1 - self.ds.ci_alpha / 2)
            df['woe_lower'] = bootstrap_mean - z * woe_std
            df['woe_upper'] = bootstrap_mean + z * woe_std
        df[['woe_lower', 'woe_upper']] = df[['woe_lower', 'woe_upper']].round(self.round_woe)
        try:
            df.loc[df.index > 0, 'overlap'] = (df['woe_lower'].shift().between(df['woe_lower'], df['woe_upper'])) | (df['woe_upper'].shift().between(df['woe_lower'], df['woe_upper']))
        except:
            pass
        return df

    def WOEOrderChecker(self, all_samples=False):
        """
        Проверка на ссохранение тренда WOE
        :param all_samples: проверять сохранение тренда WOE на всех сэмплах относительно трейна

        :return: кортеж (флаг прохождения проверки, ДатаФрейм с WOE)
        """
        self.ds.logger.info(make_header('WOE order checks', 50))
        woe_df = self.calc_woe_confint(all_samples=all_samples).reset_index()
        self.ds.logger.info('\n' + woe_df.to_string())
        result = True
        if all_samples:
            if woe_df[woe_df['correct_trend'] == False].empty:
                self.ds.logger.info('Trend holding on all samples')
            else:
                self.ds.logger.info('...Failed!\nTrend breaking on samples')
                result = False
        if result:
            if woe_df[woe_df['overlap'] == True].empty:
                self.ds.logger.info('Confidence intervals for groups do not overlap\n...Passed!')
            else:
                self.ds.logger.info(f'Confidence intervals overlap in group {woe_df[woe_df["overlap"] == True].index.to_list()}\n...Failed!')
                result = False
        return result, woe_df

    def set_round_groups(self, with_acceptable_migration=False):
        '''
        Rounds boundaries of groups. Checks if the rounding parameter is valid and extends it if necessary.
        Checks if groups do not collide and if groups' samples remain stable.

        Parameters
        -----------
        with_acceptable_migration: is it ok to allow migration between groups after rounding
        '''

        if not self.categorical:
            if with_acceptable_migration:
                # min interval between groups' boundariesf
                min_diff_b = min([1 if k < 0
                                  else (v[1] - self.ds.samples[self.ds.train_name][self.feature].min() if -np.inf in v
                                        else (self.ds.samples[self.ds.train_name][self.feature].max() - v[0] if np.inf in v
                                              else v[1] - v[0])) for (k, v) in self.groups.items()])
                change_rounds = False
                while min_diff_b < (.1)**self.round_digits:
                    self.round_digits += 1
                    change_rounds = True

                # checks changes in groups' volumes in case of rounding
                change_rounds_2 = True
                while change_rounds_2 and len(self.groups) > 1:
                    change_rounds_2 = False
                    for group in self.groups:
                        if group != -1:
                            left_from=round(self.groups[group][0], self.round_digits) if round(self.groups[group][0], self.round_digits) <= self.groups[group][0] else self.groups[group][0]
                            left_to=self.groups[group][0] if round(self.groups[group][0], self.round_digits) <= self.groups[group][0] else round(self.groups[group][0], self.round_digits)

                            right_from=round(self.groups[group][1], self.round_digits) if round(self.groups[group][1], self.round_digits) <= self.groups[group][1] else self.groups[group][1]
                            right_to=self.groups[group][1] if round(self.groups[group][1], self.round_digits) <= self.groups[group][1] else round(self.groups[group][1], self.round_digits)

                            migration = sum(self.ds.samples[self.ds.train_name][self.feature].apply(
                                                lambda x: (left_from <= x < left_to) or (right_from <= x < right_to)))
                            if migration/self.ds.samples[self.ds.train_name].shape[0] >= self.rounding_migration_coef and not change_rounds_2:
                                change_rounds_2 = True
                                self.round_digits += 1

                # rounding
                rounded_groups = {}
                for (k, v) in self.groups.items():
                    if k >= 0:
                        rounded_groups[k] = [round(v[0], self.round_digits), round(v[1], self.round_digits)]
                    else:
                        rounded_groups[k] = v
                if change_rounds:
                    self.ds.logger.info(f'The rounding parameter is too large, setting to {self.round_digits}')
                self.groups = rounded_groups
            else:
                exact_edges=[]
                rounded_edges=[]
                for (k, v) in self.groups.items():
                    if k>0:
                        exact_edges.append(v[0])
                        before_split = self.ds.samples[self.ds.train_name][self.feature][self.ds.samples[self.ds.train_name][self.feature]<v[0]].max()
                        rounded_split = (before_split+v[0])/2
                        precision = len(str(rounded_split).split('.')[1])
                        previous_rounded_split = None
                        while rounded_split > before_split and rounded_split<v[0] and previous_rounded_split!=rounded_split:
                            previous_rounded_split=rounded_split
                            final_precision = precision
                            precision-=1
                            rounded_split = int(rounded_split) if precision==0 \
                                                else int((rounded_split)*(10**precision))/(10**precision)
                        candidate_split=int((before_split+v[0])/2) if final_precision==0 \
                                                 else int(((before_split+v[0])/2)*(10**final_precision))/(10**final_precision)

                        if final_precision<len(str(v[0]).split('.')[1])-(len(str(v[0]).replace('.',''))-len(str(v[0]).replace('.','').rstrip('0'))) and \
                           (self.ds.samples[self.ds.train_name][self.feature]<v[0]).sum()==(self.ds.samples[self.ds.train_name][self.feature]<candidate_split).sum():
                            rounded_edges.append(candidate_split)
                        else:
                            rounded_edges.append(v[0])

                rounded_groups = {}
                for (k, v) in self.groups.items():
                    if k >= 0:
                        rounded_groups[k] = [-np.inf if v[0]==-np.inf else rounded_edges[exact_edges.index(v[0])],
                                              np.inf if v[1]==np.inf  else rounded_edges[exact_edges.index(v[1])]]
                    else:
                        rounded_groups[k] = v

                if rounded_edges!=exact_edges:
                    self.ds.logger.info(f'Rounding edges: {[exact_edges[i] for i in range(len(exact_edges)) if exact_edges[i]!=rounded_edges[i]]} to {[rounded_edges[i] for i in range(len(rounded_edges)) if exact_edges[i]!=rounded_edges[i]]}')
                    self.groups = rounded_groups

    def categorical_to_interval(self, data):
        '''
        Transforms categorical features into interval oned via WOE calculation for each category.
        '''
        # WOE for each category: optimal alpha - woe by folds - final woe calculation
        # turning categorical values into separate groups for pre-woe
        self.groups = {group: [value] for group, value in enumerate(data.dropna().unique())}
        self.set_groups(inplace=True)
        self.woes = self.calc_woes(simple=True)
        return self.set_avg_woes(data=data)

    def get_tree_splits(self, dtree):
        '''
        Returns list of thresholds of the deision tree.

        Parameters
        ---------------
        tree: DecisionTreeClassifier object

        Returns
        ---------------
        boundaries: list of thresholds
        '''
        children_left = dtree.tree_.children_left
        children_right = dtree.tree_.children_right
        threshold = dtree.tree_.threshold
        # boundaries of split
        boundaries = [-np.inf, np.inf]

        # determination of groups
        for i in range(dtree.tree_.node_count):
            if children_left[i] != children_right[i]:
                boundaries.append(threshold[i])

        return sorted(boundaries)

    def find_nearest(self, x, values):
        '''
        Finds in values the nearest one for x. If 'values' is dict then returns the nearest value and its key and
        if 'values' is list then returns the nearest value and its index

        Parameters
        --------------
        x: float, the value to process
        values: dict or list of possible nearest values
        '''
        if isinstance(values, dict):
            diff = {abs(v-x): k for (k, v) in values.items()}
            return diff[min(diff)]
        elif isinstance(values, list) or isinstance(values, np.ndarray):
            diff = abs(np.array(values) - x)
            return diff.index(min(diff)),

    def fit_gridsearch(self, x_train, y_train, parameters_grid, criterion, scoring):
        '''
        TECH

        Searches for the best decision tree for groups. Used in self.fit().

        Parameters
        ------------
        x_train: pd.Series to fit on
        y_train: pd.Series with targets for fit
        parameters_grid: parameters for gridsearch

        Returns
        ------------
        The best decision tree
        '''

        gridsearch = GridSearchCV(DecisionTreeClassifier(criterion=criterion, random_state=self.ds.random_state),
                                  parameters_grid, scoring=scoring, cv=7)
        gridsearch.fit(pd.DataFrame(x_train), y_train)
        return gridsearch.best_estimator_

    def categorical_recover(self, pre_groups):
        '''
        Recovers self.groups and self.woes for categorical non-predefined features because such features are pre-processed
        '''
        if self.categorical:
            final_groups = {}
            for group, vals in self.groups.items():
                if isinstance(vals, list):
                    for category, pre_woe in self.woes.items():
                        if category in pre_groups and not pd.isnull(pre_woe) and pre_woe >= vals[0] and pre_woe < vals[1]:
                            if group in final_groups:
                                final_groups[group] = final_groups[group] + pre_groups[category]
                            else:
                                final_groups[group] = pre_groups[category]
            try:
                self.groups = {g: sorted(v) for g, v in final_groups.items()}
            except:
                self.groups = final_groups
            self.woes = {group: self.woes[group] for group in self.groups if group in self.woes}

    def merge(self, groups_list):
        '''
        Merges two WOE intervals

        Parameters
        -----------
        groups_list: [group1, group2] - the groups to be merged
        '''

        # Checks for correctness of the groups to merge
        # existing groups
        for group in groups_list:
            if group not in self.groups:
                self.ds.logger.error(f'Group {group} is incorrect! Correct values are {list(self.groups)}')
                return None
        # only 2 groups per merge call
        if len(groups_list) != 2:
            self.ds.logger.error('Please enter 2 groups')
            return None

        # only neighbouring groups for ordered features
        if not self.categorical and groups_list[0] > 0 and groups_list[1] > 0 and abs(groups_list[0] - groups_list[1]) > 1:
            self.ds.logger.error('Please enter neighbouring groups')
            return None

        # merging groups in self.groups
        min_group = min(groups_list)
        max_group = max(groups_list)

        if min_group < 0:
            for sb in self.special_bins_groups:
                if self.special_bins_groups[sb] == min_group:
                    self.special_bins_groups[sb] = max_group
                    del self.groups[min_group]
        else:
            if self.categorical:
                self.groups[min_group] = self.groups[min_group] + self.groups[max_group]
            else:
                self.groups[min_group] = [min(self.groups[min_group][0], self.groups[max_group][0]), max(self.groups[min_group][1], self.groups[max_group][1])]

            for sb in self.special_bins_groups:
                if self.special_bins_groups[sb] == max_group:
                    self.special_bins_groups[sb] = min_group
                elif self.special_bins_groups[sb] > max_group:
                    self.special_bins_groups[sb] -= 1
            self.groups = {g if g < max_group + 1 else g - 1: v for g, v in self.groups.items()}
        self.fit(new_groups=False)
        gc.collect()

    def to_history(self, dupl=True):
        '''
        Writes current state of self to history.
        '''
        current = {'groups':self.groups.copy(),
                   'woes': self.woes.copy(),
                   'special_bins_groups': self.special_bins_groups,
                   'others_woe':self.others_woe}
        if dupl or current not in self.history:
            self.history.append(current)

    def curr_iteration(self):
        try:
            curr_iteration = len(self.history) - 1 - self.history[::-1].index({'groups': self.groups,
                                                                               'woes': self.woes,
                                                                               'special_bins_groups': self.special_bins_groups,
                                                                               'others_woe': self.others_woe})
        except:
            self.to_history()
            curr_iteration = len(self.history) - 1
        return curr_iteration

    def insert_subgroup(self, group, new_group):
        '''
        Makes new group and calculates WOE for categorical features

        Parameters
        --------------
        new_group: a user-defined new group of values, consists of values from the group to split and the other values of the group will be separated.
        Only for categorical features. Example: group = [1, 2, 4, 6, 9], new_group = [1, 2, 9] => the two new groups will be [1, 2, 9], [4, 6].
        For the same result we could set new_group parameter = [4, 6]
        '''
        self.groups[max(self.groups) + 1] = new_group
        self.groups[group] = [i for i in self.groups[group] if i not in new_group]
        self.fit(new_groups=False)

    def insert_new_bound(self, group, add_bound):
        '''
        Inserts new bound and calculates WOE for interval features

        Parameters
        ------------
        group: the group to insert the new bound into
        add_bound: the new bound to insert
        '''
        new_group_num = group + 1
        tmp_groups = copy.deepcopy(self.groups)
        tmp_woes = copy.deepcopy(self.woes)
        for g in sorted([g for g in self.groups if not isinstance(g, str)]):
            if g > new_group_num:
                tmp_groups[g] = self.groups[g - 1].copy()
                tmp_woes[g] = self.woes[g - 1]
        tmp_groups[max(self.groups) + 1] = self.groups[max(self.groups)].copy()
        tmp_woes[max(self.woes) + 1] = self.woes[max(self.woes)]
        for sb in self.special_bins_groups:
            if self.special_bins_groups[sb] > group:
                self.special_bins_groups[sb] += 1
        self.groups = tmp_groups
        self.woes = tmp_woes
        self.groups[new_group_num] = [add_bound, self.groups[group][1]]
        self.groups[group][1] = add_bound
        self.fit(new_groups=False)

    def split(self, group=None, to_add=None, min_bin_size=0.05, criterion='entropy', scoring='neg_log_loss'):
        '''
        Splits a WOE interval into two.

        Parameters
        -----------
        group: a group to split, integer
        to_add: in case of interval - a user-defined bound for the split (the intermediate bound of the interval), only for ordered features;
                in case of categorical -  a user-defined new group of values, consists of values from the group to split and the other values of the group will be separated. Only for categorical features.
                Example: group = [1, 2, 4, 6, 9], new_group = [1, 2, 9] => the two new groups will be [1, 2, 9], [4, 6]. For the same result we could set new_group parameter = [4, 6]
        '''
        if group is None:
            if self.categorical:
                for g in self.groups:
                    if to_add[0] in self.groups[g]:
                        group = g
                        break
            else:
                for g, v in self.groups.items():
                    if to_add >= v[0] and to_add < v[1]:
                        group = g
                        break

        if group is None:
            self.ds.logger.error('Invalid group!')
            return None

        sb_to_split = [sb for sb, g in self.special_bins_groups.items() if g == group]
        if sb_to_split:
            for sb in sb_to_split:
                self.special_bins_groups[sb] = min(self.groups) - 1
                self.groups[min(self.groups) - 1] = [self.special_bins[sb]]
                self.special_bins_process[sb] = 'separate'
                self.ds.logger.info(f'Split special bin {sb} from group {group}')
            self.fit(new_groups=False)
            return None

        if isinstance(to_add, int) or isinstance(to_add, float):
            if self.categorical:
                self.ds.logger.error('The feature is categorical so to_add should be a list of values for the new group')
                return None
            elif not (to_add >= self.groups[group][0] and to_add < self.groups[group][1]):
                self.ds.logger.error('New bound is out-of-range for the specified group')
                return None
            else:
                self.insert_new_bound(group, to_add)
        elif isinstance(to_add, list) or isinstance(to_add, np.ndarray):
            if not self.categorical:
                self.ds.logger.error('The feature is not categorical so to_add must be a float')
                return None
            else:
                for n in to_add:
                    if n not in self.groups[group]:
                        self.ds.logger.error('Invalid new_group!')
                        return None
                if self.groups[group] == to_add:
                    self.ds.logger.error(f'New_group contains all the values of group {group}')
                    return None
                else:
                    self.insert_subgroup(group, to_add)
                    gc.collect()

        # if no pre-defined bounds or groups
        else:
            self.ds.logger.info(f'Splitting started! Feature {self.feature} group: {group}')
            df = self.ds.samples[self.ds.train_name]
            if self.categorical:
                df = df[df[self.feature].isin(self.groups[group])].copy()
            else:
                df = df[(df[self.feature] >= self.groups[group][0]) & (df[self.feature] < self.groups[group][1])].copy()

            if df[self.ds.target].nunique() > 1:
                parameters_grid = {'min_samples_leaf': [min_bin_size if min_bin_size > 1 else int(round(self.ds.samples[self.ds.train_name].shape[0] * min_bin_size, 0))],
                                   'max_depth' : [1]}

                tmp_categorical = self.categorical
                self.categorical = ''
                # optimal split
                try:
                    final_tree = self.fit_gridsearch(df[self.feature], df[self.ds.target], parameters_grid, criterion, scoring)
                    final_tree.fit(df[[self.feature]], df[self.ds.target])
                except Exception:
                    self.ds.logger.error('Fitting with cross-validation failed! Possible cause: too few representatives of one of the target classes')
                    self.ds.logger.error('Try setting the bound yourself')
                    self.categorical = tmp_categorical
                    return None

                tree_splits = self.get_tree_splits(final_tree)
                self.categorical = tmp_categorical
                tree_splits = [x for x in tree_splits if x not in [-np.inf, np.inf]]
                if len(tree_splits) == 0:
                    self.ds.logger.error('No good binning found. Try setting the bound yourself')
                    return None
                else:
                    add_bound = round(sorted(tree_splits)[0], self.round_digits)
                    self.ds.logger.info(f'Additional bound {add_bound}')
                    #adding the new bound (woe for categorical) to groups
                    if self.categorical:
                        # find group by woe bound...
                        # since in self.ds.samples[self.ds.train_name][feature] we have woes calculated for each categorical value...
                        new_group = list(self.ds.samples[self.ds.train_name][(self.ds.samples[self.ds.train_name][self.feature] < add_bound) & (self.ds.samples[self.ds.train_name][self.feature].isin(self.groups[group]))][self.feature].drop_duplicates())
                        self.ds.logger.info(f'new_group: {new_group}')
                        self.insert_subgroup(group, new_group)
                    else:
                        self.insert_new_bound(group, add_bound)
                    gc.collect()
            else:
                self.ds.logger.error(f'All observations in the specified group have the same target value {df[self.ds.target].unique()[0]}')

    def merge_by_woe(self, woe_threshold=0.05):
        '''
        Merges all groups, close by WOE (for interval features only neighboring groups and missing group are checked)

        Parameters
        -----------
        woe_threshold: if woe difference between groups (neighboring groups for interval) is less then this threshold, then they are to be merged
        '''
        self.ds.logger.info(make_header('Merging by WOE', 50))
        for _ in range(len(self.groups.copy())):
            groups_stat = self.calc_groups_stat()
            groups_stat = groups_stat[groups_stat.index >= 0].reset_index()
            if self.categorical:
                groups_stat = groups_stat.sort_values(by=['woe'])
            groups_stat['woe_diff'] = (groups_stat['woe'] - groups_stat['woe'].shift(-1)).abs()
            groups_stat['merge_to'] = groups_stat['group'].shift(-1)
            groups_stat = groups_stat[(groups_stat['woe_diff'] < woe_threshold) & (groups_stat['woe_diff'] == groups_stat['woe_diff'].min())]

            if not groups_stat.empty:
                merge_groups = list(groups_stat[['group', 'merge_to']].astype('int').values[0])
                self.ds.logger.info(f'\nMerging of two groups close by WOE: {merge_groups}')
                self.merge(merge_groups)
            else:
                self.ds.logger.info('No groups with close WOE.')
                self.ds.logger.info('...Done')
                break

    def merge_by_size(self, target_threshold=5, size_threshold=100):
        '''
        Merges small groups (by target or size) to the closest by WoE (for interval features only neighboring groups and missing group are checked)

        Parameters
        -----------
        target_threshold: min number of targets for group to not be considered small
        size_threshold: min number of observations for group to not be considered small
        '''
        self.ds.logger.info(make_header('Merging by size', 50))
        for _ in range(len(self.groups.copy())):
            groups_stat = self.calc_groups_stat()
            groups_stat = groups_stat[groups_stat.index >= 0].reset_index()
            if self.categorical:
                groups_stat = groups_stat.sort_values(by=['woe'])
            if target_threshold < 1:
                groups_stat['n1'] = groups_stat['n1'] / groups_stat['n']
            groups_stat['small'] = (groups_stat['n1'] < target_threshold) | (groups_stat['n'] < size_threshold)
            groups_stat['woe_diff1'] = np.where(groups_stat['small'], (groups_stat['woe'] - groups_stat['woe'].shift(1)).abs(), np.nan)
            groups_stat['woe_diff2'] = np.where(groups_stat['small'], (groups_stat['woe'] - groups_stat['woe'].shift(-1)).abs(), np.nan)
            groups_stat['woe_diff'] = groups_stat[['woe_diff1', 'woe_diff2']].min(axis=1)
            groups_stat['merge_to'] = np.where(groups_stat['woe_diff'] != groups_stat['woe_diff'].min(), np.nan,
                                               np.where(groups_stat['woe_diff'] == groups_stat['woe_diff1'], groups_stat['group'].shift(1),
                                                        np.where(groups_stat['woe_diff'] == groups_stat['woe_diff2'], groups_stat['group'].shift(-1), np.nan)))
            groups_stat = groups_stat.dropna(subset=['merge_to'])

            if not groups_stat.empty:
                merge_groups = list(groups_stat[['group', 'merge_to']].astype('int').values[0])
                self.ds.logger.info(f'\nMerging small group {merge_groups[0]} (by target or size) to the closest by WoE group {merge_groups[1]}')
                self.merge(merge_groups)
            else:
                self.ds.logger.info('No small groups.')
                self.ds.logger.info('...Done')
                break

    def transform(self):
        '''
        Transforms a Data object according to WOE parameters fitted. Can be used only after .fit().

        Parameters
        ------------
        data: Data object to transform

        Returns
        ----------
        transformed Data object
        '''
        if self.ds.samples is not None:
            f_WOE = add_suffix(self.feature)
            for name, sample in self.ds.samples.items():
                self.ds.samples[name][f_WOE] = self.set_avg_woes(data=(self.ds.samples[name][self.feature]))
            if self.ds.bootstrap_base is not None:
                self.ds.bootstrap_base[f_WOE] = self.set_avg_woes(data=(self.ds.bootstrap_base[self.feature]))
            self.ds.features = [f_WOE]
        return self.ds

    def export_scorecard(self, iteration=None, full=True):
        '''
        Transforms self.groups, self.woes and self.special_bins_groups to dataframe.

        Returns
        ----------
        dataframe with binning information
        '''
        # searching for WOE for each interval of values
        scorecard = pd.DataFrame.from_dict(self.woes, orient='index').reset_index().set_axis(['group', 'woe'], axis=1)
        if self.ds is not None and self.ds.samples and 'group' in self.ds.samples[self.ds.train_name].columns:
            scorecard = scorecard.merge(self.calc_groups_stat().reset_index().drop(['woe'], axis=1), on='group', how='outer')
        else:
            scorecard['n'] = np.nan
            scorecard['n0'] = np.nan
            scorecard['n1'] = np.nan
        scorecard = scorecard[scorecard['group'] != -100]
        scorecard['values'] = scorecard['group'].map({g: str(v) if self.categorical else str(v).replace(", ", "; ") for g, v in self.groups.items()})
        scorecard = pd.concat([scorecard, pd.DataFrame({'group': -100, 'values': 'all others', 'woe': self.others_woe}, index=[0])])
        scorecard['feature'] = self.feature
        g_sb = {}
        for sb, group in self.special_bins_groups.items():
            try:
                g_sb[group].append(sb)
            except:
                g_sb[group] = [sb]
        scorecard['special_bins'] = scorecard['group'].map({**{g: str({k: v for k, v in self.special_bins.items() if k in l}) for g, l in g_sb.items()},  **{-100: 'others'}})
        scorecard = scorecard[['feature', 'group', 'values', 'woe', 'special_bins', 'n', 'n0', 'n1']]
        scorecard['target_rate'] = scorecard['n1'] / scorecard['n']
        scorecard['sample_part'] = scorecard['n'] / scorecard['n'].sum()
        scorecard['n0_part'] = scorecard['n0'] / scorecard['n0'].sum()
        scorecard['n1_part'] = scorecard['n1'] / scorecard['n1'].sum()
        scorecard['iteration'] = iteration if iteration is not None else len(self.history)
        scorecard[['target_rate', 'sample_part', 'n0_part', 'n1_part']] = scorecard[['target_rate', 'sample_part', 'n0_part', 'n1_part']].round(self.round_woe)
        if not full:
            scorecard.drop(['n', 'n0', 'n1', 'n0_part', 'n1_part', 'iteration'], axis=1, inplace=True)
        return scorecard

    def rollback(self, iteration=None):
        '''
        Rolls back the last operation.

        Parameters
        -----------
        iteration: number of groups iteration to return to (if None, then rollback to the previous iteration)
        '''
        if iteration is None:
            iteration = -1
        if self.history and iteration < len(self.history):
            self.groups = self.history[iteration]['groups']
            self.woes = self.history[iteration]['woes']
            self.special_bins_groups = self.history[iteration]['special_bins_groups']
            self.others_woe = self.history[iteration]['others_woe']
            self.fit(new_groups=False, woe_fit=False, to_history=False)
        else:
            self.ds.logger.error('Sorry, no changes detected or iteration found. Nothing to rollback')
            return None

    def check_values(self, s):
        '''
        Checks if any string element of the list contains comma. This method is used in parsing imported dataframes with groups, borders and woes.

        Returns
        --------
        False if there is a comma
        '''
        quotes = s.count("'")
        if quotes > 0:
            commas = s.count(',')
            if commas != (quotes/2)-1:
                return False
        return True

    def str_to_list(self, s):
        '''
        Parses ['values'] from a dataframe constructed by self.groups_to_dataframe().
        '''
        s = str(s)

        if pd.isnull(s) or s == '[nan]':
            return np.nan
        if self.check_values(s):
            v = (re.split('[\'|"]? *, *[\'|"]?', s[1:-1]))
            if v[0][0] in ("'", '"'):
                v[0]=v[0][1:]
            if v[-1][-1] in ("'", '"'):
                v[-1]=v[-1][:-1]
            return [float(x) if (x[-3:] == 'inf' or (min([y.isdigit() for y in x.split('.')]) and x.count('.') < 2)) else (x if x!='' else np.nan) for x in v]
        else:
            self.ds.logger.error(f'Error in string {s}! Delete commas from feature values!')
            return None

    def import_scorecard(self, scorecard, fit_flag=False, verbose=False):
        '''
        Sets self.groups, self.woe, self.categorical and self.special_bin_groups values from dataframe and calculates woe (by fit).

        Parameters
        ----------
        scorecard: a DataFrame with 'categorical_type', 'group', 'values' and 'missing' fields
        fit_flag: should woes be calculated or taken from input dataframe
        '''
        if 'woe' not in scorecard:
            fit_flag = True
        if 'iteration' not in scorecard:
            scorecard['iteration'] = 0
        if fit_flag:
            scorecard = scorecard[scorecard['iteration'] == scorecard['iteration'].max()]
        self.is_active = True
        for iter, woe_df in scorecard.groupby('iteration'):
            if 'missing' in woe_df:
                try:
                    self.others_woe = woe_df[woe_df['group'] == 'others']['woe'].iloc[0]
                except:
                    self.others_woe = np.nan
                woe_df = woe_df[~woe_df['group'].isin(['others'])]
                if 'categorical_type' in woe_df:
                    self.categorical = list(woe_df['categorical_type'].fillna(''))[0]
                elif 'categorical' in woe_df:
                    self.categorical = 'object' if list(woe_df['categorical'])[0] else ''
                if self.categorical == 'category':
                    self.categorical = 'object'
                values = list(woe_df['values'])
                to_convert = False
                for v in values:
                    if isinstance(v, str):
                        to_convert = True
                if self.categorical:
                    if to_convert:
                        values_corrected = []
                        for v in values:
                            if isinstance(v, str):
                                if v == 'nan':
                                    v_ = np.nan
                                else:
                                    if self.categorical == 'object':
                                        v_ = re.split('[\'|"] *, *[\'|"]', v[1:-1])
                                    else:
                                        v_ = re.split(' *, *', v[1:-1])
                                    if v_[0][0] in ("'", '"'):
                                        v_[0] = v_[0][1:]
                                    if v_[-1][-1] in ("'", '"'):
                                        v_[-1] = v_[-1][:-1]
                            else:
                                v_ = v
                            values_corrected.append(np.array(v_).astype(self.categorical).tolist())
                        values = values_corrected.copy()
                    self.categorical = True
                else:
                    self.categorical = False
                    if to_convert:
                        values_corrected = []
                        for v in values:
                            if str(v)[0] == '[':
                                values_corrected.append([float(x) if x != '' else np.nan for x in str(v)[1:-1].replace(" ", "").split(',')])
                            else:
                                values_corrected.append(float(v))
                        values = values_corrected.copy()
                woe_df['values'] = values
                if not self.special_bins:
                    self.special_bins = {g: woe_df[woe_df['group'] == g]['values'].values[0] for g in woe_df['group'].values[::-1] if isinstance(g, str) and isinstance(
                            woe_df[woe_df['group'] == g]['values'].values[0], list)}
                for i, g in enumerate(self.special_bins):
                    woe_df.loc[woe_df['group'] == g, 'group'] = -2 - i
                self.groups = woe_df.set_index('group')['values'].to_dict()
                try:
                    self.special_bins_groups['nan'] = woe_df[woe_df['missing'] == 1]['group'].values[0]
                    if self.special_bins_groups['nan'] == -1:
                        self.groups[-1] = np.nan
                except:
                    self.special_bins_groups['nan'] = -1
                self.special_bins_groups = {v: -i for i, v in enumerate(self.special_bins, start=1)}
            else:
                try:
                    self.others_woe = woe_df[woe_df['group'] == -100]['woe'].iloc[0]
                except:
                    self.others_woe = 0
                woe_df = woe_df[woe_df['group'] != -100]
                self.categorical = True
                self.groups = woe_df.set_index('group')['values'].astype('str').to_dict()
                for g, v in self.groups.items():
                    if '; ' in v and '"' not in v and "'" not in v:
                        self.groups[g] = eval(v.replace('inf', 'np.inf').replace(';', ','))
                        self.categorical = False
                    else:
                        self.groups[g] = eval(v.replace('nan', 'np.nan'))
                self.special_bins = {}
                self.special_bins_groups = {}
                for g, sb_dict in woe_df.dropna(subset=['special_bins']).set_index('group')['special_bins'].to_dict().items():
                    sb_dict = eval(sb_dict.replace(': nan', ': np.nan')) if sb_dict else {}
                    self.special_bins.update(sb_dict)
                    self.special_bins_groups.update({s: g for s in sb_dict})

            self.groups = dict(sorted(self.groups.items()))
            if fit_flag:
                self.fit(new_groups=False, woe_fit=True, to_history=False, verbose=verbose)
            else:
                self.woes = woe_df.set_index('group')['woe'].to_dict()
                if self.ds.samples is not None and iter == scorecard['iteration'].max():
                    self.fit(new_groups=False, woe_fit=False, to_history=False, verbose=verbose)
            self.to_history(dupl=False)

    def copy(self):
        return copy.deepcopy(self)

class FeatureCross:
    '''
    ВОЕ-трансформация для кросс-переменных. Для каждой переменной первого уровня должэен быть создан свой эксземпляр
    '''

    def __init__(self, feature_woe):
        """
        :param feature_woe: объект FeatureWOE переменной первого уровня
        """
        self.feature_woe = feature_woe.copy()
        self.cross = {}

    def add_f2_features(self, feature_woes):
        group_idx = {}
        if self.feature_woe.ds.samples is not None:
            for sample in self.feature_woe.ds.samples:
                if sample != self.feature_woe.ds.train_name:
                    df_s = self.feature_woe.ds.samples[sample].copy()
                    df_s['group'] = self.feature_woe.set_groups(data=df_s[self.feature_woe.feature])
                else:
                    df_s = self.feature_woe.ds.samples[sample]
                group_idx[sample] = {group: [df_s.index.get_loc(idx) for idx in df_sg.index] for group, df_sg in
                                     df_s.groupby('group')}
        if self.feature_woe.ds.bootstrap_base is not None:
            bs = self.feature_woe.ds.bootstrap_base
            bs['group'] = self.feature_woe.set_groups(data=bs[self.feature_woe.feature])
            bs_group_idx = {group: {bs.index.get_loc(idx) for idx in tmp.index} for group, tmp in bs.groupby('group')}
            bootstrap = {
                group: [list(set(idx) & bs_group_idx[group]) for idx in self.feature_woe.ds.bootstrap] for
                group in bs_group_idx}
        for fw in feature_woes:
            self.cross[fw.feature] = {}
            for group in self.feature_woe.groups:
                fw2 = fw.copy()
                fw2.history = []
                if fw.ds.samples is not None:
                    fw2.ds.samples = {name: sample.iloc[group_idx[name][group]] if group in group_idx[name] else sample[0:0] for name, sample in fw.ds.samples.items()}
                else:
                    fw2.ds.samples = None
                if fw.ds.bootstrap_base is not None and self.feature_woe.ds.bootstrap_base is not None and group in bootstrap:
                    fw2.ds.bootstrap_base = fw.ds.bootstrap_base
                    fw2.ds.bootstrap = bootstrap[group]
                self.cross[fw.feature][group] = fw2

    def auto_fit(self, ds_aux=None, cross_features=None, verbose=True, method='tree', max_n_bins=10, min_bin_size=0.05,
                 criterion='entropy', scoring='neg_log_loss', max_depth=None, solver='cp', divergence='iv',
                 WOEM_on=True, WOEM_woe_threshold=0.05, WOEM_sb=False,
                 SM_on=True, SM_target_threshold=5, SM_size_threshold=100,
                 BL_on=True, BL_allow_Vlogic_to_increase_gini=10, BL_spec_form=True,
                 G_on=True, G_gini_threshold=5, G_with_test=True, G_gini_decrease_threshold=0.4, G_min_gini_in_time=-100,
                 WOEO_on=True, WOEO_all_samples=True,
                 PSI_on=True, PSI_base_period_index=-1, PSI_threshold=0.5):
        check_list = ['Log', 'Business Logic', 'Gini', 'WOE', 'PSI', 'scorecard']
        auto_fit_parameters = {}
        for f in ['method', 'max_n_bins', 'min_bin_size', 'criterion', 'scoring', 'max_depth', 'solver', 'divergence',
                  'WOEM_on', 'WOEM_woe_threshold', 'WOEM_sb',
                  'SM_on', 'SM_target_threshold', 'SM_size_threshold',
                  'BL_on', 'BL_allow_Vlogic_to_increase_gini', 'BL_spec_form',
                  'G_on', 'G_gini_threshold', 'G_gini_decrease_threshold', 'G_with_test', 'G_min_gini_in_time',
                  'WOEO_on', 'WOEO_all_samples',
                  'PSI_on', 'PSI_base_period_index', 'PSI_threshold']:
            auto_fit_parameters[f] = eval(f)
        auto_fit_parameters['verbose'] = False
        min_bin_size = auto_fit_parameters['min_bin_size'] if auto_fit_parameters['min_bin_size'] > 1 else auto_fit_parameters['min_bin_size']*len(self.feature_woe.ds.samples[self.feature_woe.ds.train_name])
        if cross_features is None:
            cross_features = list(self.cross)
        if not cross_features:
            return {}, {}
        check_dfs = {f2: {} for f2 in cross_features}
        result = {f2: False for f2 in cross_features}
        for f2 in cross_features:
            res_group = {}
            for group, fw in self.cross[f2].items():
                try:
                    auto_fit_parameters['min_bin_size'] = min_bin_size / len(fw.ds.samples[fw.ds.train_name])
                    res_group[group], check_dfs[f2][group] = fw.auto_fit(ds_aux=ds_aux, **auto_fit_parameters)
                    for c in check_list[:-1]:
                        if not check_dfs[f2][group][c].empty:
                            check_dfs[f2][group][c] = check_dfs[f2][group][c][check_dfs[f2][group][c]['iteration'] == check_dfs[f2][group][c]['iteration'].max()].drop(['iteration'], axis=1)
                            check_dfs[f2][group][c].insert(loc=0, column='bin1', value=str(self.feature_woe.groups[group]) if self.feature_woe.categorical else str(self.feature_woe.groups[group]).replace(", ", "; "))
                            check_dfs[f2][group][c].insert(loc=0, column='group1', value=group)
                    if not res_group[group]:
                        fw.reset_groups(fit=True)
                except:
                    fw.reset_groups(fit=True)
            result[f2] = any(list(res_group.values())) and self.calc_gini(f2) > G_gini_threshold
            if result[f2]:
                self.calc_simple_woes(f2)
                if verbose:
                    self.print_woe(f2)
        try:
            check_dfs = {c: pd.concat([check_dfs[f2][group][c] for f2 in cross_features for group in check_dfs[f2]]) for c in check_list[:-1]}
            check_dfs['scorecard'] = self.export_scorecard()
        except:
            check_dfs = {c: pd.DataFrame() for c in check_list}
        return result, check_dfs

    def fit(self, cross_features=None, new_groups=True, method='tree', max_n_bins=10, min_bin_size=0.05,
            criterion='entropy', scoring='neg_log_loss', max_depth=None, monotonic=False, solver='cp', divergence='iv',
            verbose=False):
        if cross_features is None:
            cross_features = list(self.cross)
        for f2 in cross_features:
            for group in self.cross[f2]:
                self.cross[f2][group].fit(new_groups=new_groups, to_history=True, method=method, max_n_bins=max_n_bins,
                                          min_bin_size=min_bin_size, monotonic=monotonic, criterion=criterion, scoring=scoring,
                                          max_depth=max_depth, solver=solver, divergence=divergence, verbose=False)
            self.calc_simple_woes(f2)
            if verbose:
                self.print_woe(f2)

    def calc_simple_woes(self, f2):
        to_calc = self.get_data(f2, field='group')
        to_calc['group'] = '[' + to_calc['group_x'].astype('str') + ', ' + to_calc['group_y'].astype('str') + ']'
        for g, woe in self.feature_woe.calc_simple_woes(to_calc).items():
            g_l = json.loads(g)
            self.cross[f2][g_l[0]].woes[g_l[1]] = woe

    def calc_groups_stat(self, f2):
        to_calc = self.get_data(f2, field='group')
        to_calc['group'] = '[' + to_calc['group_x'].astype('str') + ', ' + to_calc['group_y'].astype('str') + ']'
        groups_stat = self.feature_woe.calc_groups_stat(to_calc)
        groups_stat['woe'] = groups_stat.index.map(self.get_woes(f2))
        return groups_stat

    def print_woe(self, f2):
        '''
        Prints WOE parameters in a standard and convenient way
        '''
        if self.feature_woe.ds.logger.level <= 20:
            groups_stat = self.calc_groups_stat(f2).reset_index()
            groups_stat[['n', 'n1']] = groups_stat[['n', 'n1']].round(0).astype('int')
            groups_stat['values'] = groups_stat['group'].map(self.get_2values(f2)).astype('str')
            self.feature_woe.ds.logger.info(f"Current binning for {cross_name(self.feature_woe.feature, f2)}:\n{groups_stat[['group', 'values', 'woe', 'n', 'n1']].to_string(index=False)}")

    def get_transform_func(self, f2, start='', lang='py'):
        s = start
        i = 0
        for cond1, g1 in zip(*self.feature_woe.get_condlist(lang=lang)):
            fw = self.cross[f2][g1]
            for cond2, woe in zip(*fw.get_condlist(groups_map=fw.woes, lang=lang)):
                if lang == 'py':
                    s += f"np.where(({cond1}) & ({cond2}), {woe}, \n" + " " * (len(start) + 9 * (i + 1))
                else:
                    s += f"        WHEN ({cond1}) and ({cond2}) THEN {woe}\n"
                i += 1
        if lang == 'py':
            s += f'{self.feature_woe.others_woe}{")" * i}'.replace('nan', 'np.nan')
        else:
            s += f'        ELSE {self.feature_woe.others_woe}\n    END'
        return s

    def set_groups(self, data, f2):
        '''
        Replaces all values of a feature to related WOE

        Parameters
        -----------
        data: DataFrame, containing initial values of features

        Returns
        -----------
        a Series of WOE-transformed feature values
        '''
        condlist = []
        cond_groups = []
        condlist_1, groups_1 = self.feature_woe.get_condlist(data[self.feature_woe.feature])
        for cond, g in zip(condlist_1, groups_1):
            condlist_2, groups_2 = self.cross[f2][g].get_condlist(data[f2])
            condlist += [c & cond for c in condlist_2]
            cond_groups += [str([g, g2]) for g2 in groups_2]
        return np.select(condlist, cond_groups, str([-100, -100]))

    def set_avg_woes(self, data, f2):
        '''
        Replaces all values of a feature to related WOE

        Parameters
        -----------
        data: DataFrame, containing initial values of features

        Returns
        -----------
        a Series of WOE-transformed feature values
        '''
        condlist = []
        cond_woes = []
        condlist_1, groups_1 = self.feature_woe.get_condlist(data[self.feature_woe.feature])
        for cond, g in zip(condlist_1, groups_1):
            condlist_2, groups_2 = self.cross[f2][g].get_condlist(data[f2])
            condlist += [c & cond for c in condlist_2]
            cond_woes += [self.cross[f2][g].woes[g2] for g2 in groups_2]
        return np.select(condlist, cond_woes, self.feature_woe.others_woe).astype('float32')

    def get_data(self, f2, sample=None, field=None):
        if sample is None:
            sample = self.feature_woe.ds.train_name
        if field is None:
            field = f2
        return self.feature_woe.ds.samples[sample].merge(pd.concat([fw.ds.samples[sample][field] for fw in self.cross[f2].values()]), left_index=True, right_index=True, how='left')

    def calc_gini(self, f2, sample=None):
        fw = self.feature_woe
        if sample is None:
            sample = fw.ds.train_name
        f_WOE = cross_name(fw.feature, f2)
        try:
            data = self.get_data(f2, sample=sample)
            fw.ds.samples[sample][f_WOE] = self.set_avg_woes(data, f2)
            return fw.ds.calc_gini(samples=[sample], features=[f_WOE]).at[f_WOE, sample]
        except:
            return 0

    def transform(self, features=None):
        '''
        Transforms a Data object according to WOE parameters fitted. Can be used only after .fit().

        Parameters
        ------------
        data: Data object to transform

        Returns
        ----------
        transformed Data object
        '''
        if features is None:
            features = list(self.cross)
        if self.feature_woe.ds.samples is not None:
            new_features = []
            for f2 in features:
                f_WOE = add_suffix(cross_name(self.feature_woe.feature, f2))
                for name, sample in self.feature_woe.ds.samples.items():
                    data = self.get_data(f2, sample=name)
                    self.feature_woe.ds.samples[name][f_WOE] = self.set_avg_woes(data, f2)
                if self.feature_woe.ds.bootstrap_base is not None:
                    data = self.feature_woe.ds.bootstrap_base.merge(list(self.cross[f2].values())[0].ds.bootstrap_base[f2], how='left', left_index=True, right_index=True)
                    self.feature_woe.ds.bootstrap_base[f_WOE] = self.set_avg_woes(data, f2)
                new_features.append(f_WOE)
            self.feature_woe.ds.features = new_features
        return self.feature_woe.ds

    def export_scorecard(self, features=None, full=True):
        '''
        Returns
        ----------
        dataframe with binning information
        '''
        if features is None:
            features = self.cross
        features = [f for f in features if f in self.cross]
        dfs = []
        for f2 in features:
            feature = f'cross_{self.feature_woe.feature}&{f2}'
            dfs_group = []
            sc1 = self.feature_woe.export_scorecard(full=False)[['group', 'values', 'special_bins']].set_axis(['group1', 'values1', 'special_bins1'], axis=1)
            for group in self.cross[f2]:
                df = self.cross[f2][group].export_scorecard(full=True)
                df = df[df['group'] != -100]
                df['group1'] = group
                df = df.merge(sc1, on=['group1'], how='left')
                df['group'] = '[' + df['group1'].astype('str') + ', ' + df['group'].astype('str') + ']'
                df['values'] = df['values1'].astype('str') + ' & ' + df['values'].astype('str')
                df['special_bins'] = (df['special_bins1'].fillna('').astype('str') + ' & ' + df['special_bins'].fillna('').astype('str')).replace(' & ', '')
                dfs_group.append(df.drop(['group1', 'values1', 'special_bins1'], axis=1))
            df = pd.concat(dfs_group).reset_index(drop=True)
            df['feature'] = feature
            df['sample_part'] = df['n'] / df['n'].sum()
            df['n0_part'] = df['n0'] / df['n0'].sum()
            df['n1_part'] = df['n1'] / df['n1'].sum()
            df[['sample_part', 'n0_part', 'n1_part']] = df[['sample_part', 'n0_part', 'n1_part']].round(self.feature_woe.round_woe)
            if not full:
                df.drop(['n', 'n0', 'n1', 'n0_part', 'n1_part', 'iteration'], axis=1, inplace=True)
            dfs.append(df)
            dfs.append(pd.DataFrame({'feature': feature, 'group': -100, 'values': 'all others', 'woe': self.feature_woe.others_woe, 'special_bins': 'others'}, index=[0]))
        if dfs:
            scorecard = pd.concat(dfs).reset_index(drop=True)
        else:
            scorecard = pd.DataFrame()
        return scorecard

    def import_scorecard(self, scorecard, fit_flag=False, verbose=False):
        '''
        Sets self.groups, self.woes, self.categorical values from dataframe and calculates woe (by fit).

        Parameters
        ----------
        scorecard: a DataFrame with 'categorical_type', 'group', 'values' and 'missing' fields
         fit_flag: should woes be calculated or taken from input dataframe
        '''
        for f2, scorecard_f2 in scorecard.groupby('f2'):
            if scorecard_f2[scorecard_f2['group1'] != -100].empty:
                continue
            for group, scorecard_f2_group in scorecard_f2.groupby('group1'):
                if group not in ['others', -100]:
                    if 'missing2' in scorecard_f2_group:
                        sc = scorecard_f2_group[['f2', 'categorical_type2', 'group2', 'values2', 'woe', 'missing2']] \
                                      .set_axis(['feature', 'categorical_type', 'group', 'values', 'woe', 'missing'], axis=1)
                    else:
                        if 'woe' in scorecard_f2_group.columns:
                            sc = scorecard_f2_group[['f2', 'group2', 'values2', 'woe', 'special_bins2']] \
                                .set_axis(['feature', 'group', 'values', 'woe', 'special_bins'], axis=1)
                        else:
                            sc = scorecard_f2_group[['f2', 'group2', 'values2', 'special_bins2']] \
                                .set_axis(['feature', 'group', 'values', 'special_bins'], axis=1)
                            fit_flag = True
                    self.cross[f2][group].import_scorecard(scorecard=sc, fit_flag=fit_flag)
                else:
                    if 'woe' in scorecard_f2_group.columns:
                        self.feature_woe.others_woe = scorecard_f2_group['woe'].values[0]
                    else:
                        self.feature_woe.others_woe = 0
            if fit_flag:
                self.calc_simple_woes(f2)
            if verbose:
                self.print_woe(f2)

    def get_woes(self, f2):
        return {str([g, g2]): woe for g, fw in self.cross[f2].items() for g2, woe in fw.woes.items()}

    def get_values(self, f2):
        return {str([g, g2]): value for g, fw in self.cross[f2].items() for g2, value in fw.groups.items()}

    def get_2values(self, f2):
        return {str([g, g2]): str(self.feature_woe.groups[g]) + ' & ' + str(value) for g, fw in self.cross[f2].items() for g2, value in fw.groups.items()}