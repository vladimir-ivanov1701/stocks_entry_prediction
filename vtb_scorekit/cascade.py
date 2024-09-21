# -*- coding: utf-8 -*-

from .data import DataSamples
from .model import LogisticRegressionModel, NpEncoder
from ._utils import make_header
import pandas as pd
import cloudpickle
import base64
import inspect
import matplotlib.pyplot as plt
import warnings
import json
import gc

warnings.simplefilter('ignore')
plt.rc('font', family='Verdana', size=12)
try:
    plt.style.use([s for s in plt.style.available if 'darkgrid' in s][0])
except:
    pass
pd.set_option('display.precision', 3)
gc.enable()


class Cascade:
    """
    Класс для работы с каскадом моделей
    """

    def __init__(self, models=None, integral=None, ds=None, name=''):
        """
        :param models: список моделей в каскаде. Элементами списка могут быть объекты класса Cascade, LogisticRegressionModel и названия полей отдельных скоров
        :param integral: функция, вычисляющая интегральный скор каскада по списку скоров входящих в него моделей. При None интегральный скор вычисляется логрегом
                         Пример интегральной функции

                         def calc_lgd(scores): # scores - список [score_0, score_1, ...], где
                                               #          score_i - скор модели i каскада (self.models[i]), имеет тип pd.Series
                             pd0 = 1 / (1 + np.exp(-scores[0]))
                             pd1 = 1 / (1 + np.exp(-scores[1]))
                             return pd1 + 0.361*(1 - pd0 - pd1)
        :param ds: ДатаСэмпл, на котором будет рассчитываться интегральный скор
        :param name: название каскада
        """
        self.models = models if models is not None else []
        self.integral = integral if integral is not None else LogisticRegressionModel(name='Integral')
        self.ds = ds
        self.name = name
        for i, model in enumerate(self.models):
            if isinstance(model, LogisticRegressionModel) and not model.name:
                model.name = f'Model {i}'
            if isinstance(model, Cascade) and not model.name:
                model.name = f'Cascade {i}'
    
    def print_log(self, s):
        if self.ds is None or self.ds.logger is None:
            print(s)
        else:
            self.ds.logger.info(s)
            
    def save_model(self, file_name='model.json', pickle_protocol=4):
        """
        Сохранение каскада в файл
        :param file_name: название json файла для сохранения каскада
        """
        cascade = {'name': self.name, 'models': []}
        for model in self.models:
            if isinstance(model, (LogisticRegressionModel, Cascade)):
                cascade['models'].append(model.save_model(file_name=None, pickle_protocol=pickle_protocol))
            else:
                cascade['models'].append(model)
        if isinstance(self.integral, LogisticRegressionModel):
            cascade['integral'] = self.integral.save_model(file_name=None, pickle_protocol=pickle_protocol)
        else:
            cascade['integral'] = base64.b64encode(cloudpickle.dumps(self.integral, protocol=pickle_protocol)).decode()
        if file_name is not None:
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(cascade, file, ensure_ascii=False, indent=4, cls=NpEncoder)
            self.print_log(f'The cascade was successfully saved to file {file_name}')
        else:
            return cascade

    def load_model(self, file_name='model.json'):
        """
        Загрузка каскада из файла
        :param file_name: название json файла для загрузки каскада
        """
        if isinstance(file_name, str):
            cascade = json.load(open(file_name, 'rt', encoding='utf-8'))
            self.print_log(f'The cascade was loaded from file {file_name}')
        elif isinstance(file_name, dict):
            cascade = file_name
        else:
            self.print_log('file_name type must be str or dict!')
            return None
        if 'name' in cascade:
            self.name = cascade['name']
        self.models = []
        if 'models' in cascade:
            for model in cascade['models']:
                if isinstance(model, dict):
                    if 'models' in model:
                        cascade_models = Cascade()
                        cascade_models.load_model(model)
                        self.models.append(cascade_models)
                    else:
                        logreg = LogisticRegressionModel()
                        logreg.load_model(model, verbose=False)
                        self.models.append(logreg)
                else:
                    self.models.append(model)
        if 'integral' in cascade:
            if isinstance(cascade['integral'], dict):
                self.integral = LogisticRegressionModel()
                self.integral.load_model(cascade['integral'], verbose=False)
            else:
                self.integral = cloudpickle.loads(base64.b64decode(cascade['integral']))

    def auto_logreg(self, ds=None, validate=False, out='auto_cascade.xlsx', save_model='auto_cascade.json'):
        """
        Построение модели в автоматическом режиме с минимальным набором параметров
        :param ds: ДатаСэмпл, на котором будет рассчитываться интегральный скор. При None берется self.ds
        :param validate: флаг для выполнения валидацонных тестов
        :param out: либо строка с названием эксель файла, либо объект pd.ExcelWriter для сохранения отчета
        :param save_model: название json файла для сохранения каскада
        """
        if ds is not None:
            self.ds = ds
        if isinstance(out, str):
            writer = pd.ExcelWriter(self.ds.result_folder + out, engine='xlsxwriter')
        else:
            writer = out
        for model in self.models:
            if isinstance(model, LogisticRegressionModel):
                self.ds.logger.info(make_header(model.name, 175))
                config = {'report': {'metrics': ['wald', 'ks', 'vif', 'iv', 'ontime']}}
                model.auto_logreg(out=writer if model.coefs else f'{model.name + "_" if model.name else ""}mfa.xlsx',
                                  save_model=None, config=config)
            if isinstance(model, Cascade):
                self.ds.logger.info(make_header(model.name, 200))
                model.auto_logreg(out=writer, save_model=None)
        if isinstance(self.integral, LogisticRegressionModel):
            self.fit()
            self.integral.report(ds=self.scoring(), out=writer, sheet_name=f'{self.name}_integral', metrics=['wald', 'ks', 'vif', 'iv', 'ontime'])
            if validate:
                self.ds.logger.info(make_header('Validation', 175))
                self.integral.validate(ds=self.ds)
        if save_model:
            self.save_model(file_name=f'{self.ds.result_folder}{save_model}')
        if isinstance(out, str):
            writer.close()

    def fit(self, ds=None, sample_name=None):
        """
        Обучение интегральной модели
        :param ds: ДатаСэмпл
        :param sample_name: название сэмпла на котором проводится обучение. При None берется ds.train_sample
        """
        if not isinstance(self.integral, LogisticRegressionModel):
            self.print_log('self.integral is not LogisticRegressionModel!')
            return None
        if ds is None:
            ds = self.ds
        features = []
        for model in self.models:
            if isinstance(model, (LogisticRegressionModel, Cascade)):
                score = f'{model.name}_score'
                ds = model.scoring(data=ds, score_field=score, pd_field=None)
                features.append(score)
            elif isinstance(model, str):
                features.append(model)
        self.integral.fit(ds=ds, sample_name=sample_name, features=features)

    def scoring(self, data=None, score_field='score', pd_field='pd', scale_field=None):
        """
        Скоринг выборки.
        :param data: ДатаСэмпл или ДатаФрейм. Возвращается объект того же типа
        :param score_field: поле, в которое должен быть записан посчитанный скор
        :param pd_field: поле, в которое должен быть записан посчитанный PD
        :param scale_field: поле, в которое должен быть записан посчитанный грейд

        :return: ДатаСэмпл или ДатаФрейм с добавленными полями скоров, PD и грейда
        """

        if data is not None:
            if isinstance(data, pd.DataFrame):
                ds = DataSamples(samples={'train': data}, features=[], cat_columns=[])
            else:
                ds = data
        else:
            ds = self.ds
        features = []
        for model in self.models:
            if isinstance(model, (LogisticRegressionModel, Cascade)):
                score = f'{model.name}_score'
                ds = model.scoring(data=ds, score_field=score, pd_field=None)
                features.append(score)
            elif isinstance(model, str):
                features.append(model)
        if isinstance(self.integral, LogisticRegressionModel):
            ds = self.integral.scoring(data=ds, score_field=score_field, pd_field=pd_field, scale_field=scale_field)
        else:
            for name in ds.samples:
                ds.samples[name][score_field] = self.integral([ds.samples[name][f] for f in features])
            if ds.bootstrap_base is not None:
                ds.bootstrap_base[score_field] = self.integral([ds.bootstrap_base[f] for f in features])
        if isinstance(data, pd.DataFrame):
            return ds.to_df(sample_field=None)
        else:
            return ds

    def get_code(self, score_field=None, pd_field=None, scale_field=None, lang='py'):
        """
        Вспомогательная функция для метода to_py
        :param score_field: поле, в которое должен быть записан посчитанный скор
        :param pd_field:  поле, в которое должен быть записан посчитанный PD
        :param scale_field:  поле, в которое должен быть записан посчитанный грейд

        :return: хардкод каскада
        """
        def clean_func(f):
            try:
                code_func = inspect.getsource(f)
            except Exception as e:
                self.print_log(f'Error! {e}')
                return '\n# Error on self.integral function code restore!\n# Please add it here manually\n'
            res = []
            pref = None
            for line in code_func.split('\n'):
                if pref is None:
                    if 'def ' in line and f.__name__ in line:
                        pref = line.split('def')[0]
                        res.append(line.replace(pref, '', 1))
                elif line.startswith(pref):
                    res.append(line.replace(pref, '', 1))
                else:
                    res.append(line)
            return '\n' + '\n'.join(res) + '\n'

        result = ''
        features = []
        for model in self.models:
            if isinstance(model, LogisticRegressionModel):
                result += model.get_code(score_field=f"'{model.name}_score'", lang=lang)
                features.append(f'{model.name}_score')
            elif isinstance(model, Cascade):
                result += model.get_code(lang=lang)
                features.append(f'{model.name}_score')
            else:
                features.append(model)

        if isinstance(self.integral, LogisticRegressionModel):
            result += self.integral.get_code(score_field='score_field' if score_field else f"'{self.name}_score'",
                                             pd_field='pd_field' if pd_field else None,
                                             scale_field='scale_field' if scale_field else None,
                                             lang=lang)
        else:
            result += clean_func(self.integral)
            result += f"df[{score_field}] = {self.integral.__name__}([df[f] for f in {features}])\n"
        return result

    def to_py(self, file_name='model.py', score_field='score', pd_field='pd', scale_field=None):
        """
        Генерация хардкода функции scoring
        :param file_name: название питоновского файла, куда должен быть сохранен код
        :param score_field: поле, в которое должен быть записан посчитанный скор
        :param pd_field:  поле, в которое должен быть записан посчитанный PD
        :param scale_field:  поле, в которое должен быть записан посчитанный грейд
        """
        result = "import pandas as pd\nimport numpy as np\n\n"
        if scale_field and isinstance(self.integral, LogisticRegressionModel):
            result += f'''
def to_scale(PD):
    scale = {self.integral.scale}
    for s in scale:
        if PD < scale[s]:
            return s
    return 'MSD'\n\n'''

        result += f'''
def scoring(df, score_field='score', pd_field='pd', scale_field=None):
    """
    Функция скоринга выборки
    Arguments:
        df: [pd.DataFrame] входной ДатаФрейм, должен содержать все нетрансформированные переменные модели
        score_field: [str] поле, в которое должен быть записан посчитанный скор
        pd_field: [str] поле, в которое должен быть записан посчитанный PD
        scale_field: [str] поле, в которое должен быть записан посчитанный грейд
    Returns:
        df: [pd.DataFrame] выходной ДатаФрейм с добавленными полями трансформированных переменных, скоров, PD и грейда
    """
'''
        result += self.get_code(score_field='score_field' if score_field else None,
                                pd_field='pd_field' if pd_field else None,
                                scale_field='scale_field' if scale_field else None,
                                lang='py').replace('\n', '\n    ')
        result += f"return df\n\n"
        result += f'''df = scoring(df, score_field={f"'{score_field}'" if score_field else None}, pd_field={f"'{pd_field}'" if pd_field else None}, scale_field={f"'{scale_field}'" if scale_field else None})'''
        if file_name:
            if self.ds is not None:
                file_name = self.ds.result_folder + file_name
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(result)
            self.print_log(f'The model code for implementation saved to file {file_name}')
        self.print_log(result)



