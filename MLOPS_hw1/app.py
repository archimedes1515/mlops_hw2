from flask import Flask, request, abort, jsonify
from flask_restx import Api, Resource, reqparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json

model_dict = {
    'logreg': LogisticRegression(),
    'RandomForest': RandomForestClassifier()
    }  # словарь с доступными моделями

'''
Для логистической регрессии доступны гиперпараметры:
тип регуляризации и коэффициент регуляризации.
Для случайного леса:
информационный критерий и максимальная глубина.
'''
hyperparams_dict = {
    'logreg': ['penalty', 'C'],
    'RandomForest': ['criterion', 'max_depth']
    }  # словарь для вывода только доступных гиперпараметров

data = [pd.DataFrame()]  # список для хранения данных

app = Flask(__name__)
api = Api(app, title='my ML API')  # оборачиваем app для документации swagger

parser = reqparse.RequestParser()  # parser для сбора гиперпараметров
parser.add_argument('params', type=str, help='model params (pass as str)',
                    location='json')

parser_data = reqparse.RequestParser()  # parser для сбора данных
parser_data.add_argument('data', type=str, help='data (list of str)',
                         location='json')


@api.route('/entry')
class EntryPage(Resource):
    """Приветствие."""

    def get(self):
        """Инфо о доступных моделях (ввод строго аналогично названиям ниже)."""
        return 'Welcome. Choose 1 of the next models: logreg, RandomForest'


@api.route('/entry/<string:model_name>')
class ClassifierModels(Resource):
    """Класс для вывода модели, тюнинга гиперпараметров, удаления модели."""

    @api.doc(responses={
        200: 'Successful prediction',
        400: 'Error',
        404: 'Model doesn\'t exist'
    }, params={'model_name': 'Name of the model'})
    def get(self, model_name):
        """Отобразить модель."""
        if model_name in model_dict:
            good_hyp_params = {
                k: v for k, v in model_dict[model_name].get_params().items()
                if k in hyperparams_dict[model_name]
                }  # оставим только доступные (для изменения) гиперпараметры
            return jsonify(model=model_name,
                           model_params=good_hyp_params)
        else:
            abort(404, 'No such model')  # будем стараться отлавливать подобное

    @api.expect(parser)
    def post(self, model_name):
        """Настройка гиперпараметров модели."""
        if model_name == 'logreg':
            params = parser.parse_args()  # считаем данные в виде строки
            try:
                penalty, C = params['params'].split()  # разделим параметры
                params_dict = {'penalty': penalty, 'C': float(C)}
            except BaseException as err:
                abort(400, f'{err}. Wrong params format or number, check docs')
        elif model_name == 'RandomForest':
            params = parser.parse_args()
            criterion, max_depth = params['params'].split()
            try:
                params_dict = {
                    'criterion': criterion,
                    'max_depth': int(max_depth)
                    }
            except BaseException as err:
                abort(400, f'{err}. Wrong params format or number, check docs')
        else:
            abort(400, 'No such model')
        try:
            model_dict[model_name].set_params(**params_dict)
        except BaseException as err:
            abort(400, f'{err}. Sklearn did not accept your params')
        return f'Initialized model {model_name} with args: {params_dict}'

    def delete(self, model_name):
        """Удаление модели."""
        if model_name in model_dict:
            if model_name == 'logreg':
                model_dict[model_name] = LogisticRegression()
            else:
                model_dict[model_name] = RandomForestClassifier()
        else:
            abort(404, 'No such model')
        return f'You have dumped model {model_name}'


@api.route('/entry/data')
class Data(Resource):
    """Класс для получения, отображения данных."""

    # def get(self):
    #     """Вывод данных."""
    #     return data

    @api.expect(parser_data)
    def post(self):
        """Подтягивание данных."""
        # data = list(map(lambda x: x.split(), data))
        # data = [list(map(float, i[:-1])) + [int(i[-1])] for i in data]
        try:
            data[0] = data[0].append(
                pd.read_csv(parser_data.parse_args()['data'],
                            header=None),
                ignore_index=True)
        except BaseException as err:
            abort(400, f'{err}. Wrong data')
        if data.isna().sum().sum() > 0:
            abort(400, 'Wrong input data (try to check shapes of rows)')
        return 'Your data accepted'


@api.route('/entry/<string:model_name>/fit_predict')
class FitPredict(Resource):
    """Класс для вывода предсказаний модели и обучения."""

    def get(self, model_name):
        """Выдача предсказаний модели."""
        try:
            df = data[0]
            X = df.iloc[:, 1:]
            # y = df[0].astype(int)
            return ' '.join(model_dict[model_name].predict(X).astype(str))
        except BaseException as err:
            abort(404,
                  f'{err}. No data given|mistakes in data or model not fitted')

    def post(self, model_name):
        """Обучение модели."""
        try:
            df = data[0]
            X = df.iloc[:, 1:]
            y = df[0].astype(int)
            model_dict[model_name] = model_dict[model_name].fit(X, y)
            return 'Model fitted, success'
        except BaseException as err:
            abort(404,
                  f'{err}. Possibly no data provided yet or mistakes in data')


if __name__ == '__main__':
    app.run(debug=True)
