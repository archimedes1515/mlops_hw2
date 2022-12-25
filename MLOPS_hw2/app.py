from flask import Flask, abort, jsonify
from flask_restx import Api, Resource, fields
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
import pickle
from flask_sqlalchemy import SQLAlchemy


model_dict = {
    'logreg': LogisticRegression(),
    'RandomForest': RandomForestClassifier(n_estimators=5)
    }  # словарь с доступными моделями

'''
Для логистической регрессии доступны гиперпараметры:
тип регуляризации и коэффициент регуляризации.
Для случайного леса:
информационный критерий и максимальная глубина.
'''
hyperparams_dict = {
    'logreg': ['solver', 'C'],
    'RandomForest': ['criterion', 'max_depth']
    }  # словарь для вывода только доступных гиперпараметров

app = Flask(__name__)
api = Api(app, title='my ML API')  # оборачиваем app для документации swagger

if __name__ == '__main__':
    app.config["SECRET_KEY"] = "pogchampsecret"
    app.config[
        "SQLALCHEMY_DATABASE_URI"
    ] = "postgresql://postgres:123@postgres:5432/homework2"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class ModelTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.LargeBinary, nullable=False)


db.create_all()

data_hyperparams = api.model('Data and Hyperparams',
                             {'Hyperparams': fields.Raw(default={'C': 1.0}),
                              'train': fields.List(fields.List(fields.Float()),
                                                   default=[[1, 2, 3],
                                                            [2, 1, 1],
                                                            [1, 1, 2],
                                                            [0, 2, 0]]),
                              'target': fields.List(fields.Integer(),
                                                    default=[1, 0, 1, 0])
                              }
                             )  # для сбора гиперпараметров и трейн-сета в post

model_choice = api.model('Model id',
                         {'model_id': fields.Raw(default={'id': 0})})

model_id_data = api.model('Model id and Train Data',
                          {'model_id': fields.Raw(default={'id': 0}),
                           'train': fields.List(fields.List(fields.Float()),
                                                default=[[1, 1, 1],
                                                         [2, 2, 1],
                                                         [1, 3, 2],
                                                         [1, 2, 4]]),
                           'target': fields.List(fields.Integer(),
                                                 default=[0, 1, 1, 1])
                           }
                          )  # для получения id модели и трейн данных в put

model_id_test = api.model('Model id and Test Data',
                          {'model_id': fields.Raw(default={'id': 0}),
                           'test': fields.List(fields.List(fields.Float()),
                                               default=[[0, 0, 1],
                                                        [1, 1, 1],
                                                        [3, 2, 2],
                                                        [2, 2, 0]])
                           }
                          )  # для получения id модели и тестовых данных


@api.route('/models')
class Models(Resource):
    """Список доступных для обучения классов моделей."""

    def get(self):
        """Инфо о доступных моделях (ввод строго аналогично названиям ниже)."""
        return jsonify(availible_model_classes=['logreg', 'RandomForest'],
                       available_hyperparams=hyperparams_dict)


@api.route('/model_config/<string:model_name>')
class ClassifierModels(Resource):
    """Класс для вывода моделей, тюнинга гиперпараметров, удаления модели."""

    @api.doc(responses={
        200: 'Success',
        400: 'Error',
        404: 'Model doesn\'t exist'
    }, params={'model_name': 'Name of the model'})
    def get(self):
        """Отображение моделей."""
        return 'logreg and RandomForest'

    @api.expect(data_hyperparams)
    def post(self, model_name):
        """Настройка гиперпараметров модели и обучение."""
        hyperparams = api.payload['Hyperparams']  # получим гиперпараметры
        if (model_name == 'logreg') and ('C' in hyperparams):
            hyperparams['C'] = float(hyperparams['C'])
        elif (model_name == 'RandomForest') and ('max_depth' in hyperparams):
            hyperparams['max_depth'] = int(hyperparams['max_depth'])
        try:
            model_dict[model_name].set_params(**hyperparams)
        except (ValueError, KeyError) as err:
            abort(400, f'Wrong model params or model name, check docs: {err}')
        train = api.payload['train']  # получим данные для обучения
        target = api.payload['target']
        try:
            new_model = model_dict[model_name].fit(train, target)
        except ValueError as err:
            abort(400, f'Failed to fit. Wrong data: {err}')
        score = new_model.score(train, target)
        if model_name == 'logreg':
            model_dict[model_name] = LogisticRegression()
        elif model_name == 'RandomForest':
            model_dict[model_name] = RandomForestClassifier()
        new_record = ModelTable(model=pickle.dumps(new_model))
        db.session.add(new_record)
        db.session.commit()
        return f'{new_model} fit on data. id: {new_record.id}. Train score: {score}'

    @api.expect(model_id_data)
    def put(self, model_name):
        """Обучение заново для выбранной модели."""
        if model_name in model_dict:
            model_id = api.payload['model_id']['id']  # считаем id модели
            train = api.payload['train']
            target = api.payload['target']
            my_query = ModelTable.query.get(model_id)
            if my_query is None:
                abort(404, f'Model {model_name} with id {model_id} not found')
            else:
                model = pickle.loads(my_query.model)
                try:
                    model = model.fit(train, target)
                except ValueError as err:
                    abort(400, f'Failed to fit. Check data. {err}')
                my_query.model = pickle.dumps(model)
                db.session.commit()
                return f'Refitted model {model_name} with id {model_id}'
        else:
            abort(404, 'No such model')

    @api.expect(model_choice)
    def delete(self, model_name):
        """Удаление модели."""
        if model_name in model_dict:
            model_id = api.payload['model_id']['id']
            effected_rows = ModelTable.query.filter(ModelTable.id == model_id).delete()
            if effected_rows == 0:
                abort(404, f'Model {model_name} with id {model_id} not found')
            else:
                db.session.commit()
        else:
            abort(404, 'No such model')
        return f'Dumped model {model_name} with id {model_id}'


@api.route('/model_predict/<string:model_name>')
class Predict(Resource):
    """Класс для вывода предсказаний конкретной модели."""

    @api.expect(model_id_test)
    def post(self, model_name):
        """Предсказания с помощью выбранной модели."""
        if model_name in model_dict:
            model_id = api.payload['model_id']['id']
            my_query = ModelTable.query.get(model_id)
            if my_query is not None:
                model = pickle.loads(my_query.model)
                try:
                    test = api.payload['test']
                    predict = model.predict(test)
                    return {'predicts': list(map(int, predict))}
                except (NotFittedError, ValueError) as err:
                    return 404
                    # abort(400, f'Failed to predict. Check data. {err}')
            else:
                abort(404, f'Model {model_name} with id {model_id} not found')
        else:
            abort(404, 'No such model')


if __name__ == '__main__':
    app.run(debug=True)
