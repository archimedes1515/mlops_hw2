from app import model_dict, app, ClassifierModels, Predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def test_model_dict(mocker):
    mocker.patch('app.db.session')
    assert len(model_dict) == 2
    assert isinstance(model_dict['logreg'], LogisticRegression)
    assert isinstance(model_dict['RandomForest'], RandomForestClassifier)


def test_logreg_fit(mocker):
    test_model = LogisticRegression(C=1.0)
    mocker.patch('app.db.session')
    train = [[1, 2, 3],
             [2, 1, 1],
             [1, 1, 2],
             [0, 2, 0]]
    target = [1, 0, 1, 0]
    test_model.fit(train, target)
    cm = ClassifierModels()
    with app.test_request_context('/model_config/logreg', json={'Hyperparams': {'C': 1.0},
                                                                'train': train,
                                                                'target': target}):
        ret = cm.post(model_name='logreg')
    assert 'fit on data.' in ret
    assert float(ret.split()[-1]) == test_model.score(train, target)


def test_pred_fit(mocker):
    test_model = LogisticRegression(C=1.0)
    mocker.patch('app.db.session')
    mocker.patch('pickle.loads', return_value=test_model)
    train = [[1, 2, 3],
             [2, 1, 1],
             [1, 1, 2],
             [0, 2, 0]]
    target = [1, 0, 1, 0]

    test_set = [[0, 0, 1],
                [1, 1, 1],
                [3, 2, 2],
                [2, 2, 0]]

    pred = Predict()
    with app.test_request_context('/model_predict/logreg', json={'model_id': {'id': 0},
                                                                 'test': test_set}):
        ret = pred.post(model_name='logreg')
        assert ret == 404
    test_model.fit(train, target)
    with app.test_request_context('/model_predict/logreg', json={'model_id': {'id': 0},
                                                                 'test': test_set}):
        ret = pred.post(model_name='logreg')
        assert ret['predicts'] == [0, 0, 1, 0]
