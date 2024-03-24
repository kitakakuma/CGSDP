# -*- coding:utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


def classifier_output(data_train, label_train, data_test, label_test):
    rf = MLPClassifier(random_state=42)
    parameters = {"hidden_layer_sizes": [(100, 100, 100)], "solver": ['adam', 'sgd'], "max_iter": [1000, 2000], }
    gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
    gsearch.fit(data_train, label_train)
    predprob = gsearch.predict(data_test)
    predprob_auc = gsearch.predict_proba(data_test)[:, 1]
    recall = metrics.recall_score(label_test, predprob)
    auc = metrics.roc_auc_score(label_test, predprob_auc)
    precision = metrics.precision_score(label_test, predprob)
    fmeasure = metrics.f1_score(label_test, predprob)
    mcc = metrics.matthews_corrcoef(label_test, predprob)
    accuracy = metrics.accuracy_score(label_test, predprob)
    return predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy
