# -*- coding:utf-8 -*-
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours

def average_value(list):
    return float(sum(list))/len(list)

def label_sum(label_train):
    label_sum=0
    for each in label_train:
        label_sum=label_sum+each
    return label_sum
def generate_imbalance_data(X_train, y_train, imbalance):
    if (label_sum(y_train) > (int(len(y_train) * 0.4))) and (label_sum(y_train) < (int(len(y_train) * 0.6))):
        print("The training data does not need balance.")
        state = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(state)
        np.random.shuffle(y_train)
        return X_train, y_train
    else:
        if imbalance == 'SMOTE':
            X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        elif imbalance == 'SMOTEENN':
            X_resampled, y_resampled = SMOTEENN().fit_resample(X_train, y_train)
        else:
            rus = RandomUnderSampler(random_state=0, replacement=True)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    state = np.random.get_state()
    np.random.shuffle(X_resampled)
    np.random.set_state(state)
    np.random.shuffle(y_resampled)

    return X_resampled, y_resampled
