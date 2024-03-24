# -*- coding:utf-8 -*-
import time
import tensorflow as tf

from classify.utils import generate_imbalance_data, average_value
from utils import *
from sklearn.model_selection import RepeatedKFold
import numpy as np
from Classifier import *
import pandas as pd

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

######################################################################################################################
baseURL = "../data/code/"
######################################################################################################################

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('classifier', 'MLP', 'Select a classifier for classification.')
flags.DEFINE_string('imbalance', 'SMOTEENN', 'Select a methods of dealing with imbalanced data.')


def run_evaluation(X_train, y_train, X_test, y_test):
    t = time.time()
    # data sample
    X_resampled, y_resampled = generate_imbalance_data(X_train, y_train, FLAGS.imbalance)

    # training classifier
    predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy = \
        classifier_output(X_resampled, y_resampled, X_test, y_test)

    print("precision=", "{:.5f}".format(precision),
          "recall=", "{:.5f}".format(recall),
          "f-measure=", "{:.5f}".format(fmeasure),
          "auc=", "{:.5f}".format(auc),
          "accuracy=", "{:.5f}".format(accuracy),
          "time=", "{:.5f}".format(time.time() - t))
    return precision, recall, fmeasure, auc, accuracy


def cvdp(datalist):
    origin_train_data = pd.read_csv(
        baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\" + datalist[0] + ".csv",
        header=0, index_col=False)
    dw_train_data = pd.read_csv(baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\cgsdp_emb_add.csv",
                                header=0, index_col=False)
    # dw_train_data = pd.read_csv(baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\cgcn_emb_add.csv",
    #                             header=0, index_col=False)
    # dw_train_data = pd.read_csv(baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\cnn_emb_add.csv",
    #                             header=0, index_col=False)

    X_train = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 1:-1]], axis=1))
    # X_train = np.array(origin_train_data.iloc[:, 1:-1])
    y_train = np.array(np.int64(origin_train_data['bug'] > 0))

    origin_test_data = pd.read_csv(
        baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\" + datalist[1] + ".csv",
        header=0, index_col=False)
    dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\cgsdp_emb_add.csv",
                               header=0, index_col=False)
    # dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\cgcn_emb_add.csv",
    #                            header=0, index_col=False)
    # dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\cnn_emb_add.csv",
    #                            header=0, index_col=False)

    X_test = np.array(pd.concat([dw_test_data, origin_test_data.iloc[:, 1:-1]], axis=1))
    # X_test = np.array(origin_test_data.iloc[:, 1:-1])
    y_test = np.array(np.int64(origin_test_data['bug'] > 0))
    return X_train, y_train, X_test, y_test


def cpdp(datalist):
    origin_train_data = pd.read_csv(
        baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\" + datalist[0] + ".csv",
        header=0, index_col=False)
    dw_train_data = pd.read_csv(baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\cgsdp_cross_emb_add.csv",
                                header=0, index_col=False)
    # dw_train_data = pd.read_csv(baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\cgcn_cross_emb_add.csv",
    #                             header=0, index_col=False)
    # dw_train_data = pd.read_csv(baseURL + datalist[0].split('-')[0] + "\\" + datalist[0] + "\\cnn_cross_emb_add.csv",
    #                             header=0, index_col=False)

    X_train = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 1:-1]], axis=1))
    # X_train = np.array(origin_train_data.iloc[:, 1:-1])
    y_train = np.array(np.int64(origin_train_data['bug'] > 0))

    origin_test_data = pd.read_csv(
        baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\" + datalist[1] + ".csv",
        header=0, index_col=False)
    dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\cgsdp_cross_emb_add.csv",
                               header=0, index_col=False)
    # dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\cgcn_cross_emb_add.csv",
    #                            header=0, index_col=False)
    # dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\cnn_cross_emb_add.csv",
    #                            header=0, index_col=False)

    X_test = np.array(pd.concat([dw_test_data, origin_test_data.iloc[:, 1:-1]], axis=1))
    # X_test = np.array(origin_test_data.iloc[:, 1:-1])
    y_test = np.array(np.int64(origin_test_data['bug'] > 0))
    return X_train, y_train, X_test, y_test


def wpdp(data):
    F1_list = []
    precision_list = []
    recall_list = []
    AUC_list = []
    accuracy_list = []
    origin_train_data = pd.read_csv(baseURL + data.split('-')[0] + "\\" + data + "\\" + data + ".csv", header=0,
                                    index_col=False)
    dw_train_data = pd.read_csv(baseURL + data.split('-')[0] + "\\" + data + "\\cgsdp_emb_add.csv", header=0,
                                index_col=False)
    # dw_train_data = pd.read_csv(baseURL + data.split('-')[0] + "\\" + data + "\\cgcn_emb_add.csv", header=0,
    #                             index_col=False)
    # dw_train_data = pd.read_csv(baseURL + data.split('-')[0] + "\\" + data + "\\cnn_emb_add.csv", header=0,
    #                             index_col=False)
    X = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 1:-1]], axis=1))
    # X = np.array(origin_train_data.iloc[:, 1:-1])
    y = np.array(np.int64(origin_train_data['bug'] > 0))
    exp_cursor = 1
    kf = RepeatedKFold(n_splits=5, n_repeats=10)  # We can modify n_repeats when debugging.
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        precision, recall, fmeasure, auc, accuracy = run_evaluation(X_train, y_train, X_test, y_test)
        F1_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)
        AUC_list.append(auc)
        accuracy_list.append(accuracy)

        exp_cursor = exp_cursor + 1

    avg = []
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(F1_list))
    avg.append(average_value(AUC_list))
    avg.append(average_value(accuracy_list))

    name = ['precision', 'recall', 'F1', 'AUC', 'Accuracy']
    results = []
    results.append(precision_list)
    results.append(recall_list)
    results.append(F1_list)
    results.append(AUC_list)
    results.append(accuracy_list)
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)
    df.to_csv("../results/cgsdp" + data + ".csv")
    # df.to_csv("../results/cgcn/" + data + ".csv")
    # df.to_csv("../results/cnn/" + data + ".csv")
    # df.to_csv("../results/tradition/" + data + ".csv")


# choose projects
# dict_file = open('./within_project.txt', 'r')
# dict_file = open('./cross_version.txt', 'r')
dict_file = open('./cross_project2.txt', 'r')
lines = dict_file.readlines()
for line in lines:
    datalist = line.strip().split(',')
    print(line.strip() + " Start!")
    dataset = datalist[0].split('-')[0]
    print(datalist[-1] + " Start!")
    wpdp(line.strip())

    # cross version/cross project
    # F1_list = []
    # precision_list = []
    # recall_list = []
    # AUC_list = []
    # accuracy_list = []
    # X_train, y_train, X_test, y_test = cvdp(datalist)
    # X_train, y_train, X_test, y_test = cpdp(datalist)
    # for i in range(50):
    #     precision, recall, fmeasure, auc, accuracy = run_evaluation(X_train, y_train, X_test, y_test)
    #     F1_list.append(fmeasure)
    #     precision_list.append(precision)
    #     recall_list.append(recall)
    #     AUC_list.append(auc)
    #     accuracy_list.append(accuracy)

    # print(datalist[-1] + "Finished!")

    # result = [line.strip(), average_value(precision_list), average_value(recall_list), average_value(F1_list),
    #           average_value(AUC_list), average_value(accuracy_list)]
    # df = pd.DataFrame([result])
    # df.to_csv("..\\results\\crossproject\\cgsdp\\crossversion.csv", mode="a", header=None,index=False)
    # df.to_csv("..\\results\\crossproject\\cgcn\\crossversion.csv", mode="a", header=None,index=False)
    # df.to_csv("..\\results\\crossproject\\cnn\\crossversion.csv", mode="a", header=None,index=False)
    # df.to_csv("..\\results\\crossproject\\tradition\\crossversion.csv", mode="a", header=None,index=False)

    # df.to_csv("..\\results\\crossproject\\cgsdp\\crossproject_all.csv", mode="a", header=None,index=False)
    # df.to_csv("..\\results\\crossproject\\cgcn\\crossproject_all.csv", mode="a", header=None,index=False)
    # df.to_csv("..\\results\\crossproject\\cnn\\crossproject_all.csv", mode="a", header=None,index=False)
    # df.to_csv("..\\results\\crossproject\\tradition\\crossproject_all.csv", mode="a", header=None,index=False)
