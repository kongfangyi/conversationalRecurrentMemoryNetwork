from keras import backend as K
from scipy.stats import pearsonr

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


import numpy as np
import pandas as pd
import sklearn.metrics as me
import tensorflow as tf


def returnMacroF1(result_csv_file):
    resultData = pd.read_csv(result_csv_file, header=[0])
    macroF1_2 = me.f1_score(np.array(resultData["test_sentiment"]),np.array(resultData["sentiment"]), average="macro")
    print(macroF1_2)
    return np.array(macroF1_2)

def returnMicroF1(result_csv_file):
    resultData = pd.read_csv(result_csv_file, header=[0])
    macroF1_2 = me.f1_score(np.array(resultData["test_sentiment"]),np.array(resultData["sentiment"]), average="micro")
    print(macroF1_2)
    return np.array(macroF1_2)

import sklearn.metrics as me
def task3returnMicroF1(result_csv_file):
    resultData = pd.read_csv(result_csv_file, header=[0])
    resultData = resultData[resultData['test_sentiment'] != 3]
    microF1_2 = me.f1_score(np.array(resultData["test_sentiment"]), np.array(resultData["sentiment"]), average="micro")
    print(microF1_2)
    return np.array(microF1_2)
