#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 20:52
# @Author  : q_y_jun
# @email   : q_y_jun@163.com
# @File    : dynamicMemoryNetwork

import os
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from kfy.metrics import f1, returnMacroF1, task3returnMicroF1
from kfy.loss_func import focal_loss
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda
import keras.backend as K

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import np_utils
from transformers import *
from transformers import TFDistilBertModel, RobertaTokenizerFast, TFRobertaModel
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, Dot, Flatten, LSTM, MaxPooling1D, Reshape, \
    Softmax, Dense, Dropout, Bidirectional
import tensorflow as tf
import random

import pandas as pd
from collections import Counter
import pickle

print(tf.__version__)

np.random.seed(100)
tf.random.set_seed(1314)
# --------------------------------------------------------------------------------------------------------------------
# global Variables

nb_epoch = 2
hidden_dim = 120
nb_filter = 60
kernel_size = 3

batch_size = 4
nb_epoch = 4
head_amount  = 2
learning_ratio = 1e-6
dropout_ratio = 0.1

# dropout为0.1
# 1e-8学习率过低，1轮valacc为0.049；val_f1基本为0
# 1e-6得到1轮和两轮的结果均为0.3084因此过拟合

# 降低dropout为0.01
# 1e-6三轮过拟合f1为0 val_acc为0.8478
# 1e-8三轮过拟合f1为0.3008
# dopout为0.00
# 1e-8三轮过拟合f1为0.3008

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 全局参数
# 设置随机值
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(1314)


# 得到bert对应的ids及mask输入
def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    inputs = tokenizer(instance, return_tensors="tf", padding="max_length", max_length=max_sequence_length)

    input_ids = inputs["input_ids"]
    input_masks = inputs["attention_mask"]

    return [input_ids, input_masks]


# 将ids及mask对应的数据，分别包装为数组[ids数组，mask数组]
def compute_input_arrays(train_data_input, tokenizer, max_sequence_length):
    input_ids, input_masks = [], []
    for instance in tqdm(train_data_input):
        ids, masks = _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)
        input_ids.append(ids[0])
        input_masks.append(masks[0])

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32)]


from collections import OrderedDict, Counter


def dynamicMemoryLayer():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    x_1 = tf.keras.layers.GlobalAveragePooling1D()(turn1_seq_output)
    x_1 = Reshape((1, hidden_dim))(x_1)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    x_2 = tf.keras.layers.GlobalAveragePooling1D()(turn2_seq_output)
    x_2 = Reshape((1, hidden_dim))(x_2)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    x_3 = tf.keras.layers.GlobalAveragePooling1D()(turn3_seq_output)
    x_3 = Reshape((1, hidden_dim))(x_3)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector_turn1 = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector_turn1 = memoryVector_turn1[0]
    from kfy.interActivateLayer_recurrent_v2 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn1, turn1_seq_output])
    print("input_size", interActivateVec_turn1)

    tanh_inter_left_turn1 = Tanh()(interActivateVec_turn1)
    inter_trans_turn1 = TransMatrix()(interActivateVec_turn1)
    tanh_inter_right_turn1 = Tanh()(inter_trans_turn1)

    scaledPool_inter_right_turn1 = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right_turn1)
    scaledPool_inter_right_turn1 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right_turn1)
    print("scaledPool_inter_right ", scaledPool_inter_right_turn1)

    softmax_inter_right_turn1 = Softmax()(scaledPool_inter_right_turn1)

    from tensorflow.keras.layers import RepeatVector
    memoryVector = Dot(axes=1)([turn2_seq_output, softmax_inter_right_turn1])
    combVecMemory = RepeatVector(121)(memoryVector)
    print("1st turn end MemoryVector is  ", combVecMemory)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory, turn2_seq_output])
    print("second period input_size", interActivateVec_turn2)

    tanh_inter_left_turn2 = Tanh()(interActivateVec_turn2)
    inter_trans_turn2 = TransMatrix()(interActivateVec_turn2)
    tanh_inter_right_turn2 = Tanh()(inter_trans_turn2)

    scaledPool_inter_right_turn2 = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right_turn2)
    scaledPool_inter_right_turn2 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right_turn2)
    print("scaledPool_inter_right ", scaledPool_inter_right_turn2)

    softmax_inter_right_turn2 = Softmax()(scaledPool_inter_right_turn2)

    from tensorflow.keras.layers import RepeatVector
    memoryVector_turn2 = Dot(axes=1)([turn3_seq_output, softmax_inter_right_turn2])
    combVecMemory_turn2 = RepeatVector(121)(memoryVector_turn2)
    print("1st turn end MemoryVector is  ", combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn2, turn3_seq_output])
    print("thrid period input_size", interActivateVec_turn3)

    tanh_inter_left_turn3 = Tanh()(interActivateVec_turn3)
    inter_trans_turn3 = TransMatrix()(interActivateVec_turn3)
    tanh_inter_right_turn3 = Tanh()(inter_trans_turn3)

    scaledPool_inter_left_turn3 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left_turn3)
    scaledPool_inter_left_turn3 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left_turn3)
    print("scaledPool_inter_left ", scaledPool_inter_left_turn3)
    '''
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)
    softmax_inter_right = Softmax()(scaledPool_inter_right)
    '''

    softmax_inter_left = Softmax()(scaledPool_inter_left_turn3)

    x = tf.keras.layers.Dense(4, activation='softmax')(softmax_inter_left)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model


# 仅使用memoryVector/turn3的最后行输出
def dynamicMemoryLayer_v2():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    x_1 = tf.keras.layers.GlobalAveragePooling1D()(turn1_seq_output)
    x_1 = Reshape((1, hidden_dim))(x_1)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    x_2 = tf.keras.layers.GlobalAveragePooling1D()(turn2_seq_output)
    x_2 = Reshape((1, hidden_dim))(x_2)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    x_3 = tf.keras.layers.GlobalAveragePooling1D()(turn3_seq_output)
    x_3 = Reshape((1, hidden_dim))(x_3)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector = Dot(axes=1)([memoryVector, extended_softmax_inter_left])
    memoryVector = Reshape([memoryVector.shape[2], memoryVector.shape[1]])(memoryVector)
    print("first processing period memoryVector is ", memoryVector)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn2_input_mid = Dot(axes=1)([turn2_seq_output, extended_softmax_inter_right])
    turn2_input_mid = Reshape([turn2_input_mid.shape[2], turn2_input_mid.shape[1]])(turn2_input_mid)
    print("second start processing period turn_ is ", turn2_input_mid)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn2_input_mid])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector = Dot(axes=1)([memoryVector, extended_softmax_inter_left])
    memoryVector = Reshape([memoryVector.shape[2], memoryVector.shape[1]])(memoryVector)
    print("first processing period memoryVector is ", memoryVector)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn3_input_mid = Dot(axes=1)([turn3_seq_output, extended_softmax_inter_right])
    turn3_input_mid = Reshape([turn3_input_mid.shape[2], turn3_input_mid.shape[1]])(turn3_input_mid)
    print("third start processing period turn_ is ", turn3_input_mid)

    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn3_input_mid])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    memoryVector = Dot(axes=1)([memoryVector, softmax_inter_left])
    print("third processing period memoryVector is ", memoryVector)

    turn3_output = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    print("third processing period turn3_output is ", turn3_output)

    # comboVec = Concatenate(axis=1)([memoryVector, turn3_output])
    # comboVec = Reshape([hidden_dim_f_dml*2])(comboVec)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(memoryVector)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-7)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model


# 使用3turns的输出进行行堆叠之后，使用LSTM
def dynamicMemoryLayer_v3():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    x_1 = tf.keras.layers.GlobalAveragePooling1D()(turn1_seq_output)
    x_1 = Reshape((1, hidden_dim))(x_1)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    x_2 = tf.keras.layers.GlobalAveragePooling1D()(turn2_seq_output)
    x_2 = Reshape((1, hidden_dim))(x_2)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    x_3 = tf.keras.layers.GlobalAveragePooling1D()(turn3_seq_output)
    x_3 = Reshape((1, hidden_dim))(x_3)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn1 = Dot(axes=1)([memoryVector, extended_softmax_inter_left])
    memoryVector_turn1 = Reshape([memoryVector_turn1.shape[2], memoryVector_turn1.shape[1]])(memoryVector_turn1)
    print("first processing period memoryVector is ", memoryVector_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn2_input_mid = Dot(axes=1)([turn2_seq_output, extended_softmax_inter_right])
    turn2_input_mid = Reshape([turn2_input_mid.shape[2], turn2_input_mid.shape[1]])(turn2_input_mid)
    print("second start processing period turn_ is ", turn2_input_mid)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn1, turn2_input_mid])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn2 = Dot(axes=1)([memoryVector_turn1, extended_softmax_inter_left])
    memoryVector_turn2 = Reshape([memoryVector_turn2.shape[2], memoryVector_turn2.shape[1]])(memoryVector_turn2)
    print("first processing period memoryVector is ", memoryVector_turn2)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn3_input_mid = Dot(axes=1)([turn3_seq_output, extended_softmax_inter_right])
    turn3_input_mid = Reshape([turn3_input_mid.shape[2], turn3_input_mid.shape[1]])(turn3_input_mid)
    print("third start processing period turn_ is ", turn3_input_mid)

    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn2, turn3_input_mid])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # -----------------------------------------------------------------------------------------------------
    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn3 = Dot(axes=1)([memoryVector_turn2, extended_softmax_inter_left])
    memoryVector_turn3 = Reshape([memoryVector_turn3.shape[2], memoryVector_turn3.shape[1]])(memoryVector_turn3)
    print("third processing period memoryVector is ", memoryVector_turn3)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn3_output = Dot(axes=1)([turn3_seq_output, extended_softmax_inter_right])
    turn3_output = Reshape([turn3_output.shape[2], turn3_output.shape[1]])(turn3_output)
    print("third start end period turn_3 is ", turn3_output)
    # --------------------------------------------------------------------------------------------------------

    # 将中间输出的turn1、turn2、turn3使用行堆叠的方式进行扩展

    comboVec = Concatenate(axis=1)([x_1, memoryVector_turn1, turn2_input_mid, memoryVector_turn2,
                                    turn3_input_mid, memoryVector_turn3, turn3_output])
    # comboVec = Reshape([hidden_dim_f_dml*2])(comboVec)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(comboVec)
    x = tf.keras.layers.LSTM(128, return_sequences=True, activation='softmax')(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False, activation='softmax')(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=5e-7)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
    return model


# 仅使用3turns的行向量输出进行行堆叠之后，使用LSTM
def dynamicMemoryLayer_v5():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    x_1 = tf.keras.layers.GlobalAveragePooling1D()(turn1_seq_output)
    x_1 = Reshape((1, hidden_dim))(x_1)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    x_2 = tf.keras.layers.GlobalAveragePooling1D()(turn2_seq_output)
    x_2 = Reshape((1, hidden_dim))(x_2)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    x_3 = tf.keras.layers.GlobalAveragePooling1D()(turn3_seq_output)
    x_3 = Reshape((1, hidden_dim))(x_3)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn1 = Dot(axes=1)([memoryVector, extended_softmax_inter_left])
    memoryVector_turn1 = Reshape([memoryVector_turn1.shape[2], memoryVector_turn1.shape[1]])(memoryVector_turn1)
    print("first processing period memoryVector is ", memoryVector_turn1)
    # 产生turn1记忆行向量
    memoryVector_turn1_row = Dot(axes=1)([memoryVector, softmax_inter_left])
    memoryVector_turn1_row = Reshape([1, hidden_dim_f_dml])(memoryVector_turn1_row)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn2_input_mid = Dot(axes=1)([turn2_seq_output, extended_softmax_inter_right])
    turn2_input_mid = Reshape([turn2_input_mid.shape[2], turn2_input_mid.shape[1]])(turn2_input_mid)
    print("second start processing period turn_ is ", turn2_input_mid)
    # 产生turn2融合行向量
    turn2_input_mid_row = Dot(axes=1)([turn2_seq_output, softmax_inter_right])
    turn2_input_mid_row = Reshape([1, hidden_dim_f_dml])(turn2_input_mid_row)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn1, turn2_input_mid])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn2 = Dot(axes=1)([memoryVector_turn1, extended_softmax_inter_left])
    memoryVector_turn2 = Reshape([memoryVector_turn2.shape[2], memoryVector_turn2.shape[1]])(memoryVector_turn2)
    print("first processing period memoryVector is ", memoryVector_turn2)
    # 产生turn2记忆行向量
    memoryVector_turn2_row = Dot(axes=1)([memoryVector_turn1, softmax_inter_left])
    memoryVector_turn2_row = Reshape([1, hidden_dim_f_dml])(memoryVector_turn2_row)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn3_input_mid = Dot(axes=1)([turn3_seq_output, extended_softmax_inter_right])
    turn3_input_mid = Reshape([turn3_input_mid.shape[2], turn3_input_mid.shape[1]])(turn3_input_mid)
    print("third start processing period turn_ is ", turn3_input_mid)
    # 产生turn3融合行向量
    turn3_input_mid_row = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    turn3_input_mid_row = Reshape([1, hidden_dim_f_dml])(turn3_input_mid_row)

    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn2, turn3_input_mid])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # -----------------------------------------------------------------------------------------------------
    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn3 = Dot(axes=1)([memoryVector_turn2, extended_softmax_inter_left])
    memoryVector_turn3 = Reshape([memoryVector_turn3.shape[2], memoryVector_turn3.shape[1]])(memoryVector_turn3)
    print("third processing period memoryVector is ", memoryVector_turn3)

    # 产生turn3记忆行向量
    memoryVector_turn3_row = Dot(axes=1)([memoryVector_turn2, softmax_inter_left])
    memoryVector_turn3_row = Reshape([1, hidden_dim_f_dml])(memoryVector_turn3_row)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn3_output = Dot(axes=1)([turn3_seq_output, extended_softmax_inter_right])
    turn3_output = Reshape([turn3_output.shape[2], turn3_output.shape[1]])(turn3_output)
    print("third start end period turn_3 is ", turn3_output)

    # 产生turn3融合行向量
    turn3_output_row = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    turn3_output_row = Reshape([1, hidden_dim_f_dml])(turn3_output_row)
    # --------------------------------------------------------------------------------------------------------

    # 将中间输出的turn1、turn2、turn3使用行堆叠的方式进行扩展

    comboVec = Concatenate(axis=1)([turn3_input_mid_row, memoryVector_turn3_row, turn3_output_row])
    # comboVec = Reshape([hidden_dim_f_dml*2])(comboVec)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(comboVec)
    x = tf.keras.layers.LSTM(128, return_sequences=True, activation='softmax')(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False, activation='softmax')(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-10)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
    return model


# 仅使用3turns再次与FusionMatrix融合，并将结果直接使用LSTM进行预测
def dynamicMemoryLayer_v6():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    x_1 = tf.keras.layers.GlobalAveragePooling1D()(turn1_seq_output)
    x_1 = Reshape((1, hidden_dim))(x_1)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    x_2 = tf.keras.layers.GlobalAveragePooling1D()(turn2_seq_output)
    x_2 = Reshape((1, hidden_dim))(x_2)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    x_3 = tf.keras.layers.GlobalAveragePooling1D()(turn3_seq_output)
    x_3 = Reshape((1, hidden_dim))(x_3)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn1 = Dot(axes=1)([memoryVector, extended_softmax_inter_left])
    memoryVector_turn1 = Reshape([memoryVector_turn1.shape[2], memoryVector_turn1.shape[1]])(memoryVector_turn1)
    print("first processing period memoryVector is ", memoryVector_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将turn1处理过的memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn2_input_mid = Dot(axes=1)([turn2_seq_output, extended_softmax_inter_right])
    turn2_input_mid = Reshape([turn2_input_mid.shape[2], turn2_input_mid.shape[1]])(turn2_input_mid)
    print("second start processing period turn_ is ", turn2_input_mid)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn1, turn2_input_mid])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    # 如果memoryVector想与产生的中间向量相乘必须将产生的中间向量进行行扩展，
    # memoryVector: seq_len X hiddendim; extend_softmax_inter_left: hiddenim X hiddendim
    extended_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    for i in range(softmax_inter_left.shape[1] - 1):
        reshaped_softmax_inter_left = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
        extended_softmax_inter_left = Concatenate(axis=1)([extended_softmax_inter_left, reshaped_softmax_inter_left])
    memoryVector_turn2 = Dot(axes=1)([memoryVector_turn1, extended_softmax_inter_left])
    memoryVector_turn2 = Reshape([memoryVector_turn2.shape[2], memoryVector_turn2.shape[1]])(memoryVector_turn2)
    print("first processing period memoryVector is ", memoryVector_turn2)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将turn2处理过的memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    turn3_input_mid = Dot(axes=1)([turn3_seq_output, extended_softmax_inter_right])
    turn3_input_mid = Reshape([turn3_input_mid.shape[2], turn3_input_mid.shape[1]])(turn3_input_mid)
    print("third start processing period turn_ is ", turn3_input_mid)

    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_turn2, turn3_input_mid])
    print("thrid period input_size", interActivateVec)

    tanhInterAct = Tanh()(interActivateVec)

    # 直接使用信息融合矩阵进行预测

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(tanhInterAct)
    x = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)
    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    return model


# 仅使用3turns再次与FusionMatrix融合，并将结果直接使用LSTM进行预测
def sequenceDynamicMemoryLayer_v1():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''
    # ---------------------------------------------------------------------------------------------------------------------
    # 将turn1处理过的memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn1_output = Dot(axes=1)([turn1_seq_output, softmax_inter_right])
    turn1_output = Reshape([1, turn1_output.shape[1]])(turn1_output)
    print("First turn processing period turn is ", turn1_output)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn2_seq_output])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''# ---------------------------------------------------------------------------------------------------------------------
    # 将turn2处理过的memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn2_output = Dot(axes=1)([turn2_seq_output, softmax_inter_right])
    turn2_output = Reshape([1, turn2_output.shape[1]])(turn2_output)
    print("Second turn end processing period turn_ is ", turn2_output)

    # ---------------------------------------------------------------------------------------------------------------------
    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn3_seq_output])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    turn3_output = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    turn3_output = Reshape([1, turn3_output.shape[1]])(turn3_output)
    print("Third turn end processing period turn_ is ", turn3_output)

    combVec = Concatenate(axis=1)([turn1_output, turn2_output, turn3_output])
    # 直接使用信息融合矩阵进行预测

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(combVec)
    x = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-7)

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['acc', f1])
    return model


# 仅使用3turns再次与FusionMatrix融合，并将结果直接使用LSTM进行预测
def sequenceDynamicMemoryLayerCap_v1():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''
    # ---------------------------------------------------------------------------------------------------------------------
    # 将turn1处理过的memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn1_output = Dot(axes=1)([turn1_seq_output, softmax_inter_right])
    turn1_output = Reshape([1, turn1_output.shape[1]])(turn1_output)
    print("First turn processing period turn is ", turn1_output)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn2_seq_output])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''# ---------------------------------------------------------------------------------------------------------------------
    # 将turn2处理过的memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn2_output = Dot(axes=1)([turn2_seq_output, softmax_inter_right])
    turn2_output = Reshape([1, turn2_output.shape[1]])(turn2_output)
    print("Second turn end processing period turn_ is ", turn2_output)

    # ---------------------------------------------------------------------------------------------------------------------
    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn3_seq_output])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    turn3_output = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    turn3_output = Reshape([1, turn3_output.shape[1]])(turn3_output)
    print("Third turn end processing period turn_ is ", turn3_output)

    combVec = Concatenate(axis=1)([turn1_output, turn2_output, turn3_output])
    # 直接使用信息融合矩阵进行预测
    print("Concatenated is ", combVec)

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(combVec)
    Num_capsule = 10
    Dim_capsule = 32
    Routings = 5
    # x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    from kfy.CapsuleNetv2 import Capsule
    x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    print("capNet processed is ", x)
    x = Flatten()(x)
    print("flattened is ", x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['acc', f1])
    return model


# 仅使用3turns进行自注意力机制和turn-wise注意力机制结合，并使用capusleNet进行预测
def sequenceDynamicMemoryLayerCap_v2():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left_turn1 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left_turn1 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left_turn1)

    print("scaledPool_inter_left ", scaledPool_inter_left_turn1)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''
    # ---------------------------------------------------------------------------------------------------------------------
    # 将turn1处理过的memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn1_output = Dot(axes=1)([turn1_seq_output, softmax_inter_right])
    turn1_output = Reshape([1, turn1_output.shape[1]])(turn1_output)
    print("First turn processing period turn is ", turn1_output)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn2_seq_output])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left_turn2 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left_turn2 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left_turn2)
    print("scaledPool_inter_left ", scaledPool_inter_left_turn2)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''# ---------------------------------------------------------------------------------------------------------------------
    # 将turn2处理过的memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn2_output = Dot(axes=1)([turn2_seq_output, softmax_inter_right])
    turn2_output = Reshape([1, turn2_output.shape[1]])(turn2_output)
    print("Second turn end processing period turn_ is ", turn2_output)

    # ---------------------------------------------------------------------------------------------------------------------
    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn3_seq_output])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left_turn3 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left_turn3 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left_turn3)
    print("scaledPool_inter_left ", scaledPool_inter_left_turn3)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    turn3_output = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    turn3_output = Reshape([1, turn3_output.shape[1]])(turn3_output)
    print("Third turn end processing period turn_ is ", turn3_output)

    combVec = Concatenate(axis=1)([turn1_output, turn2_output, turn3_output])
    # 直接使用信息融合矩阵进行预测
    print("Concatenated is ", combVec)

    combVecOnTurnLevel = Concatenate(axis=1)(
        [scaledPool_inter_left_turn1, scaledPool_inter_left_turn2, scaledPool_inter_left_turn3])
    print("ConcatenatedOnTurnLevel is ", combVecOnTurnLevel)

    weightOnTurnLevel = Dense(3, activation="softmax")(combVecOnTurnLevel)
    print('weightOnTurnLevel', weightOnTurnLevel)
    weightedCombvec = tf.keras.layers.Dot(axes=1)([combVec, weightOnTurnLevel])
    print('weightedCombvec', weightedCombvec)

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output:
    #  comboVec)
    '''
    x = tf.keras.layers.Dropout(0.15)(combVec)
    Num_capsule = 10
    Dim_capsule = 32
    Routings = 5
    #x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    from kfy.CapsuleNetv2 import Capsule
    x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    print("capNet processed is ", x)
    x = Flatten()(x)
    print("flattened is ", x)
    '''
    x = tf.keras.layers.Dense(4, activation='softmax')(weightedCombvec)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-7)

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['acc', f1])
    return model

    # 仅使用3turns进行自注意力机制和turn-wise注意力机制结合，并使用capusleNet进行预测


def sequenceDynamicMemoryLayerCap_v3():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    dropout_ratio_f_dml = dropout_ratio

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_input = np.random.random(size=(batch_size, 200, 768))

    bert_model = TFRobertaModel.from_pretrained('d:\\code\\roberta_model\\roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])
    print("input_size", interActivateVec_turn1)

    tanh_inter_left_turn1 = Tanh()(interActivateVec_turn1)
    inter_trans_turn1 = TransMatrix()(interActivateVec_turn1)
    tanh_inter_right_turn1 = Tanh()(inter_trans_turn1)

    scaledPool_inter_right_turn1 = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right_turn1)
    scaledPool_inter_right_turn1 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right_turn1)
    print("scaledPool_inter_right ", scaledPool_inter_right_turn1)

    softmax_inter_right_turn1 = Softmax()(scaledPool_inter_right_turn1)

    from tensorflow.keras.layers import RepeatVector
    memoryVector_turn1 = Dot(axes=1)([turn1_seq_output, softmax_inter_right_turn1])
    combVecMemory_turn1 = RepeatVector(200)(memoryVector_turn1)
    print("1st turn end MemoryVector is  ", combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])
    print("second period input_size", interActivateVec_turn2)

    tanh_inter_left_turn2 = Tanh()(interActivateVec_turn2)
    inter_trans_turn2 = TransMatrix()(interActivateVec_turn2)
    tanh_inter_right_turn2 = Tanh()(inter_trans_turn2)

    scaledPool_inter_right_turn2 = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right_turn2)
    scaledPool_inter_right_turn2 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right_turn2)
    print("scaledPool_inter_right ", scaledPool_inter_right_turn2)

    softmax_inter_right_turn2 = Softmax()(scaledPool_inter_right_turn2)

    from tensorflow.keras.layers import RepeatVector
    memoryVector_turn2 = Dot(axes=1)([turn2_seq_output, softmax_inter_right_turn2])
    combVecMemory_turn2 = RepeatVector(200)(memoryVector_turn2)
    print("2ed turn end MemoryVector is  ", combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])
    print("thrid period input_size", interActivateVec_turn3)

    '''
    tanh_inter_left_turn3 = Tanh()(interActivateVec_turn3)
    inter_trans_turn3 = TransMatrix()(interActivateVec_turn3)
    tanh_inter_right_turn3 = Tanh()(inter_trans_turn3)

    scaledPool_inter_left_turn3 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left_turn3)
    scaledPool_inter_left_turn3 = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left_turn3)
    print("scaledPool_inter_left ", scaledPool_inter_left_turn3)
    '''

    # softmax_inter_left = Softmax()(scaledPool_inter_left_turn3)
    bi_out = Bidirectional(LSTM(128, return_sequences=False))(interActivateVec_turn3)
    x_1 = tf.keras.layers.Dense(64, activation='softmax')(bi_out)
    # x_2 = tf.keras.layers.Dense(16, activation='softmax')(x_1)
    # x_3 = tf.keras.layers.Dense(26, activation='softmax')(x_2)
    x_2 = tf.keras.layers.Dense(4, activation='softmax')(x_1)
    print(x_2)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x_2)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model


# 不使用interActivate模块
def ablationInterAct_InterActivateDynamicMemoryLayerCap():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]

    combVec = Concatenate(axis=1)([turn1_seq_output, turn2_seq_output, turn3_seq_output])
    # 直接使用信息融合矩阵进行预测
    print("Concatenated is ", combVec)

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.3)(combVec)
    Num_capsule = 10
    Dim_capsule = 32
    Routings = 5
    # x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    from kfy.CapsuleNetv2 import Capsule
    x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    print("capNet processed is ", x)
    x = Flatten()(x)
    print("flattened is ", x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['acc', f1])
    return model


# 不使用interActivate模块，使用LSTM
def ablation_Caps_InterActivateDynamicMemoryLayerCap():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]

    combVec = Concatenate(axis=1)([turn1_seq_output, turn2_seq_output, turn3_seq_output])
    # 直接使用信息融合矩阵进行预测
    print("Concatenated is ", combVec)

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(combVec)

    x = LSTM(hidden_dim, recurrent_dropout=0.15, return_sequences=False)(x)

    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6)

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['acc', f1])
    return model


# 使用interActivate及Dense进行预测
def ablation_InterActivateDenseCap():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768

    # 产生memoryVector的初始化矩阵

    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    memoryVecor_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合
    memoryVector = bert_model(memoryVector_id, attention_mask=memoryVecor_mask)
    memoryVector = memoryVector[0]

    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn1_seq_output])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''
    # ---------------------------------------------------------------------------------------------------------------------
    # 将turn1处理过的memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn1_output = Dot(axes=1)([turn1_seq_output, softmax_inter_right])
    turn1_output = Reshape([1, turn1_output.shape[1]])(turn1_output)
    print("First turn processing period turn is ", turn1_output)

    # 开始第二轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn2_seq_output])
    print("second period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_right = Softmax()(scaledPool_inter_right)

    '''# ---------------------------------------------------------------------------------------------------------------------
    # 将turn2处理过的memoryVector和turn3进行融合
    # 对中间向量softmax_inter_right进行行扩展
    extended_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
    for i in range(softmax_inter_right.shape[1] - 1):
        reshaped_softmax_inter_right = Reshape([1, softmax_inter_right.shape[1]])(softmax_inter_right)
        extended_softmax_inter_right = Concatenate(axis=1)([extended_softmax_inter_right, reshaped_softmax_inter_right])
    '''
    turn2_output = Dot(axes=1)([turn2_seq_output, softmax_inter_right])
    turn2_output = Reshape([1, turn2_output.shape[1]])(turn2_output)
    print("Second turn end processing period turn_ is ", turn2_output)

    # ---------------------------------------------------------------------------------------------------------------------
    # 开始第三轮融合
    interActivateVec = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector, turn3_seq_output])
    print("thrid period input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_f_dml)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=hidden_dim)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH_f_dml,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    turn3_output = Dot(axes=1)([turn3_seq_output, softmax_inter_right])
    turn3_output = Reshape([1, turn3_output.shape[1]])(turn3_output)
    print("Third turn end processing period turn_ is ", turn3_output)

    combVec = Concatenate(axis=1)([turn1_output, turn2_output, turn3_output])
    # 直接使用信息融合矩阵进行预测
    print("Concatenated is ", combVec)

    # output_lstm = Bidirectional(LSTM(128, return_sequences=True))(tanhInterAct)
    # print("capsule output: ", comboVec)
    x = tf.keras.layers.Dropout(0.15)(combVec)
    Num_capsule = 10
    Dim_capsule = 32
    Routings = 5
    # x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    # from kfy.CapsuleNetv2 import Capsule
    # x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    # print("capNet processed is ", x)
    x = Flatten()(x)
    print("flattened is ", x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask, memoryVector_id, memoryVecor_mask], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=optimizer, metrics=['acc', f1])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['acc', f1])
    return model


# _____________________________________________________________________________________________
def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev, y_test = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)
            y_test.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    y_test = np.array(y_test)

    return [X_train, X_test, X_dev, y_train, y_dev, y_test]


def get_feature(pickle_path):
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_path, 'rb'))
    maxlen = 165
    X_train, X_test, X_dev, y_train, y_dev, y_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]

    n_test_sample = X_test.shape[0]

    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400

    return maxlen, max_features, num_features, W, X_train, y_train, X_dev, y_dev, X_test, y_test


turn1_pickle = "..\\pickle\\turn1_periodical.pickle3"
turn2_pickle = "..\\pickle\\turn2_periodical.pickle3"
turn3_pickle = "..\\pickle\\turn3_periodical.pickle3"
from tensorflow.keras.layers import Add, Multiply, BatchNormalization, Dense, Dropout, Embedding, LSTM, Cropping1D, GRU, \
    Bidirectional, Input, Flatten, Convolution1D, MaxPooling1D, Dot


def ablationRoberta():
    dropout_rate = 0.01
    Routings = 3  # 更改
    Num_capsule = 6

    turn1_maxlen, turn1_max_features, turn1_num_features, turn1_W, turn1_X_train, turn1_y_train, turn1_X_dev, turn1_y_dev, turn1_x_test, turn1_y_test = get_feature(
        turn1_pickle)
    turn2_maxlen, turn2_max_features, turn2_num_features, turn2_W, turn2_X_train, turn2_y_train, turn2_X_dev, turn2_y_dev, turn2_x_test, turn2_y_test = get_feature(
        turn2_pickle)
    turn3_maxlen, turn3_max_features, turn3_num_features, turn3_W, turn3_X_train, turn3_y_train, turn3_X_dev, turn3_y_dev, turn3_x_test, turn3_y_test = get_feature(
        turn3_pickle)

    # 产生随机数作为MemoryVector的初始化输入
    memoryVector = [np.random.randint(0, 20, size=(turn1_maxlen, hidden_dim))]

    turn1_sequence = Input(shape=(turn1_maxlen,), dtype='int32')
    turn1_embedded = Embedding(input_dim=turn1_max_features, output_dim=turn1_num_features, input_length=turn1_maxlen,
                               weights=[turn1_W], trainable=False)(turn1_sequence)
    turn1_embedded = Dropout(0.01)(turn1_embedded)
    # bi-lstm
    turn1_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(
        turn1_embedded)
    turn1_enc = GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True)(turn1_embedded)

    # left_capsule = Flatten()(left_capsule)

    turn2_sequence = Input(shape=(turn2_maxlen,), dtype='int32')
    turn2_embedded = Embedding(input_dim=turn2_max_features, output_dim=turn2_num_features,
                               input_length=turn2_maxlen,
                               weights=[turn2_W], trainable=False)(turn2_sequence)
    turn2_embedded = Dropout(dropout_rate)(turn2_embedded)
    turn2_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(
        turn2_embedded)
    turn2_enc = GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True)(turn2_embedded)

    # left_capsule = Flatten()(left_capsule)
    turn3_sequence = Input(shape=(turn3_maxlen,), dtype='int32')
    turn3_embedded = Embedding(input_dim=turn3_max_features, output_dim=turn3_num_features,
                               input_length=turn3_maxlen,
                               weights=[turn3_W], trainable=False)(turn3_sequence)
    turn3_embedded = Dropout(dropout_rate)(turn3_embedded)
    turn3_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(
        turn3_embedded)
    turn3_enc = GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True)(turn3_embedded)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    # right_capsule = Flatten()(right_capsule)

    # comboVec = Concatenate(axis=1)([left_enc, right_enc])
    from kfy.interActivateLayer_recurrent import interActivate, Tanh, TransMatrix

    interActivateVec = interActivate(hidden_dims=hidden_dim)([memoryVector, turn1_enc])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=165)(tanh_inter_left)
    scaledPool_inter_left = Reshape((165,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_left_turn1 = scaledPool_inter_left

    scaledPool_inter_right = MaxPooling1D(pool_size=165)(tanh_inter_right)
    scaledPool_inter_right = Reshape((165,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    softmax_inter_left = Dot(axes=1)([turn1_enc, softmax_inter_left])
    print("softmax_inter_left", softmax_inter_left, turn1_enc)
    turn1_output = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)

    # _________________________________________________________________________________________
    # 第二轮
    interActivateVec = interActivate(hidden_dims=hidden_dim)([memoryVector, turn2_enc])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=165)(tanh_inter_left)
    scaledPool_inter_left = Reshape((165,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_left_turn2 = scaledPool_inter_left

    scaledPool_inter_right = MaxPooling1D(pool_size=165)(tanh_inter_right)
    scaledPool_inter_right = Reshape((165,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    softmax_inter_left = Dot(axes=1)([turn2_enc, softmax_inter_left])
    turn2_output = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)
    print("softmax_inter_left", softmax_inter_left, turn2_enc)

    # 第三轮
    interActivateVec = interActivate(hidden_dims=hidden_dim)([memoryVector, turn3_enc])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=165)(tanh_inter_left)
    scaledPool_inter_left = Reshape((165,))(scaledPool_inter_left)
    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_left_turn3 = scaledPool_inter_left

    scaledPool_inter_right = MaxPooling1D(pool_size=165)(tanh_inter_right)
    scaledPool_inter_right = Reshape((165,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    softmax_inter_left = Dot(axes=1)([turn3_enc, softmax_inter_left])
    print("softmax_inter_left", softmax_inter_left, turn3_enc)
    turn3_output = Reshape([1, softmax_inter_left.shape[1]])(softmax_inter_left)

    # _____________________________________________________________________________________

    combVecOnTurnLevel = Concatenate(axis=1)(
        [scaledPool_inter_left_turn1, scaledPool_inter_left_turn2, scaledPool_inter_left_turn3])
    print("ConcatenatedOnTurnLevel is ", combVecOnTurnLevel)

    weightOnTurnLevel = Dense(3, activation="softmax")(combVecOnTurnLevel)

    # 将自注意力处理过的turn1与对应的contextAttention进行加权
    weightOnTurnLevel_turn1 = weightOnTurnLevel[:, 0]
    weightOnTurnLevel_turn1 = Reshape([1, 1])(weightOnTurnLevel_turn1)
    weightOnTurnLevel_turn1 = Dot(axes=1)([turn1_output, weightOnTurnLevel_turn1])
    weightOnTurnLevel_turn1 = Reshape([1, 768])(weightOnTurnLevel_turn1)
    print('weightOnTurnLevel_turn1', weightOnTurnLevel_turn1)

    # 将自注意力处理过的turn1与对应的contextAttention进行加权
    weightOnTurnLevel_turn2 = weightOnTurnLevel[:, 1]
    weightOnTurnLevel_turn2 = Reshape([1, 1])(weightOnTurnLevel_turn2)
    weightOnTurnLevel_turn2 = Dot(axes=1)([turn2_output, weightOnTurnLevel_turn2])
    weightOnTurnLevel_turn2 = Reshape([1, 768])(weightOnTurnLevel_turn2)
    print('weightOnTurnLevel_turn2', weightOnTurnLevel_turn2)

    # 将自注意力处理过的turn1与对应的contextAttention进行加权
    weightOnTurnLevel_turn3 = weightOnTurnLevel[:, 2]
    weightOnTurnLevel_turn3 = Reshape([1, 1])(weightOnTurnLevel_turn3)
    weightOnTurnLevel_turn3 = Dot(axes=1)([turn3_output, weightOnTurnLevel_turn3])
    weightOnTurnLevel_turn3 = Reshape([1, 768])(weightOnTurnLevel_turn3)
    print('weightOnTurnLevel_turn3', weightOnTurnLevel_turn3)

    combVecOnTurnLevel = Concatenate(axis=1)(
        [weightOnTurnLevel_turn1, weightOnTurnLevel_turn2, weightOnTurnLevel_turn3])
    print("weighted_ConcatenatedOnTurnLevel is ", combVecOnTurnLevel)

    comboVec = GRU(hidden_dim)(combVecOnTurnLevel)

    output = Dense(6, activation="softmax")(comboVec)
    print("output: ", output)

    model = Model(inputs=[turn1_sequence, turn2_sequence, turn3_sequence], outputs=output)

    return model


# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerCap_v5():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    head_amount_f_dml = head_amount
    dropout_ratio_f_dml = dropout_ratio
    print("dropout: "+(str(dropout_ratio_f_dml))+";head_amout: "+str(head_amount_f_dml))

    # 产生memoryVector的初始化矩阵
    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    #memoryVector_input = np.random.random(size=(batch_size, 200, 768))
    memoryVector_input = np.random.uniform(0,1,size=(batch_size, 200, 768))

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])

    softmaxInteractiveFeature_turn1 = Softmax()(interActivateVec_turn1)
    combVecMemory_turn1 = Dot(axes=1)([softmaxInteractiveFeature_turn1, turn1_seq_output])
    combVecMemory_turn1 = Tanh()(combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])

    softmaxInteractiveFeature_turn2 = Softmax()(interActivateVec_turn2)
    combVecMemory_turn2 = Dot(axes=1)([softmaxInteractiveFeature_turn2, turn2_seq_output])
    combVecMemory_turn2 = Tanh()(combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])

    # '''
    softmaxInteractiveFeature_turn3 = Softmax()(interActivateVec_turn3)
    combVecMemory_turn3 = Dot(axes=1)([softmaxInteractiveFeature_turn3, turn3_seq_output])
    combVecMemory_turn3 = Tanh()(combVecMemory_turn3)
    #'''

    from kfy.transformer import TransformerBlock
    x = TransformerBlock(768, head_amount_f_dml, 128)(combVecMemory_turn3)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    #'''
    x = Bidirectional(LSTM(32,return_sequences=False))(combVecMemory_turn3)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)

    # '''
    x = tf.keras.layers.Dense(32, activation='softmax')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    print(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model

# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerCap_Hingeloss():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    head_amount_f_dml = head_amount
    dropout_ratio_f_dml = dropout_ratio
    print("dropout: "+(str(dropout_ratio_f_dml))+";head_amout: "+str(head_amount_f_dml))

    # 产生memoryVector的初始化矩阵
    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    #memoryVector_input = np.random.random(size=(batch_size, 200, 768))
    memoryVector_input = np.random.uniform(0,1,size=(batch_size, 200, 768))

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])

    softmaxInteractiveFeature_turn1 = Softmax()(interActivateVec_turn1)
    combVecMemory_turn1 = Dot(axes=1)([softmaxInteractiveFeature_turn1, turn1_seq_output])
    combVecMemory_turn1 = Tanh()(combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])

    softmaxInteractiveFeature_turn2 = Softmax()(interActivateVec_turn2)
    combVecMemory_turn2 = Dot(axes=1)([softmaxInteractiveFeature_turn2, turn2_seq_output])
    combVecMemory_turn2 = Tanh()(combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])

    # '''
    softmaxInteractiveFeature_turn3 = Softmax()(interActivateVec_turn3)
    combVecMemory_turn3 = Dot(axes=1)([softmaxInteractiveFeature_turn3, turn3_seq_output])
    combVecMemory_turn3 = Tanh()(combVecMemory_turn3)


    from kfy.transformer import TransformerBlock
    x = TransformerBlock(768, head_amount_f_dml, 128)(combVecMemory_turn3)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)

    x = GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)

    # '''
    x = tf.keras.layers.Dense(32, activation='softmax')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    print(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_hinge', optimizer=optimizer, metrics=['acc', f1])
    return model


# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerCap_Hingeloss_ablation_transToLSTM():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    head_amount_f_dml = head_amount
    dropout_ratio_f_dml = dropout_ratio
    print("dropout: " + (str(dropout_ratio_f_dml)) + ";head_amout: " + str(head_amount_f_dml))

    # 产生memoryVector的初始化矩阵
    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    # memoryVector_input = np.random.random(size=(batch_size, 200, 768))
    memoryVector_input = np.random.uniform(0, 1, size=(batch_size, 200, 768))

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])

    softmaxInteractiveFeature_turn1 = Softmax()(interActivateVec_turn1)
    combVecMemory_turn1 = Dot(axes=1)([softmaxInteractiveFeature_turn1, turn1_seq_output])
    combVecMemory_turn1 = Tanh()(combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])

    softmaxInteractiveFeature_turn2 = Softmax()(interActivateVec_turn2)
    combVecMemory_turn2 = Dot(axes=1)([softmaxInteractiveFeature_turn2, turn2_seq_output])
    combVecMemory_turn2 = Tanh()(combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])

    # '''
    softmaxInteractiveFeature_turn3 = Softmax()(interActivateVec_turn3)
    combVecMemory_turn3 = Dot(axes=1)([softmaxInteractiveFeature_turn3, turn3_seq_output])
    combVecMemory_turn3 = Tanh()(combVecMemory_turn3)
    '''

    from kfy.transformer import TransformerBlock
    x = TransformerBlock(768, head_amount_f_dml, 128)(combVecMemory_turn3)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    '''

    x = LSTM(32,return_sequences=False)(combVecMemory_turn3)


    x = tf.keras.layers.Dropout(dropout_ratio)(x)

    # '''
    x = tf.keras.layers.Dense(32, activation='softmax')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    print(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_hinge', optimizer=optimizer, metrics=['acc', f1])
    return model

# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerCap_Hingeloss_ablation_transToCapslue():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    head_amount_f_dml = head_amount
    dropout_ratio_f_dml = dropout_ratio
    print("dropout: "+(str(dropout_ratio_f_dml))+";head_amout: "+str(head_amount_f_dml))

    # 产生memoryVector的初始化矩阵
    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    #memoryVector_input = np.random.random(size=(batch_size, 200, 768))
    memoryVector_input = np.random.uniform(0,1,size=(batch_size, 200, 768))

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])

    softmaxInteractiveFeature_turn1 = Softmax()(interActivateVec_turn1)
    combVecMemory_turn1 = Dot(axes=1)([softmaxInteractiveFeature_turn1, turn1_seq_output])
    combVecMemory_turn1 = Tanh()(combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])

    softmaxInteractiveFeature_turn2 = Softmax()(interActivateVec_turn2)
    combVecMemory_turn2 = Dot(axes=1)([softmaxInteractiveFeature_turn2, turn2_seq_output])
    combVecMemory_turn2 = Tanh()(combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])

    # '''
    softmaxInteractiveFeature_turn3 = Softmax()(interActivateVec_turn3)
    combVecMemory_turn3 = Dot(axes=1)([softmaxInteractiveFeature_turn3, turn3_seq_output])
    combVecMemory_turn3 = Tanh()(combVecMemory_turn3)
    '''

    from kfy.transformer import TransformerBlock
    x = TransformerBlock(768, head_amount_f_dml, 128)(combVecMemory_turn3)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    
    x = LSTM(32,return_sequences=False)(combVecMemory_turn3)
    #'''
    from CapsuleNetv2 import Capsule
    x = Capsule(num_capsule=200, dim_capsule=16)(combVecMemory_turn3)
    x = GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)

    # '''
    x = tf.keras.layers.Dense(32, activation='softmax')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    print(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_hinge', optimizer=optimizer, metrics=['acc', f1])
    return model


# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerTransWLstm_v5():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    dropout_ratio_f_dml = dropout_ratio
    head_amount_f_dml = 4

    # 产生memoryVector的初始化矩阵
    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    memoryVector_input = np.random.random(size=(batch_size, 200, 768))

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])

    softmaxInteractiveFeature_turn1 = Softmax()(interActivateVec_turn1)
    combVecMemory_turn1 = Dot(axes=1)([softmaxInteractiveFeature_turn1, turn1_seq_output])
    combVecMemory_turn1 = Tanh()(combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])

    softmaxInteractiveFeature_turn2 = Softmax()(interActivateVec_turn2)
    combVecMemory_turn2 = Dot(axes=1)([softmaxInteractiveFeature_turn2, turn2_seq_output])
    combVecMemory_turn2 = Tanh()(combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])

    # '''
    softmaxInteractiveFeature_turn3 = Softmax()(interActivateVec_turn3)
    combVecMemory_turn3 = Dot(axes=1)([softmaxInteractiveFeature_turn3, turn3_seq_output])
    combVecMemory_turn3 = Tanh()(combVecMemory_turn3)
    # '''



    from kfy.transformer import TransformerBlock
    x = TransformerBlock(768, head_amount_f_dml, 128)(combVecMemory_turn3 )
    print("capNet processed is ", x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    #x = GlobalAveragePooling1D()(x)
    x = LSTM(32,return_sequences=False)(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    print("flattened is ", x)
    # '''
    x = tf.keras.layers.Dense(32, activation='softmax')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    print(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model


# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerCapsule_v1():
    MAX_SEQUENCE_LENGTH_f_dml = 200
    hidden_dim_f_dml = 768
    dropout_ratio_f_dml = dropout_ratio
    head_amount_f_dml = head_amount

    # 产生memoryVector的初始化矩阵
    turn1_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn1_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn2_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn2_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    turn3_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)
    turn3_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH_f_dml,), dtype=tf.int32)

    #memoryVector_input = np.random.random(size=(batch_size, 200, 768))
    memoryVector_input = np.random.randn(batch_size, 200, 768)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    # 产生turn1的bert模型
    turn1_seq_output = bert_model(turn1_id, attention_mask=turn1_mask)
    turn1_seq_output = turn1_seq_output[0]
    turn1_seq_output = Dropout(dropout_ratio_f_dml)(turn1_seq_output)

    # 产生turn2的bert模型
    turn2_seq_output = bert_model(turn2_id, attention_mask=turn2_mask)
    turn2_seq_output = turn2_seq_output[0]
    turn2_seq_output = Dropout(dropout_ratio_f_dml)(turn2_seq_output)

    # 产生turn3的bert模型
    turn3_seq_output = bert_model(turn3_id, attention_mask=turn3_mask)
    turn3_seq_output = turn3_seq_output[0]
    turn3_seq_output = Dropout(dropout_ratio_f_dml)(turn3_seq_output)

    # -------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn1进行融合

    from kfy.interActivateLayer_recurrent_v3 import interActivate, Tanh, TransMatrix
    interActivateVec_turn1 = interActivate(hidden_dims=hidden_dim_f_dml)([memoryVector_input, turn1_seq_output])

    softmaxInteractiveFeature_turn1 = Softmax()(interActivateVec_turn1)
    combVecMemory_turn1 = Dot(axes=1)([softmaxInteractiveFeature_turn1, turn1_seq_output])
    combVecMemory_turn1 = Tanh()(combVecMemory_turn1)

    # ---------------------------------------------------------------------------------------------------------------------
    # 将第一个memoryVector和turn2进行融合
    # 对中间向量softmax_inter_right进行行扩展

    # 开始第二轮融合
    interActivateVec_turn2 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn1, turn2_seq_output])

    softmaxInteractiveFeature_turn2 = Softmax()(interActivateVec_turn2)
    combVecMemory_turn2 = Dot(axes=1)([softmaxInteractiveFeature_turn2, turn2_seq_output])
    combVecMemory_turn2 = Tanh()(combVecMemory_turn2)

    # 开始第三轮融合
    interActivateVec_turn3 = interActivate(hidden_dims=hidden_dim_f_dml)([combVecMemory_turn2, turn3_seq_output])

    # '''
    softmaxInteractiveFeature_turn3 = Softmax()(interActivateVec_turn3)
    combVecMemory_turn3 = Dot(axes=1)([softmaxInteractiveFeature_turn3, turn3_seq_output])
    combVecMemory_turn3 = Tanh()(combVecMemory_turn3)


    from kfy.transformer import TransformerBlock
    x = TransformerBlock(768, head_amount_f_dml,128)(combVecMemory_turn3)

    '''
    print("capNet processed is ", x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    print("flattened is ", x)
    '''
    from kfy.CapsuleNetv2 import Capsule
    x = Capsule(num_capsule=64, dim_capsule=8)(x)
    x = GlobalAveragePooling1D()(x)


    x = tf.keras.layers.Dense(32, activation='softmax')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    print(x)

    model = tf.keras.models.Model(inputs=[turn1_id, turn1_mask, turn2_id, turn2_mask,
                                          turn3_id, turn3_mask], outputs=x)

    # from tensorflow_addons.optimizers import AdamW
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_ratio)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model



def get_attention(sent_model, sequences):
    cnt_reviews_1 = sequences[0].shape[0]
    sent_att_w_1 = sent_model.layers[22].get_weights()
    sent_all_att_1 = []
    sent_before_att_1 = K.function([sent_model.layers[0].input, sent_model.layers[1].input, K.learning_phase()],
                                   [sent_model.layers[22].input])
    # print(sent_att_w[0].shape, sent_att_w[1].shape, sent_att_w[2].shape)

    for i in range(cnt_reviews_1):
        sent_each_att_1 = sent_before_att_1([sequences[0][i].reshape((1, sequences[0].shape[-1])),
                                             sequences[1][i].reshape((1, sequences[1].shape[-1])), 0])
        mask_1 = [0 if w == 0 else 1 for w in sequences[0][i]]

        sent_each_att_1 = cal_att_weights(sent_each_att_1, sent_att_w_1, mask_1)
        sent_each_att_1 = sent_each_att_1.ravel()
        sent_all_att_1.append(sent_each_att_1)

    return [sent_all_att_1]


def cal_att_weights(output, att_w, mask):
    eij = np.tanh(np.dot(output[0], att_w[0]) + att_w[1])
    eij = np.squeeze(eij * att_w[1])
    # eij = eij.reshape((eij.shape[0], eij.shape[1]))
    ai = np.exp(eij)
    weights = ai / np.sum(ai)
    weights = weights * mask
    return weights


# 第一步，读取格式文件0vg
# 格式：输入分为三段每段都是batch X seq_len X embSize

import pickle

if __name__ == '__main__':
    # 6G RTX1060 每轮约需1个小时

    # load train/dev/test test
    # 读取语料信息
    TRAIN_PATH = '../data/singleTurns/train.txt'
    DEV_PATH = '../data/singleTurns/dev.txt'
    TEST_PATH = '../data/singleTurns/test.txt'
    train_data = pd.read_table(TRAIN_PATH, sep='\t')
    dev_data = pd.read_table(DEV_PATH, sep='\t')
    test_data = pd.read_table(TEST_PATH, sep='\t')

    # 读取train, dev, test各阶段语料库的内容
    turn1_x_train = [i_list for i_list in train_data['turn1']]
    turn2_x_train = [i_list for i_list in train_data['turn2']]
    turn3_x_train = [i_list for i_list in train_data['turn3']]

    turn1_x_dev = [i_list for i_list in dev_data['turn1']]
    turn2_x_dev = [i_list for i_list in dev_data['turn2']]
    turn3_x_dev = [i_list for i_list in dev_data['turn3']]

    turn1_x_test = [i_list for i_list in test_data['turn1']]
    turn2_x_test = [i_list for i_list in test_data['turn2']]
    turn3_x_test = [i_list for i_list in test_data['turn3']]

    # the length of different sets
    # 统计各长度的语句的数量，并显示
    trainLenCounter = Counter(
        [len(i) for i in turn1_x_train] + [len(i) for i in turn2_x_train] + [len(i) for i in turn3_x_train])
    devLenCounter = Counter(
        [len(i) for i in turn1_x_dev] + [len(i) for i in turn2_x_dev] + [len(i) for i in turn3_x_dev])
    testLenCounter = Counter(
        [len(i) for i in turn1_x_test] + [len(i) for i in turn2_x_test] + [len(i) for i in turn3_x_test])
    # 获得数据集中每个长度出现的次数
    trainLenFre = sorted(dict(trainLenCounter).items(), key=lambda d: d[0], reverse=True)
    # 长度大于某个数值的比例,
    # 在train中长度大于120的比例为0.00087，大于100的比例为0.0020
    lengthRatio_train = sum([value_list for key_list, value_list in trainLenFre if key_list > 100]) / (
                3 * len(turn1_x_train))

    # 在dev中长度大于120的比例为0.00097，大于100的比例为0.0021
    devLenFre = sorted(dict(devLenCounter).items(), key=lambda d: d[0], reverse=True)
    lengthRatio_dev = sum([value_list for key_list, value_list in devLenFre if key_list > 100]) / (3 * len(turn1_x_dev))

    # 在test中长度大于120的比例为0.00048，大于100的比例为0.0012
    testLenFre = sorted(dict(testLenCounter).items(), key=lambda d: d[0], reverse=True)
    lengthRatio_test = sum([value_list for key_list, value_list in testLenFre if key_list > 100]) / (
                3 * len(turn1_x_test))

    TRUNCATED_LENGTH = 200

    # 得到数据中最长的序列长度
    max_train_lens = max(max([len(i) for i in turn1_x_train]), max([len(i) for i in turn2_x_train]),
                         max([len(i) for i in turn3_x_train]))

    max_dev_lens = max(max([len(i) for i in turn1_x_dev]), max([len(i) for i in turn2_x_dev]),
                       max([len(i) for i in turn3_x_dev]))
    max_test_lens = max(max([len(i) for i in turn1_x_test]), max([len(i) for i in turn2_x_test]),
                        max([len(i) for i in turn3_x_test]))

    max_seq_len = max(max_train_lens, max_dev_lens, max_test_lens)

    # 获取输入中的y
    # 读取原存储数据中的y
    emotion2label = {'sad': 0, 'happy': 1, 'angry': 2, 'others': 3}
    y_train = np_utils.to_categorical(np.array([emotion2label[i_list] for i_list in train_data['label']]))
    y_dev = np_utils.to_categorical(np.array([emotion2label[i_list] for i_list in dev_data['label']]))
    y_test = np.array([emotion2label[i_list] for i_list in test_data['label']])

    # --------------------------------------------------------------------------------------------------------------
    '''
    print("开始分别取出左右两侧")
    #生成bert词向量
    #left bertEmbedding
    #input_type means 0:left_train
    # 生成bert Embedding
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", pad_token='[PAD]')
    # 产生各个集合的不同输入序列
    turn1_train_inputs = compute_input_arrays(turn1_x_train, tokenizer, TRUNCATED_LENGTH)
    turn2_train_inputs = compute_input_arrays(turn2_x_train, tokenizer, TRUNCATED_LENGTH)
    turn3_train_inputs = compute_input_arrays(turn3_x_train, tokenizer, TRUNCATED_LENGTH)
    print("完成train集合")


    turn1_dev_inputs = compute_input_arrays(turn1_x_dev, tokenizer, TRUNCATED_LENGTH)
    turn2_dev_inputs = compute_input_arrays(turn2_x_dev, tokenizer, TRUNCATED_LENGTH)
    turn3_dev_inputs = compute_input_arrays(turn3_x_dev, tokenizer, TRUNCATED_LENGTH)


    turn1_test_inputs = compute_input_arrays(turn1_x_test, tokenizer, TRUNCATED_LENGTH)
    turn2_test_inputs = compute_input_arrays(turn2_x_test, tokenizer, TRUNCATED_LENGTH)
    turn3_test_inputs = compute_input_arrays(turn3_x_test, tokenizer, TRUNCATED_LENGTH)


    file = open('../data/singleTurns/pickle/roberta/context_bert_chinese.pickle', 'wb')
    pickle.dump([turn1_train_inputs, turn1_dev_inputs,turn1_test_inputs, turn2_train_inputs,turn2_dev_inputs,
                 turn2_test_inputs, turn3_train_inputs, turn3_dev_inputs, turn3_test_inputs], file)
    file.close()
    print("bert tokenizer has finished!!!")
    '''
    # --------------------------------------------------------------------------------------------------------------

    print("3 turns的预处理工作已经处理完成")

    pickle_file = open('../data/singleTurns/pickle/roberta/context_bert_chinese.pickle', 'rb')
    turn1_train_inputs, turn1_dev_inputs, turn1_test_inputs, turn2_train_inputs, turn2_dev_inputs, \
    turn2_test_inputs, turn3_train_inputs, turn3_dev_inputs, turn3_test_inputs = pickle.load(pickle_file)

    x_train_inputs = turn1_train_inputs + turn2_train_inputs + turn3_train_inputs
    x_dev_inputs = turn1_dev_inputs + turn2_dev_inputs + turn3_dev_inputs
    x_test_inputs = turn1_test_inputs + turn2_test_inputs + turn3_test_inputs

    x_dev_inputs = [i[0:2700] for i in x_dev_inputs[0:2700]]
    y_dev = y_dev[0:2700]

    x_test_inputs = [i[0:5500] for i in x_test_inputs[0:5500]]
    y_test = y_test[0:5500]

    # ---------------------------------------------------------------------------------------------------------------
    dropout_ratioList = [0.0,0.1, 0.2, 0.3, 0.4, 0.5]
    head_amount_list = [2,4,8,16,32,64,128]
    epoch_list = [20]
    #选定参数8head,0.1dropout,测试不同轮数下的成绩
    for i in dropout_ratioList:
        print("开始训练")
        dropout_ratio = i
        head_amount = 5
        model = sequenceDynamicMemoryLayerCap_v5()
        #model = sequenceDynamicMemoryLayerTransWLstm_v5()
        model = sequenceDynamicMemoryLayerCap_Hingeloss_ablation_transToCapslue()
        print("model summary:",model.summary())
        print("model layers description:", model.layers)

        # 训练模型
        early_stopping = EarlyStopping(monitor='acc', patience=3)

        model.fit(x_train_inputs, y_train,
                  validation_data=(x_dev_inputs, y_dev),
                  batch_size=batch_size, epochs=20,
                  #callbacks=[early_stopping],
                  verbose=1)  # verbose=0不输出日志信息;verbose=1逐条输出日志信息；verbose=2输出evlaue日志信息。

        # 使用模型进行预测
        y_pred = model.predict(x_test_inputs, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)

        result_output = pd.DataFrame(data={'test_sentiment': y_test, "sentiment": y_pred})
        # # Use pandas to write the comma-separated output file
        result_save_path = "../result/V2_5e-7_context_bert_wordEmbedding.csv"
        result_output.to_csv(result_save_path, index=False, quoting=3)
        result_outputStr = "trans_5e-7_focalLoss:" + ",the micro f1 score is: " + str(
            task3returnMicroF1(result_save_path))
        print(result_outputStr)
        result_save_path_1 = "../result/tanh_transCap_4head_lr"+str(learning_ratio )+"dr_0.1_context_bert_wordEmbedding_epochs" + str(i) + "_f1value" + str(
            task3returnMicroF1(result_save_path)) + ".csv"
        result_output.to_csv(result_save_path_1, index=False, quoting=3)

