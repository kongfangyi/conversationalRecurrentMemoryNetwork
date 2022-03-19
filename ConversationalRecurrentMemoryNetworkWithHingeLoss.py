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
head_amount = 2
learning_ratio = 2e-6
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

# 更新时直接用interActive变量进行更新
def sequenceDynamicMemoryLayerCap_Hingeloss_ablation_transToCapslue():
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
    epoch_list = [8]
    #选定参数8head,0.1dropout,测试不同轮数下的成绩
    for i in epoch_list:
        print("开始训练")
        dropout_ratio = 0.5
        head_amount = 4
        model = sequenceDynamicMemoryLayerCap_v5()
        #model = sequenceDynamicMemoryLayerTransWLstm_v5()
        model = sequenceDynamicMemoryLayerCap_Hingeloss_ablation_tanh()
        model = sequenceDynamicMemoryLayerCap_Hingeloss()
        print("model summary:",model.summary())
        print("model layers description:", model.layers)

        # 训练模型
        early_stopping = EarlyStopping(monitor='acc', patience=3)

        model.fit(x_train_inputs, y_train,
                  validation_data=(x_dev_inputs, y_dev),
                  batch_size=batch_size, epochs=i,
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



