#!/usr/bin/env python
# author:David
# Email: david_captain@sina.com
# datetime:2019/5/11 16:09
# software: PyCharm

import re
import pandas as pd
import numpy as np
import gensim
import pickle
import jieba

from collections import defaultdict

label2emotion = {0: 'sad', 1: 'happy', 2: 'angry', 3: 'others'}
emotion2label = {'sad': 0, 'happy': 1, 'angry': 2, 'others': 3}



train = pd.read_csv('../data/connected/train.txt', sep='\t')
dev = pd.read_csv('../data/connected/dev.txt', sep='\t')
test = pd.read_csv('../data/connected/test.txt', sep='\t')


def review_to_wordlist(review):
    # jieba.load_userdict('E:/python/chinese_classify/dict/tool_dict.txt')
    repeatedChars = ['.', '?', '!', ',', '。', '，', '！', '？']
    for c in repeatedChars:
        lineSplit = review.split(c)
        while True:
            try:
                lineSplit.remove('')
            except:
                break
        line = c.join(lineSplit)
    line = jieba.cut(line.strip())
    word = " ".join(line)
    word = re.sub('\\u200b', '', word)
    word = word.split()

    return word

# 1 train  0 dev -1 test
def build_data_train_dev_test(data_train, data_dev, data_test):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)

    # Pre-process train dataset
    for i in range(len(data_train)):
        rev = data_train[i]
        y = train['label'][i]
        orig_rev = ' '.join(rev)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': emotion2label[y],
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 1}
        revs.append(datum)

    # Pre-process dev dataset
    for i in range(len(data_dev)):
        rev = data_dev[i]
        y = dev['label'][i]
        orig_rev = ' '.join(rev)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': emotion2label[y],
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 0}
        revs.append(datum)

    # Pre-process train test dataset
    for i in range(len(data_test)):
        rev = data_test[i]
        y = test['label'][i]
        orig_rev = ' '.join(rev)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': emotion2label[y],
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab


def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    return word_vecs


def get_W(word_vecs, k=200):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':

    clean_train_reviews = []
    for review in train["turn1"]:
        clean_train_reviews.append(review_to_wordlist(review))

    clean_dev_reviews = []
    for review in dev["turn1"]:
        clean_dev_reviews.append(review_to_wordlist(review))

    clean_test_reviews = []
    for review in test["turn1"]:
        clean_test_reviews.append(review_to_wordlist(review))

    print(len(clean_train_reviews))


    file = open('../data/connected/pickle/word_list.pickle', 'wb')
    pickle.dump([clean_train_reviews, clean_dev_reviews, clean_test_reviews], file)
    file.close()

    file = open('../data/connected/pickle/word_list.pickle', 'rb')
    train_word, dev_word, test_word = pickle.load(file)
    file.close()

    revs, vocab = build_data_train_dev_test(train_word, dev_word, test_word)
    #最长的句子长度。
    max_l = np.max(pd.DataFrame(revs)['num_words'])

    # # Glove Common Crawl
    model_file = 'D:/data/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, encoding='utf-8')

    w2v = load_bin_vec(model, vocab)

    W, word_idx_map = get_W(w2v, k=model.vector_size)

    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = '../data/connected/pickle/periodical.pickle3'
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))




