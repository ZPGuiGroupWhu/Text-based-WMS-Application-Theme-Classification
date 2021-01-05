from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

import time
import tensorflow as tf

from sklearn import metrics
from utils import *
from models import GCN, MLP, GCNM
import random
import os
import sys
import nltk
from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix
import importlib
importlib.reload(sys)

import numpy as np
from skmultilearn.adapt import MLkNN
import sklearn.metrics as metrics
import torch
from torch.autograd import Variable
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

classes=['Agriculture','Biodiversity','Climate','Disaster','Ecosystem','Energy','Geology','Health','Water','Weather']
datasets = ['own_layer_Agriculture', 'own_wms_Biodiversity', 'own_wms_Climate', 'own_wms_Disaster', \
            'own_wms_Ecosystem', 'own_wms_Energy', 'own_wms_Geology', 'own_wms_Health', 'own_wms_Water',
            'own_wms_Weather', ]
fr = open('data/sweet/distance dic with topic and feature by sw wms.txt', 'r+')
dic = eval(fr.read())  # 读取的str转换为字典
print(dic)
fr.close()

wordkeys = []
for key in dic.keys():
    wordkeys.append(key)

classkeys = []
for key in dic[wordkeys[1]].keys():
    classkeys.append(key)

spi = dict()
for key in wordkeys:
    d_array = []
    for key2 in classkeys:
        d_array.append(dic[key][key2])
    d_min = min(d_array)+1
    spi[key] = 1.0 / d_min

def prepare_data_according_mask(mask):
    dataset_name = 'own_wms_all'
    sentence = np.load('corpus wms2.npy').tolist()

    label = np.load('train data wms all.npy').tolist()
    sentences = []
    labels = []
    row=''
    train_or_test_list = []
    for i in range(len(sentence)):
        row = ''
        sentences.append(str(sentence[i]))
        for C in range(len(classes)):
            if label[i][C] == 1:
                row=row+'1,'
            else:
                row = row + '0,'
        labels.append(row)
        if mask[i] == 1:
            train_or_test_list.append('train')
        else:
            train_or_test_list.append('test')

    meta_data_list = []

    for i in range(len(sentences)):
        meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
        meta_data_list.append(meta)

    meta_data_str = '\n'.join(meta_data_list)

    f = open('data/' + dataset_name + '.txt', 'w')
    f.write(meta_data_str)
    f.close()

    corpus_str = '\n'.join(sentences)

    f = open('data/corpus/' + dataset_name + '.txt', 'w')
    f.write(corpus_str)
    f.close()


def renew_data_according_mask(mask):
    dataset_name = 'own_wms_all'
    sentence = np.load('corpus wms2.npy').tolist()

    label = np.load('train data wms all new.npy').tolist()
    sentences = []
    labels = []
    row = ''
    train_or_test_list = []
    for i in range(len(sentence)):
        row = ''
        sentences.append(str(sentence[i]))
        for C in range(len(classes)):
            if label[i][C] == 1:
                row = row + '1,'
            else:
                row = row + '0,'
        labels.append(row)
        if mask[i] == 1:
            train_or_test_list.append('train')
        else:
            train_or_test_list.append('test')

    meta_data_list = []

    for i in range(len(sentences)):
        meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
        meta_data_list.append(meta)

    meta_data_str = '\n'.join(meta_data_list)

    f = open('data/' + dataset_name + '.txt', 'w')
    f.write(meta_data_str)
    f.close()

    corpus_str = '\n'.join(sentences)

    f = open('data/corpus/' + dataset_name + '.txt', 'w')
    f.write(corpus_str)
    f.close()


def remove_word():
    dataset = 'own_wms_all'
    stop_words = set(stopwords.words('english'))
    print(stop_words)

    # Read Word Vectors
    # word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
    # vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
    # word_embeddings_dim = len(embd[0])
    # dataset = '20ng'

    doc_content_list = []
    f = open('data/corpus/' + dataset + '.txt', 'rb')
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
    f.close()

    word_freq = {}  # to remove rare words

    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            # word not in stop_words and word_freq[word] >= 5
            if dataset == 'mr':
                doc_words.append(word)
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        # if doc_str == '':
        # doc_str = temp
        clean_docs.append(doc_str)

    clean_corpus_str = '\n'.join(clean_docs)

    f = open('data/corpus/' + dataset + '.clean.txt', 'w')
    # f = open('data/wiki_long_abstracts_en_text.clean.txt', 'w')
    f.write(clean_corpus_str)
    f.close()

    # dataset = '20ng'
    min_len = 10000
    aver_len = 0
    max_len = 0

    f = open('data/corpus/' + dataset + '.clean.txt', 'r')
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))




def build_graph():
    dataset = 'own_wms_all'
    word_embeddings_dim = 300
    word_vector_map = {}

    # shulffing
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()
    # print(doc_train_list)
    # print(doc_test_list)

    doc_content_list = []
    f = open('data/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    # print(doc_content_list)

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    print(train_ids)
    random.shuffle(train_ids)

    # partial labeled data
    # train_ids = train_ids[:int(0.2 * len(train_ids))]

    train_ids_str = '\n'.join(str(index) for index in train_ids)
    f = open('data/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    print(test_ids)
    random.shuffle(test_ids)

    test_ids_str = '\n'.join(str(index) for index in test_ids)
    f = open('data/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()

    ids = train_ids + test_ids
    #print(ids)
    print(len(ids))

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('data/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()

    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    word_doc_list = {}

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    definitions = []

    for word in vocab:
        word = word.strip()
        synsets = wn.synsets(clean_str(word))
        word_defs = []
        for synset in synsets:
            syn_def = synset.definition()
            word_defs.append(syn_def)
        word_des = ' '.join(word_defs)
        if word_des == '':
            word_des = '<PAD>'
        definitions.append(word_des)

    string = '\n'.join(definitions)

    f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
    f.write(string)
    f.close()

    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vec.fit_transform(definitions)
    tfidf_matrix_array = tfidf_matrix.toarray()
    #print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

    word_vectors = []

    for i in range(len(vocab)):
        word = vocab[i]
        vector = tfidf_matrix_array[i]
        str_vector = []
        for j in range(len(vector)):
            str_vector.append(str(vector[j]))
        temp = ' '.join(str_vector)
        word_vector = word + ' ' + temp
        word_vectors.append(word_vector)

    string = '\n'.join(word_vectors)

    f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
    f.write(string)
    f.close()

    word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    _, embd, word_vector_map = loadWord2Vec(word_vector_file)
    word_embeddings_dim = len(embd[0])

    # label list
    label_set = set()  # 不重复的序列
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    f = open('data/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()

    # x: feature vectors of training docs, no initial features
    # slect 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()

    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        temp2 = temp[2].split(',')
        one_hot=[]
        for i in range(len(temp2)):
            if temp2[i]=='0':
                one_hot.append(0)
            elif temp2[i]=='1':
                one_hot.append(1)
        y.append(one_hot)
    y = np.array(y)
    print(y)

    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        temp2 = temp[2].split(',')
        one_hot=[]
        for i in range(len(temp2)):
            if temp2[i]=='0':
                one_hot.append(0)
            elif temp2[i]=='1':
                one_hot.append(1)
        #for i in range(len(temp)):
         #   if i>1:
          #      if temp[i]=='0':
           #         one_hot.append(0)
            #    elif temp[i]=='1':
             #       one_hot.append(1)
        ty.append(one_hot)
    ty = np.array(ty)
    print(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words

    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size,
                                      word_embeddings_dim))  # vocab_size = len(vocab) word_embeddings_dim = len(embd[0])

    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        temp2 = temp[2].split(',')
        one_hot=[]
        for i in range(len(temp2)):
            if temp2[i]=='0':
                one_hot.append(0)
            elif temp2[i]=='1':
                one_hot.append(1)
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(classes))]
        ally.append(one_hot)

    ally = np.array(ally)
    print(ally)

    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    '''
    Doc word heterogeneous graph
    '''

    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # word vector cosine similarity as weights

    '''
   for i in range(vocab_size):
       for j in range(vocab_size):
           if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
               vector_i = np.array(word_vector_map[vocab[i]])
               vector_j = np.array(word_vector_map[vocab[j]])
               similarity = 1.0 - cosine(vector_i, vector_j)
               if similarity > 0.9:
                   print(vocab[i], vocab[j], similarity)
                   row.append(train_size + i)
                   col.append(train_size + j)
                   weight.append(similarity)
    '''
    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf * spi[lemma(word)])#
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    # dump objects
    f = open("data/ind.{}.x".format(dataset), 'wb')
    pkl.dump(x, f)
    f.close()

    f = open("data/ind.{}.y".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()

    f = open("data/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()

    f = open("data/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()

    f = open("data/ind.{}.allx".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open("data/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()




def train_model():
    dataset = 'own_wms_all'
    # Define model evaluation function
    # test_cost, test_acc, pred, labels, test_duration
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

    # Set random seed
    seed = random.randint(1, 200)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    lst = list(FLAGS._flags().keys())
    for key in lst:
        FLAGS.__delattr__(key)
    # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('dataset', dataset, 'Dataset string.')
    # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_string('model', 'gcn_multi', 'Model string.')
    flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0,
                       'Weight for L2 loss on embedding matrix.')  # 5e-4
    flags.DEFINE_integer('early_stopping', 10,
                         'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
        FLAGS.dataset)
    print(adj)
    # print(adj[0], adj[1])
    f = open('data/' + dataset + '_y_test.txt', 'w')
    for i in range(len(y_test)):
        f.write(str(y_test[i]) + '\n')
    f.close()
    features = sp.identity(features.shape[0])  # featureless 单位阵

    print(adj.shape)
    print(features.shape)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    elif FLAGS.model == 'gcn_multi':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = GCNM
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        # helper variable for sparse dropout
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # Create model
    print(features[2][1])
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_conf)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):
        aaa = model.pred
        bbb = placeholders['labels']
        print(bbb)

        print(str(aaa))

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(
            features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.layers[0].embedding], feed_dict=feed_dict)

        # Validation
        #print(y_val[1100])
        cost, acc, pred, labels, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(
                outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, pred, labels, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    test_pred = []
    test_labels = []
    print(len(test_mask))
    for i in range(len(test_mask)):
        # print(test_mask[i])
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(labels[i])
    print(test_labels)

    test_pred_sort0=np.arange(100*10).reshape(100, 10)
    test_labels_sort0=np.arange(100*10).reshape(100, 10)
    test_labels_sort_mask0 = [0 for i in range(100)]

    #test_pred_sort = [0 for i in range(100)]
    #test_labels_sort = [0 for i in range(100)]

    test_idx_reorder = parse_index_file("data/{}.test.index".format(dataset))
    print(test_idx_reorder)

    for i in range(len(test_idx_reorder)):
        idx = test_idx_reorder[i] - 401
        test_pred_sort0[idx] = test_pred[i][:]
        test_labels_sort0[idx] = test_labels[i][:]
        test_labels_sort_mask0[idx] = 1

    test_pred_sort = []
    test_labels_sort = []
    for i in range(100):
        if test_labels_sort_mask0[i] == 1:
            test_pred_sort.append(test_pred_sort0[i])
            test_labels_sort.append(test_labels_sort0[i])

    print(test_labels_sort)

    strlabel = 'result/label/' + dataset + '.npy'
    strpred = 'result/pred/' + dataset + '.npy'
    np.save(strlabel, test_labels_sort)
    np.save(strpred, test_pred_sort)
    '''
    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_labels, test_pred, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))
    '''

    # doc and word embeddings
    print('embeddings:')
    word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
    train_doc_embeddings = outs[3][:train_size]  # include val docs
    test_doc_embeddings = outs[3][adj.shape[0] - test_size:]

    print(len(word_embeddings), len(train_doc_embeddings),
          len(test_doc_embeddings))
    print(word_embeddings)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
    words = f.readlines()
    f.close()

    vocab_size = len(words)
    word_vectors = []
    for i in range(vocab_size):
        word = words[i].strip()
        word_vector = word_embeddings[i]
        word_vector_str = ' '.join([str(x) for x in word_vector])
        word_vectors.append(word + ' ' + word_vector_str)

    word_embeddings_str = '\n'.join(word_vectors)
    f = open('data/' + dataset + '_word_vectors.txt', 'w')
    f.write(word_embeddings_str)
    f.close()

    doc_vectors = []
    doc_id = 0
    for i in range(train_size):
        doc_vector = train_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    for i in range(test_size):
        doc_vector = test_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    doc_embeddings_str = '\n'.join(doc_vectors)
    f = open('data/' + dataset + '_doc_vectors.txt', 'w')
    f.write(doc_embeddings_str)
    f.close()



def evaluate_result():
    len0 = len(np.load('result\label\own_wms_all.npy').tolist())
    dataset = 'own_wms_all'
    label=np.arange(len0*11).reshape(len0, 11)
    pred=np.arange(len0*11).reshape(len0, 11)

    strlabel = 'result/label/' + dataset + '.npy'
    strpred = 'result/pred/' + dataset + '.npy'

    test_labels = np.load(strlabel).tolist()
    test_pred = np.load(strpred).tolist()
    #for i in range(len(test_labels)):
        #print(test_labels)
        #print(test_pred)
        #print('\n')

    for j in range(len(test_labels)):
        for i in range(len(classes)):
            label[j][i] = test_labels[j][i]
            pred[j][i] = test_pred[j][i]


    for j in range(len(test_labels)):
        label[j][10] = 1
        pred[j][10] = 1
        for i in range(10):
            if label[j][i] == 1:
                label[j][10] = 0
            if pred[j][i] == 1:
                pred[j][10] = 0

    countn=0

    for i in range(len(test_labels)):
        print(label[i])
        print(pred[i])
        error=0
        for j in range(len(label[i])):
            if label[i][j] != pred[i][j]:
                error+=1
        if error>0:
            countn +=1

    print(countn)

    strlabel2 = 'result/result_label.npy'
    strpred2 = 'result/result_predict.npy'
    np.save(strlabel2, label)
    np.save(strpred2, pred)



    def precision(test, pre):
        hamming_loss = metrics.hamming_loss(test, pre)
        accuracy_score = metrics.accuracy_score(test, pre)
        jaccard_similarity_score = metrics.jaccard_similarity_score(test, pre)
        precision_score = metrics.precision_score(test, pre, average='samples')
        recall_score = metrics.recall_score(test, pre, average='samples')
        f1_score = metrics.f1_score(test, pre, average='samples')
        print('hamming_loss:', hamming_loss)
        print('accuracy_score', accuracy_score)
        print('jaccard_similarity_score', jaccard_similarity_score)
        print('precision_score', precision_score)
        print('recall_score', recall_score)
        print('f1_score', f1_score)
        stringpath2 = 'result/gcn-multi result precision.txt'
        #stringpath2 = 'only/wms nofilter result precision.txt'
        doc = open(stringpath2, 'a')  # 打开一个存储文件，并依次写入
        print(str(hamming_loss) + ',' + str(accuracy_score) + ',' + str(jaccard_similarity_score) + ',' + str(
            precision_score) + ',' + str(recall_score) + ',' + str(f1_score) + '\n', file=doc)

        return [hamming_loss, accuracy_score, jaccard_similarity_score, precision_score, recall_score, f1_score]

    aaa = precision(csr_matrix(pred), csr_matrix(label))
    strprecision = 'result/result_precision.npy'
    np.save(strprecision, aaa)
    return pred



def tfidf(corpus):
	vector = CountVectorizer()
	x = vector.fit_transform(corpus)
	transform = TfidfTransformer()
	tfidf = transform.fit_transform(x)
	return tfidf

def mlknn(train_data_inx,y_train,test_data_inx):
	classifier = MLkNN(k=mlknn_k)
	x_train = []
	x_test = []
	for i in range(len(train_data_inx)):
		x_train.append(corpus_tfidf[train_data_inx[i]])
	for j in range(len(test_data_inx)):
		x_test.append(corpus_tfidf[test_data_inx[j]])
	classifier.fit(csr_matrix(x_train), csr_matrix(y_train))
	mlknn_pre = classifier.predict(csr_matrix(x_test))
	mlknn_pre = mlknn_pre.toarray()
	return mlknn_pre

mlknn_k = 3
path = 'E:/graduate thesis_201812_201906/code/data/data for computing/'
corpus = np.load(path+'corpus wms.npy').tolist()
train_label = np.load(path+'train data wms.npy.').tolist()
test_data = np.load(path+'test data wms.npy').tolist()
train_label_all_renew=np.load('train data wms all.npy').tolist()
mask = []
for i in range(len(corpus)):
    if i < 401:
        mask.append(1)
    else:
        mask.append(0)

prepare_data_according_mask(mask)

corpus_tfidf = tfidf(corpus).toarray()
inx = len(train_label)

train_data_inx = []
train_data_inx_copy = []
test_data_inx = []
pre = [[0] * 11] * len(test_data)
pre_inx = []
for a in range(0, inx):
    train_data_inx.append(a)
for b in range(inx, len(corpus)):
    test_data_inx.append(b)

iter_count=0
while len(test_data_inx) != 0 and train_data_inx_copy != train_data_inx and (iter_count==0 or len(pre_inx)>2):
    print('iter num', iter_count, 'this round data:')
    print('test data length', len(test_data_inx))
    print('train data length', len(train_data_inx), len(pre_inx))
    print('detail', pre_inx, test_data_inx, train_data_inx)
    train_data_inx_copy = train_data_inx[:]

    #knn
    set1 = mlknn(train_data_inx, train_label, test_data_inx)
    # gcn
    label = np.load('train data wms all.npy').tolist()

    if iter_count>0:
        renew_data_according_mask(mask)
    build_graph()
    train_model()
    set2 = evaluate_result()
    print('prediction', len(set1), len(set2))

    temp = test_data_inx[:]
    pre_inx = []
    mask2 = mask[:]
    for i in range(len(set1)):
        error = 0
        pre[test_data_inx[i] - inx] = set2[i]
        for f in range(len(set1[i])):
            if set1[i][f] != set2[i][f]:
                error += 1
                break
        if error == 0:
            train_label.append(set1[i])
            pre[test_data_inx[i] - inx] = set1[i]
            train_data_inx.append(test_data_inx[i])
            pre_inx.append(test_data_inx[i])
            temp.remove(test_data_inx[i])
            mask2[test_data_inx[i]]=1
            for tt in range(len(train_label_all_renew[test_data_inx[i]])):
                train_label_all_renew[test_data_inx[i]][tt]=set1[i][tt]

            #cnt=-1

            #for j in range(len(mask)):
                #if(mask[j]==0):
                    #cnt=cnt+1
                #if (cnt == i):
                    #mask2[j]=1
                    #break

    for i in range(len(mask)):
        if(mask2[i]==1):
            mask[i]=1


    test_data_inx = temp[:]
    np.save('train data wms all new.npy', train_label_all_renew)
    iter_count += 1

#for h in range(len(set1)):
#    pre[test_data_inx[h] - inx] = set1[h]

print(csr_matrix(test_data))
print(csr_matrix(pre))
f = open('result_wms.txt', 'w')
f.write(str(csr_matrix(test_data)))
f.write(str(csr_matrix(pre)))
f.close()


def precision(test, pre):
    hamming_loss = metrics.hamming_loss(test, pre)
    accuracy_score = metrics.accuracy_score(test, pre)
    jaccard_similarity_score = metrics.jaccard_similarity_score(test, pre)
    precision_score = metrics.precision_score(test, pre, average='samples')
    recall_score = metrics.recall_score(test, pre, average='samples')
    f1_score = metrics.f1_score(test, pre, average='samples')
    print('hamming_loss:', hamming_loss)
    print('accuracy_score', accuracy_score)
    print('jaccard_similarity_score', jaccard_similarity_score)
    print('precision_score', precision_score)
    print('recall_score', recall_score)
    print('f1_score', f1_score)
    # stringpath2 = 'sweet/word2vec/wms_du_0.' + str(num) + ' result precision.txt'
    stringpath2 = 'gcn-sw-knn/wms gcn_knn3 multi nofilter result precision.txt'
    doc = open(stringpath2, 'a')  # 打开一个存储文件，并依次写入
    print(str(hamming_loss) + ',' + str(accuracy_score) + ',' + str(jaccard_similarity_score) + ',' + str(
        precision_score) + ',' + str(recall_score) + ',' + str(f1_score) + '\n', file=doc)
    return [hamming_loss, accuracy_score, jaccard_similarity_score, precision_score, recall_score, f1_score]





precision(csr_matrix(test_data),csr_matrix(pre))






