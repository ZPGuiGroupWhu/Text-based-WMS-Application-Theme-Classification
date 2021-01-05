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
import numpy as np
'''
pred0 = []
f = open('lstm.txt', 'rb')

for line in f.readlines():
    pred0.append(line.strip().decode('utf-8'))
f.close()

pred=[]
print(pred0)
for i in range(len(pred0)):
    temp=[]
    temp0=pred0[i].split(',')
    for j in range(len(temp0)):
        if temp0[j]=='1':
            temp.append(1)
        else:
            temp.append(0)
    pred.append(temp)

strpred = 'result/pred/lstm.npy'

np.save(strpred, pred)
print(pred)
'''
def evaluate_result():
    len0 = len(np.load('result\label\own_wms_all.npy').tolist())
    dataset = 'own_wms_all'

    print(len0)
    label=np.arange(len0*11).reshape(len0, 11)
    pred=np.arange(len0*11).reshape(len0, 11)

    strlabel = 'result/label/' + dataset + '.npy'
    strpred = 'text/wms lstm result 80_8 .npy'
    #print(strpred)

    test_labels = np.load(strlabel).tolist()
    test_pred = np.load(strpred).tolist()

    print(len(test_pred))
    #for i in range(len(test_labels)):
        #print(test_labels)
        #print(test_pred)
        #print('\n')

    for j in range(len0):
        for i in range(10):
            label[j][i] = test_labels[j][i]
            pred[j][i] = test_pred[j][i]


    for j in range(len0):
        label[j][10] = 1
        pred[j][10] = 1
        for i in range(10):
            if label[j][i] == 1:
                label[j][10] = 0
            if pred[j][i] == 1:
                pred[j][10] = 0

    countn=0

    for i in range(len0):
        print(label[i])
        print(pred[i])
        error=0
        for j in range(len(label[i])):
            if label[i][j] != pred[i][j]:
                error+=1
        if error>0:
            countn +=1

    print(countn)

    strlabel2 = 'result/wms cnn result_label.npy'
    strpred2 = 'result/wms cnn result_predict.npy'
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

        stringpath2 = 'result/wms 80 lstm result precision.txt'
        #stringpath2 = 'gcn-sw/wordnet/wms gcn-multi du_0.'+str(num)+' result precision.txt'
        doc = open(stringpath2, 'a')  # 打开一个存储文件，并依次写入
        print(str(hamming_loss) + ',' + str(accuracy_score) + ',' + str(jaccard_similarity_score) + ',' + str(
            precision_score) + ',' + str(recall_score) + ',' + str(f1_score) + '\n', file=doc)

        return [hamming_loss, accuracy_score, jaccard_similarity_score, precision_score, recall_score, f1_score]

    aaa = precision(csr_matrix(pred), csr_matrix(label))
    #strprecision = 'result/result_precision.npy'
    #np.save(strprecision, aaa)
    return pred

evaluate_result()