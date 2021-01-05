from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os
import sys

import sys
import csv
import math
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix
import importlib
importlib.reload(sys)

datasets = ['own_layer_Agriculture','own_layer_Biodiversity','own_layer_Climate','own_layer_Disaster',\
            'own_layer_Ecosystem','own_layer_Energy','own_layer_Geology','own_layer_Health','own_layer_Water','own_layer_Weather']


label=np.load('test data layer.npy').tolist()
pred=np.load('test data layer.npy').tolist()
#label=np.arange(292*11).reshape(292, 11)
#pred=np.arange(292*11).reshape(292, 11)

for i in range(len(datasets)):
    dataset=datasets[i]
    strlabel= 'result/label/' + dataset + '.npy'
    strpred = 'result/pred/' + dataset + '.npy'
    test_labels=np.load(strlabel).tolist()
    test_pred=np.load(strpred).tolist()
    print(test_labels)
    print(test_pred)
    for j in range(len(test_labels)):
        label[j][i]= test_labels[j]
        pred[j][i] = test_pred[j]

for j in range(292):
	label[j][10]= 1
	pred[j][10] = 1
	for i in range(10):
		if label[j][i]==1:
			label[j][10] = 0
		if pred[j][i]==1:
			pred[j][10] = 0
print(label)
print(pred)
strlabel2 = 'result/result_label.npy'
strpred2 = 'result/result_predict.npy'
np.save(strlabel2, label)
np.save(strpred2, pred)

def precision(test,pre):
	hamming_loss = metrics.hamming_loss(test, pre)
	accuracy_score = metrics.accuracy_score(test,pre)
	jaccard_similarity_score = metrics.jaccard_similarity_score(test,pre)
	precision_score = metrics.precision_score(test,pre,average='samples')
	recall_score = metrics.recall_score(test,pre,average='samples')
	f1_score = metrics.f1_score(test,pre,average='samples')
	print('hamming_loss:',hamming_loss)
	print('accuracy_score',accuracy_score)
	print('jaccard_similarity_score',jaccard_similarity_score)
	print('precision_score',precision_score)
	print('recall_score',recall_score)
	print('f1_score',f1_score)
	return [hamming_loss,accuracy_score,jaccard_similarity_score,precision_score,recall_score,f1_score]


aaa = precision(csr_matrix(pred),csr_matrix(label))
strprecision = 'result/result_precision.npy'
np.save(strprecision, aaa)

