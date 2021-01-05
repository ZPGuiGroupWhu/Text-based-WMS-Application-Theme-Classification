#!/usr/bin/python
#-*-coding:utf-8-*-
import numpy as np
classes=['Agriculture','Biodiversity','Climate','Disaster','Ecosystem','Energy','Geology','Health','Water','Weather']
anti_classes=['Agriculture','Biodiversity','Climate','Disaster','Ecosystem','Energy','Geology','Health','Water','Weather']
for C in range(len(classes)):
    dataset_name = 'own_layer_'+classes[C]
    #sentences = ['Would you like a plain sweater or something else?​', 'Great. We have some very nice wool slacks over here. Would you like to take a look?']
    #labels = ['Yes' , 'No' ]
    #train_or_test_list = ['train', 'test']
    classes=['Agriculture','Biodiversity','Climate','Disaster','Ecosystem','Energy','Geology','Health','Water','Weather']
    anti_classes=['Agriculture','Biodiversity','Climate','Disaster','Ecosystem','Energy','Geology','Health','Water','Weather']
    for i in range(len(classes)):
        anti_classes[i]='Not_'+classes[i]
    sentence=np.load('corpus layer.npy').tolist()
    label=np.load('train data layer all.npy').tolist()
    sentences=[]
    labels=[]
    train_or_test_list = []
    for i in range(len(sentence)):
        sentences.append(str(sentence[i]))
        if label[i][C]==1:
            labels.append('1')
        else:
            labels.append('0')
        if i<1168:
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

for C in range(len(classes)):
    print('\'own_layer_'+classes[C]+'\',')
