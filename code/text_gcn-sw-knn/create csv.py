import csv
import numpy as np
classes=['Agriculture','Biodiversity','Climate','Disaster','Ecosystem','Energy','Geology','Health','Water','Weather']

sentence=np.load('corpus wms2.npy').tolist()
label=np.load('train data wms all.npy').tolist()
for C in range(len(classes)):
    sentences=[]
    labels=[]
    train_or_test_list = []
    for i in range(len(sentence)):
        sentences.append(str(sentence[i]))
        if label[i][C]==1:
            labels.append('1')
        else:
            labels.append('0')
        if i<25:
            train_or_test_list.append('train')
        else:
            train_or_test_list.append('test')


    # 1. 创建文件对象
    fn1='csv/wms/5/'+classes[C]+'train.csv'
    f1 = open(fn1,'w',encoding='utf-8',newline='')
    fn2='csv/wms/5/'+classes[C]+'test.csv'
    f2 = open(fn2,'w',encoding='utf-8',newline='')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer1 = csv.writer(f1)
    csv_writer2 = csv.writer(f2)

    # 3. 构建列表头
    csv_writer1.writerow(["label", "content"])
    csv_writer2.writerow(["label", "content"])

    # 4. 写入csv文件内容
    for i in range(len(sentence)):
        if i<25:
            csv_writer1.writerow([labels[i],sentences[i]])
        else:
            csv_writer2.writerow([labels[i], sentences[i]])


    # 5. 关闭文件
    f1.close()
    f2.close()
