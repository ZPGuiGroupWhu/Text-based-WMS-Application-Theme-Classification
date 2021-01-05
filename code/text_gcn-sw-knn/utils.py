import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # training nodes are training docs, no initial features
    # print("x: ", x)
    # test nodes are training docs, no initial features
    # print("tx: ", tx)
    # both labeled and unlabeled training instances are training docs and words
    # print("allx: ", allx)
    # training labels are training doc labels
    # print("y: ", y)
    # test labels are test doc labels
    # print("ty: ", ty)
    # ally are labels for labels for allx, some will not have labels, i.e., all 0
    # print("ally: \n")
    # for i in ally:
    # if(sum(i) == 0):
    # print(i)
    # graph edge weight is the word co-occurence or doc word frequency
    # no need to build map, directly build csr_matrix
    # print('graph : ', graph)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print('len of ally------------')
    print(len(ally))
    print('len of ty------------')
    print(len(ty))
    print('len of label------------')
    print(len(labels))
    print('shape of label------------')
    print(str(labels.shape))
    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    #for i in range(len(test_mask)):
    #    print(str(test_mask[i]))

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    print('len of y_test')
    print(len(y_test))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()#a.flatten()就是把a降到一维,默认是按行的方向降
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

###########################################################
# coding:utf-8
import nltk
import xlrd
import random
import MySQLdb
import numpy as np
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.metrics as metrics
from nltk.stem.wordnet import WordNetLemmatizer
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import BRkNNaClassifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import MultinomialNB

# define topic
topic = ['agriculture', 'biodiversity', 'climate', 'disaster', 'energy', 'ecosystem', 'geology', 'health', 'water',
         'weather', 'No Topic']
adj_noun = [
    ['geology', 'geochemical', 'geologie', 'geologic', 'geochemistry', 'geological', 'worldgeology', 'geochronological',
     'geologischen', 'geology', 'geomatic', 'geomatical', 'geophysical', 'geograph', 'geographic', 'geography',
     'geographical'],
    ['mine', 'mineralogy', 'mineral'],
    ['fire', 'fuoco', 'bruciate'],
    ['radiation', 'aeroradiometric', 'radiometric'],
    ['meteorology', 'meteorological', 'aerology'],
    ['climate', 'climatology'],
    ['environment', 'environmental', 'inhabit', 'ecologic', 'ecology', 'ecological', 'habitats'],
    ['bodendegradation', 'bodenmineralogie', 'bodendegradation', 'bodenmechanik', 'bodendekontamination',
     'bodenphysikalische', 'bodenfruchtbarkeit', 'bodendegradation', 'bodenhorizont', 'bodenverbreitung',
     'bodenskelett', 'bodenbiologie bodenchemie', 'bodennutzung', 'bodenfunktion', 'bodenausgangsgestein'],
    ['hill', 'hillshade', 'hillshades'], ['agriculture', 'agricoltura', 'agricultural'],
    ['hydrology', 'hydrography', 'hydrographical', 'hydrographic'], ['graze', 'grazing'],
    ['biodiversity', 'paleontology', 'biodiversidad', 'biological', 'biologic', 'diversity', 'biology'],
    ['specie', 'species'],
    ['fish', 'fishing', 'fisheries'], ['river', 'statescitiesrivers'],
    ['earthquake', 'seismic'],
    ['vegetation', 'vegetale', 'vegetazione'],
    ['water', 'undersea', 'ice', 'atlas', 'marine', 'marines', 'shoreline', 'bathymetry', 'rain'],
    ['coastal', 'coastlines']]
stopwordadd_for_mlcsw = ['rilievo', 'http', 'net', 'workshop', 'dot', 'get', 'map', 'start', 'noaa', 'data', 'without',
                         'www', 'also', 'null', 'gstore', 'one', '\n', 'represent', 'caused', 'limiter',
                         'com', 'wms', 'request', 'getcapabilities', 'survey', 'service', 'wmsserver', 'arcgis',
                         'mapserver', 'https', 'us', 'server', 'two', 'three', 'zinc', 'biweekly', 'content',
                         'demo', 'web', 'united', 'state', 'fop', 'deli', 'info', 'local', 'usa', 'doi', 'date', 'soap',
                         'plus', 'styling', 'aster', 'four', 'five', 'six', 'assembled', 'former', 'program',
                         'implementation', 'extension', 'dynamic', 'generate', 'basement', 'demonstration', 'showcase',
                         'support', 'digitized', 'seven', 'eight', 'integrates', 'risk', 'ultraviolet'
                                                                                         'sample', 'enjoy',
                         'observation', 'specialty', 'thailand', 'provides', 'infrastructure', 'ordnance', 'germany',
                         'gps', 'nine', 'ten', 'number', 'size', 'affect', 'colored',
                         'form', 'source', 'including', 'cia', 'code', 'germany', 'contained', 'intended', 'purpose',
                         'changed', 'time', 'lea', 'delaware', 'lead', 'investigator', 'solution', 'large',
                         'prior', 'notice', 'please', 'use', 'nobody', 'take', 'responsibility', 'fun', 'information',
                         'professional', 'wmo', 'michigan', 'south', 'wale', 'represents', 'kodiak',
                         'visit', 'provided', 'settlement', 'free', 'graticule', 'suite', 'tool', 'view', 'address',
                         'inspector', 'daily', 'approach', 'massachusetts', 'served', 'inspire', 'vector',
                         'csis', 'portal', 'haiti', 'inc', 'bin', 'cwm', 'fao', 'format', 'feed', 'type', 'atom', 'icc',
                         'cat', 'pennsylvania', 'wisconsin', 'jersey', 'humidity', 'delivered', 'sub', 'den',
                         'base', 'description', 'ensemble', 'sur', 'par', 'public', 'mexico', 'school', 'district',
                         'boundary', 'research', 'application', 'country', 'international', 'programme',
                         'public', 'mosaic', 'nasa', 'remote', 'sensing', 'imagery', 'science', 'picture', 'poverty',
                         'estimate', 'based', 'active', 'activity', 'designated', 'cooperative', 'unit',
                         'berthing', 'general', 'arctic', 'legacy', 'longer', 'maintained', 'basis', 'removed',
                         'operation', 'march', 'see', 'range', 'anomaly', 'tectonics', 'past', 'blue', 'keyhole',
                         'british', 'columbia', 'canada', 'region', 'park', 'gob', 'rep', 'argentina', 'car', 'est',
                         'hel', 'arizona', 'world', 'existed', 'north', 'degree', 'latitude', 'metadata',
                         'conformes', 'norma', 'ecuador', 'van', 'monument', 'gauge', 'watch', 'warning', 'asp', 'air',
                         'goddard', 'center', 'disc', 'west', 'station', 'established', 'aid', 'berlin',
                         'nga', 'name', 'scaled', 'locality', 'spot', 'die', 'pub', 'nad', 'pro', 'standard',
                         'cantones', 'bonn', 'row', 'version', 'calculated', 'compilation', 'protection',
                         'classification',
                         'offer', 'access', 'layer', 'mapping', 'accordance', 'open', 'consortium', 'specification',
                         'athene', 'ncdc', 'aleutian', 'element', 'part', 'observing', 'projection', 'particular',
                         'federal', 'county', 'pol', 'dam', 'steer', 'border', 'facility', 'runway', 'index', 'palmer',
                         'inventory', 'missouri', 'tract', 'america', 'age', 'search', 'award', 'language',
                         'lock', 'place', 'port', 'line', 'node', 'crossing', 'transit', 'mop', 'louisiana',
                         'mississippi', 'alabama', 'lcd', 'centre', 'imaging', 'taken', 'setup', 'composite', 'rescue',
                         'guam', 'hawaii', 'maryland', 'mercator', 'network', 'far', 'hazmat', 'ida', 'director',
                         'georgia', 'virginia', 'deposition', 'example', 'give', 'floor', 'wet', 'point', 'partner',
                         'recovery', 'feature', 'interoperability', 'prognostic', 'catalog', 'trier', 'fleet',
                         'documentation', 'order', 'restriction', 'interpretation', 'office', 'administration',
                         'mission', 'radar', 'committee', 'difference', 'accompanying', 'shear', 'split', 'orange',
                         'reflected', 'silver', 'contributed', 'found', 'european', 'generation', 'measurement',
                         'event', 'substrate', 'funding', 'method', 'safety', 'evaluation', 'foundation', 'nation',
                         'small', 'formation', 'limit', 'permafrost', 'unesco', 'signature', 'top', 'major',
                         'production', 'discover', 'blank', 'destruction', 'western', 'target', 'dredge', 'project',
                         'street', 'hue', 'concentration', 'eruption', 'chart', 'organization', 'doe', 'section',
                         'sector', 'gold', 'infrared', 'guideline', 'cell', 'original', 'boulder', 'lanai', 'barrier',
                         'observatory', 'report', 'alteration', 'bag', 'bad', 'gathered', 'set', 'reference',
                         'arc', 'observed', 'contour', 'subject', 'expert', 'future', 'whitewater', 'dal', 'con',
                         'scale', 'cover', 'band', 'total']
lda_stopword = ['ows', 'sciencebase', 'catalogmaps', 'statistic', 'dec', 'bgc', 'bfe', 'lsoa', 'msoa', 'jan', 'bfc',
                'percentage', '']


# participle words
def participle(text):
    pattern = r"""(?x)                   # set flag to allow verbose regexps 
	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe 
	              |\.\.\.                # ellipsis 
	              |(?:[.,;"'?():-_`])    # special characters with meanings 
	            """
    texts = list(text)
    for i in range(len(texts)):
        if texts[i] == '_':
            texts[i] = ','
    text = ''.join(texts)
    text = nltk.regexp_tokenize(text, pattern)
    text = [v for v in text if str(v).isalpha()]
    return text


# del stop words
stopworddic = set(stopwords.words('english'))
stopwordadd = ['http', 'net', 'mrdata', 'data', 'service', 'getcapabilities', 'wms', 'request', 'mapserver', 'server',
               'wmsserver',
               'null', 'www', 'arcgis', 'gov', 'map', 'response', 'mostly', 'center']


def del_stopword(text):
    text = [i for i in text if i not in stopworddic and i not in stopwordadd and len(i) >= 3]
    return text


# tf-idf
def tfidf(corpus):
    vector = CountVectorizer()
    x = vector.fit_transform(corpus)
    word = vector.get_feature_names()
    # print word,len(word)
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(x)
    return tfidf


# compute average of some numbers
def avg(num1, num2, num3, num4):
    sum = num1 + num2 + num3 + num4
    average = float(sum) / 4
    return average


# calculate precision of classification result
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
    return [hamming_loss, accuracy_score, jaccard_similarity_score, precision_score, recall_score, f1_score]


# database set
def database(dbname):
    conn = MySQLdb.connect(host="localhost", user="root", passwd="123456", db=dbname, charset="utf8")
    cursor = conn.cursor()
    return cursor


def preprocess(id_arr, sql, cursor, id_train, id_test, top_inx):
    corpus = []
    label_train_arr = []
    label_test_arr = []
    label_arr = []
    feature = []
    for i in id_arr:
        com = sql + str(i)
        cursor.execute(com)
        for row in cursor.fetchall():
            if row != None:
                ida = int(row[0])
                text = row[1:(len(row) - 1)]
                line = ' '.join(text)
                line = participle(line)
                line = [str(nltk.stem.WordNetLemmatizer().lemmatize(word)).lower() for word in line]
                line = del_stopword(line)
                for word in line:
                    if word not in feature:
                        feature.append(word)
                line = ' '.join(line)
                corpus.append(line)
                cato = [0] * len(topic)
                for j in range(len(topic)):
                    if topic[j] in str(row[top_inx]):
                        cato[j] = 1
                if ida in id_train:
                    label_train_arr.append(cato)
                if ida in id_test:
                    label_test_arr.append(cato)
                label_arr.append(cato)
    label_train = csr_matrix(label_train_arr)
    label_test = csr_matrix(label_test_arr)
    return [label_train, label_test, corpus, feature, label_arr, label_test_arr, label_train_arr]


# get topic related word dic
def getTopRelatedDic():
    data = xlrd.open_workbook(
        'E:/graduate thesis_201812_201906/code/data/data for computing/selected topic related dic agriculture.xlsx')
    table = data.sheets()[0]
    sys_dic = []
    for i in range(0, table.nrows):
        for j in range(0, 2):
            topic_word = str(table.cell(i, j).value).lower()
            topic_word = topic_word.split(',')
            for word in topic_word:
                sys_dic.append(word)
    return sys_dic


# text id pre-processing for train and test data choosen
def allDataIds(cursor, maxId, sql):
    ids = []
    for i in range(1, maxId):
        cursor.execute(sql + str(i))
        for row in cursor.fetchall():
            if row != None:
                ids.append(int(row[0]))
    random.shuffle(ids)
    return ids


def writeTxt(file, string):
    fo = open(file, 'w')
    fo.write(string)
    print('write finished')
    fo.close()


def predict_write(corpus_tfidf, start, train_label, test_label, csv_write, classifier, classifierStr, k):
    x_train = corpus_tfidf[:start]
    x_test = corpus_tfidf[start:]
    classifier.fit(x_train, train_label)
    pre = classifier.predict(x_test)
    result = precision(test_label, pre)
    if k:
        csv_write.writerow([classifierStr, k] + result)
    else:
        csv_write.writerow([classifierStr] + result)


def classify_multi_method(corpus_tfidf, start, train_label, test_label, csv_write):
    for k in range(1, 11):
        print(k)
        classifier = MLkNN(k=k)
        predict_write(corpus_tfidf, start, train_label, test_label, csv_write, classifier, 'ML-KNN', k)

    for k in range(1, 11):
        classifier = BRkNNaClassifier(k=k)
        predict_write(corpus_tfidf, start, train_label, test_label, csv_write, classifier, 'BR-KNN', k)

    classifier = ClassifierChain(
        classifier=MultinomialNB(),
        require_dense=[False, True]
    )
    predict_write(corpus_tfidf, start, train_label, test_label, csv_write, classifier, 'NB-ClassifierChain', None)

    classifier = MajorityVotingClassifier(
        clusterer=FixedLabelSpaceClusterer(clusters=[[1, 2, 3], [0, 2, 5], [4, 5]]),
        classifier=ClassifierChain(classifier=GaussianNB())
    )
    predict_write(corpus_tfidf, start, train_label, test_label, csv_write, classifier, 'NB-VoteClassifier', None)


def weightBySw(corpus, feature, path_dic, topic):
    weight_by_path = []
    for i in range(len(corpus)):
        line_weight = [0] * len(feature)
        for j in range(len(feature)):
            word = feature[j]
            if word in corpus[i] and word in path_dic:
                temp = []
                for top in topic:
                    temp.append(path_dic[word][top])
                min_path = np.min(temp)
                similar = float(1) / float(min_path)
                line_weight[j] = similar
        weight_by_path.append(line_weight)
    return weight_by_path

def lemma(word):
	for r in range(len(adj_noun)):
		if word in adj_noun[r]:
			word = adj_noun[r][0]
	return word