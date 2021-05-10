import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from sklearn.model_selection import train_test_split
import sys
import random
from subprocess import Popen
from config import CONFIG


if len(sys.argv) != 2:
    sys.exit("Use: python -m src.build_graph <dataset>")

datasets = ['dblp', 'M10', 'covid', 'covid_title']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

cfg = CONFIG()
random.seed(1)
np.random.seed(1)

print('Reading data...')

doc_content_list = []
with open('data/' + dataset + '.clean.txt', 'rb') as f:
    lines = f.readlines()
    for line in lines:
        temp = line.strip().decode('latin1').split(' ', 1)
        if len(temp) == 2:
            doc_content_list.append(temp[1])
        else:
            doc_content_list.append('')

adjedges = []
with open('data/' + dataset + '/adjedges.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        adjedges.append(line.strip())

index_docid_list = []
all_labels = []
with open('data/' + dataset + '/labels.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split(' ')
        index = len(index_docid_list)
        index_docid_list.append([index, temp[0]])
        all_labels.append(temp[1])

doc_train_list, doc_test_list = train_test_split(index_docid_list, train_size=cfg.train_size, stratify=all_labels, random_state=1)

docid_to_idx_map = {}

train_idxs = [index for index, _ in doc_train_list]
for ii, doc_id in enumerate(doc_train_list):
    docid_to_idx_map[doc_id[1]] = ii
train_idxs_str = '\n'.join(str(index) for index in train_idxs)
with open('data/' + dataset + '.train.index', 'w') as f:
    f.write(train_idxs_str)

test_idxs = [index for index, _ in doc_test_list]
offset = len(doc_train_list)
for ii, doc_id in enumerate(doc_test_list):
    docid_to_idx_map[doc_id[1]] = offset + ii
test_idxs_str = '\n'.join(str(index) for index in test_idxs)
with open('data/' + dataset + '.test.index', 'w') as f:
    f.write(test_idxs_str)

shuffle_all_labels = []
shuffle_doc_words_list = []
shuffle_adjedges = []
for idx in train_idxs:
    shuffle_all_labels.append(all_labels[idx])
    shuffle_doc_words_list.append(doc_content_list[idx])
    shuffle_adjedges.append(adjedges[idx])

for idx in test_idxs:
    shuffle_all_labels.append(all_labels[idx])
    shuffle_doc_words_list.append(doc_content_list[idx])
    shuffle_adjedges.append(adjedges[idx])

shuffle_all_labels_str = '\n'.join(str(index) for index in shuffle_all_labels)
with open('data/' + dataset + '_label_shuffle.txt', 'w') as f:
    f.write(shuffle_all_labels_str)

print('Building graph...')

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
vocab.sort()
random.shuffle(vocab)
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
with open('data/' + dataset + '_vocab.txt', 'w') as f:
    f.write(vocab_str)

# label list
label_set = set()
for l in shuffle_all_labels:
    label_set.add(l)
label_list = list(label_set)
label_list.sort()

label_list_str = '\n'.join(label_list)
with open('data/' + dataset + '_labels.txt', 'w') as f:
    f.write(label_list_str)

# select 90% training set
test_size = len(doc_test_list)
train_size = len(doc_train_list)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size

y = []
for i in range(real_train_size):
    label = shuffle_all_labels[i]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

ty = []
for i in range(test_size):
    label = shuffle_all_labels[i + train_size]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)

ally = []
for i in range(train_size):
    label = shuffle_all_labels[i]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

# print(y.shape, ty.shape, ally.shape)

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
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

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
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)
to_remove = len(weight)

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
        weight.append(freq * idf)
        doc_word_set.add(word)

if cfg.doc_doc_edge or cfg.neg_loss:
    adj_pos = []
    adj_pos_dict = {}
    ii = None
    tote = 0
    for i in range(len(shuffle_adjedges)):
        edges = shuffle_adjedges[i]
        edges = edges.split(' ')
        tote += len(edges) - 1
        if len(edges) > 1:
            for e in edges[1:]:
                if e in docid_to_idx_map.keys():
                    ii = docid_to_idx_map[e]
                    if cfg.doc_doc_edge:
                        if i < train_size:
                            row.append(i)
                        else:
                            row.append(i + vocab_size)
                        if ii < train_size:
                            col.append(ii)
                        else:
                            col.append(ii + vocab_size)
                        weight.append(1)
                    if cfg.neg_loss and i < real_train_size and ii < real_train_size:
                        adj_pos.append(str(i) + ',' + str(ii))
                        if i in adj_pos_dict.keys():
                            adj_pos_dict[i].append(ii)
                        else:
                            adj_pos_dict[i] = [ii]

    if cfg.neg_loss:
        adj_pos_str = '\n'.join(adj_pos)
        with open('data/' + dataset + '_adj_pos.txt', 'w') as f:
            f.write(adj_pos_str)

        adj_pos_set = set(adj_pos)
        alll = list(range(real_train_size))
        label_d = dict()
        for idx, al in enumerate(shuffle_all_labels[:real_train_size]):
            if al in label_d.keys():
                label_d[al].append(idx)
            else:
                label_d[al] = [idx]

        adj_neg = []
        for pos_pair in adj_pos:
            nn = random.sample(list(set(alll) - set(adj_pos_dict[int(pos_pair.split(',')[0])]) - set(label_d[shuffle_all_labels[int(pos_pair.split(',')[0])]])), cfg.num_neg_samples)
            for neg in nn:
                adj_neg.append(pos_pair.split(',')[0] + ',' + str(neg))

        adj_neg_str = '\n'.join(adj_neg)
        with open('data/' + dataset + '_adj_neg.txt', 'w') as f:
            f.write(adj_neg_str)

node_size = train_size + vocab_size + test_size

adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

print('Dumping files required for training...')

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)

if cfg.node2vec:
    adj = sp.csr_matrix(
        (weight[to_remove:], (row[to_remove:], col[to_remove:])), shape=(node_size, node_size))

    adj = adj.tocoo()

    print('Running node2vec on the pruned graph...')
    with open("node2vec_cpp/graph/{}.edgelist".format(dataset), 'w') as f:
        for i, j, v in zip(adj.row, adj.col, adj.data):
            f.write(str(i) + ' ' + str(j) + ' ' + str(v) + '\n')
    p = Popen(['./node2vec_cpp/node2vec', '-i:node2vec_cpp/graph/{}.edgelist'.format(dataset), '-o:node2vec_cpp/emb/{}.emd'.format(dataset),
               '-d:128', '-e:1', '-p:1', '-q:1', '-v', '-w'])
    p.wait()
