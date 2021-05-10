from __future__ import division
from __future__ import print_function
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import time
import torch
import torch.nn as nn
from src.utils.utils import *
from src.models.gcn import GCN
from config import CONFIG


cfg = CONFIG()

if len(sys.argv) != 2:
    sys.exit("Use: python -m src.train <dataset>")

datasets = ['dblp', 'M10', 'covid', 'covid_title']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")
cfg.dataset = dataset

# Set random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

# Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
print('Starting training...')
# Load data
adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size, adj_pos_i, adj_pos_j, adj_neg_i, adj_neg_j = load_corpus(
    cfg.dataset, cfg.neg_loss)

if cfg.node2vec:
    features = sp.identity(adj.shape[0])
    n2v_features = {}
    with open("node2vec_cpp/emb/{}.emd".format(dataset), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            tmp = line.split(' ', 1)
            n2v_features[int(tmp[0])] = np.array(list(map(float, tmp[1].split(' '))))
    all_zeros = [1e-5] * 128

    if cfg.node2vec_interaction_initialize:
        neighbors = np.split(adj.indices, adj.indptr[1:-1])
        hist_features = []
        for idx, neighbor in enumerate(neighbors):
            if idx < train_size or idx >= (adj.shape[0] - test_size):
                local_interactions = []
                wts = []
                try:
                    cur = n2v_features[idx]
                except:
                    cur = np.zeros(128)
                for n in neighbor:
                    try:
                        local_interactions.append(
                            cosine_similarity(cur.reshape(1, -1), n2v_features[n].reshape(1, -1))[0][0])
                        wts.append(adj[idx, n])
                    except:
                        local_interactions.append(0.0)
                        wts.append(0.0)
                tmmp = np.histogram(local_interactions, bins=cfg.bins, range=(-1.0, 1.0))[0]
                if cfg.hist_map == 'lch':
                    hist_features.append(np.log10(tmmp, where=(tmmp != 0)))
                elif cfg.hist_map == 'nh':
                    hist_features.append(tmmp)
                else:
                    print('Invalid value of hist_map')
                    exit()
            else:
                hist_features.append(np.zeros(cfg.bins))
        hist_features = np.array(hist_features, dtype=float)
        if cfg.hist_map == 'nh':
            from sklearn.preprocessing import normalize
            hist_features = normalize(hist_features, axis=1, norm='l2')
        hist_features = sp.csr_matrix(hist_features)
        features = sp.hstack((features, hist_features))
    else:
        features = [n2v_features[i] if i in n2v_features.keys() else all_zeros for i in range(adj.shape[0])]
        features = np.array(features)
        from sklearn.preprocessing import normalize
        features = normalize(features, axis=1, norm='l1')
        features = sp.coo_matrix(features)
else:
    features = sp.coo_matrix(sp.identity(adj.shape[0]))  # featureless

support = None
num_supports = None
model_func = None
if cfg.model == 'gcn':
    support = preprocess_adj(adj)
    num_supports = 1
    model_func = GCN

values = features.data
indices = np.vstack((features.row, features.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = features.shape
t_features = torch.sparse.FloatTensor(i, v, torch.Size(shape))
t_features = t_features.float()
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

values = support.data
indices = np.vstack((support.row, support.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = support.shape
t_support = torch.sparse.FloatTensor(i, v, torch.Size(shape))

# if torch.cuda.is_available():
#     model_func = model_func.cuda()
#     t_features = t_features.cuda()
#     t_y_train = t_y_train.cuda()
#     t_y_val = t_y_val.cuda()
#     t_y_test = t_y_test.cuda()
#     t_train_mask = t_train_mask.cuda()
#     tm_train_mask = tm_train_mask.cuda()
#     for i in range(len(support)):
#         t_support = [t.cuda() for t in t_support if True]

if cfg.model == 'gcn':
    model = model_func(input_dim=features.shape[1], support=t_support, num_classes=y_train.shape[1])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def NEG_loss(embeddings):
    loss_pos = (embeddings[adj_pos_i] * embeddings[adj_pos_j]).sum(1).sigmoid().log()
    loss_neg = (embeddings[adj_neg_i] * embeddings[adj_neg_j]).sum(1).neg().sigmoid().log()

    return cfg.lamb * (-(loss_pos.sum() / len(adj_pos_i)) - (loss_neg.sum() / len(adj_neg_i)))


# Define model evaluation function
def evaluate(features, labels, mask):
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        if cfg.neg_loss:
            N_loss = NEG_loss(model.layer1.embedding)
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

    if cfg.neg_loss:
        return loss.numpy() + N_loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)
    else:
        return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)

val_losses = []
PATH = 'models/model_' + dataset + '.pt'
best_loss = 1000000

# Train model
for epoch in range(cfg.epochs):
    t = time.time()

    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
    if cfg.neg_loss:
        N_loss = NEG_loss(model.layer1.embedding)
    else:
        N_loss = 0
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    if cfg.neg_loss:
        N_loss.backward()

    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    if best_loss >= val_loss:
        best_loss = val_loss
        torch.save(model, PATH)

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss+N_loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1]):
        print_log("Early stopping...")
        break

print_log("Optimization Finished!")

model = torch.load(PATH)

# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))

print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# doc and word embeddings
tmp = model.layer1.embedding.numpy()
word_embeddings = tmp[train_size: adj.shape[0] - test_size]
train_doc_embeddings = tmp[:train_size]  # include val docs
test_doc_embeddings = tmp[adj.shape[0] - test_size:]

print_log('Embeddings:')
print_log('\rWord_embeddings:'+str(len(word_embeddings)))
print_log('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings))) 
print_log('\rTest_doc_embeddings:'+str(len(test_doc_embeddings))) 
print_log('\rWord_embeddings:') 
print(word_embeddings)

with open('data/' + dataset + '_vocab.txt', 'r') as f:
    words = f.readlines()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
with open('data/' + dataset + '_word_vectors.txt', 'w') as f:
    f.write(word_embeddings_str)

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
with open('./data/' + dataset + '_doc_vectors.txt', 'w') as f:
    f.write(doc_embeddings_str)
