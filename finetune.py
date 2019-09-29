class Config:
    def __init__(self):
        self.dev_path = 'data/dev.csv'
        self.train_path = 'data/train.csv'
        self.max_vocab = 5000
        self.max_sent_len = 30
        self.lr = 1e-4
        self.epoch = 3
        self.batch_size = 64

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch as T
# from pytorch_pretrained_bert import BertAdam, BertTokenizer
from keras.preprocessing.text import Tokenizer
from transformers import *
device = T.device('cuda')
import torch.nn.functional as F
conf = Config()

def load_data():
    with open(conf.dev_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:1000]
        lines = [line.strip().split('\t')[1:] for line in lines]
        text = [line[0] for line in lines]
        text2 = [line[1] for line in lines]
        labels = [int(line[2]) for line in lines]
    return text, text2, labels

text, text2, labels = load_data()
print(text[0])

# tknzr = Tokenizer(conf.max_vocab,
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
# tokenized_sents = bert_tokenizer.tokenize(text)
# print(tokenized_sents[0])
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# data = [tokenizer.encode()]
import numpy as np
data1 = [tokenizer.encode(sent, add_special_tokens=True) for sent in text]
data2 = [tokenizer.encode(sent, add_special_tokens=True) for sent in text2]
data1 = pad_sequences(data1, maxlen=30, padding='post', truncating='post')
data2 = pad_sequences(data2, maxlen=30, padding='post', truncating='post')

data = list(zip(data1, data2))

# data = np.asarray(data)
# print(data.shape)
model.to(device)
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.1, stratify=\
    labels, shuffle=True)
# trainX1 = [x[0] for x in trainX]
# trainX2 = [x[1] for x in trainX]
# testX1 = [x[0] for x in testX]
# testX2 = [x[1] for x in testX]
trainX = T.tensor(trainX, dtype=T.long)
trainY = T.tensor(trainY, dtype=T.float)
testX = T.tensor(testX, dtype=T.long)
testY = T.tensor(testY, dtype=T.float)
train_data = TensorDataset(trainX, trainY)
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
test_data = TensorDataset(testX, testY)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

import torch.nn as nn
# top_module = nn.Sequential(
#     nn.Linear(768, 64),
#     nn.Linear(64, 1),
#     nn.Sigmoid()
# )



# top_module.to(device)

from sklearn.metrics import roc_auc_score

optimizer = T.optim.Adam(lr=conf.lr, params=model.parameters())
for e in range(conf.epoch):
    
    print("epoch : {}".format(e))
    accu_loss = 0.0
    for i, (instances, targets) in enumerate(train_loader):
        # print("{}/{}".format(i, len(train_loader)), end=' ,', flush=True)
        Xa = instances[:,0,:]
        Xb = instances[:,1,:]
        Xa = Xa.to(device)
        Xb = Xb.to(device)
        targets = targets.to(device)
        targets = targets.view(targets.size()[0], 1)
        # print(targets.size())
        model.train()
        # top_module.train()
        res_a = model(Xa)[0]
        pooled_a = res_a.max(dim=1)[0]
        res_b = model(Xb)[0]
        pooled_b = res_b.max(dim=1)[0]
        # print(pooled)
        output = T.tensordot(pooled_a, pooled_b, (1,1))
        print(output.size())
        loss = F.binary_cross_entropy(output, targets)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accu_loss += loss
        if i % 10 == 0:
            print("{}/{}".format(i, len(train_loader)))
            print(accu_loss / (i + 1), flush=True)
    accu_roc = 0.0
    for j, (instances, targets) in enumerate(test_loader):
        model.eval()
        Xa = instances[:, 0, :]
        Xb = instances[:, 1, :]
        Xa = Xa.to(device)
        Xb = Xb.to(device)
        # top_module.eval()
        with T.no_grad():
            res_a = model(Xa)[0]
            pooled_a = res_a.max(dim=1)[0]
            res_b = model(Xb)[0]
            pooled_b = res_b.max(dim=1)[0]
            output = F.sigmoid(T.dot(pooled_a, pooled_b))
            # output = top_module(pooled)
        output = output.cpu()
        # testY = testY.cpu()
        output = np.squeeze(output)
        targets = np.squeeze(targets)
        roc = roc_auc_score(targets, output)
        accu_roc += roc
        # print("ROC:{}".format(roc))
    print(accu_roc / len(test_loader))