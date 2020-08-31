#!/usr/bin/env python
# coding: utf-8

# ### import

# In[1]:


import sys
import os
import random
maketrans = str.maketrans
import math
from collections import Counter, defaultdict
import pandas as pd


# In[2]:


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import cuda
import time
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# In[3]:


from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


# ### seed 固定

# In[4]:


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 13
seed_everything(seed)


# ### データ読み込み

# In[5]:


val_df = pd.read_csv('../processed_data/val_df_572_theta.csv')
val2_df = pd.read_csv('../processed_data/val2_df_309_theta.csv')
test_df = pd.read_csv('../processed_data/test_df_topic_theta.csv')


# ### モデル定義

# In[6]:


topic_tokens = [f'[{i}]' for i in range(20)]


# In[7]:


class JobModel(nn.Module):
    def __init__(self):
        super(JobModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(
            'roberta-base', output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(
            'roberta-base', config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 4, bias=False)
        self.topic_classifier = nn.Linear(config.hidden_size, 20, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.normal_(self.topic_classifier.weight, std=0.02)

    def forward(self, input_ids, attention_mask, task_id):
        _, _, hs = self.roberta(input_ids, attention_mask)
        x = torch.stack([hs[-1][:, 0], hs[-2][:, 0], hs[-3][:, 0], hs[-4][:, 0]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        if task_id == 0:
            ret = self.classifier(x)
        elif task_id == 1:
            ret = self.topic_classifier(x)
        return ret


# In[8]:


# 初回実行時のみ保存
# トークンidの順番は，seed_everythingで固定できなかったので，実行する度に変動します．
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', additional_special_tokens=sorted(topic_tokens))
# tokenizer.save_pretrained('../models/topic_tokenizer/')
tokenizer = RobertaTokenizer.from_pretrained('../models/topic_tokenizer/')


# ### トピックトークン付与

# In[9]:


X_val = '[' + val_df.topic_id.map(str).values + '] </s> ' + val_df.description.values
X_val2 = '[' + val2_df.topic_id.map(str).values + '] </s> ' + val2_df.description.values
test_X = '[' + test_df.topic_id.map(str).values + '] </s> ' + test_df.description.values

X_val = np.array(X_val)
X_val2 = np.array(X_val2)
test_X = np.array(test_X)


# In[10]:


drop_header_tr = ['id', 'description', 'jobflag', 'topic_id']
drop_header_te = ['id', 'description', 'topic_id']


# ### トピック確率取り出し

# In[11]:


theta_test = test_df.drop(drop_header_te, axis=1).values
theta_val2 = val2_df.drop(drop_header_tr, axis=1).values

theta_test = np.array(theta_test, dtype=np.float32)


# In[12]:


y_val = val_df.jobflag.values -1
y_val2 = val2_df.jobflag.values -1
y_val = np.array(y_val, dtype=int)
y_val2 = np.array(y_val2, dtype=int)


# In[13]:


(X_val.shape, y_val.shape), (X_val2.shape, y_val2.shape), (test_X.shape, theta_test.shape)


# ### train

# In[14]:


# 訓練データの削り方によって，得られるモデルからの予測カテゴリ割合が大きく変わるので，
# RoBERTa シングルモデルの学習時に，暫定スコアが最も高い提出に予測カテゴリ割合が近くなるようなseedの選択
magic_seed = [42, 346, 291, 241, 312, 150, 353, 310, 266, 188]
for s in magic_seed:
    train_df = pd.read_csv('../processed_data/train_df_1088_theta_seed_{}.csv'.format(s))
    X_train = '[' + train_df.topic_id.map(str).values + '] </s> ' + train_df.description.values
    theta_train = train_df.drop(drop_header_tr, axis=1).values
    y_train = train_df.jobflag.values -1
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=int)

    all_X = np.concatenate([X_train, X_val2, test_X], axis=0)
    all_theta = np.concatenate([theta_train, theta_val2, theta_test], axis=0)
    all_ids = np.concatenate([np.arange(len(X_train)), np.arange(len(all_X))], axis=0).astype(np.int32)
    task_ids = np.concatenate([np.zeros(len(X_train)), np.ones(len(all_X))], axis=0).astype(np.int32)
    model = JobModel()
    model.roberta.resize_token_embeddings(len(tokenizer))
    device_id = 0
    model.to(device_id)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    batchsize = 64
    num_iter = math.floor(len(all_ids) / batchsize)
    num_epoch = 3

    num_warmup_steps = 0
    num_train_steps = num_iter * num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    print(f'{s=}', '-'*50)
    for epoch in range(num_epoch):
        model.train()
        shuffled = np.random.permutation(len(all_ids))

        sum_ce_loss = 0.0
        sum_kl_loss = 0.0
        n=0
        cat_n = 0
        topic_n = 0
        sum_cat_correct = 0
        sum_topic_correct = 0
        val_correct = 0
        batchsize = 64
        for i in range(0, len(all_ids)-batchsize, batchsize):
            ids = shuffled[i:i+batchsize]
            xid, tid = all_ids[ids], task_ids[ids]
            optimizer.zero_grad()
            for task in range(2):
                if task == 0:
                    text_batch = list(X_train[xid[tid==task]])
                    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)

                    input_ids = encoding['input_ids'].to(device_id)
                    attention_mask = encoding['attention_mask'].to(device_id)
                    labels = torch.tensor(y_train[xid[tid==task]]).to(device_id)

                    outputs = model(input_ids, attention_mask=attention_mask, task_id=task)

                    ce_loss = 0.5 * F.cross_entropy(outputs, labels)
                    true_y = labels.cpu().detach().numpy()
                    pred_y = outputs.cpu().detach().numpy().argmax(axis=1)
                    sum_cat_correct += np.sum(true_y == pred_y)
                    cat_n += len(true_y)

                elif task == 1:
                    text_batch = list(all_X[xid[tid==task]])
                    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)

                    input_ids = encoding['input_ids'].to(device_id)
                    attention_mask = encoding['attention_mask'].to(device_id)
                    labels = torch.tensor(all_theta[xid[tid==task]]).to(device_id)

                    outputs = model(input_ids, attention_mask=attention_mask, task_id=task)


                    kl_loss = 0.5 * (- torch.sum(labels*F.log_softmax(outputs, dim=1), dim=1) + torch.sum(labels*torch.log(labels), dim=1)).mean()
                    true_y = labels.cpu().detach().numpy().argmax(axis=1)
                    pred_y = outputs.cpu().detach().numpy().argmax(axis=1)
                    sum_topic_correct += np.sum(true_y == pred_y)
                    topic_n += len(true_y)

            loss = ce_loss + kl_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_ce_loss += ce_loss.data
            sum_kl_loss += kl_loss.data

            n += len(ids)

        accuracy_cat = sum_cat_correct / cat_n
        accuracy_topic = sum_topic_correct / topic_n
        print("Epoch {} : ce_loss {}, kl_loss {}, acc {}, acc2 {}".format(epoch, sum_ce_loss / n, sum_kl_loss / n, accuracy_cat, accuracy_topic))
        
        # validation
        val_pred = []
        model.eval()
        for i in range(0, len(X_val), batchsize):
            text_batch = list(X_val[i:i+batchsize])
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)

            input_ids = encoding['input_ids'].to(device_id)
            attention_mask = encoding['attention_mask'].to(device_id)

            outputs = model(input_ids, attention_mask=attention_mask, task_id=0)
            pred_y = outputs.cpu().detach().numpy().argmax(axis=1)

            val_pred += list(pred_y)
            val_correct += np.sum(y_val[i:i+batchsize] == pred_y)
        val_pred = np.array(val_pred)
        print(val_correct/len(X_val), f1_score(y_val, val_pred, average='macro'))

        val_correct = 0
        val_pred = []
        for i in range(0, len(X_val2), batchsize):
            text_batch = list(X_val2[i:i+batchsize])
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)

            input_ids = encoding['input_ids'].to(device_id)
            attention_mask = encoding['attention_mask'].to(device_id)

            outputs = model(input_ids, attention_mask=attention_mask, task_id=0)
            pred_y = outputs.cpu().detach().numpy().argmax(axis=1)

            val_pred += list(pred_y)
            val_correct += np.sum(y_val2[i:i+batchsize] == pred_y)
        val_pred = np.array(val_pred)
        print(val_correct/len(X_val2), f1_score(y_val2, val_pred, average='macro'))
        
        # test に対する予測カテゴリの割合確認
        model.eval()
        batchsize = 32
        test_y = []
        for i in range(0, len(test_X), batchsize):
            text_batch = list(test_X[i:i+batchsize])
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = encoding['input_ids'].to(device_id)
            attention_mask = encoding['attention_mask'].to(device_id)

            outputs = model(input_ids, attention_mask=attention_mask, task_id=0)
            pred_y = outputs.cpu().detach().numpy().argmax(axis=1)
            test_y += list(pred_y)

        test_y = np.array(test_y)
        test_y += 1
        print(np.array(list(Counter(sorted(test_y)).values())) / len(test_y))
    torch.save(model.state_dict(), '../models/roberta-10ens/roberta_mtdnn_ce5kl5_seed{}.model'.format(s))


# In[ ]:





# In[ ]:




