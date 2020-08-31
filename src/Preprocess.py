#!/usr/bin/env python
# coding: utf-8

# ### import

# In[1]:


import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[2]:


import sys
import os
import random
maketrans = str.maketrans
import math
from collections import Counter, defaultdict
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# In[3]:


import numpy as np
import torch
import time
from tqdm import tqdm
import pickle


# In[4]:


import lda
from sklearn.feature_extraction.text import CountVectorizer
import joblib


# ### seed 固定

# In[5]:


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

# In[6]:


train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')


# In[7]:


train_y = train_df.jobflag.values - 1


# ### テキスト正規化

# In[8]:


'''https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L26'''
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n``,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


# In[9]:


'''https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python'''
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# In[10]:


train_sequence = list(map(text_to_word_sequence, train_df.description.values))
test_sequence = list(map(text_to_word_sequence, test_df.description.values))


# In[11]:


stop_words = frozenset(stopwords.words('english'))
remove_sw = lambda x: [word for word in x if word not in stop_words]

train_sequence = list(map(remove_sw, train_sequence))
test_sequence = list(map(remove_sw, test_sequence))


# In[12]:


train_sequence = list(map(nltk.pos_tag, train_sequence))
test_sequence = list(map(nltk.pos_tag, test_sequence))


# In[13]:


wnl = WordNetLemmatizer()
lemmatize = lambda x: [wnl.lemmatize(w, get_wordnet_pos(pos)) if get_wordnet_pos(pos) else w for w, pos in x]
train_sequence = list(map(lemmatize, train_sequence))
test_sequence = list(map(lemmatize, test_sequence))


# In[14]:


word_counter = Counter([word for seq in train_sequence for word in seq])


# In[15]:


train_X = list(map(lambda x: [word for word in x if word_counter[word] >= 2], train_sequence))
test_X = list(map(lambda x: [word for word in x if word_counter[word] >= 2], test_sequence))
train_X = np.array(list(map(' '.join, train_X)))
test_X = np.array(list(map(' '.join, test_X)))
all_X = np.concatenate([train_X, test_X], axis=0)


# ### 重複文の削除

# In[16]:


x2y = defaultdict(list)
for x, y in zip(train_X, train_y):
    x2y[x] += [y]


# In[17]:


ctrain_X = []
ctrain_y = []
ids = []
exist_sentence = {}

for sentence_id, (x, y) in enumerate(zip(train_X, train_y)):
    if len(set(x2y[x])) == 1 and exist_sentence.get(x) is None:
        ids += [sentence_id]
        ctrain_X += [x]
        ctrain_y += [y]
        exist_sentence[x] = y
    else:
        print(x, set(x2y[x]))

ctrain_X = np.array(ctrain_X)
ctrain_y = np.array(ctrain_y)
ids = np.array(ids)


# In[18]:


len(train_X), len(ctrain_X)


# ### トピックモデルの作成

# In[19]:


bow_model = CountVectorizer(stop_words='english')
bow = bow_model.fit_transform(all_X)
# 初回実行時のみ保存
# joblib.dump(bow_model, '../models/bow_model.pkl')


# In[20]:


bow_model = joblib.load('../models/bow_model.pkl')


# In[21]:


n = 20
n_iter = 2000
start = time.time()
lda_model = lda.lda.LDA(n_topics=n, n_iter=n_iter, random_state=0, refresh=100)
lda_model.fit(bow)
# 初回実行時のみ保存
# joblib.dump(lda_model, '../models/lda_model_{}_{}iter.pkl'.format(n, n_iter))
end = time.time()
print("topic_N =", str(n), "train time", end - start)


# In[22]:


lda_model = joblib.load('../models/lda_model_20_2000iter.pkl')


# In[23]:


bow = bow_model.transform(all_X)
theta_docs_20 = lda_model.transform(bow)


# In[24]:


train_theta = theta_docs_20[:len(train_X)]
test_theta = theta_docs_20[len(train_X):]


# In[25]:


train_topic = train_theta.argmax(axis=1)
test_topic = test_theta.argmax(axis=1)


# In[26]:


train_df['topic_id'] = train_topic
test_df['topic_id'] = test_topic


# In[27]:


train_theta_df = pd.DataFrame(train_theta)
train_theta_df.columns = [f'topic{i}' for i in range(train_theta.shape[1])]
test_theta_df = pd.DataFrame(test_theta)
test_theta_df.columns = [f'topic{i}' for i in range(test_theta.shape[1])]

train_df = pd.concat([train_df, train_theta_df], axis=1)
test_df = pd.concat([test_df, test_theta_df], axis=1)


# In[28]:


train_df = train_df.iloc[ids]


# In[29]:


train_df.to_csv('../processed_data/train_df_2865_topic_theta.csv', index=False)
test_df.to_csv('../processed_data/test_df_topic_theta.csv', index=False)


# ### validationの分割

# In[30]:


train_y = train_df.jobflag.values - 1


# In[31]:


topic2valn = {}
for topic_id, count in ((573 * test_df.topic_id.value_counts(normalize=True) * 2 + 1) // 2).items():
    topic2valn[topic_id] = int(count)


# In[32]:


train_ids = train_df.id.values


# In[33]:


ids_train = []
y_val = []
ids_val = []
train_topic = train_df.topic_id.values
for i in range(20):
# for i in range(10):
    sub_ids = train_ids[train_topic == i]
    sub_train_y = train_y[train_topic == i]
    valn = topic2valn[i]
    shuffle_idx = np.random.permutation(len(sub_ids))
    ids_train += list(sub_ids[shuffle_idx][valn:])
    y_val += list(sub_train_y[shuffle_idx][:valn])
    ids_val += list(sub_ids[shuffle_idx][:valn])

ids_train = np.array(ids_train, dtype=int)
y_val = np.array(y_val, dtype=int)
ids_val = np.array(ids_val, dtype=int)


# In[34]:


test_rate = np.array([0.2314, 0.1833, 0.1982, 0.3872])
val_n = Counter(y_val)[1] / test_rate[1]
cat2valn = ((val_n * test_rate * 2 + 1) // 2).astype(np.int32)
cat2valn


# In[35]:


ids_val2 = []
for i in range(4):
    sub_ids = ids_val[y_val == i]
    valn = cat2valn[i]
    shuffle_idx = np.random.permutation(len(sub_ids))
    ids_val2 += list(sub_ids[shuffle_idx][:valn])

ids_val2 = np.array(ids_val2, dtype=int)


# In[36]:


in_train = train_df.id.apply(lambda x: x in ids_train)
train_df[in_train].to_csv('../processed_data/train_df_2293_theta.csv', index=False)
in_val = train_df.id.apply(lambda x: x in ids_val)
train_df[in_val].to_csv('../processed_data/val_df_572_theta.csv', index=False)
in_val2 = train_df.id.apply(lambda x: x in ids_val2)
train_df[in_val2].to_csv('../processed_data/val2_df_309_theta.csv', index=False)


# In[37]:


train_df[in_train].shape, train_df[in_val].shape, train_df[in_val2].shape


# In[38]:


magic_seed = [42, 346, 291, 241, 312, 150, 353, 310, 266, 188]
for s in magic_seed:
    train_df = pd.read_csv('../processed_data/train_df_2293_theta.csv')
    mins = []
    for k,v in Counter(train_df.jobflag).items():
        print(k,v)
        mins.append(v)
    mins = min(mins)
    train_ids = []
    for i in range(1,5):
        train_ids += train_df[train_df["jobflag"]==i].sample(mins,random_state=s).id.values.tolist()
    train_df = train_df[train_df.id.apply(lambda x: x in set(train_ids))]
    print(len(train_df))
    train_df.to_csv('../processed_data/train_df_1088_theta_seed_{}.csv'.format(s), index=False)


# In[ ]:




