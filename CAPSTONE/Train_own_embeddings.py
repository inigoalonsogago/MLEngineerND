PATH_DRIVE = "Datasets/"

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import io

from keras.datasets import imdb
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, LSTM
from keras.layers.embeddings import Embedding

from string import punctuation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.utils import lemmatize
nltk.download('punkt')
nltk.download('stopwords')

from collections import Counter
import random

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

PATH_IMDB_TRAIN = PATH_DRIVE + "Imdb_train.csv"
PATH_IMDB_TEST = PATH_DRIVE + "Imdb_test.csv"

PATH_CORPUS_CINE = PATH_DRIVE + "CorpusCine.csv"

PATH_MUSE_EN = PATH_DRIVE + "wiki.multi.en.vec"
PATH_MUSE_ES = PATH_DRIVE + "wiki.multi.es.vec"

PATH_DATA = PATH_DRIVE + "data/"

cols = ["review","label"]
df = pd.read_csv(PATH_IMDB_TRAIN,header=None, names=cols)
df_test = pd.read_csv(PATH_IMDB_TEST,header=None, names=cols)
df = df.append(df_test)

df["label"].count()

df.review[0]

"""# Preprocessing

## Tokenize — Encode the words
"""

def clean(sentence):
  
  # lowercase
  sentence = sentence.lower()
  
  # punctuation
  sentence = ''.join([c for c in sentence if c not in punctuation])
  
  # tokenize
  tokens = word_tokenize(sentence)
  
  return tokens

def preprocessing(df):
  
  cleaned_rev = []
  count = Counter()
  for review in df["review"]:
#     print(review)
    rev = clean(review)
    cleaned_rev.append(rev)
    count.update(rev)
   
  vocab_to_int = {w:i for i, (w,c) in enumerate(count.most_common())}
  
  vocab_to_word = {i:w for (w,i) in vocab_to_int.items()}
    
  return cleaned_rev, vocab_to_int, vocab_to_word

cleaned_rev, vocab_to_int, vocab_to_word = preprocessing(df)

reviews_int = []
for review in cleaned_rev:
    r = [vocab_to_int[w] for w in review]
    reviews_int.append(r)
print (reviews_int[0:3])

labels = []
for label in df["label"]:
    labels.append(label)

"""## Analyzing review length"""

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.show()
pd.Series(reviews_len).describe()

"""### Outliers"""

reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
labels = [ labels[i] for i, l in enumerate(reviews_len) if l> 0 ]

"""### padding"""

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

len_feat = 300
features = pad_features(reviews_int, len_feat)

feature_s = features.copy()
labels_s = labels.copy()

labels_a = np.array(labels)
indices = np.arange(features.shape[0])
np.random.shuffle(indices)

features_a = features[indices]
labels_a = labels_a[indices]

features = features_a
labels =labels_a

split_frac = 0.8

train_x = features[0:int(split_frac*len(features))]
train_y = labels[0:int(split_frac*len(features))]

remaining_x = features[int(split_frac*len(features)):]
remaining_y = labels[int(split_frac*len(features)):]

valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]

test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

"""# Embeddings

## load
"""

def load_vec(emb_path, nmax):
    '''
    INPUT:
    emb_path: path of the txt file where the word embeddings are
    nmax: maxium number of word embeddings you want to load (I guess that
    the words are ordered from more frequent to least frequent)
    
    OUTPUT:
    embedings: array of dimensions words(nmax) x embedding dimensions (300)
    id2word: dictionary id (keys) word(values)
    word2id: dictionary word (keys) id(values)
    '''
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

nmax = 50000
src_embeddings, src_id2word, src_word2id = load_vec(PATH_MUSE_EN, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(PATH_MUSE_ES, nmax)

print("muse eng dim : ", src_embeddings.shape)
print("muse es dim : ", tgt_embeddings.shape)

"""## Nearest neighbors"""

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))

get_nn("cat", src_embeddings, src_id2word, src_embeddings, src_id2word, K=1)

"""## Update Embeddings Indexing"""

voc_emb_id={}
wordsno=[]
for i, word in vocab_to_word.items():
    try:
        src_word2id[word]          
        voc_emb_id[word] = i
    except KeyError:
        wordsno.append(word)
        pass
      
nb_features = 300
ordered_emb = np.zeros((len(vocab_to_int), nb_features))

for word, index in voc_emb_id.items():
    en_id = src_word2id[word]
    ordered_emb[index] = src_embeddings[en_id][:nb_features]

print(src_id2word[2])
for i, emb in enumerate(ordered_emb):
  if emb[0] == src_embeddings[2][0]:
      print(vocab_to_word[i])

ordered_emb.shape

"""# Model"""

embedding_size=200
model=Sequential()
model.add(Embedding(len(vocab_to_int), embedding_size, input_length=len_feat))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

def plot(history, nb_epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    first_epoch = 0
    epochs = range(first_epoch + 1, nb_epochs + 1)

    plt.plot(epochs, loss_values[first_epoch:], 'b', label='Training loss')
    plt.plot(epochs, val_loss_values[first_epoch:], 'r', label='Validation loss')
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False);

    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc_values[first_epoch:], 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc_values[first_epoch:], 'r', label='Validation Accuracy')
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=False);

num_epochs = 10
history = model.fit(train_x, train_y, batch_size = 32, epochs=num_epochs, validation_data=(valid_x, valid_y), verbose=1)

plot(history, num_epochs)

loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
print('Accuracy: %f' % (accuracy*100))