

PATH_DRIVE = "Datasets/"

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import random
from collections import Counter
import io

import collections
import string
from string import punctuation

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from gensim.utils import lemmatize

from keras.datasets import imdb
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, LSTM
from keras.layers.embeddings import Embedding

PATH_IMDB_TRAIN = PATH_DRIVE + "Imdb_train.csv"
PATH_IMDB_TEST = PATH_DRIVE + "Imdb_test.csv"

PATH_CORPUS_CINE = PATH_DRIVE + "CorpusCine.csv"

PATH_MUSE_EN = PATH_DRIVE + "wiki.multi.en.vec"
PATH_MUSE_ES = PATH_DRIVE + "wiki.multi.es.vec"

PATH_DATA = PATH_DRIVE + "data/"

import pickle
def save_obj(obj, name ):
    with open(PATH_DRIVE+ name + '.pkl', 'wb') as f:
      pickle.dump(obj, f)#, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
  #loads from the folder datasets
    with open(PATH_DRIVE+ name + '.pkl', 'rb') as f:
        return pickle.load(f)


cols = ["review","label"]
df = pd.read_csv(PATH_IMDB_TRAIN,header=None, names=cols)
df_test = pd.read_csv(PATH_IMDB_TEST,header=None, names=cols)
df = df.append(df_test)



df_es = pd.read_csv(PATH_CORPUS_CINE,header=None, names=cols)

print(df["label"].count())
print(df_es["label"].count())



"""# Spanish Preprocessing"""

df_es.review[0]

del_jump = lambda x: x.replace('\n','')
df_es['review'] = df_es['review'].apply(del_jump)
df_es.review[0]

table = collections.defaultdict(lambda: None)
table.update({
    ord('á'):'Ã¡',
    ord('é'):'Ã©',
    ord('í'):'Ãa',
    ord('ó'):'Ã³',
    ord('ú'):'Ãº',
    ord('ü'):'Ã¼Ãs',
    ord('ñ'):'e',
    ord('ô'):'Ã±',
    ord('ế'):'áº¿',
    ord(' '):' '
    })
# map aplica a todo los caracteres ascii la funcion ord
#crea un diccionario en el que cada numero ascii se asocia a unos caracteres
table.update(dict(zip(map(ord,string.ascii_uppercase), string.ascii_lowercase)))
table.update(dict(zip(map(ord,string.ascii_lowercase), string.ascii_lowercase)))
table.update(dict(zip(map(ord,string.digits), string.digits)))
#transforma cada caracter
convert_char = lambda x: x.translate(table)

print('are están also también you tú had tenía score puntuación years años football fútbol linguistic lingüístico film película'.translate(table,))

print(table)

df_es.review[0]

#transforma cada caracter en la review
df_es.review[0].translate(table)

df_es['review'] = df_es['review'].apply(convert_char)
df_es.review[0]

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

def clean_es(sentence):
  
  # lowercase
  sentence = sentence.lower()
  
  # punctuation
  sentence = ''.join([c.translate(table,) for c in sentence if c not in punctuation])
  
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

def preprocessing_es(df):
  
  cleaned_rev = []
  count = Counter()
  for review in df["review"]:
#     print(review)
    rev = clean_es(review)
    cleaned_rev.append(rev)
    count.update(rev)
   
  vocab_to_int = {w:i for i, (w,c) in enumerate(count.most_common())}
  
  vocab_to_word = {i:w for (w,i) in vocab_to_int.items()}
    
  return cleaned_rev, vocab_to_int, vocab_to_word

cleaned_rev, vocab_to_int, vocab_to_word = preprocessing(df)

es_cleaned_rev, es_vocab_to_int, es_vocab_to_word = preprocessing(df_es)

list(es_vocab_to_int.items())[:3]

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



"""## Update Embeddings Indexing"""

list_src = list(src_word2id.keys())
list_imdb = list(vocab_to_int.keys())
list_common = []
for word in list_imdb:
  if word in list_src:
    list_common.append(word)

print('Number of words on both, the embeddings and the imbd dataset', len(list_common))
#it can be seen how the punctuation symbols are not in the common set
print(list(src_word2id.keys())[:5])
print(list_common[:5])

#create dictionaries for transformations
voc_to_id = {w:i for i,w in enumerate(list_common)}
voc_to_word = {i:w for i,w in enumerate(list_common)}

es_voc_to_id = load_obj("es_voc_to_id" )
voc_to_id = load_obj("en_voc_to_id" )
print(es_voc_to_id)
print(len(es_voc_to_id))
len(voc_to_id)

voc_to_word = {i:w for (w,i) in voc_to_id.items()}
es_voc_to_word = {i:w for (w,i) in es_voc_to_id.items()}

print(list(voc_to_id.items())[:10])

print(list(vocab_to_int.items())[:10])

voc_emb_id={}
wordsno=[]
for i, word in voc_to_word.items():
    try:
        src_word2id[word]          
        voc_emb_id[word] = i
    except KeyError:
        wordsno.append(word)
        pass
      
print(len(voc_emb_id))
      
nb_words = 300
ordered_emb = np.zeros((len(voc_to_id), nb_words))

for word, index in voc_emb_id.items():
    en_id = src_word2id[word]
    ordered_emb[index] = src_embeddings[en_id][:nb_words]
print(ordered_emb.shape)

print((len(wordsno)/len(list(vocab_to_word.items())))*100)
len(list(vocab_to_word.items()))-len(wordsno)

print(src_id2word[2])
for i, emb in enumerate(ordered_emb):
  if emb[0] == src_embeddings[2][0]:
      print(vocab_to_word[i])

ordered_emb.shape

"""## Update spanish embeddings"""
''' The alignment of the spanish embeddings is commented, the output file is available on datasets'''

# print(len(list(es_vocab_to_int.items())))

# print(es_vocab_to_int.keys())
# print(es_vocab_to_int["historia"])
# print(tgt_word2id["historia"])
# emb_es_hist = tgt_embeddings[tgt_word2id["historia"]]

# list_w = ['historia', 'que', 'la', 'y']
# list_emb = [tgt_embeddings[tgt_word2id[i]] for i in list_w]

# def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
#     print("Nearest neighbors of \"%s\":" % word)
#     word2id = {v: k for k, v in src_id2word.items()}
#     word_emb = src_emb[word2id[word]]
#     scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
#     k_best = scores.argsort()[-K:][::-1]
#     for i, idx in enumerate(k_best):
#       print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
# #       print(idx)

# def get_closest(word_emb, src_emb, tgt_emb):
#     scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
#     k_best = scores.argsort()[-1:][::-1]
#     return k_best[0]

# nb_features = 300

# en_emb_50 = np.zeros((len(src_embeddings), nb_features))
# for i,emb in enumerate(src_embeddings):
#   en_emb_50[i] = emb[:nb_features]
  
# es_emb_50 = np.zeros((len(tgt_embeddings), nb_features))
# for i,emb in enumerate(tgt_embeddings):
#   es_emb_50[i] = emb[:nb_features]
  
# ord_emb_50 = np.zeros((len(ordered_emb), nb_features))
# for i,emb in enumerate(ordered_emb):
#   ord_emb_50[i] = emb[:nb_features]

# get_nn("historia", en_emb_50, src_id2word, es_emb_50, tgt_id2word, K=1)

# for w in list_w:
#   emb = tgt_embeddings[tgt_word2id[w]]
# #   print("{} :".format(w))#
#   #closest_word(voc_emb_id, ordered_emb, vocab_to_word, emb)))
#   get_nn(w, es_emb_50, tgt_id2word, ord_emb_50, voc_to_word, K=1)
# #   print(get_closest(emb, tgt_embeddings, ordered_emb))

# es_voc_to_id = {}
# miss=[]
# for i,word in enumerate(es_vocab_to_int.keys()):
#   if not i%300:
#     print("{}% : {}% missing".format( (i/len(es_vocab_to_int.keys()))*100, ((len(miss)+1)/(i+1)*100)))
#   try : 
#     emb = es_emb_50[tgt_word2id[word]]
#     es_voc_to_id[word] = get_closest(emb,es_emb_50, ord_emb_50)
#   except KeyError:
#     miss.append(word)
#     pass

# save_obj(es_voc_to_id, "es_voc_to_id_300" )
# save_obj(voc_to_id, "en_voc_to_id_300" )

# print(len(miss))

# es_vocab_to_id = load_obj("es_voc_to_id" )
# en_voc_to_id = load_obj("en_voc_to_id" )
# print(len(es_vocab_to_id))
# len(en_voc_to_id)

es_reviews_int = []
for review in es_cleaned_rev:
    r = [es_voc_to_id[w] for w in review if w in es_voc_to_id.keys()]
    es_reviews_int.append(r)
print (es_reviews_int[0:3])

es_labels = []
for label in df_es["label"]:
    es_labels.append(label)

"""# Indexing English dataset"""

vocab_to_int = voc_to_id
vocab_to_word = voc_to_word

reviews_int = []
for i,review in enumerate(cleaned_rev):
    r = [vocab_to_int[w] for w in review  if w in vocab_to_int.keys()]
    reviews_int.append(r)
print (reviews_int[0:3])

labels = []
for label in df["label"]:
    labels.append(label)

"""## Analyzing length"""

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.show()
pd.Series(reviews_len).describe()

es_reviews_len = [len(x) for x in es_reviews_int]
pd.Series(es_reviews_len).hist()
plt.show()
pd.Series(es_reviews_len).describe()

"""## Outlier"""

reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
labels = [ labels[i] for i, l in enumerate(reviews_len) if l> 0 ]

es_reviews_int = [ es_reviews_int[i] for i, l in enumerate(es_reviews_len) if l>0 ]
es_labels = [ es_labels[i] for i, l in enumerate(es_reviews_len) if l> 0 ]

"""## Padding"""

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
es_features = pad_features(es_reviews_int, len_feat)

feature_s = features.copy()
labels_s = labels.copy()

labels_a = np.array(labels)
indices = np.arange(features.shape[0])
np.random.shuffle(indices)

features = features[indices]
labels = labels_a[indices]

es_feature_s = es_features.copy()
es_labels_s = es_labels.copy()

es_labels_a = np.array(es_labels)
es_indices = np.arange(es_features.shape[0])
np.random.shuffle(es_indices)

es_features = es_features[es_indices]
es_labels = es_labels_a[es_indices]

split_frac = 0.8

train_x = features[0:int(split_frac*len(features))]
train_y = labels[0:int(split_frac*len(features))]

remaining_x = features[int(split_frac*len(features)):]
remaining_y = labels[int(split_frac*len(features)):]

valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]

test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

"""# Model"""

# define model
model = Sequential()
e = Embedding(ordered_emb.shape[0], ordered_emb.shape[1], weights=[ordered_emb], input_length=len_feat, trainable=False)
model.add(e)
model.add(LSTM(50, dropout = 0.6))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

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

def stats(model, x_train, y_train):
  y_predict = model.predict(x_train)
  l_pred = [round(x[0]) for x in list(y_predict)]
  tp = [i for i, val in enumerate(l_pred) if val == y_train[i] and val]
  tn = [i for i, val in enumerate(l_pred) if val == y_train[i] and not val]
  fn = [i for i, val in enumerate(l_pred) if val != y_train[i] and val]
  fp = [i for i, val in enumerate(l_pred) if val != y_train[i] and not val]
  TP, TN, FP, FN = len(tp), len(tn), len(fp), len(fn)
  #accuracy = (TP + TN)/(TP + TN + FP + FN)
  #F1_score = 2*TP / (2*TP + FP + FN + keras.backend.epsilon())
  accuracy = (TP + TN)/(TP + TN + FP + FN)
  Precision = TP / (TP+FP)
  Recall = TP / (TP+FN)
  F1_score = 2*(Recall * Precision) / (Recall + Precision)


  return tp,tn,fn,fp,accuracy, F1_score

nb_epochs = 5
history = model.fit(train_x, train_y, batch_size = 32, epochs=nb_epochs, validation_data=(valid_x, valid_y), verbose=1)

plot(history, nb_epochs)

loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
print('Accuracy: %f' % (accuracy*100))

tp,tn,fn,fp,accuracy, F1 = stats(model, test_x, test_y)
print(accuracy)
print(F1)

loss, accuracy = model.evaluate(es_features, es_labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))

tp,tn,fn,fp,accuracy, F1 = stats(model, es_features, es_labels)
print(accuracy)
print(F1)

