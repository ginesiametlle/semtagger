#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
sys.path.append(sys.argv[1])

import random
import numpy as np

#import os
#import time
#import argparse
#from collections import defaultdict
#import pandas as pd
#from keras.utils import to_categorical
#from sklearn.model_selection import train_test_split
#from keras.models import Model, Input
#from keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
#from keras_contrib.layers import CRF
#import matplotlib.pyplot as plt

from models.argparser import get_args
from models.loader import load_word_embeddings, load_conll, write_conll
from utils.input2feats import tagged_sents2feats

# set random seeds to ensure comparability of results
random.seed(7937)
np.random.seed(7937)

# define padding words to use and their tags
# these are appended to the beginning and the end of the input sentences
pad_words = {}
pad_words['begin'] = '<s>'
pad_words['end'] = '</s>'
pad_words['pad'] = '<pad>'

# define aliases to use for out-of-vocabulary (oov) words
# these are employed to replace oov words in the input sentences
oov_words = {}
oov_words['number'] = '<num>'
oov_words['unknown'] = '<unk>'

# default sem-tag to which special words are mapped
DEFAULT_TAG = 'NIL'

# obtain arguments
args = get_args()

# load word embedding vectors
word2idx, emb_matrix, emb_dim = load_word_embeddings(args.embeddings,
                                                oovs = list(oov_words.values()),
                                                pads = list(pad_words.values()),
                                                sep = ' ',
                                                lower = False)

# read and pad input sentences and their tags
tag2idx, sents, max_len = load_conll(args.raw_data,
                            set(word2idx.keys()),
                            oovs = oov_words,
                            pads = pad_words,
                            default_tag = DEFAULT_TAG,
                            len_perc = args.max_len_perc,
                            lower = False)

# output processed sentences for reference
write_conll(args.data, sents)

# map read sentences and their tags to a feature vector
X, y = tagged_sents2feats(sents)
sys.exit()

# START THE FUN

# LOOK OUT FOR A LOT OF OOVS
X = [[word2idx[w[0]] if w[0] in word2idx else 0 for w in s] for s in sents]
X = pad_sequences(maxlen=args.max_sent_len, sequences=X, padding="post", value=0)
#print(X)

n_tags = len(tag2idx)
#print(n_tags)
y = [[tag2idx[w[1]] for w in s] for s in sents]
y = pad_sequences(maxlen=args.max_sent_len, sequences=y, padding="post", value=tag2idx["NIL"])
y = [to_categorical(i, num_classes=n_tags) for i in y]
#print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)


input = Input(shape=(args.max_sent_len,))
model = Embedding(input_dim=n_words, output_dim=50,
                  input_length=args.max_sent_len, mask_zero=True)(input)  # 50-dim embedding
model = Bidirectional(GRU(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=20, validation_split=0.1, verbose=1)


hist = pd.DataFrame(history.history)

plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()


# do some silly predictions
#i = 50

p = model.predict(np.array(X_test))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test, -1)
lengths = [len(s) for s in sents]
total = 0
correct = 0

for triple in zip(p, true, lengths):
    pred_tags = triple[0]
    true_tags = triple[1]
    l = triple[2]
    for n in range(min(args.max_sent_len,l)):
        total += 1
        if pred_tags[n] == true_tags[n]:
            correct += 1

print('Accuracy on the test set: ', correct/total)

#for predtags in p:
#    for truetags in true:
        # determine the real length of the sentence

#print(p)
#print(true)
#print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
#print(30 * "=")
#for w, t, pred in zip(X_test[i], true, p[0]):
#    if w != 0:
#        print("{:15}: {:5} {}".format(str(w), str(t), str(pred)))



