#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
sys.path.append(sys.argv[1])

import random
import numpy as np

from sklearn.model_selection import train_test_split

#import os
#import time
#import argparse
#from collections import defaultdict
#import pandas as pd
#from keras.utils import to_categorical
#from keras.models import Model, Input
#from keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
#from keras_contrib.layers import CRF
#import matplotlib.pyplot as plt

from models.argparser import get_args
from models.loader import load_word_embeddings, load_conll, write_conll

from utils.input2feats import wordsents2sym
from utils.mapper import get_optimizer, get_loss

from models.nnmodels import get_model


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
tag2idx, word_sents, max_len = load_conll(args.raw_data,
                                     set(word2idx.keys()),
                                     oovs = oov_words,
                                     pads = pad_words,
                                     default_tag = DEFAULT_TAG,
                                     len_perc = args.max_len_perc,
                                     lower = False)

# output processed sentences for reference
write_conll(args.data, word_sents)

# map word sentences and their tags to a symbolic representation
X_word, y_word, nb_classes = wordsents2sym(word_sents, max_len, word2idx, tag2idx,
                   oov_words['unknown'], DEFAULT_TAG,
                   pad_words['pad'], DEFAULT_TAG)

# split word data into training and test
print('[INFO] Splitting word data into training and test...')
X_word_train, X_word_test, y_word_train, y_word_test = train_test_split(X_word, y_word, test_size=args.test_size)
print('[INFO] Training split for words contains:', X_word_train.shape, '-->', y_word_train.shape)
print('[INFO] Test split for words contains:', X_word_test.shape, '-->', y_word_test.shape)

# build inputs and outputs for the model
X_train = [X_word_train, ]
y_train = [y_word_train, ]
X_test = [X_word_test, ]
y_test = [y_word_test, ]

# build the specified neural model
model = get_model(args.model, args.model_size, args.num_layers, args.noise_sigma, args.hidden_activation,
                  args.output_activation, args.dropout, args.batch_normalization)


# WE CAN HAVE VARIOUS LOSSES, WITH WEIGHTS, AND VARIOUS METRICS

#model.compile(optimizer = )

#model_outputs = [y_train, ]
#model_losses = ['categorical_crossentropy', ]
#model_loss_weights = [1.0, ]
#model_metrics = [actual_accuracy, ]

#model = build_model()

#model.compile(optimizer='adam',
#              loss=model_losses,
#              loss_weights=model_loss_weights,
#              metrics=model_metrics)
#model.summary()

print(X_train)
print(y_train)
sys.exit()

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



