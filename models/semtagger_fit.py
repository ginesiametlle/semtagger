#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
sys.path.append(sys.argv[1])

import os
import random
import itertools
import operator

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from models.argparser import get_args
from models.loader import load_embeddings, load_conll
from models.loader import make_char_seqs, write_conll, write_chars
from models.nnmodels import get_model

from utils.convert_input2feats import wordsents2sym, charsents2sym
from utils.data_stats import plot_dist_tags, plot_confusion_matrix, plot_accuracy
from utils.data_stats import plot_confusion_matrix


###################
### DEFINITIONS ###
###################

# set random seeds to ensure comparability of results
rnd_seed = 7937
random.seed(rnd_seed)
np.random.seed(rnd_seed)

# define padding symbols to use and their tags
# these are appended to the beginning and the end of the input sentences
pad_sym = {}
pad_sym['begin'] = '<s>'
pad_sym['end'] = '</s>'
pad_sym['pad'] = '<pad>'

# define aliases to use for out-of-vocabulary (oov) symbols
# these are employed to replace oov items in the input sentences
oov_sym = {}
oov_sym['number'] = '<num>'
oov_sym['unknown'] = '<unk>'

# default sem-tag to which special words are mapped
DEFAULT_TAG = 'NIL'

# obtain arguments
args = get_args()

if not args.use_words and not args.use_chars:
    print('[ERROR] Cannot use words nor character features')
    print('[ERROR] A neural model will not be trained...')
    sys.exit()

#### MODIFICATIONS FOR SMALL EXPERIMENTS
args.use_chars = 0
args.test_size = 0.2
args.dev_size = 0.1
args.grid_search = 0
args.epochs = 5
args.model_size = 40
args.num_layers = 1
args.noise_sigma = 0.0
args.batch_normalization = 0


#############################
### LOAD AND PROCESS DATA ###
#############################

# load word embedding vectors
word2idx, wemb_matrix, wemb_dim = load_embeddings(args.word_embeddings,
                                                  oovs = list(oov_sym.values()),
                                                  pads = list(pad_sym.values()),
                                                  sep = ' ',
                                                  lower = False)

# read and pad input sentences and their tags
tag2idx, word_sents, max_wlen = load_conll(args.raw_pmb_data,
                                           extra = args.raw_extra_data,
                                           vocab = set(word2idx.keys()),
                                           oovs = oov_sym,
                                           pads = pad_sym,
                                           default_tag = DEFAULT_TAG,
                                           len_perc = args.max_len_perc,
                                           lower = False,
                                           mwe = args.multi_word)


#### INFO
# plot distribution over tags
plot_dist_tags(word_sents,
               set(word2idx.keys()),
               os.path.dirname(args.output) + '/tagdist.svg',
               set(pad_sym.values()))


# load character embedding vectors and input sequences
if args.use_chars:
    char2idx, cemb_matrix, cemb_dim = load_embeddings(args.char_embeddings,
                                                      oovs = list(oov_sym.values()),
                                                      pads = list(pad_sym.values()),
                                                      sep = ' ',
                                                      lower = False)

    char_sents, max_clen = make_char_seqs(word_sents,
                                          vocab = set(char2idx.keys()),
                                          oovs = oov_sym,
                                          pads = pad_sym,
                                          lower = False)


#### INFO
# output processed word sentences for reference
write_conll(args.data_words, word_sents)
if args.use_chars:
    write_chars(args.data_chars, char_sents)


# map word sentences and their tags to a symbolic representation
if args.use_words:
    X_word, y_word, nb_classes = wordsents2sym(word_sents, max_wlen,
                                               word2idx, tag2idx,
                                               oov_sym['unknown'], DEFAULT_TAG,
                                               pad_sym['pad'], DEFAULT_TAG)
if args.use_chars:
    X_char = charsents2sym(char_sents, max_clen,
                           char2idx, oov_sym['unknown'], pad_sym['pad'])


# split word data into training and test
if args.use_words:
    print('[INFO] Splitting word data into training and test...')
    X_word_train, X_word_test, y_word_train, y_word_test = train_test_split(X_word, y_word, test_size=args.test_size)
    print('[INFO] Training split for words contains:', X_word_train.shape, '-->', y_word_train.shape)
    print('[INFO] Test split for words contains:', X_word_test.shape, '-->', y_word_test.shape)

if args.use_chars:
    print('[INFO] Splitting character data into training and test...')
    X_char_train, X_char_test = train_test_split(X_char, test_size=args.test_size)
    print('[INFO] Training split for characters contains:', X_char_train.shape)
    print('[INFO] Test split for characters contains:', X_char_test.shape)


#############################
### PREPARE TAGGING MODEL ###
#############################

# build input and output data for the model
y_train = y_word_train
y_test = y_word_test

if args.use_words and args.use_chars:
    X_train = [X_word_train, X_char_train]
    X_test = [X_word_test, X_char_test]
elif args.use_words:
	X_train = X_word_train
	X_test = X_word_test
elif args.use_chars:
	X_train = X_char_train
	X_test = X_word_test


# compute size of the word and character vocabulary
num_words = 0
num_chars = 0
if args.use_words:
    num_words = len(word2idx.keys())
if args.use_chars:
    num_chars = len(char2idx.keys())


# obtain model object
if not args.use_words:
    max_wlen = 0
    num_words = 0
    wemb_dim = 0
    wemb_matrix = None
if not args.use_chars:
    max_clen = 0
    num_chars = 0
    cemb_dim = 0
    cemb_matrix = None

model = get_model(args, max_wlen, num_words, wemb_dim, wemb_matrix, nb_classes, max_clen, num_chars, cemb_dim, cemb_matrix)
model.summary()


# train the model
history = model.fit(X_train, np.array(y_train), batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, verbose=args.verbose)


#### INFO
# plot how the training went
plot_accuracy(history,
              ['strict_accuracy', 'val_strict_accuracy'],
              ['Training data', 'Dev. data'],
              0.6,
              os.path.dirname(args.output) + '/acc.svg')


#### INFO
# predict on the test set and plot confusion matrix
p = model.predict(np.array(X_test), verbose=min(1, args.verbose))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test, -1)
lengths = [len(s) for s in word_sents]y

total = 0
correct = 0

for triple in zip(p, true, lengths):
    pred_tags = triple[0]
    true_tags = triple[1]
    l = triple[2]
    for n in range(min(max_wlen,l)):
        total += 1
        if pred_tags[n] == true_tags[n]:
            correct += 1

print('Accuracy on the test set: ', correct/total)

# CONFUSION MATRIX FOR TEST
plot_confusion_matrix(p, true, lenghts, word2idx, True)


p = model.predict(np.array(X_train), verbose=min(1, args.verbose))
p = np.argmax(p, axis=-1)
true = np.argmax(y_train, -1)
lengths = [len(s) for s in word_sents]y

# CONFUSTION MATRIX FOR TRAIN
plot_confusion_matrix(p, true, lenghts, word2idx, True)


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

