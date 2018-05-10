#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
import os
sys.path.append(sys.argv[1])
#sys.stderr = open('/dev/null', 'w')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import operator
import pickle
import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from models.argparser import get_args
from models.loader import load_embeddings, load_conll
from models.loader import make_char_seqs, write_conll, write_chars
from models.nnmodels import get_model

from utils.input2feats import wordsents2sym, charsents2sym
from utils.data_stats import plot_dist_tags, plot_accuracy, plot_confusion_matrix


#sys.stderr = sys.__stderr__


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
PADDING_TAG = 'NIL'
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
args.epochs = 10
args.model_size = 40
args.num_layers = 1
args.noise_sigma = 0.0
args.batch_normalization = 0
args.use_words = 1
args.use_chars = 0


#############################
### LOAD AND PROCESS DATA ###
#############################

# load word embedding vectors
print('[INFO] Loading word embeddings...')
word2idx, wemb_matrix, wemb_dim = load_embeddings(args.word_embeddings,
                                                  oovs = list(oov_sym.values()),
                                                  pads = list(pad_sym.values()),
                                                  sep = ' ',
                                                  lower = False)

# read and pad input sentences and their tags
print('[INFO] Loading word sentences...')
tag2idx, word_sents, max_wlen = load_conll(args.raw_pmb_data,
                                           extra = args.raw_extra_data,
                                           vocab = set(word2idx.keys()),
                                           oovs = oov_sym,
                                           pads = pad_sym,
                                           padding_tag = PADDING_TAG,
                                           len_perc = args.max_len_perc,
                                           lower = False,
                                           mwe = args.multi_word)

# map word sentences and their tags to a symbolic representation
print('[INFO] Reshaping word data...')
random.shuffle(word_sents)
X_word, y_tag = wordsents2sym(word_sents, max_wlen,
                              word2idx, tag2idx,
                              oov_sym['unknown'], DEFAULT_TAG,
                              pad_sym['pad'], PADDING_TAG)

# compute word-based inputs
if args.use_words:
    # split word data into training and test
    print('[INFO] Splitting word data into training and test...')
    X_word_train, X_word_test, y_tag_train, y_tag_test = train_test_split(X_word, y_tag, test_size=args.test_size, shuffle=False)
    print('[INFO] Training split for words contains:', X_word_train.shape, '-->', y_tag_train.shape)
    print('[INFO] Test split for words contains:', X_word_test.shape, '-->', y_tag_test.shape)

    #### INFO
    # plot distribution over tags
    plot_dist_tags(word_sents,
                   set(word2idx.keys()),
                   os.path.dirname(args.output_model) + '/semtag_dist.svg',
                   os.path.dirname(args.output_model) + '/semtag_dist.txt',
                   set(pad_sym.values()))

    # output processed word sentences for reference
    write_conll(args.output_words, word_sents)


# compute character-based inputs
if args.use_chars:
    # load character embedding vectors
    print('[INFO] Loading character embeddings...')
    char2idx, cemb_matrix, cemb_dim = load_embeddings(args.char_embeddings,
                                                      oovs = list(oov_sym.values()),
                                                      pads = list(pad_sym.values()),
                                                      sep = ' ',
                                                      lower = False)

    # read and pad input sentences
    print('[INFO] Loading character sentences...')
    char_sents, max_clen = make_char_seqs(word_sents,
                                          vocab = set(char2idx.keys()),
                                          oovs = oov_sym,
                                          pads = pad_sym,
                                          lower = False)

	# map character sentences and their tags to a symbolic representation
    print('[INFO] Reshaping character data...')
    X_char = charsents2sym(char_sents, max_clen,
                           char2idx, oov_sym['unknown'], pad_sym['pad'])

    # split character data into training and test
    print('[INFO] Splitting character data into training and test...')
    X_char_train, X_char_test, y_tag_train, y_tag_test = train_test_split(X_char, y_tag, test_size=args.test_size, shuffle=False)
    print('[INFO] Training split for characters contains:', X_char_train.shape, '-->', y_tag_train.shape)
    print('[INFO] Test split for characters contains:', X_char_test.shape, '-->', y_tag_test.shape)

    #### INFO
    # output proce wssed character sequences for reference
    write_chars(args.output_chars, char_sents)


#############################
### PREPARE TAGGING MODEL ###
#############################

# build input and output data for the model
y_train = y_tag_train
y_test = y_tag_test
if args.use_words and args.use_chars:
    X_train = [X_word_train, X_char_train]
    X_test = [X_word_train, X_char_train]
if args.use_words:
    X_train = X_word_train
    X_test = X_word_test
if args.use_chars:
    X_train = X_char_train
    X_test = X_char_test

# compute size target classes and word / character vocabulary
num_words = 0
num_chars = 0
num_tags = len(tag2idx.keys())
if args.use_words:
    num_words = len(word2idx.keys())
if args.use_chars:
    num_chars = len(char2idx.keys())

# obtain model object
if not args.use_words:
    max_wlen = 0
    wemb_dim = 0
    wemb_matrix = None
if not args.use_chars:
    max_clen = 0
    cemb_dim = 0
    cemb_matrix = None

model = get_model(args, num_tags, max_wlen, num_words, wemb_dim, wemb_matrix, max_clen, num_chars, cemb_dim, cemb_matrix)
model.summary()


#########################
### FIT TAGGING MODEL ###
#########################

# train the model
history = model.fit(X_train, np.array(y_train), batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, verbose=args.verbose)

#### INFO
# predict on the training and test sets
classes = [x[0] for x in sorted(tag2idx.items(), key=operator.itemgetter(1))]
idx2tag = {v: k for k, v in tag2idx.items()}
lengths = [len(s) for s in word_sents]


p_train = model.predict(np.array(X_train), verbose=min(1, args.verbose))
p_train = np.argmax(p_train, axis=-1)
true_train = np.argmax(y_train, -1)
total_train = 0
correct_train = 0
sent_index_train = 0


for triple in zip(p_train, true_train, lengths):
    pred_tags = triple[0]
    true_tags = triple[1]
    l = lengths[sent_index_train]

    #print(word_sents[sent_index_train])
    #print(l)
    sent_index_train += 1
    for n in range(min(max_wlen,l)):
        total_train += 1
        if pred_tags[n] == true_tags[n]:
            correct_train += 1
print('Accuracy on the training set: ', correct_train/total_train)


p_test = model.predict(np.array(X_test), verbose=min(1, args.verbose))
p_test = np.argmax(p_test, axis=-1)
true_test = np.argmax(y_test, -1)
total_test = 0
correct_test = 0
sent_index_test = 0

for triple in zip(p_test, true_test, lengths):
    pred_tags = triple[0]
    true_tags = triple[1]
    l = lengths[sent_index_train + sent_index_test]

    #print(true_tags)
    #print(word_sents[sent_index_train + sent_index_test])
    #print(l)

    sent_index_test += 1
    for n in range(min(max_wlen,l)):
        total_test += 1
        if pred_tags[n] == true_tags[n]:
            correct_test += 1
print('Accuracy on the test set: ', correct_test/total_test)


#### INFO
# plot confusion matrix (train + test)
plot_confusion_matrix(p_train, true_train, lengths[:sent_index_train], classes, os.path.dirname(args.output_model) + '/cmat_train_oov.svg', idx2tag, set(word2idx.keys()), True)
plot_confusion_matrix(p_test, true_test, lengths[sent_index_train:], classes, os.path.dirname(args.output_model) + '/cmat_test_oov.svg', idx2tag, set(word2idx.keys()), True)


#### INFO
# plot how the training went
plot_accuracy(history,
              ['strict_accuracy', 'val_strict_accuracy'],
              ['Training data', 'Dev. data'],
              correct_test/total_test,
              os.path.dirname(args.output_model) + '/semtag_acc.svg')


#######################
### SAVE MODEL INFO ###
#######################
model.save_weights(args.output_model)

minfo = {}
minfo['args'] = args
minfo['pad_sym'] = pad_sym
minfo['oov_sym'] = oov_sym
minfo['DEFAULT_TAG'] = DEFAULT_TAG
minfo['tag2idx'] = tag2idx
minfo['idx2tag'] = idx2tag
minfo['num_tags'] = num_tags
if args.use_words:
    minfo['word2idx'] = word2idx
    minfo['max_wlen'] = max_wlen
    minfo['num_words'] = num_words
    minfo['wemb_dim'] = wemb_dim
    minfo['wemb_matrix'] = wemb_matrix
if args.use_chars:
    minfo['char2idx'] = char2idx
    minfo['max_clen'] = max_clen
    minfo['num_chars'] = num_chars
    minfo['cemb_dim'] = cemb_dim
    minfo['cemb_matrix'] = cemb_matrix

with open(args.output_model_info, 'wb') as f:
    pickle.dump(minfo, f, pickle.HIGHEST_PROTOCOL)

