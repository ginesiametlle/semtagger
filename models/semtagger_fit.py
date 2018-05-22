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

from sklearn.model_selection import train_test_split, GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

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

# define sentence padding symbols to use and their tags
# these are appended to the beginning and the end of the input sentences
pad_word = {}
pad_word['begin'] = '<s>'
pad_word['end'] = '</s>'
pad_word['pad'] = '<pad>'

# define word padding symbols to use and their tags
# these are appended to the beginning and the end of the input words
pad_char = {}
pad_char['begin'] = '<w>'
pad_char['end'] = '</w>'
pad_char['pad'] = '<c>'

# define aliases to use for out-of-vocabulary (oov) symbols
# these are employed to replace oov items in the input sentences
oov_sym = {}
oov_sym['number'] = '<num>'
oov_sym['unknown'] = '<unk>'
oov_sym['UNKNOWN'] = '<UNK>'

# default sem-tag used for padding
PADDING_TAG = 'PAD'
# default sem-tag to which special words are mapped
DEFAULT_TAG = 'NIL'
# list of sem-tags which we do not want to consider when displaying results
# these correspond to coarse sem-tags from the Universal Semantic Tagset
IGNORE_TAGS = set(['ANA', 'ACT', 'ATT', 'COM', 'UNE', 'DXS', 'LOG', 'MOD', 'DSC', 'NAM', 'EVE', 'TNS', 'TIM', 'UNK'])


# parse input arguments
args = get_args()

if not args.use_words and not args.use_chars:
    print('[ERROR] Cannot use words nor character features')
    print('[ERROR] A neural model will not be trained...')
    sys.exit()


#############################
### LOAD AND PROCESS DATA ###
#############################

# load word embedding vectors
print('[INFO] Loading word embeddings...')
word2idx, wemb_matrix, wemb_dim = load_embeddings(args.word_embeddings,
                                                  oovs = list(oov_sym.values()),
                                                  pads = list(pad_word.values()),
                                                  sep = ' ',
                                                  lower = False,
                                                  case_dim = True)

# read and pad input sentences and their tags
print('[INFO] Loading word sentences...')
tag2idx, word_sents, max_slen = load_conll(args.raw_pmb_data,
                                           extra = args.raw_extra_data,
                                           vocab = set(word2idx.keys()),
                                           oovs = oov_sym,
                                           pads = pad_word,
                                           padding_tag = PADDING_TAG,
                                           default_tag = DEFAULT_TAG,
                                           ignore_tags = IGNORE_TAGS,
                                           len_perc = args.sent_len_perc,
                                           lower = False,
                                           mwe = args.multi_word,
                                           unk_case = True)


# randomize the input data
random.shuffle(word_sents)

# map word sentences and their tags to a symbolic representation
print('[INFO] Reshaping word data...')
X_word, y_tag = wordsents2sym(word_sents, max_slen,
                              word2idx, tag2idx,
                              oov_sym['unknown'], DEFAULT_TAG,
                              pad_word['pad'], PADDING_TAG)

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
                   set(pad_word.values()))

    # output processed word sentences for reference
    write_conll(args.output_words, word_sents)
    write_conll(args.output_words + str('.map'), (zip(x[0], x[1]) for x in zip(X_word.tolist(), y_tag.tolist())))

# compute character-based inputs
if args.use_chars:
    # load character embedding vectors
    print('[INFO] Loading character embeddings...')
    char2idx, cemb_matrix, cemb_dim = load_embeddings(args.char_embeddings,
                                                      oovs = list(oov_sym.values()),
                                                      pads = list(pad_char.values()),
                                                      sep = ' ',
                                                      lower = False,
                                                      case_dim =False)

    # read and pad input sentences
    print('[INFO] Loading character sentences...')
    char_sents, max_wlen = make_char_seqs(word_sents,
                                          vocab = set(char2idx.keys()),
                                          oovs = oov_sym,
                                          pads = pad_char,
                                          len_perc = args.word_len_perc,
                                          lower = False,
                                          mwe = args.multi_word)

	# map character sentences and their tags to a symbolic representation
    print('[INFO] Reshaping character data...')
    X_char = charsents2sym(char_sents,
                           max_slen,
                           max_wlen,
                           char2idx,
                           oov_sym['unknown'],
                           pad_char)

    # split character data into training and test
    print('[INFO] Splitting character data into training and test...')
    X_char_train, X_char_test, y_tag_train, y_tag_test = train_test_split(X_char, y_tag, test_size=args.test_size, shuffle=False)
    print('[INFO] Training split for characters contains:', X_char_train.shape, '-->', y_tag_train.shape)
    print('[INFO] Test split for characters contains:', X_char_test.shape, '-->', y_tag_test.shape)

    #### INFO
    # output proce wssed character sequences for reference
    write_chars(args.output_chars, (zip(x[0], x[1]) for x in zip(char_sents, [[z[1] for z in s] for s in word_sents])))
    write_chars(args.output_chars + str('.map'), (zip(x[0], x[1]) for x in zip(X_char.tolist(), y_tag.tolist())))


#############################
### PREPARE TAGGING MODEL ###
#############################
#print(X_word_train[0])
#print(X_char_train[0])
#print(y_tag_train[0])
#sys.exit()

# build input and output data for the model
y_train = y_tag_train
y_test = y_tag_test
if args.use_words and args.use_chars:
    X_train = [X_word_train, X_char_train]
    X_test = [X_word_test, X_char_test]
elif args.use_words:
    X_train = X_word_train
    X_test = X_word_test
elif args.use_chars:
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
    max_slen = 0
    wemb_dim = 0
    wemb_matrix = None
if not args.use_chars:
    max_wlen = 0
    cemb_dim = 0
    cemb_matrix = None



###############################################
### FIND MODEL PARAMETERS USING GRID SEARCH ###
###############################################
if args.grid_search:
    # create model
    model = KerasClassifier(build_fn=get_model)

    # define the grid search parameters
    base_args_gs = [args]
    epochs_gs = [20, 30, 40]
    batch_size_gs = [1024, 2048]
    optimizer_gs = ['rmsprop', 'adam']
    dropout_gs = [0.1, 0.3]
    model_size_gs = [200, 300]
    num_layers_gs = [1, 2]

    param_grid = dict(base_args=base_args_gs, epochs=epochs_gs, batch_size=batch_size_gs, optimizer=optimizer_gs, dropout=dropout_gs, model_size=model_size_gs, num_layers=num_layers_gs)

    #print(X_train.shape)
    #print(np.argmax(np.array(y_train), axis=-1).shape)
    # search in the grid space
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#########################
### FIT TAGGING MODEL ###
#########################

# create a new model
model = get_model(args, num_tags, max_slen, num_words, wemb_dim, wemb_matrix, max_wlen, num_chars, cemb_dim, cemb_matrix)
model.summary()

# train the model
history = model.fit(X_train, np.array(y_train), batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, verbose=args.verbose)

#### INFO
# predict using the model
classes = [x[0] for x in tag2idx.items() if x[0] != PADDING_TAG]
idx2tag = {v: k for k, v in tag2idx.items()}
lengths = [len(s) for s in word_sents]

# predictions on the training set
p_train = model.predict(X_train, verbose=min(1, args.verbose))
p_train = np.argmax(p_train, axis=-1) + 1
true_train = np.argmax(y_train, axis=-1) + 1
total_train = 0
correct_train = 0
sent_index_train = 0

for triple in zip(p_train, true_train, lengths):
    pred_tags = triple[0]
    true_tags = triple[1]
    l = lengths[sent_index_train]

    sent_index_train += 1
    for n in range(min(max_slen,l)):
        if true_tags[n] != tag2idx[PADDING_TAG]:
            total_train += 1
            if pred_tags[n] == true_tags[n]:
                correct_train += 1
print('Accuracy on the training set: ', correct_train/total_train)

# predictions on the test set
p_test = model.predict(X_test, verbose=min(1, args.verbose))
p_test = np.argmax(p_test, axis=-1) + 1
true_test = np.argmax(y_test, axis=-1) + 1
total_test = 0
correct_test = 0
sent_index_test = 0

for triple in zip(p_test, true_test, lengths):
    pred_tags = triple[0]
    true_tags = triple[1]
    l = lengths[sent_index_train + sent_index_test]

    sent_index_test += 1
    for n in range(min(max_slen,l)):
        if true_tags[n] != tag2idx[PADDING_TAG]:
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
minfo['params'] = args
minfo['pad_word'] = pad_word
minfo['pad_char'] = pad_char
minfo['oov_sym'] = oov_sym
minfo['DEFAULT_TAG'] = DEFAULT_TAG
minfo['PADDING_TAG'] = PADDING_TAG
minfo['tag2idx'] = tag2idx
minfo['idx2tag'] = idx2tag
minfo['num_tags'] = num_tags
if args.use_words:
    minfo['word2idx'] = word2idx
    minfo['max_slen'] = max_slen
    minfo['num_words'] = num_words
    minfo['wemb_dim'] = wemb_dim
    minfo['wemb_matrix'] = wemb_matrix
if args.use_chars:
    minfo['char2idx'] = char2idx
    minfo['max_wlen'] = max_wlen
    minfo['num_chars'] = num_chars
    minfo['cemb_dim'] = cemb_dim
    minfo['cemb_matrix'] = cemb_matrix

with open(args.output_model_info, 'wb') as f:
    pickle.dump(minfo, f, pickle.HIGHEST_PROTOCOL)

