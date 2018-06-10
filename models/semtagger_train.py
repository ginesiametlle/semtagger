#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
sys.path.append(sys.argv[1])

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import random
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import keras
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from models.argparser import get_args
from models.loader import load_embeddings, load_conll, write_conll
from models.loader import make_char_seqs, write_chars
from models.nn import get_model
from models.optimizer import grid_search_params
from models.metrics import strict_accuracy_N

from utils.input2feats import wordsents2sym, charsents2sym
from utils.data_stats import plot_dist_tags, plot_dist_lengths
from utils.data_stats import plot_accuracy, plot_confusion_matrix

#sys.stderr = sys.__stderr__


###################
### DEFINITIONS ###
###################

# set random seeds to ensure comparability of results
rnd_seed = 7393
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
                                           len_perc = args.sent_len_perc,
                                           lower = False,
                                           mwe = args.multi_word,
                                           unk_case = True)

# randomize the input data
random.shuffle(word_sents)

# map word sentences and their tags to a symbolic representation
print('[INFO] Reshaping word data...')
X_word, y_tag = wordsents2sym(word_sents,
                              max_slen,
                              word2idx,
                              tag2idx,
                              oov_sym['unknown'],
                              DEFAULT_TAG,
                              pad_word['pad'],
                              PADDING_TAG)

#### INFO
# plot length distribution
plot_dist_lengths([len(s) for s in word_sents], max_slen,
                  10, 140, 1000, 13000,
                  os.path.dirname(args.output_model) + '/length_dist.svg')

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
                                          lower = False)

	# map character sentences and their tags to a symbolic representation
    print('[INFO] Reshaping character data...')
    X_char = charsents2sym(char_sents,
                           max_slen,
                           max_wlen,
                           char2idx,
                           oov_sym['unknown'],
                           pad_char['begin'],
                           pad_char['end'],
                           pad_char['pad'])

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

# compute size target classes and word/character vocabulary
num_words = 0
num_chars = 0
num_tags = len(tag2idx.keys())
if args.use_words:
    num_words = len(word2idx.keys())
if args.use_chars:
    num_chars = len(char2idx.keys())

# set not used model parameters to dummy values
if not args.use_words:
    wemb_dim = 0
    wemb_matrix = None
if not args.use_chars:
    max_wlen = 0
    cemb_dim = 0
    cemb_matrix = None


###########################
### PERFORM GRID SEARCH ###
###########################
if args.grid_search:
    print('[INFO] Performing grid-search...')
    # number of samples for cross-validation
    n_samples = 3
    args = grid_search_params(args,
                              n_samples,
                              X_train,
                              y_train,
                              tag2idx[PADDING_TAG],
                              num_tags,
                              max_slen, num_words, wemb_dim, wemb_matrix,
                              max_wlen, num_chars, cemb_dim, cemb_matrix)


#########################
### FIT TAGGING MODEL ###
#########################

# create a new model
model = get_model(args,
                  num_tags,
                  max_slen, num_words, wemb_dim, wemb_matrix,
                  max_wlen, num_chars, cemb_dim, cemb_matrix)
model.summary()

# train the model
history = model.fit(X_train, np.array(y_train),
                    batch_size = args.batch_size, epochs = args.epochs,
                    validation_split = args.dev_size, verbose = args.verbose)

#### INFO
# predict using the model
classes = [x[0] for x in tag2idx.items() if x[0] != PADDING_TAG]
idx2tag = {v: k for k, v in tag2idx.items()}

# predictions on the training set
p_train = model.predict(X_train, verbose = min(1, args.verbose))
p_train = np.argmax(p_train, axis=-1) + 1
true_train = np.argmax(y_train, axis=-1) + 1
train_acc = strict_accuracy_N(true_train, p_train, tag2idx[PADDING_TAG])
print('Accuracy on the training set: ', train_acc)

# predictions on the test set
if args.test_size > 0:
    p_test = model.predict(X_test, verbose = min(1, args.verbose))
    p_test = np.argmax(p_test, axis=-1) + 1
    true_test = np.argmax(y_test, axis=-1) + 1
    test_acc = strict_accuracy_N(true_test, p_test, tag2idx[PADDING_TAG])
    print('Accuracy on the test set: ', test_acc)
else:
    test_acc = 0

#### INFO
# plot confusion matrix (train + test)
plot_confusion_matrix(true_train, p_train, classes, tag2idx[PADDING_TAG], idx2tag,
                      os.path.dirname(args.output_model) + '/cmat_train_oov.svg',
                      vocab = set(word2idx.keys()),
                      normalize = True)
if args.test_size > 0:
    plot_confusion_matrix(true_test, p_test, classes, tag2idx[PADDING_TAG], idx2tag,
                          os.path.dirname(args.output_model) + '/cmat_test_oov.svg',
                          vocab = set(word2idx.keys()),
                          normalize = True)

#### INFO
# plot how the training went
plot_keys = ['strict_accuracy_K']
plot_labels = ['Training data']
if args.dev_size > 0:
    plot_keys += ['val_strict_accuracy_K']
    plot_labels += ['Dev. data']
plot_accuracy(history,
              plot_keys,
              plot_labels,
              test_acc,
              os.path.dirname(args.output_model) + '/semtag_acc.svg',
              os.path.dirname(args.output_model) + '/semtag_acc.txt')


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
minfo['word2idx'] = word2idx
minfo['max_slen'] = max_slen
if args.use_words:
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

