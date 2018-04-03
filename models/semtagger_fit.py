#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
sys.path.append(sys.argv[1])
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from models.argparser import get_args
from models.loader import load_embeddings, load_conll, write_conll
from models.nnmodels import get_model
from models.metrics import strict_accuracy

from utils.convert_input2feats import wordsents2sym


# set random seeds to ensure comparability of results
rnd_seed = 7937
random.seed(rnd_seed)
np.random.seed(rnd_seed)

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

if not args.use_words and not args.use_chars:
    print('[ERROR] Cannot use words nor character features')
    print('[ERROR] A neural model will not be trained...')
    sys.exit()

# load word embedding vectors
if args.use_words:
    word2idx, wemb_matrix, wemb_dim = load_embeddings(args.word_embeddings,
                                                      oovs = list(oov_words.values()),
                                                      pads = list(pad_words.values()),
                                                      sep = ' ',
                                                      lower = False)

# load character embedding vectors
if args.use_chars:
    char2idx, cemb_matrix, cemb_dim = load_embeddings(args.char_embeddings,
                                                      oovs = list(oov_words.values()),
                                                      pads = list(pad_words.values()),
                                                      sep = ' ',
                                                      lower = False)

# read and pad input sentences and their tags
tag2idx, word_sents, max_wlen = load_conll_words(args.raw_pmb_data,
                                           extra = args.raw_extra_data,
                                           vocab = set(word2idx.keys()),
                                           oovs = oov_words,
                                           pads = pad_words,
                                           default_tag = DEFAULT_TAG,
                                           len_perc = args.max_len_perc,
                                           lower = False)

#char_sents, max_clen = load_conll_chars()

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

# build the specified neural model with Keras
model = get_model(args, max_len, len(vocab), emb_dim, nb_classes, rnd_seed)

# compile the Keras model
# HOW TO ACCOUNT FOR VARIOUS INPUTS (CHARS, AUX LOSS)
model_losses = [get_loss(args.loss), ]   # when the model has multiple outputs
model_loss_weights = [1.0, ]
model_metrics = [strict_accuracy, ]
model.compile(optimizer=get_optimizer(args.optimizer), loss=model_losses, loss_weights = model_loss_weights, metrics = model_metrics)
model.summary()

sys.exit()


history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=20, validation_split=0.1, verbose=args.verbose)


hist = pd.DataFrame(history.history)

plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()


# do some silly predictions
#i = 50

p = model.predict(np.array(X_test), verbose=min(1, args.verbose))
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



