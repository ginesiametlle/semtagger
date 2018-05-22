#!/usr/bin/python3
# this script trains a neural model for semantic tagging

import sys
import os
sys.path.append(sys.argv[1])
#sys.stderr = open('/dev/null', 'w')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np

from models.argparser import get_args
from models.loader import load_conll_notags
from models.nnmodels import get_model

from utils.input2feats import wordsents2sym, charsents2sym

#sys.stderr = sys.__stderr__


# obtain arguments and trained model information
args = get_args()
OFF_FILE = args.input_pred_file
ON_FILE = args.output_pred_file
MODEL_FILE = args.output_model

minfo = pickle.load(open(args.output_model_info, 'rb'))
USE_WORDS = minfo['args'].use_words
USE_CHARS = minfo['args'].use_chars

# read and featurize unlabelled data
word_sents = load_conll_notags(OFF_FILE, minfo['word2idx'].keys(), minfo['oov_sym'], minfo['pad_sym'], False, True, True)
word_sents_maps = [[y[0] for y in x] for x in word_sents]
word_sents_originals = [[y[1] for y in x] for x in word_sents]

# transform inputs to a symbolic representation
word_sym, _ = wordsents2sym(word_sents_maps, minfo['max_wlen'], minfo['word2idx'], minfo['tag2idx'], minfo['oov_sym']['unknown'], minfo['DEFAULT_TAG'], minfo['pad_sym']['pad'], minfo['DEFAULT_TAG'])

# use a trained model to predict the corresponding tags
if USE_WORDS and USE_CHARS:
    model = get_model(minfo['args'], minfo['num_tags'], minfo['max_wlen'], minfo['num_words'], minfo['wemb_dim'], minfo['wemb_matrix'], minfo['max_clen'], minfo['num_chars'], minfo['cemb_dim'], minfo['cemb_matrix'])
if USE_WORDS:
    model = get_model(minfo['args'], minfo['num_tags'], minfo['max_wlen'], minfo['num_words'], minfo['wemb_dim'], minfo['wemb_matrix'])
if USE_CHARS:
    model = get_model(minfo['args'], minfo['num_tags'], minfo['max_clen'], minfo['num_chars'], minfo['cemb_dim'], minfo['cemb_matrix'])
model.load_weights(MODEL_FILE)
model.summary()

# write results to a file
p = model.predict(np.array(word_sym), verbose=min(1, minfo['args'].verbose))
p = np.argmax(p, axis=-1)

with open(ON_FILE, 'w') as ofile:
    for i in range(len(word_sents_originals)):
        words = word_sents_originals[i]
        tags = p[i]
        for j in range(len(words)):
            if words[j] not in minfo['pad_sym']['begin'] and words[j] not in minfo['pad_sym']['end']:
                wordtag = minfo['DEFAULT_TAG']
                if j < len(tags):
                    wordtag = minfo['idx2tag'][tags[j]]
                ofile.write(words[j] + '\t' + str(wordtag) + '\n')
        if i < len(word_sents) - 1:
            ofile.write('\n')

