#!/usr/bin/python3
# this script writes out the Polyglot embeddings for a given language

import sys
import pickle

# target language
lang = sys.argv[1]

# pickle file containing compressed word embeddings
emb_pkl = sys.argv[2]

# output text file
emb_file = sys.argv[3]

# unpickle embedding file
words, embeddings = pickle.load(open(emb_pkl, 'rb'), encoding='latin-1')
print("[INFO] Loaded Polyglot (" + str(lang) + ") embeddings with shape {}".format(embeddings.shape))

# write out in text format
with open(emb_file, 'w') as ofile:
    for i in range(len(words)):
        vectorstr = ' '.join([str(x) for x in embeddings[i].tolist()])
        ofile.write(words[i] + ' ' + vectorstr + '\n')

