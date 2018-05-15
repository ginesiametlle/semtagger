#!/usr/bin/python3
# this script writes out the lm_1b character embeddings for a given language

import sys
import string
import numpy as np


# target language
lang = sys.argv[1]

# input numpy binary file
ifile = sys.argv[2]

# output text file
emb_file = sys.argv[3]

# load embeddings
embeddings = np.load(ifile)
char_idxs = [idx for idx in range(embeddings.shape[0]) if not chr(idx).isspace() chr(idx).isprintable()]
print("[INFO] Loaded lm_1b (" + str(lang) + ") embeddings with shape {}".format(embeddings.shape))

# write out in text format
with open(emb_file, 'w') as ofile:
    for idx in char_idxs:
        c = chr(idx)
        vectorstr = ' '.join([str(x) for x in embeddings[idx]])
        ofile.write(c + ' ' + vectorstr + '\n')

