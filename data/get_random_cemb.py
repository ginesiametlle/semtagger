#!/usr/bin/python3
# this script writes out embedding vectors for a set of characters
# the vectors are initialized by sampling from a normal distribution

import sys
import string
import numpy.random as npr


# target language
lang = sys.argv[1]

# list characters to embed
emb_chars = [str(x) for x in sys.argv[2]]

# output text file
emb_file = sys.argv[3]

# determine the number of dimensions
ndims = 32
print("[INFO] Character vectors for (" + str(lang) + ") have dimensionality " + str(ndims))

# write out in text format
with open(emb_file, 'w') as ofile:
    for c in emb_chars:
        if c.isprintable() and (ord(c) == 32 or not c.isspace()):
        	vectorstr = ' '.join([str(x) for x in npr.normal(0, 1, ndims)])
        	ofile.write(c + ' ' + vectorstr + '\n')

