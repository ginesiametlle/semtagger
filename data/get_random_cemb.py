#!/usr/bin/python3
# this script initializes character embedding vectors for a set of characters

import sys
import string
import numpy.random


# target language
lang = sys.argv[1]

# list characters to embed
emb_chars = [str(x) for x in sys.argv[2]]

# output text file
emb_file = sys.argv[3]

# determine the number of dimensions
ndims = 64
print("[INFO] Character vectors for (" + str(lang) + ") have dimensionality " + str(ndims))

# write out in text format
with open(emb_file, 'w') as ofile:
    for c in emb_chars:
        if c.isprintable() and (ord(c) == 32 or not c.isspace()):
        	vectorstr = ' '.join([str(x) for x in numpy.random.normal(0, 0.5, ndims)])
        	ofile.write(c + ' ' + vectorstr + '\n')

