#!/usr/bin/python3
# this extracts character embeddings from a word embedding file

import sys
import numpy as np

# word embedding input file
wemb_file = sys.argv[1]

# character embedding output file
cemb_file = sys.argv[2]

vectors = {}
for line in open(wemb_file, 'r').readlines():
    fields = line.split(' ')
    word = fields[0]
    vec = np.array(fields[1:], dtype=float)
    for char in word:
        if char in vectors:
            vectors[char] = (vectors[char][0] + vec, vectors[char][1] + 1)
        else:
            vectors[char] = (vec, 1)

with open(cemb_file, 'w') as ofile:
    for word in vectors:
        avg_vector = np.round((vectors[word][0] / vectors[word][1]), 6).tolist()
        ofile.write(word + ' ' + ' '.join(str(x) for x in avg_vector) + '\n')

