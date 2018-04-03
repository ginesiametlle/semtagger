#!/usr/bin/python3
# this script provides procedures for reading embedding and input files

import sys
import codecs
import math
import re
import numpy as np


def load_embeddings(emb_file, oovs = [], pads = [], sep = ' ', lower = False):
    """
    Loads pre-trained word (or other units) embeddings
        Inputs:
            - emb_file: path to a file with pre-trained embeddings
            - pads: list with delimiter words to include in the embeddings
            - oovs: list with oovs aliases to include in the embeddings
			- sep: separator between embedding dimensions
            - lower: lowercase (or not) words in the embedding vocabulary
        Outputs:
            - word2idx: maps words to an index in the embedding matrix
            - emb_matrix: Embedding matrix
    """
    word2emb = {}
    word2idx = {}

    # read and store all word vectors
    for line in open(emb_file, errors = 'ignore', encoding = 'utf-8'):
        try:
            fields = line.strip().split(sep)
            vec = np.asarray(fields[1:], dtype='float32')
            word = fields[0]
            if lower:
                word = word.lower()
            word2emb[word] = vec
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        except Exception as e:
            print('[WARNING] Exception in `load_embeddings`:', e)

    # get dimensions from the last vector added
    emb_dim = word2emb[word].shape[0]

    # add custom embeddings for special characters
    mu = 0
    sigma = 0.01

    for word in pads:
        if word not in word2idx:
            vec = np.random.normal(mu, sigma, emb_dim)
            word2emb[word] = vec
            word2idx[word] = len(word2idx)
        else:
            print('[WARNING] Padding word ' + word + ' has an embedding vector!')

    for word in oovs:
        if word not in word2idx:
            vec = np.random.normal(mu, sigma, emb_dim)
            word2emb[word] = vec
            word2idx[word] = len(word2idx)
        else:
            print('[WARNING] OOV alias ' + word + ' has an embedding vector!')

    # create an embedding matrix
    vocab_size = len(word2emb)
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word2idx.items():
        if word in word2emb:
            vec = word2emb[word]
            if vec is not None and vec.shape[0] == emb_dim:
                emb_matrix[idx] = np.asarray(vec)

    # print embedding data and return results
    print('[INFO] Loaded pre-trained embeddings')
    print('[INFO] Embedding vocabulary:', emb_matrix.shape[0], '(lowercase: ' + str(lower) + ')')
    print('[INFO] OOV aliases:', oovs)
    print('[INFO] Padding words:', pads)
    print('[INFO] Embedding dimensions:', emb_dim)
    return word2idx, np.asarray(emb_matrix), emb_dim




### NEED MORE WORK HERE! SPLIT SENTENCES IN MULTIPLE ONES!
### IF A SENTENCE CONTAINS A . IN THE MIDDLE AND MORE THAN 3 TOKENS AFTER IT, IT CAN BE USED FOR SPLITTING!
### SPLIT SENTENCES SO THAT LENGTHS ARE MORE UNIFORM

def load_conll(conll_file, extra = '', vocab = {}, oovs = {}, pads = {}, default_tag = 'NIL', len_perc = 1.0, lower = False):
    """
    Reads a file in the conll format and produces processed sentences
        Inputs:
            - conll_file: path to a file with data
            - extra: path to a file with extra data
            - vocab: set containing all words to use as vocabulary
            - len_perc: percentile of allowed sentence lengths
            - oovs: dictionary with aliases to replace oovs with (valid keys: number, unknown)
            - pads: dictionary with delimiter words to include in the sentences (valid keys: begin, end)
            - lower: lowercase (or not) words in the input data
        Outputs:
            - tag2idx: maps each tag to a numerical index
            - sents: list of sentences represented as [(w1, tag1), ..., (wN, tagN)]
    """

    # note: all padding words are assigned the NIL (empty semantics) semantic tag
    tag2idx = {}
    tag2idx[default_tag] = len(tag2idx)

    # special characters used for splitting words
    split_chars = set([',', '.', ':', '-', '~', "'", '"'])

    sents = []
    lengths = []
    if 'begin' in pads:
        next_words = [pads['begin']]
        next_tags = [default_tag]
    else:
        next_words = []
        next_tags = []

    # iterate over lines in the input file
    for line in codecs.open(conll_file, mode = 'r', errors = 'ignore', encoding = 'utf-8'):
        # discard newline character
        line = line[:-1]

        # keep adding words while in the middle of a sentence
        if line:
            if len(line.split('\t')) != 2:
                raise IOError('[WARNING] Exception in `load_conll`: Input file has the wrong format')
            else:
                word, tag = line.split('\t')
                if word != 'ø':
                    if lower:
                        word = word.lower()

                    # use an heuristic and try to map oov words
                    if vocab and word not in vocab and word not in split_chars:
                        if re.match('^[0-9\.\,-]+$', word):
                            word = oovs['number']
                        elif word.lower() in vocab:
                            word = word.lower()
                        elif word.upper() in vocab:
                            word = word.upper()
                        elif word.capitalize() in vocab:
                            word = word.capitalize()
                        elif '~' in word or '-' in word:
                            # attempt to split multi-word expressions
                            constituents = re.split('[-~]+', word)
                            if all([True if c in vocab else False for c in constituents]):
                                next_words += constituents[:-1]
                                next_tags += ([tag] * (len(constituents) - 1))
                                word = constituents[-1]
                            else:
                                print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                word = oovs['unknown']
                        else:
                            print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                            word = oovs['unknown']

                    next_words.append(word)
                    next_tags.append(tag)
                    if tag not in tag2idx:
                        tag2idx[tag] = len(tag2idx)

        # stack the current sentence upon seeing an empty line
        else:
            if len(next_words) > 1:
                if 'end' in pads:
                    next_words.append(pads['end'])
                    next_tags.append(default_tag)
                sents.append(list(zip(next_words, next_tags)))
                lengths.append(len(sents[-1]))
            if 'begin' in pads:
                next_words = [pads['begin']]
                next_tags = [default_tag]
            else:
                next_words = []
                next_tags = []

    # double check the last sentence
    if len(next_words) > 1:
        if 'end' in pads:
            next_words.append(pads['end'])
            next_tags.append(default_tag)
        sents.append(list(zip(next_words, next_tags)))
        lengths.append(len(sents[-1]))
        
    # find the allowed sentence length
    max_len = sorted(lengths)[math.ceil((len(lengths)-1) * len_perc)]
    print('[INFO] Sentence length percentile: ' + str(len_perc))
    print('[INFO] Max allowed sentence length: ' + str(max_len))

    return tag2idx, sents, max_len


def write_conll(conll_file, sents):
    """
    Produces a conll file
        Inputs:
            - conll_file: path to the output file
            - sents: list of sentences, each one as a list of (word, tag) pairs
    """
    with codecs.open(conll_file, mode = 'w', errors = 'ignore', encoding = 'utf-8') as ofile:
        for sent in sents:
            if sent:
                for word, tag in sent:
                    ofile.write(word + '\t' + tag + '\n')
                ofile.write('\n')

