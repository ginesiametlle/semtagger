import codecs
import numpy as np
import sys


def load_conll(conll_file, lower = False):
    """
    Reads a file in the conll format
        Inputs:
            - conll_file: path to a file with data
            - lower: lowercase (or not) words in the input data
        Outputs:
            - tag2idx: maps each tag to a numerical index
            - sents: list of sentences represented as [(w1, tag1), ..., (wN, tagN)]
    """
    sents = []
    tag2idx = {}
    next_words = []
    next_tags = []

    # iterate over lines in the input file
    for line in codecs.open(conll_file, errors = 'ignore', encoding = 'utf-8'):
        # discard newline character
        line = line[:-1]

        # keep adding words while in the middle of a sentence
        if line:
            if len(line.split('\t')) != 2:
                raise IOError('Exception in `load_conll`: Input file has the wrong format')
            else:
                word, tag = line.split('\t')
                if lower:
                    word = word.lower()
                next_words.append(word)
                next_tags.append(tag)
                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)

        # stack the current sentence upon seeing an empty line
        else:
            if next_words:
                sents.append(list(zip(next_words, next_tags)))
            next_words = []
            next_tags = []

    # double check the last sentence
    if next_words:
        sents.append(list(zip(next_words, next_tags)))
    return tag2idx, sents


def load_embeddings(emb_file, sep = ' ', lower = False):
    """
    Loads pre-trained word embeddings (GloVe)
        Inputs:
            - emb_file: path to a file with pre-trained embeddings
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
                word2idx[word] = len(word2idx) + 1
        except Exception as e:
            print('Exception in `load_embeddings`:', e)

    # add an empty word to the embedding with index 0
    word2idx[''] = 0

    # create an embedding matrix
    vocab_size = len(word2emb) + 1
    emb_dim = word2emb[word].shape[0]
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word2idx.items():
        if word in word2emb:
            vec = word2emb[word]
            if vec is not None and vec.shape[0] == emb_dim:
                emb_matrix[idx] = np.asarray(vec)

    # print embedding data and return results
    print('[INFO] Loaded pre-trained embeddings')
    print('[INFO] Embedding vocabulary:', emb_matrix.shape[0], '(lowercase: ' + str(lower) + ')')
    print('[INFO] Embedding dimensions:', emb_matrix.shape[1])
    return word2idx, np.asarray(emb_matrix)

