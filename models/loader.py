#!/usr/bin/python3
# this script provides procedures for reading and transforming embedding and input files

import sys
import os.path
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
            - emb_dim: Dimension of the embedding vectors
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


def load_conll(conll_file, extra = '', vocab = {}, oovs = {}, pads = {}, padding_tag = 'PAD', len_perc = 1.0, lower = False, mwe = True):
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
            - mwe: handle multi-word expressions
        Outputs:
            - tag2idx: maps each tag to a numerical index
            - sents: list of sentences represented as [(w1, tag1, x1), ..., (wN, tagN, x1)],
                     where wi: mapped word, tagi: sem-tag, xi: original word
            - max_len: maximum number of words per sentence allowed
    """

    # note: all padding words are assigned the default (empty semantics) semantic tag
    tag2idx = {}
    tag2idx[padding_tag] = len(tag2idx)

    # special characters used for splitting words
    split_chars = set([',', '.', ':', '-', '~', "'", '"'])

    sents = []
    lengths = []
    if 'begin' in pads:
        next_words = [pads['begin']]
        next_tags = [padding_tag]
        next_syms = ['']
        sent_base_length = 1
    else:
        next_words = []
        next_tags = []
        next_syms = []
        sent_base_length = 0

    # select files to use
    input_files = [conll_file]
    if extra and os.path.isfile(extra):
        input_files += [extra]

    # counters
    num_raw_sents = 0
    num_sents = 0
    num_words = 0
    num_oovs = 0

    # iterate over lines in the input files
    for ifile in input_files:
        for line in codecs.open(ifile, mode = 'r', errors = 'ignore', encoding = 'utf-8'):
            # discard newline character
            line = line[:-1]

            # keep adding words while in the middle of a sentence
            if line:
                if len(line.split('\t')) != 2:
                    raise IOError('[WARNING] Exception in `load_conll`: Input file has the wrong format')
                else:
                    word, tag = line.split('\t')
                    sym = word
                    if word != 'ø':
                        num_words += 1
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
                            elif '~' in word or '-' in word and mwe:
                                # attempt to split multi-word expressions
                                constituents = re.split('[ -~]+', word)
                                if all([True if c in vocab else False for c in constituents]):
                                    next_words += constituents[:-1]
                                    next_tags += ([tag] * (len(constituents) - 1))
                                    next_syms += constituents[:-1]
                                    word = constituents[-1]
                                    sym = constituents[-1]
                                else:
                                    #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                    word = oovs['unknown']
                                    num_oovs += 1
                            else:
                                #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                word = oovs['unknown']
                                num_oovs += 1

                        next_words.append(word)
                        next_tags.append(tag)
                        next_syms.append(sym)
                        if tag not in tag2idx:
                            tag2idx[tag] = len(tag2idx)

            # stack the current sentence upon seeing an empty line or a full stop
            if not line or (len(next_words) > 3 and next_words[-4] == '.'):
                if len(next_words) > sent_base_length:
                    if not line:
                        if 'end' in pads:
                            next_words.append(pads['end'])
                            next_tags.append(padding_tag)
                            next_syms.append('')
                        sents.append(list(zip(next_words, next_tags, next_syms)))
                        lengths.append(len(sents[-1]))
                        next_words = []
                        next_tags = []
                        next_syms = []
                        num_raw_sents += 1
                        num_sents += 1
                    else:
                        split_words = next_words[:-3]
                        split_tags = next_tags[:-3]
                        split_syms = next_syms[:-3]
                        if 'end' in pads:
                            split_words.append(pads['end'])
                            split_tags.append(padding_tag)
                            split_syms.append('')
                        sents.append(list(zip(split_words, split_tags, split_syms)))
                        lengths.append(len(sents[-1]))
                        next_words = next_words[-3:]
                        next_tags = next_tags[-3:]
                        next_syms = next_syms[-3:]
                        num_sents += 1
                    if 'begin' in pads:
                        next_words = [pads['begin']] + next_words
                        next_tags = [padding_tag] + next_tags
                        next_syms = [''] + next_syms

        # double check the last sentence
        if len(next_words) > sent_base_length:
            if 'end' in pads:
                next_words.append(pads['end'])
                next_tags.append(padding_tag)
                next_syms.append('')
            sents.append(list(zip(next_words, next_tags, next_syms)))
            lengths.append(len(sents[-1]))

    # find the allowed sentence length
    max_len = sorted(lengths)[math.ceil((len(lengths)-1) * len_perc)]
    print('[INFO] Sentence length percentile: ' + str(len_perc))
    print('[INFO] Max allowed sentence word length: ' + str(max_len))
    print('[INFO] Number of OOV words: ' + str(num_oovs) + ' / ' + str(num_words))
    print('[INFO] Original number of sentences: ' + str(num_raw_sents))
    print('[INFO] Number of extracted sentences ' + str(num_sents))
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
                for word, tag, sym in sent:
                    ofile.write(word + '\t' + tag + '\n')
                ofile.write('\n')


def make_char_seqs(word_seqs, vocab={}, oovs={}, pads={}, lower = False):
    """
    Turns word sequences into char sequences
        Inputs:
            - word_seqs: list of sequences of words
            - vocab: set containing all words to use as vocabulary
            - oovs: dictionary with aliases to replace oovs with (valid keys: number, unknown)
            - pads: dictionary with delimiter words to include in the sentences (valid keys: begin, end)
            - lower: lowercase (or not) words in the input data
        Outputs:
            - char_sents: list of sentences represented as [c1,c2,...,cn]
            - max_len: maximum number of characters per sentence allowed
    """
    char_seqs = []
    max_len = 0
    num_chars = 0
    num_oovs = 0
    num_spaces = 0

    for sent in word_seqs:
        chars = []
        if 'begin' in pads:
            chars.append(pads['begin'])
        filterlist = filter(lambda x: x, [w[2] if isinstance(w, (list,tuple)) else w for w in sent])
        for c in ' '.join(filterlist):
            num_chars += 1
            if c == '~':
                c = ' '
            if lower:
                c = c.lower()
            if vocab and c not in vocab:
                if c.isspace():
                    num_spaces += 1
                if c.isdigit():
                    c = oovs['number']
                else:
                    c = oovs['unknown']
                num_oovs += 1
            chars.append(c)
        if 'end' in pads:
            chars.append(pads['end'])
        if len(chars) > max_len:
            max_len = len(chars)
        char_seqs.append(chars)

    print('[INFO] Max allowed sentence character length: ' + str(max_len))
    print('[INFO] Number of OOV characters: ' + str(num_oovs) + ' / ' + str(num_chars) + ' [' + str(num_spaces) + ' are spaces]')
    return char_seqs, max_len


def write_chars(ofile, char_sents):
    """
    Produces a conll file
        Inputs:
            - ofile: path to the output file
            - char_sents: list of sentences, each one as a list of characters
    """
    with codecs.open(ofile, mode = 'w', errors = 'ignore', encoding = 'utf-8') as ofile:
        for sent in char_sents:
            if sent:
                ofile.write(sent[0])
                for char in sent[1:]:
                    ofile.write(' ' + char)
                ofile.write('\n')


def load_conll_notags(unfile, vocab = {}, oovs = {}, pads = {}, lower = False, mwe = True):
    """
    Reads a file containing unlabelled data and produces processed sentences
        Inputs:
            - unfile: path to a file with data
            - vocab: set containing all words to use as vocabulary
            - oovs: dictionary with aliases to replace oovs with (valid keys: number, unknown)
            - pads: dictionary with delimiter words to include in the sentences (valid keys: begin, end)
            - lower: lowercase (or not) words in the input data
            - mwe: handle multi-word expressions
        Outputs:
            - sents: list of sentences represented as [w1, ..., wN], where wi is a mapped words
    """
    # special characters used for splitting words
    split_chars = set([',', '.', ':', '-', '~', "'", '"'])

    sents = []
    if 'begin' in pads:
        next_words = [pads['begin']]
        next_syms = ['']
        sent_base_length = 1
    else:
        next_words = []
        next_syms = []
        sent_base_length = 0

    # select files to use
    input_files = [unfile]

    # counters
    num_raw_sents = 0
    num_sents = 0
    num_words = 0
    num_oovs = 0

    # iterate over lines in the input files
    for ifile in input_files:
        for line in codecs.open(ifile, mode = 'r', errors = 'ignore', encoding = 'utf-8'):
            # discard newline character
            line = line[:-1]

            # keep adding words while in the middle of a sentence
            if line:
                word = line.split('\t')[0]
                sym = word
                if word != 'ø':
                    num_words += 1
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
                        elif '~' in word or '-' in word and mwe:
                            # attempt to split multi-word expressions
                            constituents = re.split('[ -~]+', word)
                            if all([True if c in vocab else False for c in constituents]):
                                next_words += constituents[:-1]
                                next_syms += constituents[:-1]
                                word = constituents[-1]
                                sym = constituents[-1]
                            else:
                                #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                word = oovs['unknown']
                                num_oovs += 1
                        else:
                            #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                            word = oovs['unknown']
                            num_oovs += 1

                    next_words.append(word)
                    next_syms.append(sym)

            # stack the current sentence upon seeing an empty line or a full stop
            if not line or (len(next_words) > 3 and next_words[-4] == '.'):
                if len(next_words) > sent_base_length:
                    if not line:
                        if 'end' in pads:
                            next_words.append(pads['end'])
                            next_syms.append('')
                        sents.append(list(zip(next_words, next_syms)))
                        next_words = []
                        next_syms = []
                        num_raw_sents += 1
                        num_sents += 1
                    else:
                        split_words = next_words[:-3]
                        split_syms = next_syms[:-3]
                        if 'end' in pads:
                            split_words.append(pads['end'])
                            split_syms.append('')
                        sents.append(list(zip(split_words, split_syms)))
                        next_words = next_words[-3:]
                        next_syms = next_syms[-3:]
                        num_sents += 1
                    if 'begin' in pads:
                        next_words = [pads['begin']] + next_words
                        next_syms = [''] + next_syms

        # double check the last sentence
        if len(next_words) > sent_base_length:
            if 'end' in pads:
                next_words.append(pads['end'])
                next_syms.append('')
            sents.append(list(zip(next_words, next_syms)))

    # find the allowed sentence length
    print('[INFO] Number of unlabelled OOV words: ' + str(num_oovs) + ' / ' + str(num_words))
    print('[INFO] Original number of unlabelled sentences: ' + str(num_raw_sents))
    print('[INFO] Number of extracted unlabelled sentences ' + str(num_sents))
    return sents

