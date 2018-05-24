#!/usr/bin/python3
# this script provides procedures for reading and transforming embedding and input files

import sys
import os.path
import codecs
import math
import re

import numpy as np
import numpy.random as npr

from collections import OrderedDict


def load_embeddings(emb_file, oovs=[], pads=[], sep=' ', lower=False, case_dim=True):
    """
    Loads pre-trained word (or other units) embeddings
        Inputs:
            - emb_file: path to a file with pre-trained embeddings
            - oovs: list with oovs aliases to include in the embeddings
            - pads: list with delimiter words to include in the embeddings
			- sep: separator between embedding dimensions
            - lower: lowercase words in the embedding vocabulary
            - case_dim: append an extra case dimension to the embedding vectors
        Outputs:
            - word2idx: maps words to an index in the embedding matrix
            - emb_matrix: embedding matrix
            - emb_dim: dimension of the embedding vectors
    """
    word2emb = {}
    word2idx = {}

    # read and store all word vectors
    for line in open(emb_file, errors = 'ignore', encoding = 'utf-8'):
        try:
            fields = line.strip().split(sep)
            word = fields[0]
            vec = np.asarray(fields[1:], dtype='float32')
            if case_dim:
                is_upper = float(word[0].isupper())
                vec = np.insert(vec, 0, is_upper, axis=0)
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
            if case_dim:
                vec = npr.normal(mu, sigma, emb_dim-1)
                vec = np.insert(vec, 0, 0., axis=0)
            else:
                vec = npr.normal(mu, sigma, emb_dim)
            word2emb[word] = vec
            word2idx[word] = len(word2idx)
        else:
            print('[WARNING] Padding item ' + word + ' has an embedding vector')

    for word in oovs:
        if word not in word2idx:
            if case_dim:
                vec = npr.normal(mu, sigma, emb_dim-1)
                is_upper = float(word[0].isupper())
                vec = np.insert(vec, 0, is_upper, axis=0)
            else:
                vec = npr.normal(mu, sigma, emb_dim)
            word2emb[word] = vec
            word2idx[word] = len(word2idx)
        else:
            print('[WARNING] OOV alias ' + word + ' has an embedding vector')

    # create an embedding matrix
    vocab_size = len(word2emb)
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word2idx.items():
        if word in word2emb:
            vec = word2emb[word]
            if vec is not None and vec.shape[0] == emb_dim:
                emb_matrix[idx] = np.asarray(vec)

    # print feedback data and return mappings
    print('[INFO] Embedding vocabulary:', emb_matrix.shape[0], '(lowercase: ' + str(lower) + ')')
    print('[INFO] OOV aliases:', oovs)
    print('[INFO] Padding items:', pads)
    print('[INFO] Embedding dimensions:', emb_dim, '(extra case dimension: ' + str(case_dim) + ')')
    return word2idx, np.asarray(emb_matrix), emb_dim


def load_conll(conll_file, extra='', vocab=[], oovs={}, pads={}, padding_tag='PAD', default_tag='NIL', ignore_tags=[], len_perc=1.0, lower=False, mwe=True, unk_case=True):
    """
    Reads a file in the conll format and produces processed unique sentences
        Inputs:
            - conll_file: path to a file with data
            - extra: path to a file with extra data
            - vocab: set containing all words to use as vocabulary
            - oovs: dictionary with aliases to replace oovs with (valid keys: number, unknown, UNKNOWN)
            - pads: dictionary with delimiter words to include in the sentences (valid keys: begin, end)
            - padding_tag: tag to use for padding
            - default_tag: tag to use by default
            - ignore_tags: list of input tags to be ignored and mapped to the default tag
            - len_perc: percentile of allowed sentence lengths
            - lower: lowercase words in the input data
            - mwe: handle multi-word expressions
            - unk_case: take into account case in OOV words
        Outputs:
            - tag2idx: maps each tag to a numerical index
            - sents: list of sentences represented as [(w1, tag1, x1), ..., (wN, tagN, x1)],
                     where wi: mapped word, tagi: sem-tag, xi: original word
            - max_len: maximum number of words per sentence allowed
    """
    # note: all padding words are assigned the default (empty semantics) semantic tag
    tag2idx = {}
    tag2idx[padding_tag] = len(tag2idx) + 1
    tag2counts = {}

    # number of duplicated sentences in the input data
    num_dup = 0
    dup_sents = set()

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
                    # map ignored tags to the default tag
                    if tag in ignore_tags:
                        tag = default_tag

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
                                constituents = re.split('[\s~ | \s-]+', word)
                                if all([True if c in vocab else False for c in constituents]):
                                    next_words += constituents[:-1]
                                    next_tags += ([tag] * (len(constituents) - 1))
                                    next_syms += constituents[:-1]
                                    word = constituents[-1]
                                    sym = constituents[-1]
                                else:
                                    #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                    if unk_case and word[0].isupper():
                                        word = oovs['UNKNOWN']
                                    else:
                                        word = oovs['unknown']
                                    num_oovs += 1
                            else:
                                #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                if unk_case and word[0].isupper():
                                    word = oovs['UNKNOWN']
                                else:
                                    word = oovs['unknown']
                                num_oovs += 1

                        next_words.append(word)
                        next_tags.append(tag)
                        next_syms.append(sym)
                        if tag not in tag2idx:
                            tag2idx[tag] = len(tag2idx) + 1

            # stack the current sentence upon seeing an empty line or a full stop
            if not line or (len(next_words) > 3 and next_words[-4] == '.'):
                if len(next_words) > sent_base_length:
                    if not line:
                        if 'end' in pads:
                            next_words.append(pads['end'])
                            next_tags.append(padding_tag)
                            next_syms.append('')
                        if ''.join(next_words) not in dup_sents:
                            sents.append(list(zip(next_words, next_tags, next_syms)))
                            dup_sents.add(''.join(next_words))
                            lengths.append(len(sents[-1]))
                            for stag in next_tags:
                                if stag not in tag2counts:
                                    tag2counts[stag] = 0
                                tag2counts[stag] += 1
                        else:
                            num_dup += 1
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
                        if ''.join(next_words) not in dup_sents:
                            sents.append(list(zip(split_words, split_tags, split_syms)))
                            dup_sents.add(''.join(next_words))
                            lengths.append(len(sents[-1]))
                            for stag in split_tags:
                                if stag not in tag2counts:
                                    tag2counts[stag] = 0
                                tag2counts[stag] += 1
                        else:
                            num_dup += 1
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
            if ''.join(next_words) not in dup_sents:
                sents.append(list(zip(next_words, next_tags, next_syms)))
                dup_sents.add(''.join(next_words))
                lengths.append(len(sents[-1]))
                for stag in next_tags:
                    if stag not in tag2counts:
                        tag2counts[stag] = 0
                    tag2counts[stag] += 1
            else:
                num_dup += 1

    # find the allowed sentence length
    max_len = sorted(lengths)[math.ceil((len(lengths)-1) * len_perc)]
    print('[INFO] Sentence length percentile: ' + str(len_perc))
    print('[INFO] Max allowed sentence length: ' + str(max_len))
    print('[INFO] Number of OOV words: ' + str(num_oovs) + ' / ' + str(num_words))
    print('[INFO] Original number of sentences: ' + str(num_raw_sents))
    print('[INFO] Number of extracted sentences: ' + str(num_sents))
    print('[INFO] Number of duplicated sentences removed: ' + str(num_dup))

    # sort tags in non-decreasing order
    sorted_tag2idx = OrderedDict(sorted(tag2idx.items(), key=lambda t: tag2counts[t[0]], reverse=True))
    return sorted_tag2idx, sents, max_len


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
                for element in sent:
                    word = element[0]
                    tag = element[1]
                    ofile.write(str(word) + '\t' + str(tag) + '\n')
                ofile.write('\n')


def make_char_seqs(word_seqs, vocab=[], oovs={}, pads={}, len_perc=1.0, lower=False):
    """
    Turns word sequences into char sequences
        Inputs:
            - word_seqs: list of sequences of words, represented as [(w1, tag1, x1), ..., (wN, tagN, xN)]
            - vocab: set containing all characters to use as vocabulary
            - oovs: dictionary with aliases to replace oovs with (valid keys: number, unknown)
            - pads: dictionary with delimiter characters to include in words (valid keys: begin, end)
            - len_perc: percentile of allowed word lengths
            - lower: lowercase words in the input data
        Outputs:
            - char_sents: list of sentences represented as [[c1_1, c2_1, ...], ..., [c1_N, c2_N, ...]]
            - max_len: maximum number of characters per word allowed
    """
    char_seqs = []
    lengths = []
    num_chars = 0
    num_oovs = 0
    num_spaces = 0

    for sent in word_seqs:
        # decompose each one of the original words
        chars = []
        wordlist = [w[2] if isinstance(w, (list,tuple)) else w for w in sent]

        # include each word in the sentences as a list of characters
        for word in wordlist:
            chars.append([])
            if lower:
                word = word.lower()
            if 'begin' in pads:
                chars[-1].append(pads['begin'])
            for c in word:
                num_chars += 1
                if vocab and c not in vocab:
                    if c.isspace():
                        num_spaces += 1
                    if c.isdigit():
                        c = oovs['number']
                    else:
                        c = oovs['unknown']
                    num_oovs += 1
                chars[-1].append(c)
            if 'end' in pads:
                chars[-1].append(pads['end'])

        char_seqs.append(chars)
        lengths += [len(x) for x in chars]

    # find the allowed word length
    max_len = sorted(lengths)[math.ceil((len(lengths)-1) * len_perc)]
    print('[INFO] Word length percentile: ' + str(len_perc))
    print('[INFO] Max allowed word length: ' + str(max_len))
    print('[INFO] Number of OOV characters: ' + str(num_oovs) + ' / ' + str(num_chars))
    print('[INFO] Number of OOV spacing characters: ' + str(num_spaces) + ' / ' + str(num_oovs))
    return char_seqs, max_len


def write_chars(ofile, char_sents):
    """
    Produces a file where each line represents a character-decomposed word within a sentence
        Inputs:
            - ofile: path to the output file
            - char_sents: list of sentences, each one as a list of lists of characters
    """
    with codecs.open(ofile, mode = 'w', errors = 'ignore', encoding = 'utf-8') as ofile:
        for sent in char_sents:
            if sent:
                for element in sent:
                    chars = element[0]
                    tag = element[1]
                    ofile.write(' '.join([str(x) for x in chars]) + '\t' + str(tag) + '\n')
                ofile.write('\n')


def load_conll_notags(unfile, vocab=[], oovs={}, pads={}, lower=False, mwe=True, unk_case=True):
    """
    Reads a file containing unlabelled data and produces processed sentences
        Inputs:
            - unfile: path to a file with data
            - vocab: set containing all words to use as vocabulary
            - oovs: dictionary with aliases to replace oovs with (valid keys: number, unknown, UNKNOWN)
            - pads: dictionary with delimiter words to include in the sentences (valid keys: begin, end)
            - lower: lowercase words in the input data
            - mwe: handle multi-word expressions
            - unk_case: take into account case in OOV words
        Outputs:
            - input_sents: list of unchanged input sentences represented as [(w1, i1), ..., (wN, iN)],
                     where wi: word, ii: word order index
            - sents: list of sentences represented as [(w1, i1, x1), ..., (wN, iN, xN)],
                     where wi: mapped word, xi: original word, ii: word order index
    """
    # special characters used for splitting words
    split_chars = set([',', '.', ':', '-', '~', "'", '"'])

    input_sents = []
    input_words = []
    windex = -1

    sents = []

    if 'begin' in pads:
        next_words = [pads['begin']]
        next_syms = ['']
        next_indexs = [windex]
        sent_base_length = 1
    else:
        next_words = []
        next_syms = []
        next_indexs = []
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
                # add new original word
                windex += 1
                input_words.append(word)

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
                            constituents = re.split('[\s~ | \s-]+', word)
                            if all([True if c in vocab else False for c in constituents]):
                                next_words += constituents[:-1]
                                next_syms += constituents[:-1]
                                next_indexs += [windex] * len(constituents[:-1])
                                word = constituents[-1]
                                sym = constituents[-1]
                            else:
                                #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                                if unk_case and word[0].isupper():
                                    word = oovs['UNKNOWN']
                                else:
                                    word = oovs['unknown']
                                num_oovs += 1
                        else:
                            #print('[INFO] Word ' + word + ' not in the embedding vocabulary')
                            if unk_case and word[0].isupper():
                                word = oovs['UNKNOWN']
                            else:
                                word = oovs['unknown']
                            num_oovs += 1

                    next_words.append(word)
                    next_syms.append(sym)
                    next_indexs.append(windex)


            # stack the current sentence upon seeing an empty line or a full stop
            if not line or (len(next_words) > 3 and next_words[-4] == '.'):
                if len(next_words) > sent_base_length:
                    if not line:
                        if 'end' in pads:
                            next_words.append(pads['end'])
                            next_syms.append('')
                            next_indexs.append(-1)
                        sents.append(list(zip(next_words, next_indexs, next_syms)))
                        input_sents.append(input_words)
                        input_words = []
                        windex = -1
                        next_words = []
                        next_syms = []
                        next_indexs = []
                        num_raw_sents += 1
                        num_sents += 1
                    else:
                        split_words = next_words[:-3]
                        split_syms = next_syms[:-3]
                        split_indexs = next_indexs[:-3]
                        if 'end' in pads:
                            split_words.append(pads['end'])
                            split_syms.append('')
                            split_indexs.append(-1)
                        sents.append(list(zip(split_words, split_indexs, split_syms)))
                        next_words = next_words[-3:]
                        next_syms = next_syms[-3:]
                        next_indexs = next_indexs[-3:]
                        num_sents += 1
                    if 'begin' in pads:
                        next_words = [pads['begin']] + next_words
                        next_syms = [''] + next_syms
                        next_indexs = [-1] + next_indexs

        # double check the last sentence
        if len(next_words) > sent_base_length:
            if 'end' in pads:
                next_words.append(pads['end'])
                next_syms.append('')
                next_indexs.append(-1)
            sents.append(list(zip(next_words, next_indexs, next_syms)))
            input_sents.append(input_words)
            input_words = []
            windex = -1

    # find the allowed sentence length
    print('[INFO] Number of unlabelled OOV words: ' + str(num_oovs) + ' / ' + str(num_words))
    print('[INFO] Original number of unlabelled sentences: ' + str(num_raw_sents))
    print('[INFO] Number of extracted unlabelled sentences ' + str(num_sents))
    return input_sents, sents

