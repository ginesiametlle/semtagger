#!/usr/bin/python3
# this script compares the accuracy of a conll tagged file against a provided gold standard

import sys
import string


# semantically tagged file
pred_file = sys.argv[1]

# gold standard tagged file
gold_file = sys.argv[2]

# compare the two files and compute tagging accuracy
sentences = 0
total_tags = 0
correct_tags = 0

with open(pred_file) as p:
    with open(gold_file) as g:
        pred_sents = p.readlines()
        gold_sents = g.readlines()

        for i in range(len(pred_sents)):
            sentences += 1
            if len(pred_sents[i].split()) != len(gold_sents[i].split()):
                print('[ERROR] The tagged files are not properly formatted')
                sys.exit()
            else:
                if len(pred_sents[i].split()) == 2:
                    pword, ptag = pred_sents[i].split()
                    gword, gtag = gold_sents[i].split()
                    if pword != gword:
                        print('[ERROR] The tagged file and the gold standard file do not match')
                        sys.exit()
                    total_tags += 1
                    if ptag == gtag:
                        correct_tags += 1

print('[INFO] THE TAGGING ACCURACY IS', correct_tags / total_tags)
print('[INFO]', sentences, 'sentences')
print('[INFO]', total_tags, 'sem-tags')
print('[INFO]', correct_tags, 'correct sem-tags')

