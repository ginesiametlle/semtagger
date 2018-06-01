#!/usr/bin/python3
# this script computes the accuracy of a CoNLL tagged file against a gold standard file

import sys
import string


# tagged file
pred_file = sys.argv[1]

# gold standard file
gold_file = sys.argv[2]

# compare the two files and compute tagging accuracy
sentences = 0
total_tags = 0
correct_tags = 0

with open(pred_file) as p:
    with open(gold_file) as g:
        pred_lines = p.readlines()
        gold_lines = g.readlines()

        for i in range(len(gold_lines)):
            if len(pred_lines) < i or len(pred_lines[i].split()) != len(gold_lines[i].split()):
                print('[ERROR] The tagged files are not properly formatted')
                sys.exit(1)
            elif len(pred_lines[i].split()) > 1:
                ptag, pword = pred_lines[i].split()[:2]
                gtag, gword = gold_lines[i].split()[:2]
                if pword != gword:
                    print('[ERROR] The tagged file and the gold standard file do not match')
                    sys.exit(1)
                if ptag == gtag:
                    correct_tags += 1
                total_tags += 1
            else:
                sentences += 1

        if len(pred_lines[-1].split()) > 1:
            sentences += 1

print('[INFO] THE TAGGING ACCURACY IS', correct_tags / total_tags)
print('[INFO]', sentences, 'sentences')
print('[INFO]', total_tags, 'tags')
print('[INFO]', correct_tags, 'correct tags')
print('[INFO]', total_tags-correct_tags, 'incorrect tags')
sys.exit(0)

