#!/usr/bin/python3
# this script computes the accuracy of a CoNLL tagged file against a gold standard file

import sys
import string


# tagged file
pred_file = sys.argv[1]

# gold standard file
gold_file = sys.argv[2]

# compare the two files and compute tagging accuracy
print('[INFO] Predicted semantic tags --> ' + pred_file)
print('[INFO] Gold semantic tags --> ' + gold_file)
sentences = 0
total_tags = 0
correct_tags = 0

# question words mapped to the QUE sem-tag
question_words = set(['what', 'where', 'when', 'who', 'why', 'how'])

# indicates when a predicted QUE sem-tag occurs at the start of a sentence
question_tag = False

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
                elif ptag == 'QUE' and pword.lower() in question_words and i > 0 and len(pred_lines[i-1].split()) <= 1:
                    question_tag = True
                elif ptag == 'QUE' and pword.lower() in question_words and i == 0:
                    question_tag = True
                total_tags += 1
            else:
                if i > 0 and pred_lines[i-1].split()[1] == '?' and question_tag:
                    correct_tags += 1
                sentences += 1
                question_tag = False

        if len(gold_lines[-1].split()) > 1:
            if gold_lines[-1].split()[1] == '?' and question_tag:
                correct_tags += 1
            sentences += 1
            question_tag = False

print('[INFO] THE TAGGING ACCURACY IS', correct_tags / total_tags)
print('[INFO] Number of sentences:', sentences)
print('[INFO] Number of tags:', total_tags)
print('[INFO] Number of correct tags:', correct_tags)
print('[INFO] Number of incorrect tags:', total_tags - correct_tags)
sys.exit(0)

