#!/usr/bin/python3
# this script extracts tokens and their tags from the PMB xml files

import sys
import string
import xml.etree.ElementTree as ET

# source xml file for a d-part in the PMB
ifile = sys.argv[1]

# output file for POS-tag data
pos_file = sys.argv[2]

# output file for sem-tag data
sem_file = sys.argv[3]


# pos-tags for the current sentence
pos_sent = []

# sem-tags for the current sentence
sem_sent = []

# navigate the tags contained in the xml tree
tree = ET.parse(ifile)
root = tree.getroot()

for token in root.iter('tagtoken'):
    tok = []
    pos = []
    sem = []

    for tag in token[0].findall('tag'):
        if tag.get('type') == 'tok':
            tok += [tag.text]
        elif tag.get('type') == 'pos':
            pos += [tag.text]
        elif tag.get('type') == 'sem':
            sem += [tag.text]

    if len(tok) == len(pos):
        pos_sent += list(zip(tok, pos))
    if len(tok) == len(sem):
        sem_sent += list(zip(tok, sem))

with open(pos_file, 'a+') as ofile:
    if pos_sent:
        ofile.write(' '.join([w[0] for w in pos_sent]) + '\n')
        ofile.write(' '.join([w[1] for w in pos_sent]) + '\n\n')
    else:
        ofile.write('')

with open(sem_file, 'a+') as ofile:
    if sem_sent:
        ofile.write(' '.join([w[0] for w in sem_sent]) + '\n')
        ofile.write(' '.join([w[1] for w in sem_sent]) + '\n\n')
    else:
        ofile.write('')

