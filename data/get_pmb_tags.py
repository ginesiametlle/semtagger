#!/usr/bin/python3
# this script extracts tokens and their tags from the PMB xml files

import sys
import string
import xml.etree.cElementTree as ET


# source xml file for a d-part in the PMB
ifile = sys.argv[1]

# output file for sem-tag data
sem_file = sys.argv[2]

# sem-tags for the current sentence
sem_sent = []

# navigate the tags contained in the xml tree
tree = ET.parse(ifile)
root = tree.getroot()

for token in root.iter('tagtoken'):
    tok = []
    sem = []

    for tag in token[0].findall('tag'):
        if tag.get('type') == 'tok':
            tok += [tag.text]
        elif tag.get('type') == 'sem':
            sem += [tag.text]

    if len(tok) == len(sem):
        sem_sent += [x for x in zip(tok, sem)]

with open(sem_file, 'a+') as ofile:
    if sem_sent:
        for i in range(len(sem_sent)):
            ofile.write(sem_sent[i][0] + '\t' + sem_sent[i][1] + '\n')
        ofile.write('\n')
    else:
        ofile.write('')

