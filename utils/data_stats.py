#!/usr/bin/python3
# this script produces graphical visualizations

import sys
import pygal
import numpy as np
import pandas as pd
import cairosvg
from matplotlib import pyplot as plt
import operator

from functools import reduce
from itertools import chain
import operator

def plot_dist_tags(sents, vocab, outfile, padwords):
    """
    Plot the amount of words per tag in a dataset
            Inputs:
                - sent: list of sentences represented as a list of (word, tag, sym) items
                - words: set containing words in the vocabulary
                - outfile: output file
                - padwords: set of words to be ignored
    """
    count = {}
    for sent in sents:
        if sent:
            for word, tag, sym in sent:
                if word not in padwords:
                    if tag not in count:
                        count[tag] = [0,0]
                    count[tag][0] += 1
                    if sym not in vocab:
                        count[tag][1] += 1

    xdata = sorted(count.keys(), key=lambda x: count[x][0])
    ydata_oov = list([count[x][1] for x in xdata])
    ydata_rest = list([count[x][0] - count[x][1] for x in xdata])

    line_chart = pygal.StackedBar(width=1600, height=800, x_label_rotation=-45, y_title='Number of words')
    line_chart.x_labels = xdata
    line_chart.add('OOVs', ydata_oov)
    line_chart.add('IVs + OOVs', ydata_rest)

    # circumvent potential svg css styling problems
    line_chart.render_to_file(outfile)
    cairosvg.svg2svg(url=outfile, write_to=outfile)

    # output text
    with open(outfile + '.txt', 'w') as tfile:
        tfile.write("tag\t#OOVs\t#IVs\t#words\tratio\n")
        for i in range(len(xdata)):
            tfile.write(str(xdata[i]) + '\t')
            tfile.write(str(ydata_oov[i]) + '\t')
            tfile.write(str(ydata_rest[i]) + '\t')
            tfile.write(str(ydata_oov[i] + ydata_rest[i]) + '\t')
            tfile.write(str(ydata_oov[i] / (ydata_oov[i] + ydata_rest[i])) + '\n')



def plot_accuracy(history, keys, labels, test_acc, outfile):
    hist = pd.DataFrame(history.history)

    chart = pygal.Line(width=1600, height=800, x_label_rotation=-45, x_title='Number of training epochs', y_title='Sem-tagging accuracy')
    xdata = [x+1 for x in range(len(hist[keys[0]]))]
    chart.x_labels = xdata

    for i in range(len(keys)):
        key = keys[i]
        label = labels[i]
        chart.add(label, hist[key], show_dots=False, stroke_style={'width': 6, 'dasharray': '3, 8', 'linecap': 'round', 'linejoin': 'round'})

    # plot line for test data
    ytest=[test_acc] * len(xdata)
    chart.add(None, ytest, show_dots=False, stroke_style={'width': 2})

    chart.render_to_file(outfile)
    cairosvg.svg2svg(url=outfile, write_to=outfile)



def plot_confusion_matrix(predicted, true, lengths, outdir, vocab=[], include_oovs = True):
    
# Compute confusion matrix
#cnf_matrix = confusion_matrix(true.flatten(), p.flatten())
#np.set_printoptions(precision=2)

# Plot normalized confusion matrix
#plt.figure()


#tagnames = [x[0] for x in sorted(tag2idx.items(), key=operator.itemgetter(1))]
#write_confusion_matrix(cnf_matrix, classes=tagnames, normalize=True,
#                      title='Normalized confusion matrix')

#plt.show()

    return 0

'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ w
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
