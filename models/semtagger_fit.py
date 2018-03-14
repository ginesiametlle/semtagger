import codecs
import numpy as np
import sys
import os
import time
import argparse
from collections import defaultdict

from loader import load_embeddings, load_conll

import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF


import matplotlib.pyplot as plt

#from models import lstm, lstm_crf, bi_lstm, bi_lstm_crf
# here we will load the embeddings, train and perform cross validaiton, plus saving the model 


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='File containing sem-tagged sentences', type=str, required=True)
parser.add_argument('--embeddings', help='File containing pretrained GloVe embeddings', type=str, required=True)
parser.add_argument('--output', help='Output hdf5 model file', type=str, required=True)
parser.add_argument('--lang', help='Language code as in ISO 639-1', type=str, default='en')
parser.add_argument('--test_size', help='Proportion of the sentences to use as a test set', type=float, default=0.2)
parser.add_argument('--cross-validate', help='Estimate hyperparameters using cross-validation', type=bool, default=False)
parser.add_argument('--model', help='Type of neural model', type=str, default='bi-lstm')
parser.add_argument('--iterations', help='Number of iterations', type=int, default=30)
parser.add_argument('--num-hidden', help='Number of hidden units', type=int, default=100)
parser.add_argument('--activation-hidden', help='Activation function for the hidden units', type=str, default='tanh')
parser.add_argument('--activation-output', help='Activation function for the output unit', type=str, default='sigmoid')
parser.add_argument('--loss', help='Loss function', type=str, default='mse')
parser.add_argument('--optimizer', help='Optimization algorithm to use', type=str, default='sgd')
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.1)
parser.add_argument('--dropout', help='Dropout rate to use', type=float, default=0.2)
parser.add_argument('--batch-size', help='Batch size', type=int, default=32)
parser.add_argument('--max_sent_len', help='Maximum length of a word sequence', type=int, default=10)
args = parser.parse_args()

# for padding
# use index 0 for words
# use NIL tag for tags


print(args.data)
print(args.embeddings)
print(args.max_sent_len)
#print(args.output)
#print(args.lang)
#print(args.test_size)

tag2idx, sents = load_conll(args.data)
#print(sorted([len(x) for x in sents]))
#print(sents[0])

word2idx, emb_matrix = load_embeddings(args.embeddings)
n_words = len(word2idx)
#print(word2idx[''])
#print(emb_matrix[word2idx['']])
#print(emb_matrix[word2idx['cat']])


# LOOK OUT FOR A LOT OF OOVS
X = [[word2idx[w[0]] if w[0] in word2idx else 0 for w in s] for s in sents]
X = pad_sequences(maxlen=args.max_sent_len, sequences=X, padding="post", value=0)
#print(X)

n_tags = len(tag2idx)
#print(n_tags)
y = [[tag2idx[w[1]] for w in s] for s in sents]
y = pad_sequences(maxlen=args.max_sent_len, sequences=y, padding="post", value=tag2idx["NIL"])
y = [to_categorical(i, num_classes=n_tags) for i in y]
#print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)



input = Input(shape=(args.max_sent_len,))
model = Embedding(input_dim=n_words, output_dim=50,
                  input_length=args.max_sent_len, mask_zero=True)(input)  # 50-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=15, validation_split=0.1, verbose=1)


hist = pd.DataFrame(history.history)

plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()


# do some silly predictions
i = 50

p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_test[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(str(w), str(t), str(pred)))



