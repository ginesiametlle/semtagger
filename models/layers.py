

import keras.layers
import keras.initializers

# define layers on the network. THINGS I NEED IN PARAMS
#num_words: number of words in the embedding vocab
#num_feats: dimensionality of the embedding vectors
#max_length: maximum allowed length of a sentence





input = Input(shape=(args.max_sent_len,))
model = Embedding(input_dim=n_words, output_dim=50,
                  input_length=args.max_sent_len, mask_zero=True)(input)  # 50-dim embedding
model = Bidirectional(GRU(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()




def embedding_layer(params):
    return keras.layers.Embedding(
        input_dim = params.num_words,
        output_dim = params.num_feats,
        input_length = params.max_length,
        mask_zero = True
    )
