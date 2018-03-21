
# define a generic class for model
# with extensions for lstm, bi-lstm, etc

# PARAMS THAT ARE NEEDED

from layers import embedding_layer, dense_layer, lstm_layer, gru_layer
from keras.layers import TimeDistributed, Bidirectional


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




def get_bilstm(params):
    model = Sequential()
    model.add(Embedding(
        input_dim = params.num_words,
        output_dim = params.num_feats,
        input_length = params.max_sent_len,
        mask_zero = True
    )
    model.add(Bidirectional)


def get_model(model_name, model_params):
    # use bidirectional gru as a default model
    if model_name == 'rnn':
        return get_rnn(model_params)
    if model_name == 'lstm':
        return get_lstm(model_params)
    elif model_name == 'gru':
        return get_gru(model_name)
    elif model_name == 'bi-lstm':
        return get_bilstm(model_params)
    else:
        return get_bigru(model_params)

