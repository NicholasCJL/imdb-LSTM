import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.regularizers import L1L2

HIDDEN_SIZE = 256
EMBED_SIZE = 32
INPUT_SIZE = 500
DROPOUT = 0.1
REC_DROPOUT = 0.0
VOCAB_SIZE = 4000

# model tuner
def get_tuned_model(hp):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE,
        output_dim=EMBED_SIZE,
        input_length=INPUT_SIZE))
    model.add(LSTM(
        units=hp.Int("units", min_value=8, max_value=HIDDEN_SIZE, step=1),
        dropout=DROPOUT, recurrent_dropout=REC_DROPOUT,
        kernel_regularizer=L1L2(
            hp.Choice("regulariserL1", values=[0.00, 0.001]),
            hp.Choice("regulariserL2", values=[0.00, 0.001])
        )
    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(5e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# vocab_size: size of vocabulary, also size of embedding input dimension
# embed_size: number of dimensions in vector space to embed vocabulary into
# input_length: length of each input (length of sequence)
# hidden_size: size of LSTM layer
def get_model(hidden_size, L1, L2):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE,
        output_dim=EMBED_SIZE,
        input_length=INPUT_SIZE))
    model.add(LSTM(
        units=hidden_size,
        dropout=DROPOUT, recurrent_dropout=REC_DROPOUT,
        kernel_regularizer=L1L2(L1, L2)
    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(5e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)