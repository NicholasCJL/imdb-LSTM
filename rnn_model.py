import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

HIDDEN_SIZE = 64
EMBED_SIZE = 32
INPUT_SIZE = 80
DROPOUT = 0.2
REC_DROPOUT = 0.0
VOCAB_SIZE = 4000

# model tuner
def get_tuned_model(hp):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE,
        output_dim=EMBED_SIZE,
        input_length=INPUT_SIZE))
    model.add(LSTM(
        units=hp.Int("units", min_value=16, max_value=HIDDEN_SIZE, step=1),
        dropout=DROPOUT, recurrent_dropout=REC_DROPOUT
    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# vocab_size: size of vocabulary, also size of embedding input dimension
# embed_size: number of dimensions in vector space to embed vocabulary into
# input_length: length of each input (length of sequence)
# hidden_size: size of LSTM layer
def get_model(vocab_size, embed_size, input_length, hidden_size, dropout=0.0, recurrent_dropout=0.0):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=input_length))
    model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)