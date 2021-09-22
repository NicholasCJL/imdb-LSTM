import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import data_processing as dp
import rnn_model

# remove GPU from tensorflow visibility, set device to CPU (use for small networks)
# tf.config.set_visible_devices([], 'GPU')

HIDDEN_SIZE = 64
EMBED_SIZE = 128
NUM_EPOCHS = 10
INPUT_SIZE = 100

# obtain data
dataset = dp.DataSet(30000, maxlen=INPUT_SIZE, train_portion=0.8)
(x_train, y_train), (x_test, y_test) = dataset.get_data()
vocab_size = dataset.get_vocab_length()

# get model
model = rnn_model.get_model(vocab_size, EMBED_SIZE, INPUT_SIZE, HIDDEN_SIZE)

# checkpoint location
filepath = "model/LSTM64_D1_weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
