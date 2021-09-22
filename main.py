from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import data_processing as dp
import rnn_model

# remove GPU from tensorflow visibility, set device to CPU (use for small networks)
# tf.config.set_visible_devices([], 'GPU')

HIDDEN_SIZE = 64
EMBED_SIZE = 128
NUM_EPOCHS = 10
INPUT_SIZE = 100
BATCH_SIZE = 128

model_name = "LSTM64_D1"

# obtain data
dataset = dp.DataSet(30000, maxlen=INPUT_SIZE, train_portion=0.8)
(x_train, y_train), (x_test, y_test) = dataset.get_data()
vocab_size = dataset.get_vocab_length()

# get model
model = rnn_model.get_model(vocab_size, EMBED_SIZE, INPUT_SIZE, HIDDEN_SIZE)

# checkpoint location
filepath = f"model/{model_name}/weights-improvement-{{epoch:02d}}-{{loss:.4f}}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
time_history = rnn_model.TimeHistory()
csv_logger = CSVLogger(f"model/{model_name}/training.csv")
callbacks_list = [checkpoint, time_history, csv_logger]

# fitting
model.fit(x_train, y_train,
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)
