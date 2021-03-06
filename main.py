import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import data_processing as dp
import rnn_model
from keras_tuner import Hyperband, HyperParameters

# remove GPU from tensorflow visibility, set device to CPU (use for small networks)
# tf.config.set_visible_devices([], 'GPU')

NUM_EPOCHS = 35
INPUT_SIZE = 500
BATCH_SIZE = 128
VOCAB_SIZE = 4000

model_name = "hyperband500_small"

# obtain data
dataset = dp.DataSet(VOCAB_SIZE, maxlen=INPUT_SIZE, train_portion=0.7)
(x_train, y_train), (x_test, y_test) = dataset.get_data()
vocab_size = dataset.get_vocab_length()

# # tuning
# tuner = Hyperband(
#     rnn_model.get_tuned_model,
#     objective="val_accuracy",
#     max_epochs=20,
#     hyperband_iterations=10,
#     overwrite=False,
#     directory=f'model/{model_name}',
#     project_name="imdbLSTM"
# )
#
# print(tuner.search_space_summary())
#
# csv_logger = CSVLogger(f"model/{model_name}/training.csv")
# early_stopping = EarlyStopping(
#     monitor="val_accuracy",
#     patience=5,
#     restore_best_weights=True
# )
# callbacks_list = [csv_logger, early_stopping]
#
# tuner.search(x_train, y_train,
#              epochs=NUM_EPOCHS,
#              batch_size=BATCH_SIZE,
#              validation_data=(x_test, y_test),
#              callbacks=callbacks_list)

# get best hyperparams and retrain model
# hyperparams = tuner.get_best_hyperparameters()[0]
# print(hyperparams.get('units'), hyperparams.get('regulariserL1'), hyperparams.get('regulariserL2'))
# model = tuner.hypermodel.build(hyperparams)
# model.summary()
# filepath = f"model/{model_name}/weights-improvement-{{epoch:03d}}-{{val_loss:.4f}}-{{val_accuracy:.4f}}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint, csv_logger]
# model.fit(x_train, y_train,
#           epochs=NUM_EPOCHS,
#           batch_size=BATCH_SIZE,
#           validation_data=(x_test, y_test),
#           callbacks=callbacks_list)
# model.save(f"model/{model_name}/best_model.hdf5")


# checkpoint location
filepath = f"model/{model_name}/weights-improvement-{{epoch:03d}}-{{val_loss:.4f}}-{{val_accuracy:.4f}}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# time_history = rnn_model.TimeHistory()
csv_logger = CSVLogger(f"model/{model_name}/training.csv")
callbacks_list = [checkpoint, csv_logger]
model = rnn_model.get_model(60, 0.0, 0.0001)

# fitting
model.fit(x_train, y_train,
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)
