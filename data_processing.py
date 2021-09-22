from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import numpy as np

class DataSet():
    def __init__(self, num_words=None, seed=113, maxlen=100, train_portion=0.5):
        # retrieve IMDb data, x is a sequence containing movie review,
        # y is a label indicating if it is positive or negative sentiment
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=num_words, seed=seed)

        # padding sequences to all be of the same length
        self.x_train = pad_sequences(self.x_train, maxlen=maxlen)
        self.x_test = pad_sequences(self.x_test, maxlen=maxlen)

        self.split_data(train_portion)

        self.word_to_index = imdb.get_word_index()
        self.index_to_word = {i:word for (word, i) in self.word_to_index.items()}

    def get_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def get_vocab_length(self):
        return len(self.word_to_index)

    # splits data into ratio train:test -> (train_portion:1-train_portion)
    def split_data(self, train_portion):
        x = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test), axis=0)
        self.x_train, self.x_test = x[:math.floor(train_portion * len(x))], x[math.floor(train_portion * len(x)):]
        self.y_train, self.y_test = y[:math.floor(train_portion * len(y))], y[math.floor(train_portion * len(y)):]
        return None