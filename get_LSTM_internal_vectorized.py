import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import models, Model
import numpy as np

from numba import guvectorize

class LSTM_layer():
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x): # for consistency
        return np.tanh(x)

    def __init__(self, weights):
        """
        :param weights: weights of LSTM layer
        """
        # transposing matrices for dot product
        self.W, self.U, self.b = np.transpose(weights[0]), np.transpose(weights[1]), np.transpose(weights[2])
        self.num_units = int(self.U.shape[1])
        self.split_weights()
        # LSTM trained stateless, initial C and h are zero vectors
        self.C = np.zeros((self.num_units), dtype=np.float32)
        self.h = np.zeros((self.num_units), dtype=np.float32)

    def split_weights(self):
        # weights are stored as (neuron_num, (i, f, c, o))
        self.W_i = np.ascontiguousarray(self.W[:self.num_units, :])
        self.W_f = np.ascontiguousarray(self.W[self.num_units:self.num_units * 2, :])
        self.W_c = np.ascontiguousarray(self.W[self.num_units * 2:self.num_units * 3, :])
        self.W_o = np.ascontiguousarray(self.W[self.num_units * 3:, :])

        self.U_i = np.ascontiguousarray(self.U[:self.num_units, :])
        self.U_f = np.ascontiguousarray(self.U[self.num_units:self.num_units * 2, :])
        self.U_c = np.ascontiguousarray(self.U[self.num_units * 2:self.num_units * 3, :])
        self.U_o = np.ascontiguousarray(self.U[self.num_units * 3:, :])

        self.b_i = np.ascontiguousarray(self.b[:self.num_units])
        self.b_f = np.ascontiguousarray(self.b[self.num_units:self.num_units * 2])
        self.b_c = np.ascontiguousarray(self.b[self.num_units * 2:self.num_units * 3])
        self.b_o = np.ascontiguousarray(self.b[self.num_units * 3:])

    def step(self, x_t):
        """
        Performs a timestep (propagating new input through layer)
        :return: array of activations [ft, it, cc, cc_update, c_out, ot, ht]
        """
        activations = []
        # forget step
        ft = self.get_ft(x_t)
        activations.append(ft)
        self.forget(ft)

        # "remembering" step
        it = self.get_it(x_t)
        activations.append(it)
        cc = self.get_CC(x_t)
        activations.append(cc)
        cc_update = self.get_CC_update(it, cc)
        activations.append(cc_update)
        self.remember(cc_update)

        # output step
        c_out = self.get_C_output()
        activations.append(c_out)
        ot = self.get_ot(x_t)
        activations.append(ot)
        output = self.output(c_out, ot)
        activations.append(output)

        return activations

    def reset(self):
        # call when done with one input (with all timesteps completed)
        # resets internal cell state and starting hidden state
        self.C = np.zeros((self.num_units), dtype=np.float32)
        self.h = np.zeros((self.num_units), dtype=np.float32)


    # vectorized activation propagation
    @staticmethod
    @guvectorize(
        ["float32[:, :], float32[:, :], float32[:], float32[:], float32[:], float32[:]"],
        "(n, m),(n, n),(m),(n),(n)->(n)",
        nopython=True
    )
    def get_ft_vec(W_f, U_f, x_t, h, b_f, res):
        wfx = W_f.dot(x_t)
        ufh = U_f.dot(h)
        sum_int = wfx + ufh
        sum_f = sum_int + b_f
        res[:] = 1 / (1 + np.exp(-sum_f))

    @staticmethod
    @guvectorize(
        ["float32[:, :], float32[:, :], float32[:], float32[:], float32[:], float32[:]"],
        "(n, m),(n, n),(m),(n),(n)->(n)",
        nopython=True
    )
    def get_it_vec(W_i, U_i, x_t, h, b_i, res):
        wix = W_i.dot(x_t)
        uih = U_i.dot(h)
        sum_int = wix + uih
        sum_f = sum_int + b_i
        res[:] = 1 / (1 + np.exp(-sum_f))

    @staticmethod
    @guvectorize(
        ["float32[:, :], float32[:, :], float32[:], float32[:], float32[:], float32[:]"],
        "(n, m),(n, n),(m),(n),(n)->(n)",
        nopython=True
    )
    def get_CC_vec(W_c, U_c, x_t, h, b_c, res):
        wcx = W_c.dot(x_t)
        uch = U_c.dot(h)
        sum_int = wcx + uch
        sum_f = sum_int + b_c
        res[:] = np.tanh(sum_f)

    @staticmethod
    @guvectorize(
        ["float32[:, :], float32[:, :], float32[:], float32[:], float32[:], float32[:]"],
        "(n, m),(n, n),(m),(n),(n)->(n)",
        nopython=True
    )
    def get_ot_vec(W_o, U_o, x_t, h, b_o, res):
        wox = W_o.dot(x_t)
        uoh = U_o.dot(h)
        sum_int = wox + uoh
        sum_f = sum_int + b_o
        res[:] = 1 / (1 + np.exp(-sum_f))

    # activations start
    def get_ft(self, x_t):
        # sigmoid(W_f . x_t + U_f . h_(t-1) + b_f) . is dot product
        # wfx = self.W_f.dot(x_t)
        # ufh = self.U_f.dot(self.h)
        # return LSTM_layer.sigmoid(wfx + ufh + self.b_f)
        return LSTM_layer.get_ft_vec(self.W_f, self.U_f, x_t, self.h, self.b_f)

    def get_it(self, x_t):
        # sigmoid(W_i . x_t + U_i . h_(t-1) + b_i)
        # wix = self.W_i.dot(x_t)
        # uih = self.U_i.dot(self.h)
        # return LSTM_layer.sigmoid(wix + uih + self.b_i)
        return LSTM_layer.get_it_vec(self.W_i, self.U_i, x_t, self.h, self.b_i)

    def get_CC(self, x_t):
        # candidate cell state before proportion
        # tanh(W_c . x_t + U_c . h_(t-1) + b_c)
        wcx = self.W_c.dot(x_t)
        uch = self.U_c.dot(self.h)
        return LSTM_layer.tanh(wcx + uch + self.b_c)
        # return LSTM_layer.get_CC_vec(self.W_c, self.U_c, x_t, self.h, self.b_c)

    def get_ot(self, x_t):
        # sigmoid(W_o . x_t + U_o . h_(t-1) + b_o)
        # wox = self.W_o.dot(x_t)
        # uoh = self.U_o.dot(self.h)
        # return LSTM_layer.sigmoid(wox + uoh + self.b_o)
        return LSTM_layer.get_ot_vec(self.W_o, self.U_o, x_t, self.h, self.b_o)

    def get_C_output(self):
        # cell state output before proportion
        # tanh(C_t)
        return LSTM_layer.tanh(self.C)

    def get_CC_update(self, it, cc):
        # candidate cell state after proportion, for updating cell state
        # it * cc, * is Hadamard product
        return it * cc
    # activations end


    # state updates start
    def forget(self, ft):
        # update old cell state in the forget step
        self.C = self.C * ft

    def remember(self, cc_update):
        # update old cell state with new information
        self.C = self.C + cc_update

    def output(self, c_output, ot):
        # proportionate the cell output vector for new output and hidden state
        self.h = c_output * ot
        return self.h
    # state updates end




