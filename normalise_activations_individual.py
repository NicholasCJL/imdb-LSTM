import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.stats import linregress
from get_LSTM_internal import LSTM_layer
from tensorflow.keras import models, Model
import data_processing as dp
import numpy as np
import os
import pickle

class RegressionData():
    def __init__(self, type):
        self.type = type
        self.slope = []
        self.intercept = []
        self.r = []
        self.stderr = []
        self.iscorrect = []
        self.input_length = []

    def add_slope(self, slope):
        self.slope.append(slope)

    def add_intercept(self, intercept):
        self.intercept.append(intercept)

    def add_r(self, r):
        self.r.append(r)

    def add_stderr(self, stderr):
        self.stderr.append(stderr)

    def add_result(self, correct):
        self.iscorrect.append(correct)

    def add_length(self, length):
        self.input_length.append(length)

def main():
    power = True # get power spectrum
    num_timesteps = 500
    num_cells = 60

    path = "Results/timesteps500_embed32_hidden60_vocab4000_5/norm_activation_by_type_length"

    model = models.load_model('model/hyperband500_small_5/weights-improvement-028-0.3471-0.8830.hdf5')
    dataset = dp.DataSet(4000, maxlen=num_timesteps, train_portion=0.7)
    # x, _, length = dataset.get_data()
    # length, _ = length
    # x, y = x
    _, y, length = dataset.get_data()
    _, length = length
    x, y = y

    i2w = dataset.i2w_vocab

    # isolating embedding layer, input sequence and obtain word embeddings (LSTM input) for manual processing
    embed_layer = Model(inputs=model.input, outputs=model.layers[0].output)

    # LSTM layer from original model to verify manual computation is correct
    test_layer = Model(inputs=model.input, outputs=model.layers[1].output)

    lstm = LSTM_layer(model.layers[1].get_weights())

    types = ['ft', 'it', 'cc', 'cc_update', 'c_out', 'ot', 'ht']
    norm_types = ['ft', 'it', 'ot']  # value goes from 0 to 1
    mirror_norm_types = ['cc', 'cc_update', 'c_out', 'ht']  # value goes from -1 to 1

    reg_data = [RegressionData(types[i]) for i in range(7)]

    print(f"Number of datapoints: {len(y)}")

    for index in range(len(y)):
        if index % 1000 == 0:
            print(index)

        # os.mkdir(f'{path}/{index}')

        x_in = x[index].reshape((1, num_timesteps))
        length_in = length[index]
        lstm_in = embed_layer.predict(x_in)

        # check if model is correct for this datapoint
        model_prediction = model.predict(x_in, batch_size=1)[0]
        model_prediction = 1 if model_prediction >= 0.5 else 0
        match = y[index] == model_prediction

        activations_list = [] # [ft, it, cc, cc_update, c_out, ot, ht]

        for k in range(num_timesteps): # number of timesteps
            activations_list.append(lstm.step(lstm_in[0][k]))
        # reset LSTM
        lstm.reset()

        for j in range(7): # number of activation types
            # get all activation values (for activation j) at every timestep, (timestep, array of activations for each neuron)
            activations_selected = np.asarray([activations[j] for activations in activations_list])
            # changes to (neuron, array of activations (type j) for each timestep)
            cell_level_activations = np.transpose(activations_selected)

            # check type
            if types[j] in mirror_norm_types:
                cell_level_activations = (cell_level_activations + 1) / 2 # compress -1 to 1 -> 0 to 1

            cell_level_activations = cell_level_activations / num_cells # normalise

            # Fourier transform of each cell's activations
            frequencies = rfftfreq(cell_level_activations[0].size, d=1)
            total_spectrum = np.asarray([0] * frequencies.size)
            for cell in cell_level_activations:
                cell_level_k = np.abs(rfft(cell))
                if power:
                    cell_level_k = cell_level_k ** 2
                total_spectrum = total_spectrum + cell_level_k

            cell_level_k = total_spectrum

            # cell_level_activations = np.asarray(sum(cell_level_activations)) # sum
            #
            # # Fourier transform
            # cell_level_k = np.abs(rfft(cell_level_activations)) # activations in frequency domain
            # if power:
            #     cell_level_k = cell_level_k ** 2 # power spectrum
            # frequencies = rfftfreq(cell_level_activations.size, d=1)
            # #plt.plot(frequencies[1:], abs(cell_level_k[1:]))
            # #plt.show()

            # convert to log-log
            log_cell_level_k = np.log10(cell_level_k[1:] + 0.00000001)
            log_frequencies = np.log10(frequencies[1:] + 0.00000001)
            # plt.plot(log_frequencies, log_cell_level_k, 'r')

            # linear regression
            slope, intercept, r_value, _, stderr = linregress(log_frequencies, log_cell_level_k)
            yfit = intercept + slope * log_frequencies
            #plt.plot(log_frequencies, yfit, 'g')

            # store regression data
            reg_data[j].add_slope(slope)
            reg_data[j].add_intercept(intercept)
            reg_data[j].add_r(r_value)
            reg_data[j].add_stderr(stderr)
            reg_data[j].add_result(match)
            reg_data[j].add_length(length_in)


    with open(f'{path}/power_fft_reg_data_test_028.pkl', 'wb') as file:
        pickle.dump(reg_data, file)

if __name__ == "__main__":
    main()