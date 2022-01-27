from get_LSTM_internal import LSTM_layer
import math
import pickle
import numpy as np
from tensorflow.keras import models, Model
import matplotlib.pyplot as plt

def get_mean_vector(h_set):
    """
    :param h_set: list of vectors h, each vector is output of LSTM layer at a timestep
    :return: mean vector hbar
    """
    hbar = h_set[0]
    for i in range(1, len(h_set)):
        hbar += h_set[i]
    hbar = hbar / len(h_set)
    return hbar

def get_magnitude(vector):
    """
    :param vector: 1D numpy array
    :return: magnitude of vector
    """
    magnitude = 0
    for element in vector:
        magnitude += element ** 2
    return math.sqrt(magnitude)

def get_norm(vector):
    """
    :param vector: vector to normalise
    :return: norm of vector
    """
    return vector / get_magnitude(vector)

def project(vector, basis):
    """
    :param vector: vector to project onto basis
    :param basis: basis for poincare map
    :return: vector projected onto basis (dot product)
    """
    return vector.dot(basis)

def get_poincare_mapping(lstm, start, num_steps, intermediate_inputs=None):
    """
    get poincare mapping (projections at h_t, projections at h_{t+1})
    :param lstm: trained LSTM_layer
    :param start: starting input
    :param num_steps: number of iterations to perform, length of intermediate_inputs has to be num_steps - 1
    :param intermediate_inputs: list of x_t to input at each timestep, zero vectors if None, each vector has to be length start
    :return: poincare mapping
    """
    if intermediate_inputs is None:
        intermediate_inputs = [np.zeros(len(start), dtype=np.float32) for _ in range(num_steps - 1)]

    # get h_t at each timestep
    h_t = [lstm.step(start)[-1]]
    h_t_1 = [] # h_{t+1}
    for i in range(num_steps - 1):
        curr_h = lstm.step(intermediate_inputs[i])[-1]
        h_t.append(curr_h)
        h_t_1.append(curr_h)

    h_t.pop() # remove last element so h_t and h_{t+1} aligns
    return h_t, h_t_1

def main():
    num_timesteps = 500
    num_cells = 60
    data_index = 3 # which review to use

    path = "" # fill in path here
    with open("dataset_4000_500_07.pkl", 'rb') as file:
        dataset = pickle.load(file)

    model = models.load_model('model/hyperband500_small_2/weights-improvement-015-0.2867-0.8877.hdf5')
    embed_layer = Model(inputs=model.input, outputs=model.layers[0].output)
    fig, ax = plt.subplots()

    # pick one review
    x, _, length = dataset.get_data()
    length, _ = length
    x, y = x
    for i in range(1):
        x_in = np.zeros((1, num_timesteps))
        # x_in = x[i].reshape((1, num_timesteps))
        print(length[i])

        lstm_in = embed_layer.predict(x_in)
        lstm = LSTM_layer(model.layers[1].get_weights())
        start = lstm_in[0][0]
        intermediate_steps = lstm_in[0][1:]

        h_t, h_t_1 = get_poincare_mapping(lstm, start, 500, intermediate_steps)
        hbar = get_mean_vector(h_t)
        for j in range(len(h_t)):
            h_t[j] = project(get_norm(h_t[j]), hbar)
            h_t_1[j] = project(get_norm(h_t_1[j]), hbar)
        # print(h_t[:5])
        # print(h_t_1[:5])
        # print(len(h_t))
        print(len(h_t_1))


        # colour = np.asarray([i for i in range(499)])
        im = ax.scatter(h_t[100:], h_t_1[100:], s=0.5, cmap='viridis')
        # fig.colorbar(im, orientation='vertical')
        #plt.xlim([2.6, 3])
        #plt.ylim([2.6, 3])
    plt.show()

if __name__ == "__main__":
    main()