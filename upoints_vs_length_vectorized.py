from get_LSTM_internal_vectorized import LSTM_layer
import math
import pickle
import numpy as np
from tensorflow.keras import models, Model
from reviewunique import ReviewUnique

def get_mean_vector(h_set):
    """
    :param h_set: list of vectors h, each vector is output of LSTM layer at a timestep
    :return: mean vector hbar
    """
    # hbar = h_set[0]
    # for i in range(1, len(h_set)):
    #     hbar += h_set[i]
    # hbar = hbar / len(h_set)
    # return hbar
    return np.array(h_set, dtype='float64').mean(axis=0)

def get_magnitude(vector):
    """
    :param vector: 1D numpy array
    :return: magnitude of vector
    """
    # magnitude = 0
    # for element in vector:
    #     magnitude += element ** 2
    # return math.sqrt(magnitude)
    return np.sqrt(vector.dot(vector))

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
    # numbers setup
    num_timesteps = 500
    len_sequence = 120000
    num_count = 9000 # last num_count points will be counted
    interval_tuple = (10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 119999)

    path = "D:/Thesis/IMDb LSTM/Results/hyperband500_small_NoL2_1000_5-4/unique_points" # fill in path here
    with open("dataset_4000_500_07.pkl", 'rb') as file:
        dataset = pickle.load(file)

    model_opt = models.load_model('model/hyperband500_small_NoL2_1000_5-4/weights-improvement-095-0.2917-0.8842.hdf5')
    model_late = models.load_model('model/hyperband500_small_NoL2_1000_5-4/weights-improvement-883-2.5239-0.8641.hdf5')
    embed_layer_opt = Model(inputs=model_opt.input, outputs=model_opt.layers[0].output)
    lstm_opt = LSTM_layer(model_opt.layers[1].get_weights())
    embed_layer_late = Model(inputs=model_late.input, outputs=model_late.layers[0].output)
    lstm_late = LSTM_layer(model_late.layers[1].get_weights())

    # pick one review
    _, x, length = dataset.get_data()
    _, length = length
    x, y = x
    start_rev, end_rev = 0, 49
    while start_rev < 15000:
        data = []
        for i in range(start_rev, end_rev+1):
            opt_data = ReviewUnique(95, i, length[i], num_count, interval_tuple)
            late_data = ReviewUnique(883, i, length[i], num_count, interval_tuple)

            # current review
            x_in = x[i].reshape((1, num_timesteps))

            # review after passing through embed layer (optimal)
            lstm_in_opt = embed_layer_opt.predict(x_in)

            # creating entire review (repeated until len_sequence)
            start = lstm_in_opt[0][0]
            intermediate_steps = np.concatenate((lstm_in_opt[0][1:], np.tile(lstm_in_opt[0], (int((len_sequence-num_timesteps) / num_timesteps), 1))))

            # generating poincare map and taking last n points, then taking only unique points
            print("pre_poincare")
            h_t_opt, h_t_1_opt = get_poincare_mapping(lstm_opt, start, len_sequence, intermediate_steps)
            hbar_opt = get_mean_vector(h_t_opt)
            print("poincare_opt")
            for interval in interval_tuple:
                opt_set = set()
                # optimisation preprocessing
                h_t_opt[interval-num_count] = project(get_norm(h_t_opt[interval-num_count]),
                                                      hbar_opt)
                h_t_1_opt[interval-num_count] = project(get_norm(h_t_1_opt[interval-num_count]),
                                       hbar_opt)
                prev_h = h_t_1_opt[interval-num_count]
                for j in range(interval-num_count+1, interval):
                    h_t_opt[j] = prev_h
                    h_t_1_opt[j] = project(get_norm(h_t_1_opt[j]), hbar_opt)
                    prev_h = h_t_1_opt[j]
                    opt_set.add((h_t_opt[j], h_t_1_opt[j]))
                opt_data.add_data(interval, len(opt_set))
                # print(i, interval, len(opt_set))
            print("project_opt")

            # repeat above for late network
            lstm_in_late = embed_layer_late.predict(x_in)
            start = lstm_in_late[0][0]
            intermediate_steps = np.concatenate((lstm_in_late[0][1:], np.tile(lstm_in_late[0], (int((len_sequence-num_timesteps) / num_timesteps), 1))))

            h_t_late, h_t_1_late = get_poincare_mapping(lstm_late, start, len_sequence, intermediate_steps)
            hbar_late = get_mean_vector(h_t_late)
            for interval in interval_tuple:
                late_set = set()
                # optimisation preprocessing
                h_t_late[interval-num_count] = project(get_norm(h_t_late[interval-num_count]),
                                                        hbar_late)
                h_t_1_late[interval-num_count] = project(get_norm(h_t_1_late[interval-num_count]),
                                                          hbar_late)
                prev_h = h_t_1_late[interval-num_count]

                for j in range(interval-num_count+1, interval):
                    h_t_late[j] = prev_h
                    h_t_1_late[j] = project(get_norm(h_t_1_late[j]), hbar_late)
                    prev_h = h_t_1_late[j]
                    late_set.add((h_t_late[j], h_t_1_late[j]))
                late_data.add_data(interval, len(late_set))

            print(f"Review {i}: length {length[i]} \n\t\t  Total Entries: {len_sequence}"
                  f"\n\t\t  Last {num_count} points"
                  f"\n\t\t  Optimal epoch unique points: {opt_data.get_data(interval)}"
                  f"\n\t\t  Late epoch unique points: {late_data.get_data(interval)}")

            data.append((opt_data, late_data))

            # reset internal states of LSTM layers
            lstm_opt.reset()
            lstm_late.reset()

        with open(f"{path}/{start_rev}_{end_rev}_{len_sequence}.pkl", 'wb') as file:
            pickle.dump(data, file)

        print(f"Saved reviews {start_rev} to {end_rev} in {path}/{start_rev}_{end_rev}_{len_sequence}.pkl")

        start_rev += 50
        end_rev += 50

if __name__ == "__main__":
    main()