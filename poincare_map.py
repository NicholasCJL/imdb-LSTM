from get_LSTM_internal import LSTM_layer
import math
import pickle
import numpy as np
from tensorflow.keras import models, Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

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
    def animate(frame):
        """
        plots <frame> in animation
        :param frame: frame number
        :return: None
        """
        ax.clear()
        colour = np.asarray([i for i in range(ppf*frame)])
        size = [0.9] * (ppf*frame-1) + [9]
        to_trace = trace_length if start_point + ppf*frame >= trace_length else start_point + ppf*frame
        ax.plot(h_t_opt[start_point+ppf*frame-to_trace:start_point+ppf*frame],
                h_t_1_opt[start_point+ppf*frame-to_trace:start_point+ppf*frame],
                   markersize=0.2, linewidth=1.1, color='g', zorder=1, label='Optimal epoch')
        ax.plot(h_t_late[start_point+ppf*frame-to_trace:start_point+ppf*frame],
                h_t_1_late[start_point+ppf*frame-to_trace:start_point+ppf*frame],
                   markersize=0.2, linewidth=1.1, color='b', zorder=2, label='Late epoch')
        ax.scatter(h_t_opt[start_point:start_point+ppf*frame], h_t_1_opt[start_point:start_point+ppf*frame],
                   s=size, c=colour, cmap=newcmp_g, zorder=3)
        ax.scatter(h_t_late[start_point:start_point+ppf*frame], h_t_1_late[start_point:start_point+ppf*frame],
                   s=size, c=colour, cmap=newcmp_b, zorder=4)
        plt.title(str(start_point+ppf*frame))
        ax.legend()
        return

    # plotting setup
    fig, ax = plt.subplots()
    fig.set_dpi(100)
    fig.set_size_inches(12.8, 7.2)
    fig.set_tight_layout(True)
    mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
    viridis = cm.get_cmap('viridis', 256)
    newcolors_k = viridis(np.linspace(0, 1, 256))
    newcolors_g = viridis(np.linspace(0, 1, 256))
    newcolors_b = viridis(np.linspace(0, 1, 256))
    black = np.array([0 / 256, 0 / 256, 0 / 256, 1])
    green = np.array([0 / 256, 256 / 256, 0 / 256, 1])
    blue = np.array([0 / 256, 0 / 256, 256 / 256, 1])
    pink = np.array([256 / 256, 25 / 256, 148 / 256, 1])
    newcolors_k[:255, :] = black
    newcolors_g[:255, :] = green
    newcolors_b[:255, :] = blue
    newcolors_k[255, :] = pink
    newcolors_g[255, :] = pink
    newcolors_b[255, :] = pink
    newcmp_k = ListedColormap(newcolors_k)
    newcmp_g = ListedColormap(newcolors_g)
    newcmp_b = ListedColormap(newcolors_b)

    # numbers setup
    trace_length = 25 # number of newest lines to draw
    ppf = 1 # datapoints per frame
    fps = 25
    num_timesteps = 500
    len_sequence = 100000
    start_point = 98000
    end_point = 100000
    num_cells = 60
    data_index = 3 # which review to use
    start_count = 95000 # number to start counting points from
    precision = 12 # precision for data point counting

    path = "" # fill in path here
    with open("dataset_4000_500_07.pkl", 'rb') as file:
        dataset = pickle.load(file)

    model_opt = models.load_model('model/hyperband500_small_NoL2_1000_5-4/weights-improvement-095-0.2917-0.8842.hdf5')
    model_late = models.load_model('model/hyperband500_small_NoL2_1000_5-4/weights-improvement-883-2.5239-0.8641.hdf5')
    embed_layer_opt = Model(inputs=model_opt.input, outputs=model_opt.layers[0].output)
    embed_layer_late = Model(inputs=model_late.input, outputs=model_late.layers[0].output)

    # pick one review
    _, x, length = dataset.get_data()
    _, length = length
    x, y = x
    for i in range(0, 5):
        # x_in = np.zeros((1, num_timesteps))
        x_in = x[i].reshape((1, num_timesteps))
        # print(length[i])
        # if length[i] < 350:
        #     continue
        lstm_in_opt = embed_layer_opt.predict(x_in)
        lstm_opt = LSTM_layer(model_opt.layers[1].get_weights())
        start = lstm_in_opt[0][0]
        # intermediate_steps = np.zeros((len_sequence-1, 32))
        #print(np.tile(lstm_in_opt[0], (int((len_sequence-num_timesteps) / num_timesteps), 1)).shape)
        intermediate_steps = np.concatenate((lstm_in_opt[0][1:], np.tile(lstm_in_opt[0], (int((len_sequence-num_timesteps) / num_timesteps), 1))))
        #print(intermediate_steps.shape)

        h_t_opt, h_t_1_opt = get_poincare_mapping(lstm_opt, start, len_sequence, intermediate_steps)
        hbar_opt = get_mean_vector(h_t_opt)
        opt_set = set()
        for j in range(start_count, len(h_t_opt)):
            h_t_opt[j] = project(get_norm(h_t_opt[j]), hbar_opt)
            h_t_1_opt[j] = project(get_norm(h_t_1_opt[j]), hbar_opt)
            opt_set.add((round(h_t_opt[j], precision), round(h_t_1_opt[j], precision)))
            # opt_set.add((h_t_opt[j], h_t_1_opt[j]))
        # print(h_t[:5])
        # print(h_t_1[:5])
        # print(len(h_t))
        #print(len(h_t_1_opt))

        lstm_in_late = embed_layer_late.predict(x_in)
        lstm_late = LSTM_layer(model_late.layers[1].get_weights())
        start = lstm_in_late[0][0]
        # intermediate_steps = np.zeros((len_sequence-1, 32))
        #print(np.tile(lstm_in_late[0], (int((len_sequence-num_timesteps) / num_timesteps), 1)).shape)
        intermediate_steps = np.concatenate((lstm_in_late[0][1:], np.tile(lstm_in_late[0], (int((len_sequence-num_timesteps) / num_timesteps), 1))))
        #print(intermediate_steps.shape)

        h_t_late, h_t_1_late = get_poincare_mapping(lstm_late, start, len_sequence, intermediate_steps)
        hbar_late = get_mean_vector(h_t_late)
        late_set = set()
        for j in range(start_count, len(h_t_late)):
            h_t_late[j] = project(get_norm(h_t_late[j]), hbar_late)
            h_t_1_late[j] = project(get_norm(h_t_1_late[j]), hbar_late)
            late_set.add((round(h_t_late[j], precision), round(h_t_1_late[j], precision)))
            # late_set.add((h_t_late[j], h_t_1_late[j]))
        # print(h_t[:5])
        # print(h_t_1[:5])
        # print(len(h_t))
        #print(len(h_t_1_late))

        print(f"Review {i}: length {length[i]} \n\t\t  Total Entries: {len_sequence}"
              f"\n\t\t  Last {len_sequence-start_count} points"
              f"\n\t\t  Optimal epoch unique points: {len(opt_set)}"
              f"\n\t\t  Late epoch unique points: {len(late_set)}")


        # colour = np.asarray([i for i in range(len_sequence-start_point-1)])
        # colour = np.asarray([i for i in range(100)])
        # plt.plot(h_t, h_t_1, markersize=0.5)
        # im = ax.scatter(h_t[start_point:], h_t_1[start_point:], s=1, c=colour, cmap='viridis')
        # fig.colorbar(im, orientation='vertical')
        #plt.xlim([2.454, 2.456])
        #plt.ylim([2.454, 2.456])
        # print(h_t_1[1800:])
        # animation = FuncAnimation(fig, animate, frames=(end_point-start_point)//ppf,
        #                         interval=1000/fps, repeat=False, save_count=(len_sequence-start_point-1)//ppf)
        # animation.save(f'D:/Thesis/IMDb LSTM/Results/hyperband500_small_NoL2_1000_5-4/test_095-883_{i}_{length[i]}_{start_point}-{end_point}.mp4')
    # plt.show()

if __name__ == "__main__":
    main()
