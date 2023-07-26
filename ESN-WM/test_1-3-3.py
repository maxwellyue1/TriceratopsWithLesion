# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen, str_to_bmp, convert_data 
from model_modified import generate_model, train_model, test_model
from identify_neurons import identify_neurons
from lesion import lesion
import sys
import os

if __name__ == '__main__':
    # Display
    fig = plt.figure(figsize=(10,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 1

    # 1-3-3 scalar task
    # Random generator initialization
    task = "1-3-3-scalar"
    n_gate = 3
    print(task)

    np.random.seed(1)

    # Build memory
    model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5,
                            radius=0.1, scaling=(1.0, 0.33), leak=1.0,
                            noise=(0.000, 0.0001, 0.000))

    # Training data
    n = 25000
    values = np.random.uniform(-1, +1, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    # Testing data
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    error = train_model(model, train_data)
    print("Training error : {0}".format(error))

    # identify the neurons to be lesioned
    num_lesion_neurons = 1
    neurons_lesion_dict = identify_neurons(model['W_out'], num_lesion_neurons)   # a dictionary of lesioned neurons, choices made on output/method
    neurons_lesion = neurons_lesion_dict['random, output 2']

    # lesion correspoing weights of selected neurons
    lesioned_model = lesion(model, neurons_lesion)

    error_wo_lesion = test_model(model, test_data, 42)
    error_w_lesion = test_model(lesioned_model, test_data, 42)
    print("Testing error without lesion : {0}".format(error_wo_lesion))
    print("Testing error with lesion : {0}".format(error_w_lesion))
    # np.save(files[0], test_data)
    # np.save(files[1], model["output"])
    # np.save(files[2], model["state"])

    # Display
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data = test_data

    ax2 = plt.subplot(n_subplots, 1, 1)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(data["input"][:,0],  color='0.75', lw=1.0)

    X, Y = np.arange(len(data)), np.ones(len(data))
    for i in range(n_gate):
        C = np.zeros((len(data),4))
        r = eval("0x"+colors[i][1:3])
        g = eval("0x"+colors[i][3:5])
        b = eval("0x"+colors[i][5:7])
        C[:,0] = r/255
        C[:,1] = g/255
        C[:,2] = b/255
        C[:,3] = data["input"][:,1+i]
        ax2.scatter(X, -1.05*Y-0.04*i, s=1.5, facecolors=C, edgecolors=None)
        ax2.plot(data["output"][:,i],  color='0.75', lw=1.0)
        ax2.plot(model["output"][:,i], lw=1.5, zorder=10)
        ax2.plot(lesioned_model["output"][:,i], lw=1.5, zorder=10)

    ax2.text(-25, -1.05, "Ticks:",
             fontsize=8, transform=ax2.transData,
             horizontalalignment="right", verticalalignment="center")
    ax2.set_ylim(-1.25,1.25)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Input & Output")
    ax2.text(0.01, 0.9, "B",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")
    plt.show()