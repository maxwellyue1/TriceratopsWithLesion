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
import copy

if __name__ == '__main__':
    # SET RANDOM SEED
    rand_seed = 0
    np.random.seed(seed=rand_seed)
    rand_seed += 1

    # Display
    fig = plt.figure(figsize=(10,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 1

    # 1-3-3 scalar task
    # Random generator initialization
    task = "1-3-3-scalar"
    n_gate = 3
    print(task)

    save_err = {}
    save_w_out = {}
    save_model = {}

    # TODO: look at noise
    # SEED SET WITHIN GENERATE MODEL, USED FOR GENERATING INITIAL WEIGHTS
    initial_model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5,
                                radius=0.1, scaling=(1.0, 0.33), leak=1.0,
                                noise=(0.000, 0.0000, 0.000), seed = rand_seed)
    rand_seed += 1
    
    # Training data (WILL BE THE SAME FOR ALL LESIONED MODELS)
    n = 25000
    values = np.random.uniform(-1, +1, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    # Testing data (WILL BE THE SAME FOR ALL LESIONED MODELS)
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])
    
    # Build memory
    # TODO: START FROM LESIONING 0 NEURONS
    for i in np.arange(100, 1001, 100):

        print(f"REDUCE MODEL SIZE TO {i} NEURONS")
        model = copy.deepcopy(initial_model)
        model["shape"] = (model["shape"][0], i, model["shape"][2])
        assert model["shape"][1] == i
        model["W_in"] = model["W_in"][0:i, :]
        model["W_rc"] = model["W_rc"][0:i, 0:i]
        model["W_fb"] = model["W_fb"][0:i, :]
        assert model["W_in"].shape[0] == i
        assert model["W_rc"].shape == (i, i)
        assert model["W_fb"].shape[0] == i

        error = train_model(model, train_data, seed = rand_seed )
        assert model["W_out"].shape[1] == i 

        rand_seed += 1
        print("Training error : {0}".format(error))

        # # identify the neurons to be lesioned
        # num_lesion_neurons = 1
        # neurons_lesion_dict = identify_neurons(model['W_out'], num_lesion_neurons)   # a dictionary of lesioned neurons, choices made on output/method
        # neurons_lesion = neurons_lesion_dict['random, output 2']

        # lesion correspoing weights of selected neurons
        # lesioned_model = lesion(model, neurons_lesion)

        model_error = test_model(model, test_data, 1)
        save_err[ f'{i} neurons'] = model_error
        save_w_out[f'{i} neurons'] = model['W_out']
        save_model[f'{i} neurons'] = model

        #error_w_lesion = test_model(lesioned_model, test_data, 42)
        print("Testing error without lesion : {0}".format(model_error))
        #print("Testing error with lesion : {0}".format(error_w_lesion))
        # np.save(files[0], test_data)
        # np.save(files[1], model["output"])
        # np.save(files[2], model["state"])

        # Display
        # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # data = test_data

        # ax2 = plt.subplot(n_subplots, 1, 1)
        # ax2.tick_params(axis='both', which='major', labelsize=8)
        # ax2.plot(data["input"][:,0],  color='0.75', lw=1.0)

        # X, Y = np.arange(len(data)), np.ones(len(data))
        # for i in range(n_gate):
        #     C = np.zeros((len(data),4))
        #     r = eval("0x"+colors[i][1:3])
        #     g = eval("0x"+colors[i][3:5])
        #     b = eval("0x"+colors[i][5:7])
        #     C[:,0] = r/255
        #     C[:,1] = g/255
        #     C[:,2] = b/255
        #     C[:,3] = data["input"][:,1+i]
        #     ax2.scatter(X, -1.05*Y-0.04*i, s=1.5, facecolors=C, edgecolors=None)
        #     ax2.plot(data["output"][:,i],  color='0.75', lw=1.0)
        #     ax2.plot(model["output"][:,i], lw=1.5, zorder=10)
        #     #ax2.plot(lesioned_model["output"][:,i], lw=1.5, zorder=10)

        # ax2.text(-25, -1.05, "Ticks:",
        #         fontsize=8, transform=ax2.transData,
        #         horizontalalignment="right", verticalalignment="center")
        # ax2.set_ylim(-1.25,1.25)
        # ax2.yaxis.tick_right()
        # ax2.set_ylabel("Input & Output")
        # ax2.text(0.01, 0.9, "B",
        #         fontsize=16, fontweight="bold", transform=ax2.transAxes,
        #         horizontalalignment="left", verticalalignment="top")
        # plt.show()
    np.save("save_err.npy", save_err)
    np.save("save_model.npy", save_model)
    np.save("save_w_out.npy", save_w_out)


    w = np.load('save_err.npy',allow_pickle=True)
    w_d = w.item()
    y = []

    for i in range(10):
        y.append(list(list(w_d.values())[i].values()))

    x=np.arange(100, 1001, 100)
    # ax.plot([1, 2, 3], label=)
    # ax.legend()
    plt.plot(x[1:],y[1:], label=['output 0','output 1','output 2','total'])
    plt.legend()
    plt.show()
