# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen, str_to_bmp, convert_data 
from model import generate_model, train_model, test_model
import sys
import os

if __name__ == '__main__':
    # Display
    fig = plt.figure(figsize=(10,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 1

    # -------------------------------------------------------------------------
    # 1-1-1 scalar task
    task = "1-1-1-scalar"
    n_gate = 1
    print(task)

    # Random generator initialization
    np.random.seed(1)

    # Build memory
    # builds a reservoir model with:
    #       1+n_gate input neurons
    #       1000 reservoir neurons
    #       n_gate output neurons
    # See generate_model in model.py
    model = generate_model(shape=(1+n_gate,1000,n_gate),
                           sparsity=0.5, radius=0.1, scaling=(1.0,1.0),
                           leak=1.0, noise=(0.0000, 0.0001, 0.0001))

    # Making training data
    n = 25000 # 300000
    values = np.random.uniform(-1, +1, n) #the input to the network 
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01 #the gating indicators
    train_data = generate_data(values, ticks) #format the data

    # Making testing data
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    # Train model
    error = train_model(model, train_data)
    print("Training error : {0}".format(error))

    # Test model
    error = test_model(model, test_data)
    print("Testing error : {0}".format(error))


    # Plotting stuff
    data = test_data

    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    ax1.plot(data["output"],  color='0.75', lw=1.0)
    ax1.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,1]
    ax1.scatter(X, -0.9*Y, s=1, facecolors=C, edgecolors=None)
    ax1.text(-25, -0.9, "Ticks:",
             fontsize=8, transform=ax1.transData,
             horizontalalignment="right", verticalalignment="center")
    ax1.set_ylim(-1.1,1.1)
    ax1.yaxis.tick_right()
    ax1.set_ylabel("Input & Output")
    ax1.text(0.01, 0.9, "A",
             fontsize=16, fontweight="bold", transform=ax1.transAxes,
             horizontalalignment="left", verticalalignment="top")

    
    plt.show()
