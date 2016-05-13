#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 erilyth <vishalvenkat71@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np

#To keep the randomness the same each time we run the code
#This can be removed if needed
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def run_network(input, network_shape, network_weights):
    
    #print network_weights
    current_input = input
    outputs = []
    for network_weight in network_weights:
        current_output_temp = np.dot(network_weight, current_input)
        #Apply the sigmoid function to smooth out and range the outputs
        current_output = sigmoid(current_output_temp)
        outputs.append(current_output)
        current_input = current_output

    print current_output


def train_network(input, output, network_shape, network_weights):
    
    #print network_weights
    current_input = input
    outputs = []
    for network_weight in network_weights:
        current_output_temp = np.dot(network_weight, current_input)
        #Apply the sigmoid function to smooth out and range the outputs
        current_output = sigmoid(current_output_temp)
        outputs.append(current_output)
        current_input = current_output

    #This will be in the reverse order
    deltas = []

    final_error = output - outputs[len(outputs)-1]
    final_delta = final_error * sigmoid_derivative(outputs[len(outputs)-1])
    deltas.append(final_delta)

    cur_delta = final_delta
    back_idx = len(outputs) - 2

    #Using the output of a layer that we get after multiplying with a weight matrix, for modifying that weight matrix
    for network_weight in network_weights[::-1][:-1]:
        #Going backwards (Backprop)
        next_error = np.dot(network_weight.T, cur_delta)
        next_delta = next_error * sigmoid_derivative(outputs[back_idx])
        deltas.append(next_delta)
        cur_delta = next_delta
        back_idx -= 1

    cur_weight_idx = len(network_weights) - 1

    for delta in deltas:
        input_used = None
        if cur_weight_idx - 1 < 0:
            input_used = input
        else:
            input_used = outputs[cur_weight_idx - 1]

        network_weights[cur_weight_idx] += np.dot(delta, input_used.T)
        cur_weight_idx -= 1

    #print network_weights
    return network_weights


#Sample training set
training_set_inputs = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1]]).T
training_set_outputs = np.array([[0, 1, 1, 0]])

#Let us model a 4,3,2,1 network
weight_arrays = []
network_shape = [4, 3, 2, 1]
for i in range(0, 3):
    cur_idx = i
    next_idx = i + 1
    #Rows correspond to next set of nodes
    #Columns correspond to current set of nodes
    weight_array = 2*np.random.rand(network_shape[next_idx], network_shape[cur_idx]) - 1
    weight_arrays.append(weight_array)

for i in range(1000):
    weight_arrays = train_network(training_set_inputs, training_set_outputs, network_shape, weight_arrays)

run_network(training_set_inputs, network_shape, weight_arrays)
