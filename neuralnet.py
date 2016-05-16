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
    '''Keeps the output in the range of -1 to 1 with a smooth transition'''
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    '''Derivative of the sigmoid function'''
    return x * (1 - x)


def generate_network(network_shape):
	'''
	Given a shape of the network, generate randomized weight matrices for the network
	'''
	weight_arrays = []
	for i in range(0, len(network_shape) - 1):
	    cur_idx = i
	    next_idx = i + 1
	    #Rows correspond to next set of nodes
	    #Columns correspond to current set of nodes
	    weight_array = 2*np.random.rand(network_shape[next_idx], network_shape[cur_idx]) - 1
	    weight_arrays.append(weight_array)

	return weight_arrays


def run_network(input, network_shape, network_weights):
    '''
    Given a trained network and the input(s), predict the possible output
    '''
    #Rows in the weight matrix correspond to nodes of the next layer 
    #whereas columns correspond to nodes of the previous layer
    #print network_weights
    current_input = input
    outputs = []
    for network_weight in network_weights:
        current_output_temp = np.dot(network_weight, current_input)
        #Apply the sigmoid function to smooth out and range the outputs
        current_output = sigmoid(current_output_temp)
        outputs.append(current_output)
        current_input = current_output

    return current_output.T


def train_network_main(input, output, training_rate, network_shape, network_weights):
	'''
	Take untrained weights and return trained weights for the neural network
	'''
	#Train the network multiple times to make it more accurate
	weight_arrays = network_weights
	for i in range(10000):
	    weight_arrays = train_network(input, output, training_rate, network_shape, weight_arrays)
	return weight_arrays


def train_network(input, output, training_rate, network_shape, network_weights):
    '''
    Given an untrained network, inputs and expected outputs, train the network
    '''
    #print network_weights
    current_input = input
    #Our predicted outputs
    outputs = []
    for network_weight in network_weights:
        current_output_temp = np.dot(network_weight, current_input)
        #Apply the sigmoid function to smooth out and range the outputs
        current_output = sigmoid(current_output_temp)
        outputs.append(current_output)
        current_input = current_output

    #This will be in the reverse order
    #Deltas will contain the error along with a few other terms which we come across
    # due to how we formulate gradient descent of the neural network
    deltas = []

    #We get these deltas according to the formula for gradient descent
    final_error = output - outputs[len(outputs)-1]
    final_delta = final_error * sigmoid_derivative(outputs[len(outputs)-1])
    deltas.append(final_delta)

    cur_delta = final_delta
    back_idx = len(outputs) - 2

    #Delta for layer i requires the weight matrix, delta of layer i+1 and expected output of layer i
    #Going backwards (Backprop)
    for network_weight in network_weights[::-1][:-1]:
        next_error = np.dot(network_weight.T, cur_delta)
        next_delta = next_error * sigmoid_derivative(outputs[back_idx])
        deltas.append(next_delta)
        cur_delta = next_delta
        back_idx -= 1

    cur_weight_idx = len(network_weights) - 1

    #These deltas will be in the reverse order, so we move backwards through the layers
    for delta in deltas:
        input_used = None
        if cur_weight_idx - 1 < 0:
            input_used = input
        else:
            input_used = outputs[cur_weight_idx - 1]

        #The weights of layer i are changed based on the input to layer i (or the output of layer i-1) and the delta of layer i
        #This is again due to the formulation of gradient descent
        network_weights[cur_weight_idx] += training_rate*np.dot(delta, input_used.T)
        cur_weight_idx -= 1

    #print network_weights
    return network_weights