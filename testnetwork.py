#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 erilyth <vishalvenkat71@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Use the neural network designed (Sample Code)
"""

from neuralnet import generate_network, train_network_main, run_network
import numpy as np

inputs = np.array([[0, 0, 1, 1], 
                   [1, 1, 1, 1], 
                   [1, 0, 1, 1]]).T

outputs = np.array([[0, 1, 1]])

shape = [4, 3, 2, 1]

weights = generate_network(shape)
weights = train_network_main(inputs, outputs, shape, weights)

test_input = np.array([[0, 0, 1, 1]]).T
run_network(test_input, shape, weights)