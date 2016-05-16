#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 erilyth <vishalvenkat71@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Use the neural network for image recognition
"""

from neuralnet import generate_network, train_network_main, run_network
import numpy as np
from PIL import Image
import copy


def normalize_image(imgArr):
    '''
    RGB to greyscale normalization
    '''
    newImgArr = copy.deepcopy(imgArr)
    for row in range(len(imgArr)):
        for col in range(len(imgArr[row])):
            new_value = (int(imgArr[row][col][0]) + int(imgArr[row][col][1]) + int(imgArr[row][col][2])) / 3
            if new_value != 0.0:
                #Values are either 0 or 255
                new_value = 255.0
            newImgArr[row][col][0] = new_value
            newImgArr[row][col][1] = new_value
            newImgArr[row][col][2] = new_value

    return newImgArr


def generate_dataset():
    '''
    Generate the images dataset which will be used by run_dataset
    '''
    imgArrList = []
    numbersWeHave = range(0, 10) #We have 10 digits 
    versionsWeHave = range(1, 16) #We have 15 versions of each digit
    outputs = []

    for number in numbersWeHave:
        for version in versionsWeHave:
            cur_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            cur_output[int(number)] = 1.0
            outputs.append(cur_output)
            imgPath = 'images/numbers/' + str(number) + '.' + str(version) + '.png'
            imgCur = Image.open(imgPath)
            #Make it black and white
            imgCurArr = normalize_image(np.array(imgCur))
            imgArrList.append(imgCurArr)

    inputs = []
    for imgArr in imgArrList:
        pixels = []
        for row in imgArr:
            for col in row:
                #Use normalized images here, so considering either R or G or B will be the same
                pixels.append(col[0]/255.0)
        inputs.append(pixels)

    return [inputs, outputs]


def generate_test_data(imgPath):
    '''
    Take an image and generate the required inputs for the network
    '''
    imgCur = Image.open(imgPath)
    imgCurArr = normalize_image(np.array(imgCur))
    inputs = []
    pixels = []
    for row in imgCurArr:
        for col in row:
            pixels.append(col[0]/255.0)
    inputs.append(pixels)
    return np.array(inputs).T



data = generate_dataset()

inputs = np.array(data[0]).T
outputs = np.array(data[1]).T

#8x8 pixel images, we consider each pixel has only a single greyscale value so 8x8 = 64 inputs
#Consider a neural net with 1 hidden layer of 32 nodes with 64 input nodes and 10 output nodes
shape = [64, 32, 10]

weights = generate_network(shape)
weights = train_network_main(inputs, outputs, 0.01, shape, weights)

#Training is done now, we can test the network

for i in range(10):
    test_inputs = generate_test_data('images/test_numbers/test' + str(i) + '.png')
    test_outputs = run_network(test_inputs, shape, weights)

    print "Test " + str(i+1) + ": Input - " + str(i) + ", Output - ",
    for output in test_outputs:
        best_match = max(output)
        for idx in range(len(output)):
            if output[idx] == best_match:
                print idx,
        print ""
