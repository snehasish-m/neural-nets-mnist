# -*- coding: utf-8 -*-
"""
Created on Fri Dec 8 10:42:45 2017

@author: Snehasish
"""

#import required packages
import tflearn as tfl
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
#mnist dataset from tflearn
import tflearn.datasets.mnist as mnist

#learning rate
LR = 0.001

#creating the dataset (both training and validation set)
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

#reshape into 4D tensor as required by tflearn
X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# Building convolutional neural network (cnn)
cnn = input_data(shape=[None, 28, 28, 1], name='input')

cnn = conv_2d(cnn, 32, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, keep_prob=0.8)   

cnn = fully_connected(cnn, 10, activation='softmax')

acc= Accuracy()
cnn = regression(cnn, optimizer='adam', learning_rate=LR, metric = acc, loss='categorical_crossentropy', name='targets')


my_model = tfl.DNN(cnn)

my_model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set= ({'input': test_x}, {'targets': test_y}), show_metric=True, snapshot_epoch=True)



