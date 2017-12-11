# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 00:17:09 2017

@author: Snehasish
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/tmp/data/", one_hot=True)

#learning rate
LR = 0.001

batch_size = 128
n_classes = 10 # MNIST total classes (0-9 digits)
no_of_epochs = 10
total_no_of_data = data.train.num_examples

#for testing data
test_x = data.test.images
test_y = data.test.labels

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])

def CNN_model(x):
    #randomly initialze weights and biases
    weights = {
        #5x5 conv, 1 input, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        
        #5x5 conv, 32 inputs, 64 outputs 
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        
        #fully connected, 7*7*64 inputs, 1024 outputs
        #28x28 is compressed down to feature maps of 7*7 size
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    #Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    #double-layered convolution
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate = 0.8)
    
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  
def train_model(x):
    #predicting output for the data
    pred = CNN_model(x)
    
    #error after predicting the digit
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) )
    
    #optimizer to reduce the error term (or cost)
    optimizer = tf.train.AdamOptimizer(learning_rate= LR).minimize(cost)
    
    # upto this point the model is only defined, but to run the model  
    # we have to run it within a session.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(no_of_epochs):
            epoch_loss = 0
            for _ in range(int(total_no_of_data/batch_size)):
                #batch-wise data is trained
                ep_x, ep_y = data.train.next_batch(batch_size)
                
                #cost(c) for this batch is calaculated
                _, c = sess.run([optimizer, cost], feed_dict={x: ep_x, y: ep_y})
                epoch_loss += c

            print('Epoch: ', ep+1, '/',no_of_epochs,'  loss:',epoch_loss)
        
        #no of correct predictions
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        #calculating the final accuracy
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y })*100, '%' )

    
train_model(x)

   