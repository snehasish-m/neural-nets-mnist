import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn 

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#learning rate
LR = 0.001

total_no_of_data = mnist.train.num_examples

no_of_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

#for testing data
test_x = mnist.test.images
test_y = mnist.test.labels

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def RNN_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size) 
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_my_model(x):
    #predicting output for the data
    pred = RNN_model(x)
    
    #error after predicting the digit
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) )
    
    #optimizer to reduce the error term (or cost)
    optimizer = tf.train.AdamOptimizer(learning_rate= LR).minimize(cost)
    
    # upto this point the model is only defined, but to run the model  
    # we have to run it within a session.
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(no_of_epochs):
            epoch_loss = 0
            for _ in range(int(total_no_of_data/batch_size)):
                #batch-wise data is trained
                ep_x, ep_y = mnist.train.next_batch(batch_size)
                ep_x = ep_x.reshape((batch_size,n_chunks,chunk_size))

                #cost(c) for this batch is calaculated
                _, c = sess.run([optimizer, cost], feed_dict={x: ep_x, y: ep_y})
                epoch_loss += c

            print('Epoch: ', epoch+1, '/',no_of_epochs,'loss:',epoch_loss)
        
        #no of correct predictions
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))

train_my_model(x)




