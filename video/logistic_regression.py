#!/usr/bin/env python

"""
A logistic regression learning algorithm example using TensorFlow library.
"""

import tensorflow as tf
import numpy as np

# Import  data
data = np.load('./trainset.npy')
norm = lambda t: t / 255.
norm = np.array([norm(xi) for xi in data])
X = norm.reshape(-1, 35)

y_ = np.load('./trainground.npy')

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 35]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([35, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        i_start = 0
        i_end = batch_size
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = X[i_start:i_end]
            batch_ys = y_[i_start:i_end]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
            # update indexes
            i_start += batch_size
            i_end += batch_size
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    data = np.load('./testset.npy')
    norm = lambda t: t / 255.
    norm = np.array([norm(xi) for xi in data])
    test_X = norm.reshape(-1, 35)

    test_y = np.load('./testround.npy')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: test_X, y: test_y}))

    for i in range(0,30):
#        print ("predicted:", pred.eval({x: test_X[i, None, :]}))
        p = sess.run(tf.argmax(pred,1), feed_dict={x: test_X[i, None]})
        print ("predicted:", p[0])
        print test_y[i]

