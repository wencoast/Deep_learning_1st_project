from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow import keras
import keras
import tensorflow as tf
import numpy as np
from resnet import resnet 
import matplotlib.pyplot as plt

print('The version of tensorflow:',tf.__version__)
print('The version of keras:', keras.__version__)

# access the Fashion MNIST directly from TensorFlow,just import and load the data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# Parameters
#learning_rate = 0.001
batch_size = 64
epochs = 10
#display_step = 20
#dropout = 0.8

# Creating placeholders
x = tf.placeholder(tf.float32, shape = [None, 784])
y_true = tf.placeholder(tf.float32, shape = [None, 10])

#If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant.
# In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Create the ResNet model
model = resnet(x = x_image, n = 20, num_classes = 10)

#define activation of last layer as score
score = model.out

# Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = score))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

# To measure accuracy
correct_prediction = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for j in range(epochs):
        for i in range(0, 60000, batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[: 5000],
                                                 y_true: mnist.test.labels[: 5000]})
        print('After %d epochs, accuracy: %g' % (j + 1, test_accuracy))