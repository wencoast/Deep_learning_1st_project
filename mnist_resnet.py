from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow import keras
import keras
import tensorflow as tf
import numpy as np
from resnet import resnet 
import matplotlib.pyplot as plt

import argparse

"""parsing and configuration"""
def parse_args():

    desc="TensorFlow implementation of Resnet on Fashion_mnist! author: Hazard Wen"
    parser=argparse.ArgumentParser(description=desc)

    # 1st add dropout or not. Dropout, probability to keep units
    parser.add_argument('--dropout_keep_prob', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    # 2nd add weight decay or not
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    # 3rd add arguments for different optimizer
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    # 4th add argument for learning rate
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.1)
    parser.add_argument('--max_epochs', type=int, help='Number of epochs to run.', default=10)
    return check_args(parser.parse_args())


def check_args(args):

    return args

    #parser.add_argument()
# secondly add data argumentation

# thirdly add different optimizers including SGD, SGD with momentum, Adam

#

# add batch normalization or not
# from taki/ops.py
#def batch_norm(x, is_training=True, scope='batch_norm'):
#    return tf_contrib.layers.batch_norm(x,
#                                        decay=0.9, epsilon=1e-05,
#                                        center=True, scale=True, updates_collections=None,
#                                       is_training=is_training, scope=scope)





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

#args=parse_arguments()
# Create the ResNet model
# model = resnet(x = x_image, n = 20, num_classes = 10, keep_probability=args.dropout_keep_prob,weight_decay=args.weight_decay)

#define activation of last layer as score




'''
main function
'''

def main():

    # specific parse arguments
    args=parse_args()

    if args is None:
        exit()
    print('\n The type of args is:',type(args))
    print('\n This is the existing arguments:', args)
    print('\n ')
    model = resnet(args, x=x_image, n=20, num_classes=10)
    score = model.out
    # Loss Function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=score))

    # Optimizer
    """def optimizer_choose(args.optimizer):

        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train = optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()

    # To measure accuracy
    correct_prediction = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)

        for j in range(args.max_epochs):
            for i in range(0, 60000, batch_size):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
            # Try to add training accuracy here
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[: 5000],
                                                     y_true: mnist.test.labels[: 5000]})
            print('After %d epochs, accuracy: %g' % (j + 1, test_accuracy))

if __name__=='__main__':
    main()