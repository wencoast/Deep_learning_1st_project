from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow import keras
import tensorflow as tf
import numpy as np
from data_augmentation import *
from resnet import resnet
import matplotlib.pyplot as plt

import argparse

"""parsing and configuration"""''
'''Use arguments to set up parameter accordingly'''
def parse_args():

    desc="TensorFlow implementation of Resnet on Fashion_mnist! author: Hazard Wen"
    parser=argparse.ArgumentParser(description=desc)
    # all the outside arguments can be passed successfully.
    # Bool data type can be input by keyboard.
    # add dropout or not. Dropout, probability to keep units
    parser.add_argument('--dropout_keep_prob', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.1)
    parser.add_argument('--max_epochs', type=int, help='Number of epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--activation_function', type=str, choices=['elu', 'relu', 'relu6', 'leaky_relu', 'sigmoid'],
                        help='which activation function we are going to use', default='None')
    # add argument for data_augmention with bool data type
    parser.add_argument('--data_augmention', type=str, choices=['True','False'], help='excute data augmention or not',default='True')
    #******************************************************************************************************************#
    # add arguments for different optimizer
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='None')
    # add argument for batch_normalization or not with bool data type
    parser.add_argument('--is_batch_normalization',type=str,choices=['True','False'], help='excute batch_normalization or not',default='True')

    return check_args(parser.parse_args())

def check_args(args):

    return args

'''
main function
'''

def main():

    print('The version of tensorflow:', tf.__version__)
    print('The version of keras:', keras.__version__)

    # specific parse arguments
    args=parse_args()

    if args is None:
        exit()

    print('\n The type of args is:',type(args))
    print('\n This is the existing arguments:', args)
    print('\n ')

    # access the Fashion MNIST directly using TensorFlow method
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('data/fashion', one_hot=True)

    # Creating placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant.
    # In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Create the ResNet model
    model = resnet(args, x=x_image, n=20, num_classes=10) # we are going to use 20 layers.
    score = model.out

    # Loss Function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=score))

    # different Optimizer
    def optimizer_choose(optimizer,learning_rate):

        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
            return opt
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
            return opt
        elif optimizer == 'ADAM':
            #opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
            opt = tf.train.AdamOptimizer(learning_rate)
            return opt
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
            return opt
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            return opt
        else:
            raise ValueError('Invalid optimization algorithm')

    choosed_optimizer= optimizer_choose(args.optimizer,args.learning_rate)

    train = choosed_optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()

    # To measure accuracy
    correct_prediction = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)

        for j in range(args.max_epochs):
            for i in range(0, 60000, args.batch_size):
                # mnist.train.next_batch returns a tuple of two arrays
                batch_x, batch_y = mnist.train.next_batch(args.batch_size)
                batch_x=data_augmentation(batch_x,28,is_data_augmention=args.data_augmention)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
            # Try to add training accuracy here
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[: 5000],
                                                     y_true: mnist.test.labels[: 5000]})
            print('After %d epochs, accuracy: %g' % (j + 1, test_accuracy))

if __name__=='__main__':
    main()