from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
import tensorflow as tf
import numpy as np
from ops import *
from resnet import resnet

# The later one could be function and class type.

import argparse

"""parsing and configuration"""''
'''Use arguments to set up parameter accordingly'''

# def parse_args():
def parse_flags():
    
    desc="TensorFlow implementation of Resnet on Fashion_mnist! author: Hazard Wen"
    parser=argparse.ArgumentParser(description=desc)
    # How to confirm whether this desc works well or not.
    
    # all the outside arguments can be passed successfully.
    # Bool data type can not be input by keyboard.
    
    '''

	In Deep_learning_prac2, we define one function to convert keyboard input into boolean.

	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

	'''
    
    '''
	
	check if is newest!!!
	
	'''
    # add dropout or not. Dropout, probability to keep units
    parser.add_argument('--dropout_keep_prob', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.1)
    parser.add_argument('--max_epochs', type=int, help='Number of epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--activation_function', type=str, choices=['elu', 'relu', 'relu6', 'leaky_relu', 'sigmoid', 'tanh', 'linear'],
                        help='which activation function we are going to use', default='None')
    # add argument for data_augmentation with bool data type
    parser.add_argument('--data_augmentation', type=str, choices=['True','False'], help='excute data augmentation or not', default='False')
    #******************************************************************************************************************#
    # Be careful on spelling when you type flag outside, otherwise it will occur one error.
    # error: unrecognized arguments: --data_augmention True
    # add arguments for different optimizer
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='None')
    # add argument for batch_normalization or not with bool data type
    parser.add_argument('--is_batch_normalization',type=str,choices=['True','False'], help='excute batch_normalization or not',default='False')
    
    # The following one without check_args is also correct.
    # return parser.parse_args()
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
    #args=parse_args()
    #args=parser.parse_args()
    args=parse_flags()
    if args is None:
        exit()

    print('\n The type of args is:',type(args))
    print('\n This is the existing arguments:', args)
    print('\n ')

    # access the Fashion MNIST directly using TensorFlow method
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('data/fashion', one_hot=True)

    # Creating placeholders
    # Explore the format of the dataset before training the model.
    # The following shows there are 60,000 images in the training set,
    # with each image represented as 28 x 28 (784) pixels
    # train_images.shape = (60000, 28, 28)
    # Likewise, there are 60,000 labels in the training set
    # len(train_labels) = 60000
    x = tf.placeholder(tf.float32, shape=[None, 784])
    #
    print('\n The shape of x placeholder is', tf.shape(x))
    array_1 = np.array([1,2])
    print('\n The shape of array_1 is', tf.shape(array_1))
    print('\n like x=np.array([1,2]), Here is tensor-like [None,784]')
    # They seems that they are different.
    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant.
    # In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    x_image = tf.identity(x_image,'input')
    print('\n The shape of x_image is', tf.shape(x_image ))
    print('\n like x=np.array([none,1,2,3]), Here is tensor-like [None,28,28,1]')
    # Create the ResNet model
    '''x=x_image means that [none,28,28,1]'''
    # The instance name should be model.
    # The class name shoulde be Resnet, uppercase should be better.
    model = resnet(args, x=x_image, n=20, num_classes=10) # we are going to use 20 layers.
    score = model.out

    # using pre-defined function to get training loss and training accuracy
    training_loss,accuracy=classification_loss(score,y_true)

    # define reg loss
    # reg_loss = tf.losses.get_regularization_loss()
    # training_loss += reg_loss

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

    choosed_optimizer_opt= optimizer_choose(args.optimizer,args.learning_rate)

    train = choosed_optimizer_opt.minimize(training_loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for j in range(args.max_epochs):
            for i in range(0, 60000, args.batch_size):
                # mnist.train.next_batch returns a tuple of two arrays
                batch_x, batch_y = mnist.train.next_batch(args.batch_size)
                #print('The tf.shape of batch_x is', tf.shape(batch_x))
                #print('The np.shape of batch_x is', np.shape(batch_x))
                #print('The type of batch_x is', type(batch_x))
                '''batch_x looks like ([64,784])'''
                batch_x=data_augmentation(batch_x, 28, is_data_augmentation=args.data_augmentation)
                # perform minimize loss per batch
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
            # Try to add training loss
            epoch_train_loss=sess.run(training_loss,feed_dict={x: batch_x, y_true: batch_y})
            # Try to add training accuracy here per epoch
            training_accuracy=sess.run(accuracy,feed_dict={x: batch_x, y_true: batch_y})

            # change test accuracy using sess.run instead of eval
            #test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[: 5000],
            #                                         y_true: mnist.test.labels[: 5000]})
            testing_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[: 5000],
                                                             y_true: mnist.test.labels[: 5000]})
            print('After %d epochs, testing_accuracy: %g, training_accuracy: %g, epoch_train_loss: %g, %d epochs' % (j + 1, testing_accuracy, training_accuracy,epoch_train_loss,j+1))

if __name__=='__main__':
    main()
