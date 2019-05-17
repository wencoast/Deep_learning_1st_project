"""

This is a TensorFlow implementation of ResNet for Fashion_mnist.

The architecture is based on the ResNet-n architecture
(where n = 20, 32, 44, 56) described in the paper to
perform analysis and tests on mnist_fashion dataset.

Paper: Deep Residual Learning for Image Recognition
(https://arxiv.org/abs/1512.03385)

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import conv_layer, max_pool, fc_layer, global_average
from residual import residual_block

class resnet(object):

	""" Implementation of ResNet Architecture """

	def __init__(self, args, x, n, num_classes):

		""" ResNet-n architecture
		{20:3, 32:5, 44:7, 56:9}
		"""

		if((n < 20) or ((n - 20) % 6 != 0)):
			print("ResNet DEPTH INVALID!\n")
			return

		self.NUM_CONV = int(((n - 20) / 6) + 3) # For n = 20, each block will have 3 residual units.
		self.X = x
		self.NUM_CLASSES = num_classes

		self.dropout_keep_prob=args.dropout_keep_prob # Keep consistent
		self.is_batch_normalization=args.is_batch_normalization
		print('\n The type of self.is_batch_normalization is',type(self.is_batch_normalization))
		self.activation_function=args.activation_function
		print('\n The type of self.activation_function is', type(self.activation_function))
		print('\n The value of self.activation_function is',self.activation_function)
		#self.checkpoint_dir = args.checkpoint_dir
		#self.log_dir = args.log_dir

		#self.epoch_num= args.epoch
		#self.batch_size=args.batch_size

		# Store the last layer of the network graph.
		self.out = None
		self.create(self.dropout_keep_prob,is_training=True)

	def create(self,dropout_keep_prob,is_training=True):
		# 1 layer
		conv1 = conv_layer(self.X, 3, 3, 16, name = 'conv1',activation_function=self.activation_function,is_batch_normalization=self.is_batch_normalization)
		self.out = conv1

		""" All residual blocks use zer-padding
		for shortcut connections """
		# i=0,1,2 every block has 2 conv layer.
		# one for loop has 6 conv layer.
		for i in range(self.NUM_CONV):
			resBlock2 = residual_block(self.out, 16, name = 'resBlock2_{}'.format(i + 1), block_activation_function=self.activation_function,block_is_batch_normalization=self.is_batch_normalization)
			self.out = resBlock2
		# 1 max_pool layer
		pool2 = max_pool(self.out, name = 'pool2')
		self.out = pool2
		# i=0,1,2 every block has 2 conv layer.
		# one for loop has 6 conv layer.
		for i in range(self.NUM_CONV):
			resBlock3 = residual_block(self.out, 32, name = 'resBlock3_{}'.format(i + 1),block_activation_function=self.activation_function,block_is_batch_normalization=self.is_batch_normalization)
			self.out = resBlock3
		# 1 max_pool layer
		pool3 = max_pool(self.out, name = 'pool3')
		self.out = pool3
		# i=0,1,2 every block has 2 conv layer.
		# one for loop has 6 conv layer.
		for i in range(self.NUM_CONV):
			resBlock4 = residual_block(self.out, 64, name = 'resBlock4_{}'.format(i + 1),block_activation_function=self.activation_function,block_is_batch_normalization=self.is_batch_normalization)
			self.out = resBlock4
		# 1 global pool layer
		# Perform global average pooling to make spatial dimensions as 1x1
		global_pool = global_average(self.out, name = 'gap')
		self.out = global_pool
		# flatten is not layer
		flatten = tf.contrib.layers.flatten(self.out)
		# 1 fully connected layer.
		# @Hazard
		# dropout_keep_prob: float, the fraction to keep before final layer.
		dpot_net = slim.dropout(flatten,dropout_keep_prob,is_training=is_training,scope='Dropout')
		fc5 = fc_layer(dpot_net, input_size = 64, output_size = self.NUM_CLASSES,relu = False, name = 'fc5')
		self.out = fc5



