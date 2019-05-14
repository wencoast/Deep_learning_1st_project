import tensorflow as tf

def conv_layer(x, filter_height, filter_width,
	num_filters, name, stride = 1, padding = 'SAME',activation_function='relu'):

	"""Create a convolution layer."""
	
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])

	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases of the conv layer
		W = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [num_filters], initializer = tf.constant_initializer(0.0))

		# Perform convolution.
		conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
		# Add the biases.
		z = tf.nn.bias_add(conv, b)

		# Perform batch normalization
		batch_norm = tf.layers.batch_normalization(z, axis = 1, beta_initializer = tf.constant_initializer(0.0),
			gamma_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		if activation_function=='relu':
			print('which kind of activation function it is', activation_function)
			relu_out = tf.nn.relu(batch_norm)
			return relu_out
		elif activation_function=='relu6':
			print('which kind of activation function it is', activation_function)
			# Computes Rectified Linear 6: min(max(input, 0), 6)
			relu6_out = tf.nn.relu6(batch_norm,name=None)
			return relu6_out
		elif activation_function=='leaky_relu':
			print('which kind of activation function it is', activation_function)
			# Apply Leaky ReLU activation function.
			# alpha represents the slope of the activation function at x<0.
			# Returns the activation value.
			leaky_relu_out = tf.nn.leaky_relu(batch_norm, alpha=0.2, name=None)
			return leaky_relu_out
		elif activation_function=='sigmoid':
			print('which kind of activation function it is', activation_function)
			# computes sigmoid of x element-wise
			# specially, y = 1/(1+exp(-x))
			sigmoid_activation_out= tf.nn.sigmoid(batch_norm,name=None)
			return sigmoid_activation_out
		else:
			print('\n Keep linear output without using non-linear activation function')
			return batch_norm
		#elif activation_function==''
		# if args.batch_normalization = True:

		# batch_norm = tf.layers.batch_normalization(z, axis = 1, beta_initializer = tf.constant_initializer(0.0),
		#	gamma_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		# Apply ReLu non linearity.

		# print('This is already excuted batch_normalization.')

		'''
		else:

		print('This is a network model without batch_normlization.')

		if relu:
			# Apply ReLu non linearity.
			a = tf.nn.relu(z)
			return a
		elif elu:
			# input: must be a tensor.  Must be one of the following types: half, bfloat16, float32, float64.
			# Computes exponential linear: exp(z)-1 if i<0, z otherwise
			# See Fast and accurate Deep Network Learning by Exponential linear Units(ELUs)
			c = tf.nn.elu(z)
			return c
		elif crelu:
			# Computes Concatenate ReLU
			# Concatenates a ReLU which selects only the positive part of the activation with
			# a ReLU which selects only the negative part of the activation.
			# Note that as a result this non-linearity doubles the depth of the activations.
			# axis The axis that the output values are concatenated along. Default is -1.
			d= tf.nn.crelu(z,axis=-1,name=None)
			return d
		
		elif selu:
			# Computes scaled exponential linear: scale * alpha * (exp(features) - 1)
			# if < 0, scale * features otherwise.
			# To be used together with initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN').
			# For correct dropout, use tf.contrib.nn.alpha_dropout.
			f= tf.nn.selu(z,name=None)
			return f
		else:
			return z
		'''



def max_pool(x, name, filter_height = 2, filter_width = 2,
	stride = 2, padding = 'VALID'):

	"""Create a max pooling layer."""

	return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
		strides = [1, stride, stride, 1], padding = padding, name = name)

def global_average(x, name):

	"""Create a global average pooling layer"""

	filter_hw = int(x.get_shape()[1])
	gap = tf.nn.avg_pool(x, ksize = [1, filter_hw, filter_hw, 1],
		strides = [1, 1, 1, 1], padding = 'VALID', name = name)

	return gap

def fc_layer(x, input_size, output_size, name, relu = True):

	"""Create a fully connected layer."""
	
	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases.
		W = tf.get_variable('weights', shape = [input_size, output_size],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))

		# Matrix multiply weights and inputs and add biases.
		z = tf.nn.bias_add(tf.matmul(x, W), b)

		if relu:
			# Apply ReLu non linearity.
			a = tf.nn.relu(z)
			return a
		else:
			return z
		'''
		elif leaky_relu:
			# Apply Leaky ReLU activation function.
			# alpha represents the slope of the activation function at x<0.
			# Returns the activation value.
			b = tf.nn.leaky_relu(z,alpha=0.2,name=None)
			return b
		elif elu:
			# input: must be a tensor.  Must be one of the following types: half, bfloat16, float32, float64.
			# Computes exponential linear: exp(z)-1 if i<0, z otherwise
			# See Fast and accurate Deep Network Learning by Exponential linear Units(ELUs)
			c = tf.nn.elu(z)
			return c
		elif crelu:
			# Computes Concatenate ReLU
			# Concatenates a ReLU which selects only the positive part of the activation with
			# a ReLU which selects only the negative part of the activation.
			# Note that as a result this non-linearity doubles the depth of the activations.
			# axis The axis that the output values are concatenated along. Default is -1.
			d= tf.nn.crelu(z,axis=-1,name=None)
			return d
		elif relu6:
			# Computes Rectified Linear 6: min(max(z, 0), 6)
			e= tf.nn.relu6(z,name=None)
			return e
		elif selu:
			# Computes scaled exponential linear: scale * alpha * (exp(features) - 1)
			# if < 0, scale * features otherwise.
			# To be used together with initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN').
			# For correct dropout, use tf.contrib.nn.alpha_dropout.
			f= tf.nn.selu(z,name=None)
			return f
		'''