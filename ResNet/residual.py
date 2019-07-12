import tensorflow as tf

from layers import conv_layer

def residual_block(x, out_channels, projection = False, name = 'residual', block_activation_function='relu', block_is_batch_normalization='True'):

	"""Create a Residual Block consisting of 3 Residual Unit with two conv layers"""

	# Get the input channels
	input_channels = int(x.get_shape()[-1])

	conv1 = conv_layer(x, 3, 3, out_channels, stride = 1, name = '{}_conv1'.format(name),activation_function=block_activation_function,is_batch_normalization=block_is_batch_normalization)
	conv2 = conv_layer(conv1, 3, 3, out_channels, stride = 1, name = '{}_conv2'.format(name),activation_function=block_activation_function,is_batch_normalization=block_is_batch_normalization)

	# What type of shortcut connection to use
	if input_channels != out_channels:
		if projection:
			# Option B: Projection Shortcut
			# This introduces extra parameters.
			shortcut = conv_layer(x, 1, 1, out_channels, stride = 1, name = '{}_shortcut'.format(name),activation_function=block_activation_function,is_batch_normalization=block_is_batch_normalization)
		else:
			# Option A: Identity mapping with Zero-Padding
			# This method doesn't introduce any extra parameters.
			shortcut = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, out_channels - input_channels]])
	else:
		# Identity mapping.
		shortcut = x


	# Element wise addition.
	out = conv2 + shortcut

	return out
