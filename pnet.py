import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np


WEIGHT_INIT_STDDEV = 0.05

n = 44

class PNet(object):
	def __init__(self, sco):
		self.p_pan = P_pan(sco)
		self.p_ms = P_ms(sco)
		self.p_fuse = P_fuse(sco)
		self.features = []

	def transform(self, PAN, ms):
		# f_pan = self.p_pan.trans(PAN)
		# f_ms = self.p_ms.trans(ms)
		# f = tf.concat([f_pan, f_ms], 3)
		MS = up_sample(up_sample(ms))
		f = tf.concat([MS, PAN], axis=-1)
		generated_img = self.p_fuse.trans(f)
		# self.var_list.extend(self.encoder.var_list)
		# self.var_list.extend(self.decoder.var_list)
		# self.var_list.extend(tf.trainable_variables())
		return generated_img



class P_pan(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []

		with tf.variable_scope(self.scope):
			with tf.variable_scope('p_pan'):
				self.weight_vars.append(self._create_variables(1, 18, 3, scope = 'conv1'))
				self.weight_vars.append(self._create_variables(18, 32, 3, scope = 'conv2'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def trans(self, image):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, Scope = self.scope + '/p_pan/b' + str(i))
		return out



class P_ms(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []

		with tf.variable_scope(self.scope):
			with tf.variable_scope('p_ms'):
				self.weight_vars.append(self._create_variables(4, 18, 3, scope = 'conv1'))
				self.weight_vars.append(self._create_variables(18, 48, 3, scope = 'conv2'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def trans(self, image):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = up_sample(out)
			out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, Scope = self.scope + '/p_ms/b' + str(i))
		return out




class P_fuse(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []

		with tf.variable_scope(self.scope):
			with tf.variable_scope('p_fuse'):
				self.weight_vars.append(self._create_variables(5, 48, 3, scope = 'conv1'))
				self.weight_vars.append(self._create_variables(48, 48, 3, scope = 'conv2'))
				self.weight_vars.append(self._create_variables(96, 48, 3, scope = 'conv3'))
				self.weight_vars.append(self._create_variables(144, 48, 3, scope = 'conv4'))
				self.weight_vars.append(self._create_variables(192, 48, 3, scope = 'conv5'))
				# self.weight_vars.append(self._create_variables(230, 36, 5, scope = 'conv6'))
				# self.weight_vars.append(self._create_variables(240, 48, 3, scope = 'dense_block_conv6'))
				self.weight_vars.append(self._create_variables(240, 128, 3, scope = 'conv6'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv7'))
				self.weight_vars.append(self._create_variables(64, 4, 3, scope = 'conv8'))
				# self.weight_vars.append(self._create_variables(16, 4, 3, scope = 'conv9'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def trans(self, image):
		dense_indices = [1,2, 3, 4]
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense = True, use_lrelu = True, Scope = self.scope + '/fuse/b' + str(i))
			elif i == len(self.weight_vars):
					out = conv2d(out, kernel, bias, dense = False, use_lrelu = False, Scope = self.scope + '/fuse/b' + str(i))
					out = tf.nn.tanh(out) / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, Scope = self.scope + '/fuse/b' + str(i))
		return out



def conv2d(x, kernel, bias, use_lrelu = True, dense = False, Scope = None, stride = 1):
	# padding image with reflection mode
	ks, ks, _, _ = kernel.get_shape().as_list()
	if ks == 5:
		x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		x_padded = tf.pad(x_padded, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	else:
		x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(input = x_padded, filter = kernel, strides = [1, stride, stride, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	# if BN:
	# 	with tf.variable_scope(Scope):
	# 		# print("Scope", Scope)
	# 		# print("reuse", not is_training)
	# 		# out = tf.contrib.layers.batch_norm(out, decay = 0.9, updates_collections = None, epsilon = 1e-5, scale = True, reuse = reuse)
	#
	# 		out = tf.layers.batch_normalization(out, training = is_training, reuse= reuse, trainable=is_training)
	if use_lrelu:
		# out = tf.nn.relu(out)
		out = tf.maximum(out, 0.2 * out)
	if dense:
		out = tf.concat([out, x], 3)
	return out


def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)