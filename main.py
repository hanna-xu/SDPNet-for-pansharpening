from __future__ import print_function
import time
import os
import h5py
import numpy as np
import scipy.ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import cv2
from pnet_tradition import PNet
from spat_ED import ED1
from spec_ED import ED2

from P2MSnet import pMS_ED
from MS2Pnet import pP_ED

# from tensorflow.python import pywrap_tensorflow

pan_path = 'PAN.h5'
gt_path = 'GT.h5'

EPOCHES = 10
BATCH_SIZE = 4
patch_size = 264
logging_period = 10
LEARNING_RATE = 0.002
DECAY_RATE = 0.8
ratio = 20

# W0 = [0.35, 0.05, 0.35, 0.25]

# W0 = [0.3, 0.1, 0.3, 0.3]

MODEL1_SAVE_PATH = './spat_models/2000/2000.ckpt'
MODEL2_SAVE_PATH = './spec_models/2000/2000.ckpt'
MS2P_MODEL_SAVEPATH = './MS2P_models/2000/2000.ckpt'
P2MS_MODEL_SAVEPATH = './P2MS_models/2000/2000.ckpt'


SPAT_INDEX = np.loadtxt("spat_diff.txt", dtype = np.int32)
# print("SPAT_INDEX:", SPAT_INDEX)
SPEC_INDEX = np.loadtxt("spec_diff.txt", dtype = np.int32)
# print("SPEC_INDEX:", SPEC_INDEX)

FEA_NUM = 20

def main():
	with tf.device('/cpu:0'):
		pan_data = h5py.File(pan_path, 'r')
		pan_data = pan_data['data'][:]
		pan_data = np.transpose(pan_data, (0, 3, 2, 1)) / 255.0

		gt_data = h5py.File(gt_path, 'r')
		gt_data = gt_data['data'][:]
		gt_data = np.transpose(gt_data, (0, 3, 2, 1)) / 255.0

		data = np.concatenate([gt_data, pan_data], axis = -1)
		print("data shape:", data.shape)
		del pan_data, gt_data

	start_time = datetime.now()
	print('Epoches: %d, Batch_size: %d' % (EPOCHES, BATCH_SIZE))

	num_imgs = data.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = data[:-mod]
	else:
		source_imgs = data

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		PAN = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'PAN')
		ms = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size / 4, patch_size / 4, 4), name = 'ms')
		GT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 4), name = 'GT')
		with tf.device('/gpu:0'):
			Pnet = PNet('pnet')
			X = Pnet.transform(PAN = PAN, ms = ms)
		theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pnet')


		with tf.device('/gpu:0'):
			X2Pnet = pP_ED('pP_ED')
			X_c_PAN = X2Pnet.transform(I = X, is_training = False, reuse = True)
		X2P_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pP_ED')


		"feature loss"
		"spatial"
		with tf.device('/gpu:1'):
			NET1 = ED1('spatial_ED')
			_ = NET1.transform(PAN, is_training = False, reuse = False)
			SPAF1= NET1.features
			_ = NET1.transform(X_c_PAN, is_training = False, reuse = True)
			SPAF2= NET1.features
		t_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')

		with tf.device('/cpu:0'):
			for num in range(FEA_NUM):
				index = SPAT_INDEX[num+10]
				print(index)
				if num == 0:
					LOSS_spat_fea = tf.reduce_mean(tf.abs(SPAF1[:, :, :, index] - SPAF2[:, :, :, index]))
				else:
					LOSS_spat_fea = LOSS_spat_fea + tf.reduce_mean(
						tf.square(SPAF1[:, :, :, index] - SPAF2[:, :, :, index]))
			LOSS_spat_fea = LOSS_spat_fea / FEA_NUM

		"spectral"
		with tf.device('/gpu:1'):
			NET2 = ED2('spectral_ED')
			_ = NET2.transform(GT, is_training = False, reuse = False)
			SPEF1 = NET2.features
			_ = NET2.transform(X, is_training = False, reuse = True)
			SPEF2 = NET2.features
		t_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spectral_ED')

		with tf.device('/cpu:0'):
			for num in range(FEA_NUM):
				index = SPEC_INDEX[num+10]
				print(index)
				if num == 0:
					LOSS_spec_fea = tf.reduce_mean(tf.square(SPEF1[:, :, :, index] - SPEF2[:, :, :, index]))
				else:
					LOSS_spec_fea = LOSS_spec_fea + tf.reduce_mean(tf.abs(SPEF1[:, :, :, index] - SPEF2[:, :, :, index]))
			LOSS_spec_fea = LOSS_spec_fea / FEA_NUM


		alpha = 25
		LOSS0 = 1 - SSIM(tf.expand_dims(X[:, :, :, 0], axis = -1),
		                 tf.expand_dims(GT[:, :, :, 0], axis = -1)) + alpha * tf.reduce_mean(
			tf.square(X[:, :, :, 0] - GT[:, :, :, 0]), axis = (1, 2))
		LOSS1 = 1 - SSIM(tf.expand_dims(X[:, :, :, 1], axis = -1),
		                 tf.expand_dims(GT[:, :, :, 1], axis = -1)) + alpha * tf.reduce_mean(
			tf.square(X[:, :, :, 1] - GT[:, :, :, 1]), axis = (1, 2))
		LOSS2 = 1 - SSIM(tf.expand_dims(X[:, :, :, 2], axis = -1),
		                 tf.expand_dims(GT[:, :, :, 2], axis = -1)) + alpha * tf.reduce_mean(
			tf.square(X[:, :, :, 2] - GT[:, :, :, 2]), axis = (1, 2))
		LOSS3 = 1 - SSIM(tf.expand_dims(X[:, :, :, 3], axis = -1),
		                 tf.expand_dims(GT[:, :, :, 3], axis = -1)) + alpha * tf.reduce_mean(
			tf.square(X[:, :, :, 3] - GT[:, :, :, 3]), axis = (1, 2))

		LOSS_X_GT = tf.reduce_mean((LOSS0 + LOSS1 + LOSS2 + LOSS3) / 4, axis = 0)
		LOSS_fea = (LOSS_spat_fea + 0.5 * LOSS_spec_fea) # 800 * (LOSS_spat_fea + 0.2 * LOSS_spec_fea)
		LOSS = LOSS_X_GT + 20 * LOSS_fea

		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)
		solver = tf.train.AdamOptimizer(learning_rate).minimize(LOSS, global_step = current_iter,
		                                                           var_list = theta)


		sess.run(tf.global_variables_initializer())
		saver0 = tf.train.Saver(var_list = X2P_list)
		saver1 = tf.train.Saver(var_list = t_list)
		saver2 = tf.train.Saver(var_list = t_list2)
		saver0.restore(sess, MS2P_MODEL_SAVEPATH)
		saver1.restore(sess, MODEL1_SAVE_PATH)
		saver2.restore(sess, MODEL2_SAVE_PATH)
		saver = tf.train.Saver(max_to_keep = 5)

		tf.summary.scalar('Loss_fea', LOSS_fea)
		tf.summary.scalar('Loss_x_GT', LOSS_X_GT)
		tf.summary.scalar('Loss', LOSS)
		tf.summary.image('PAN', PAN, max_outputs = 2)
		tf.summary.image('ms', ms[:, :, :, 0:3], max_outputs = 1)
		tf.summary.image('X', X[:, :, :, 0:3], max_outputs = 1)
		tf.summary.image('X_c_PAN', X_c_PAN, max_outputs = 1)
		tf.summary.image('GT', GT[:, :, :, 0:3], max_outputs = 1)

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0
		for epoch in range(EPOCHES):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step
				ms_batch = np.zeros(shape=(BATCH_SIZE, int(patch_size/4), int(patch_size/4), 4), dtype=np.float32)
				GT_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0:4]
				PAN_batch = np.expand_dims(data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 4],
				                           axis = -1)
				for b in range(BATCH_SIZE):
					for c in range(4):
						ms_batch[b, :, :, c] = cv2.resize(GT_batch[b, :, :, c],
						                                  (int(patch_size / 4), int(patch_size / 4)))

				# run the training step
				FEED_DICT = {PAN: PAN_batch, ms: ms_batch, GT: GT_batch}
				sess.run(solver, feed_dict = FEED_DICT)

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)
				if step % 100 == 0:
					saver.save(sess, 'models/' + str(step) + '/' + str(step) + '.ckpt')

				is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					elapsed_time = datetime.now() - start_time
					loss = sess.run(LOSS_X_GT, feed_dict = FEED_DICT)
					lr = sess.run(learning_rate)
					print('Epoch: %d/%d, Step: %d/%d, Loss: %s, Lr: %s, Time: %s\n' % (
						epoch + 1, EPOCHES, step % n_batches, n_batches, loss, lr, elapsed_time))

			saver.save(sess, 'models/' + str(step) + '/' + str(step) + '.ckpt')



def SSIM(img1, img2, size = 11, sigma = 1.5):
	window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
	k1 = 0.01
	k2 = 0.03
	L = 1  # depth of image (255 in case the image has a different scale)
	c1 = (k1 * L) ** 2
	c2 = (k2 * L) ** 2
	mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
	sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

	# value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
	value = tf.reduce_mean(ssim_map, axis = [1, 2, 3])
	return value


def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)

	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)

	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)

	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)


def smooth(I):
	kernel = tf.constant([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	B, H, W, C= I.get_shape().as_list()
	for c in range(C):
		if c == 0:
			img = tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                   padding = 'SAME')
		else:
			img = tf.concat([img, tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                                   padding = 'SAME')], axis = -1)
	return img


def grad(I):
	kernel = tf.constant([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	B, H, W, C= I.get_shape().as_list()
	for c in range(C):
		if c == 0:
			grad = tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                   padding = 'SAME')
		else:
			grad = tf.concat([grad, tf.nn.conv2d(tf.expand_dims(I[:, :, :, c], axis = -1), kernel, strides = [1, 1, 1, 1],
			                                   padding = 'SAME')], axis = -1)
	return grad


def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)


# def ERGAS(x, y):
# 	C1 = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, :, 0] - y[:, :, :, 0]), axis = [1, 2]))
# 	C2 = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, :, 1] - y[:, :, :, 1]), axis = [1, 2]))
# 	C3 = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, :, 2] - y[:, :, :, 2]), axis = [1, 2]))
# 	C4 = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, :, 3] - y[:, :, :, 3]), axis = [1, 2]))
# 	S1 = tf.square(C1 / tf.reduce_mean(x[:, :, :, 0], axis = [1, 2]))
# 	S2 = tf.square(C2 / tf.reduce_mean(x[:, :, :, 1], axis = [1, 2]))
# 	S3 = tf.square(C3 / tf.reduce_mean(x[:, :, :, 2], axis = [1, 2]))
# 	S4 = tf.square(C4 / tf.reduce_mean(x[:, :, :, 3], axis = [1, 2]))
# 	S = S1 + S2 + S3 + S4
# 	return tf.sqrt(S / 4) * 25

if __name__ == '__main__':
	main()
