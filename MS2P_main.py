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
from MS2Pnet import pP_ED
import cv2
from scipy.misc import imsave, imread
pan_path = 'PAN.h5'
gt_path = 'GT.h5'

EPOCHES = 8
BATCH_SIZE = 8
patch_size = 264
logging_period = 10
LEARNING_RATE = 0.002
DECAY_RATE = 0.8

def main():
	with tf.device('/cpu:0'):
		source_pan_data = h5py.File(pan_path, 'r')
		source_pan_data = source_pan_data['data'][:]
		source_pan_data = np.transpose(source_pan_data, (0, 3, 2, 1)) / 255.0
		print("source_pan_data shape:", source_pan_data.shape)

		gt_data = h5py.File(gt_path, 'r')
		gt_data = gt_data['data'][:]
		gt_data = np.transpose(gt_data, (0, 3, 2, 1)) / 255.0
		print("gt_data shape:", gt_data.shape)

		data = np.concatenate([gt_data, source_pan_data], axis = -1)
		print("data shape:", data.shape)
		del source_pan_data, gt_data

		start_time = datetime.now()
		print('Epoches: %d, Batch_size: %d' % (EPOCHES, BATCH_SIZE))

		num_imgs = data.shape[0]
		mod = num_imgs % BATCH_SIZE
		n_batches = int(num_imgs // BATCH_SIZE)
		print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
		if mod > 0:
			print('Train set has been trimmed %d samples...\n' % mod)
			source_imgs = data[:-mod]

		# create the graph
		with tf.Graph().as_default(), tf.Session() as sess:
			MS = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 4), name = 'MS')
			PAN = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'PAN')

			pP_NET = pP_ED('pP_ED')
			MS_converted_PAN = pP_NET.transform(I = MS, is_training = True, reuse = False)
			print("MS_converted_PAN shape:", MS_converted_PAN.shape)

			LOSS = 40 * tf.reduce_mean(tf.square(PAN - MS_converted_PAN)) + (1 - tf.reduce_mean(SSIM_LOSS(PAN, MS_converted_PAN), axis=0))
			current_iter = tf.Variable(0)
			theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pP_ED')
			for i in theta:
				print(i)

			learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
			                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
			                                           staircase = False)

			solver = tf.train.AdamOptimizer(learning_rate).minimize(LOSS, global_step = current_iter, var_list = theta)

			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(max_to_keep = 5)
			tf.summary.scalar('Loss', LOSS)
			tf.summary.scalar('Learning rate', learning_rate)
			tf.summary.image('PAN', PAN, max_outputs = 4)
			tf.summary.image('MS', MS, max_outputs = 4)
			tf.summary.image('converted_PAN', MS_converted_PAN, max_outputs = 4)

			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter("MS2P_logs/", sess.graph)

			# ** Start Training **
			step = 0
			for epoch in range(EPOCHES):
				np.random.shuffle(source_imgs)
				for batch in range(n_batches):
					step += 1
					current_iter = step
					pan_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 4]
					ms_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0:4]
					pan_batch = np.expand_dims(pan_batch, -1)
					FEED_DICT = {MS: ms_batch, PAN: pan_batch}

					# run the training step
					sess.run(solver, feed_dict = FEED_DICT)
					result = sess.run(merged, feed_dict = FEED_DICT)
					writer.add_summary(result, step)

					if step % 100 == 0:
						saver.save(sess, 'MS2P_models/' + str(step) + '/' + str(step) + '.ckpt')

					is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
					if is_last_step or step % logging_period == 0:
						elapsed_time = datetime.now() - start_time
						loss, pPAN_max, pPAN_min = sess.run([LOSS, tf.reduce_max(MS_converted_PAN), tf.reduce_min(MS_converted_PAN)], feed_dict = FEED_DICT)
						lr = sess.run(learning_rate)
						print('Epoch:%d/%d: step:%d, lr:%s, loss:%s, pPAN_max:%s, pPAN_min:%s, elapsed_time:%s\n' % (
							epoch + 1, EPOCHES, step, lr, loss, pPAN_max, pPAN_min, elapsed_time))
		writer.close()




def test():
	file_name1 = './test_imgs/pan/17.png'
	file_name2 = './test_imgs/ms/17.tif'

	pan = imread(file_name1) / 255.0
	ms = imread(file_name2) / 255.0
	print('file1:', file_name1, 'shape:', pan.shape)
	print('file2:', file_name2, 'shape:', ms.shape)

	h1, w1 = pan.shape
	pan = pan.reshape([1, h1, w1, 1])
	h2, w2, c = ms.shape

	ms_us = np.zeros([h1, w1, 4], dtype=np.float32)
	for c in range(4):
		ms_us[:, :, c] = cv2.resize(ms[ :, :, c], (h1, w1))
	ms_us = ms_us.reshape([1, h1, w1, 4])

	with tf.Graph().as_default(), tf.Session() as sess:
		MS = tf.placeholder(tf.float32, shape = (1, h1, w1, 4), name = 'MS')
		pP_NET = pP_ED('pP_ED')
		MS_converted_PAN = pP_NET.transform(I = MS, is_training = False, reuse = False)
		print("MS_converted_PAN shape:", MS_converted_PAN.shape)

		t_list = tf.trainable_variables()
		saver = tf.train.Saver(var_list = t_list)
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, './MS2P_models/2000/2000.ckpt')

		output = sess.run(MS_converted_PAN, feed_dict = {MS: ms_us})
		imsave('17.png', output[0, :, :, 0])




def SSIM_LOSS(img1, img2, size = 11, sigma = 1.5):
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



if __name__ == '__main__':
	## for train
	main()
	## for test
	# test()