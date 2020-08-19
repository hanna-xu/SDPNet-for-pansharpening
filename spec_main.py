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
from spec_ED import ED2
from P2MSnet import pMS_ED

pan_path = 'PAN.h5'
gt_path = 'GT.h5'

EPOCHES = 10
BATCH_SIZE = 10
patch_size = 264
logging_period = 10
LEARNING_RATE = 0.002
DECAY_RATE = 0.85

P2MS_MODEL_PATH = './P2MS_models/2000/2000.ckpt'

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
			PAN = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size/4, patch_size/4, 1), name = 'MS')
			with tf.device('/gpu:0'):
				pMS_NET = pMS_ED('pMS_ED')
				PAN_converted_MS = pMS_NET.transform(I = PAN, is_training = False, reuse = True)

			pMS_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pMS_ED')

			INPUT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size/4, patch_size/4, 4), name = 'INPUT')
			NET=ED2('spectral_ED')
			OUTPUT = NET.transform(INPUT, is_training=True, reuse=False)

			SSIM_LOSS0 = 1 - SSIM_LOSS(tf.expand_dims(OUTPUT[:, :, :, 0], axis=-1), tf.expand_dims(INPUT[:, :, :, 0], axis=-1))
			SSIM_LOSS1 = 1 - SSIM_LOSS(tf.expand_dims(OUTPUT[:, :, :, 1], axis=-1), tf.expand_dims(INPUT[:, :, :, 1], axis=-1))
			SSIM_LOSS2 = 1 - SSIM_LOSS(tf.expand_dims(OUTPUT[:, :, :, 2], axis=-1), tf.expand_dims(INPUT[:, :, :, 2], axis=-1))
			SSIM_LOSS3 = 1 - SSIM_LOSS(tf.expand_dims(OUTPUT[:, :, :, 3], axis=-1), tf.expand_dims(INPUT[:, :, :, 3], axis=-1))
			S_LOSS = tf.reduce_mean((SSIM_LOSS0 + SSIM_LOSS1 + SSIM_LOSS2 + SSIM_LOSS3) / 4, axis = 0)

			MSE_LOSS = tf.reduce_mean(tf.square(OUTPUT - INPUT))
			LOSS = 25 * MSE_LOSS + S_LOSS

			current_iter = tf.Variable(0)
			learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
			                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
			                                           staircase = False)

			theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spectral_ED')
			# solver = tf.train.RMSPropOptimizer(learning_rate).minimize(LOSS, global_step = current_iter, var_list = theta)
			solver = tf.train.AdamOptimizer(learning_rate).minimize(LOSS, global_step = current_iter, var_list = theta)

			sess.run(tf.global_variables_initializer())
			saver0 = tf.train.Saver(var_list = pMS_list)
			saver0.restore(sess, P2MS_MODEL_PATH)

			saver = tf.train.Saver(max_to_keep = 50)
			tf.summary.scalar('Loss', LOSS)
			tf.summary.scalar('Loss_mse', MSE_LOSS)
			tf.summary.scalar('Loss_ssim', S_LOSS)
			tf.summary.scalar('Learning rate', learning_rate)
			tf.summary.image('input', INPUT[:, :, :, 0:3], max_outputs = 3)
			tf.summary.image('output', OUTPUT[:, :, :, 0:3], max_outputs = 3)

			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter("spec_logs/", sess.graph)

			# ** Start Training **
			step = 0
			for epoch in range(EPOCHES):
				np.random.shuffle(source_imgs)
				for batch in range(n_batches):
					step += 1
					current_iter = step
					pan_batch = np.zeros(shape=(BATCH_SIZE, int(patch_size/4), int(patch_size/4)), dtype=np.float32)
					ms_batch = np.zeros(shape=(BATCH_SIZE, int(patch_size/4), int(patch_size/4), 4), dtype=np.float32)
					MS_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0:4]
					PAN_batch = data[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 4]
					for b in range(BATCH_SIZE):
						pan_batch[b, :, :] = cv2.resize(PAN_batch[b, :, :], (int(patch_size / 4), int(patch_size / 4)))
						for c in range(4):
							ms_batch[b, :, :, c] = cv2.resize(MS_batch[b, :, :, c], (int(patch_size / 4), int(patch_size / 4)))
					pan_batch = np.expand_dims(pan_batch, axis = -1)
					pan_c_ms_batch = sess.run(PAN_converted_MS, feed_dict={PAN: pan_batch}) # np.concatenate([pan_batch, pan_batch, pan_batch, pan_batch], axis = -1)
					del MS_batch, PAN_batch


					# run the training step
					if step % 2:
						FEED_DICT = {INPUT: pan_c_ms_batch}
						sess.run(solver, feed_dict = FEED_DICT)
					else:
						FEED_DICT = {INPUT: ms_batch}
						sess.run(solver, feed_dict = FEED_DICT)
					result = sess.run(merged, feed_dict = FEED_DICT)
					writer.add_summary(result, step)
					if step % 100 == 0:
						saver.save(sess, 'spec_models/' + str(step) + '/' + str(step) + '.ckpt')

					is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
					if is_last_step or step % logging_period == 0:
						elapsed_time = datetime.now() - start_time
						loss = sess.run(LOSS, feed_dict = FEED_DICT)
						lr = sess.run(learning_rate)
						print('Epoch:%d/%d: step:%d, lr:%s, loss:%s, elapsed_time:%s\n' % (
							epoch + 1, EPOCHES, step, lr, loss, elapsed_time))

				saver.save(sess, 'spec_models/' + str(step) + '/' + str(step) + '.ckpt')





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
	main()
