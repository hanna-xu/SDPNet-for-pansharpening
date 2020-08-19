from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
# from Net import Generator, WeightNet
from scipy.misc import imread, imsave
from skimage import transform, data
from glob import glob
import matplotlib.image as mpimg
import scipy.io as scio
import cv2
from spat_ED import ED1
from spec_ED import ED2

W = [0.35, 0.05, 0.35, 0.25]

MODEL1_SAVE_PATH = './spat_models/2000/2000.ckpt'
MODEL2_SAVE_PATH = './spec_models/2000/2000.ckpt'

path1 = 'test_imgs/pan/'
path2 = 'test_imgs/ms/'

MAX_15_INDEX1 = [60, 16, 69, 63, 36, 119, 43, 39, 92, 50, 14, 46, 59, 94, 9]
MIN_15_INDEX1 = [21, 1, 80, 17, 19, 113, 66, 40, 98, 61, 53, 57, 72, 109, 115]

## spatial
def main():
	for i in range(1):
		file_name1 = path1 + str(i + 17) + '.png'
		file_name2 = path2 + str(i + 17) + '.tif'

		pan = imread(file_name1) / 255.0
		ms = imread(file_name2) / 255.0
		print('file1:', file_name1, 'shape:', pan.shape)
		print('file2:', file_name2, 'shape:', ms.shape)

		# img1_Y = transform.resize(img1_Y, (h1, w1))
		# img2_Y = transform.resize(img2_Y, (h1, w1))
		h1, w1 = pan.shape
		pan = pan.reshape([1, h1, w1, 1])

		for c in range(4):
			ms_us = cv2.resize(ms[:, :, c], (264, 264))
			if c == 0:
				cpan = W[c] * np.expand_dims(ms_us, axis = -1)
			else:
				cpan = cpan + W[c] * np.expand_dims(ms_us, axis = -1)
		cpan = cpan.reshape([1, h1, w1, 1])
		print('cpan shape:', cpan.shape)

		with tf.Graph().as_default(), tf.Session() as sess:
			INPUT1 = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'INPUT')
			NET1 = ED1('spatial_ED')
			OUTPUT1 = NET1.transform(INPUT1, is_training = False, reuse = False)

			t_list = tf.trainable_variables()
			# for i in t_list:
			# 	print(i.name)
			saver = tf.train.Saver(var_list = t_list)
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, MODEL1_SAVE_PATH)
			spat_features1 = sess.run(NET1.features, feed_dict = {INPUT1: pan})
			spat_features2 = sess.run(NET1.features, feed_dict = {INPUT1: cpan})

			# diff = np.mean(np.abs(spat_features1 - spat_features2), axis = (1, 2))
			# if i == 0:
			# 	Diff = diff
			# else:
			# 	Diff = np.concatenate([Diff, diff], axis = 0)

	# scio.savemat('spat_diff.mat', {'D': Diff})
	# Diff = np.mean(Diff, axis=0)
	# print("diff:", Diff)


			'''save the perceptual features'''
			for ii in range(15):
				channel_index = MAX_15_INDEX1[ii] - 1
				if not os.path.exists('save_features/spat/max_15_of_17/pan'):
					os.makedirs('save_features/spat/max_15_of_17/pan')
				scio.savemat('save_features/spat/max_15_of_17/pan/' + str(ii + 1) + '.mat', {'a': spat_features1[0, :, :, channel_index]})
				if not os.path.exists('save_features/spat/max_15_of_17/cpan'):
					os.makedirs('save_features/spat/max_15_of_17/cpan')
				scio.savemat('save_features/spat/max_15_of_17/cpan/' + str(ii + 1) + '.mat', {'b': spat_features2[0, :, :, channel_index]})
				D = np.mean(np.abs(spat_features1[0, :, :, channel_index] - spat_features2[0, :, :, channel_index]))
				print(str(ii + 1) + ':', D)

			for ii in range(15):
				channel_index = MIN_15_INDEX1[ii] - 1
				if not os.path.exists('save_features/spat/min_15_of_17/pan'):
					os.makedirs('save_features/spat/min_15_of_17/pan')
				scio.savemat('save_features/spat/min_15_of_17/pan/' + str(ii + 1) + '.mat', {'a':spat_features1[0, :, :, channel_index]})
				if not os.path.exists('save_features/spat/min_15_of_17/cpan'):
					os.makedirs('save_features/spat/min_15_of_17/cpan')
				scio.savemat('save_features/spat/min_15_of_17/cpan/' + str(ii + 1) + '.mat', {'b': spat_features2[0, :, :, channel_index]})
				# scio.savemat('save_features/spat/min_10_of_17/cpan/' + str(ii + 1) + '.png',
				#              {'fcpan': spat_features2[0, :, :, channel_index]})
				D = np.mean(np.abs(spat_features1[0, :, :, channel_index] - spat_features2[0, :, :, channel_index]))
				print(str(ii+1) + ':', D)













# MAX_15_INDEX2=[115, 62, 109, 116, 9, 39, 97, 122, 78, 15, 2, 56, 28, 68, 19]
# MIN_15_INDEX2=[59, 63, 58, 52, 112, 12, 5, 37, 4, 36, 64, 31, 98, 25, 111]
#
# def main():
# 	for i in range(1):
# 		file_name1 = path1 + str(i + 17) + '.png'
# 		file_name2 = path2 + str(i + 17) + '.tif'
#
# 		pan = imread(file_name1) / 255.0
# 		ms = imread(file_name2) / 255.0
# 		print('file1:', file_name1, 'shape:', pan.shape)
# 		print('file2:', file_name2, 'shape:', ms.shape)
# 		h1, w1 = pan.shape
#
# 		for c in range(4):
# 			pan_ds = np.expand_dims(cv2.resize(pan, (int(h1 / 4), int(w1 / 4))), axis = -1)
# 		cms = np.concatenate([pan_ds, pan_ds, pan_ds, pan_ds], axis = -1)
# 		cms = np.expand_dims(cms, axis = 0)
# 		ms = np.expand_dims(ms, axis = 0)
# 		print('cms shape:', cms.shape)
#
# 		with tf.Graph().as_default(), tf.Session() as sess:
# 			INPUT2 = tf.placeholder(tf.float32, shape = (1, int(h1/4), int(w1/4), 4), name = 'INPUT')
# 			NET2 = ED2('spectral_ED')
# 			OUTPUT2 = NET2.transform(INPUT2, is_training = False, reuse = False)
#
# 			t_list = tf.trainable_variables()
# 			saver = tf.train.Saver(var_list = t_list)
# 			sess.run(tf.global_variables_initializer())
# 			saver.restore(sess, MODEL2_SAVE_PATH)
# 			spec_features1 = sess.run(NET2.features, feed_dict = {INPUT2: cms})
# 			spec_features2 = sess.run(NET2.features, feed_dict = {INPUT2: ms})
#
# 			# diff = np.mean(np.abs(spec_features1 - spec_features2), axis = (1, 2))
# 			# if i == 0:
# 			# 	Diff = diff
# 			# else:
# 			# 	Diff = np.concatenate([Diff, diff], axis = 0)
#
# 	# scio.savemat('spec_diff.mat', {'D': Diff})
# 	# Diff = np.mean(Diff, axis=0)
# 	# print("diff:", Diff)
#
# 			'''save the perceptual features'''
# 			for ii in range(15):
# 				channel_index = MAX_15_INDEX2[ii] - 1
# 				if not os.path.exists('save_features/spec/max_15_of_17/cms'):
# 					os.makedirs('save_features/spec/max_15_of_17/cms')
# 				scio.savemat('save_features/spec/max_15_of_17/cms/' + str(ii + 1) + '.mat', {'cms_spec': spec_features1[0, :, :, channel_index]})
# 				if not os.path.exists('save_features/spec/max_15_of_17/ms/'):
# 					os.makedirs('save_features/spec/max_15_of_17/ms/')
# 				scio.savemat('save_features/spec/max_15_of_17/ms/' + str(ii + 1) + '.mat', {'ms_spec': spec_features2[0, :, :, channel_index]})
# 				D = np.mean(np.abs(spec_features1[0, :, :, channel_index] - spec_features2[0, :, :, channel_index]))
# 				print(str(ii + 1) + ':', D)
#
# 			for ii in range(15):
# 				channel_index = MIN_15_INDEX2[ii] - 1
# 				if not os.path.exists('save_features/spec/min_15_of_17/cms'):
# 					os.makedirs('save_features/spec/min_15_of_17/cms')
# 				scio.savemat('save_features/spec/min_15_of_17/cms/' + str(ii + 1) + '.mat', {'cms_spec': spec_features1[0, :, :, channel_index]})
# 				if not os.path.exists('save_features/spec/min_15_of_17/ms'):
# 					os.makedirs('save_features/spec/min_15_of_17/ms')
# 				scio.savemat('save_features/spec/min_15_of_17/ms/' + str(ii + 1) + '.mat', {'ms_spec': spec_features2[0, :, :, channel_index]})
# 				D = np.mean(np.abs(spec_features1[0, :, :, channel_index] - spec_features2[0, :, :, channel_index]))
# 				print(str(ii+1) + ':', D)




if __name__ == '__main__':
	main()
