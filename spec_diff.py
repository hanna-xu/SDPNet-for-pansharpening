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
from pnet import PNet

from spat_ED import ED1
from spec_ED import ED2
from P2MSnet import pMS_ED
from MS2Pnet import pP_ED

from tensorflow.python import pywrap_tensorflow

MS2P_MODEL_SAVEPATH = './MS2P_models/2000/2000.ckpt'
P2MS_MODEL_SAVEPATH = './P2MS_models/2000/2000.ckpt'
SPAT_MODEL_SAVEPATH = './spat_models/2000/2000.ckpt'
SPEC_MODEL_SAVEPATH = './spec_models/2000/2000.ckpt'

path1 = 'test_imgs/pan/'
path2 = 'test_imgs/ms/'
output_path = 'features/'


def main():
	# print('\nBegin to generate pictures ...\n')
	"save features for examples"
	for i in range(100):
		file_name1 = path1 + str(i + 1) + '.png'
		file_name2 = path2 + str(i + 1) + '.tif'

		pan = imread(file_name1) / 255.0
		ms = imread(file_name2) / 255.0
		print('file1:', file_name1, 'shape:', pan.shape)
		print('file2:', file_name2, 'shape:', ms.shape)
		h1, w1 = pan.shape
		h2, w2, c = ms.shape

		with tf.Graph().as_default(), tf.Session() as sess:
			INPUT = tf.placeholder(tf.float32, shape = (1, h2, w2, 4), name = 'INPUT')
			with tf.device('/gpu:0'):
				specnet = ED2('spectral_ED')
				OUTPUT = specnet.transform(INPUT, is_training = False, reuse = False)
			spec_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spectral_ED')

			PAN = tf.placeholder(tf.float32, shape = (1, h2, w2, 1), name = 'MS')
			with tf.device('/gpu:1'):
				pMSnet = pMS_ED('pMS_ED')
				PAN_converted_MS = pMSnet.transform(I = PAN, is_training = False, reuse = False)
			pMS_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pMS_ED')

			t_list = tf.trainable_variables()
			sess.run(tf.global_variables_initializer())
			saver1 = tf.train.Saver(var_list = spec_list)
			saver1.restore(sess, SPEC_MODEL_SAVEPATH)
			saver2 = tf.train.Saver(var_list = pMS_list)
			saver2.restore(sess, P2MS_MODEL_SAVEPATH)

			pan_ds = cv2.resize(pan, (h2, w2))
			pan_ds = pan_ds.reshape([1, h2, w2, 1])
			cms = sess.run(PAN_converted_MS, feed_dict = {PAN: pan_ds})
			ms = ms.reshape([1, h2, w2, 4])
			spec_features1 = sess.run(specnet.features, feed_dict = {INPUT: ms})
			spec_features2 = sess.run(specnet.features, feed_dict = {INPUT: cms})

			diff = np.mean(np.abs(spec_features1 - spec_features2), axis = (1, 2))
			if i == 0:
				Diff = diff
			else:
				Diff = np.concatenate([Diff, diff], axis = 0)

	Diff = np.mean(Diff, axis=0)
	channel_sort = np.flip(np.argsort(Diff), axis=0)
	sorted_Diff = sorted(Diff, reverse=True)

	f = "spec_diff.txt"
	for i in range(len(channel_sort)):
		if i==0:
			with open(f, "w") as file:
				file.write(str(channel_sort[i]) + "\n")
		else:
			with open(f, "a") as file:
				file.write(str(channel_sort[i]) + "\n")
	# scio.savemat('spat_diff.mat', {'D': Diff})



if __name__ == '__main__':
	main()