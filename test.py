from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
from scipy.misc import imread, imsave
from skimage import transform, data
from glob import glob
import matplotlib.image as mpimg
import scipy.io as scio
import cv2
from pnet import PNet #_tradition
import time

from tensorflow.python import pywrap_tensorflow
from tqdm import tqdm

MODEL_SAVE_PATH = './models/5130.ckpt'
path1 = 'test_imgs/pan/'
path2 = 'test_imgs/ms/'
output_path = 'results/'

def main():
	print('\nBegin to generate pictures ...\n')
	t=[]
	for i in tqdm(range(1)):
		file_name1 = path1 + str(i + 1) + '.png'
		file_name2 = path2 + str(i + 1) + '.tif'

		pan = imread(file_name1) / 255.0
		ms = imread(file_name2) / 255.0
		print('file1:', file_name1, 'shape:', pan.shape)
		print('file2:', file_name2, 'shape:', ms.shape)

		h1, w1 = pan.shape
		pan = pan.reshape([1, h1, w1, 1])
		h2, w2, c = ms.shape
		ms = ms.reshape([1, h2, w2, 4])


		with tf.Graph().as_default(), tf.Session() as sess:
			MS = tf.placeholder(tf.float32, shape = (1, h2, w2, 4), name = 'MS')
			PAN = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'PAN')
			Pnet = PNet('pnet')
			X = Pnet.transform(PAN = PAN, ms = MS)


			t_list = tf.trainable_variables()


			saver = tf.train.Saver(var_list = t_list)
			begin = time.time()
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, MODEL_SAVE_PATH)

			output = sess.run(X, feed_dict = {PAN: pan, MS: ms})

			if not os.path.exists(output_path):
				os.makedirs(output_path)
			scio.savemat(output_path + str(i + 1) + '.mat', {'i': output[0, :, :, :]})
			end=time.time()
			t.append(end-begin)
	print("Time: mean: %s,, std: %s" % (np.mean(t), np.std(t)))


if __name__ == '__main__':
	main()
