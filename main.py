# Author: Chulwoo Pack
# Credits: Paul Quint

import tensorflow as tf
import numpy as np
# these ones let us draw images in our notebook
#import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.python.framework import tensor_util
import math
import util
import model

# flags
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/cpack/Assignment_3/dataset/', 'directory where CIFAR-100 and CIFAR-10 is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/cpack/Assignment_3/models/1', 'directory where VAE model graph and weights are saved')
flags.DEFINE_integer('batch_size', 500, '')
flags.DEFINE_integer('latent_size', 32, '')
flags.DEFINE_integer('early_stop', 15, '')
FLAGS = flags.FLAGS

#############
# CIFAR 100 #
#############
cifar100_test = {}
cifar100_train = {}
# Load the raw CIFAR-100 data.
cifar100_test = util.unpickle(FLAGS.data_dir + 'cifar-100-python/test')
cifar100_train = util.unpickle(FLAGS.data_dir + 'cifar-100-python/train')

train_data = cifar100_train[b'data']
test_data = cifar100_test[b'data']

train_data = np.reshape(train_data,(50000, 3, 32, 32)).transpose(0,2,3,1).astype(float)
test_data = np.reshape(test_data,(10000, 3, 32, 32)).transpose(0,2,3,1).astype(float)
# Normalization: Input image should be normalized to have 0 mean and unit variation for the VAE.
for i in range(50000):
    #train_data[i] = train_data[i]-np.mean(train_data[i])
    #train_data[i] = train_data[i]/np.std(train_data[i])
    train_data[i] = train_data[i]/255.0

for i in range(10000):
    test_data[i] = test_data[i]/255.0

################
# VAE Modeling #
################
img_shape = [32, 32, 3]
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, shape=[None] + img_shape, name='encoder_input')
## ENCODER
means, log_scales = model.gaussian_encoder(inputs, FLAGS.latent_size)  # (?, 4, 4, 8)
codes = model.gaussian_sample(means, log_scales)                 # (?, 4, 4, 8)
tf.identity(codes,name='encoder_output')
## DECODER
outputs = model.decoder(codes, name='decoder_output')

# calculate loss with learnable parameter for output log_scale
with tf.name_scope('loss') as scope:
    output_log_scale = tf.get_variable("output_log_scale", initializer=tf.constant(0.0, shape=img_shape))
    reconstruction_loss, latent_loss = util.vae_loss(inputs, outputs, means, log_scales, 'bernoulli', output_log_scale)
    total_loss = reconstruction_loss + latent_loss
   
################
# Training VAE #
################
global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
saver = tf.train.Saver()
best_loss = 1e+10
stop_tol = 0
# setup optimizer
with tf.name_scope('optimizer') as scope:
	#rate = tf.train.exponential_decay(0.15, step, 1, 0.001)
    train_op = tf.train.AdamOptimizer(0.0005).minimize(loss=total_loss, global_step=global_step_tensor)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(30):
	    print("epoch... ", epoch)
	    for i in range(train_data.shape[0] // FLAGS.batch_size):
	        batch_xs = train_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
	        _, gen_loss, lat_loss = sess.run([train_op, reconstruction_loss, latent_loss], {inputs: batch_xs})
	        tot_loss = gen_loss + lat_loss
	        # SAVE THE BEST MODEL
	        if tot_loss < best_loss:
	        	best_loss = tot_loss
	        	saver.save(sess, FLAGS.save_dir) 
	        	stop_tol = 0
	        	print ("Best model is found and saved... tot_loss:",tot_loss)
	        # EARLY STOPPING
	        else:
	        	stop_tol = stop_tol+1
	        	if stop_tol > FLAGS.early_stop:
	        		print ("No more improvement... Early-stopping is triggered...")
	        		break
	       	# LOG LOSS
	        if i%10==0:
	            print ("gen_loss: " , np.average(gen_loss), " lat_loss: ", np.average(lat_loss))
	print("Done!")

	# VISUALIZATION
	#input_out, output_out = session.run([inputs, outputs], {inputs: my_test_data})

