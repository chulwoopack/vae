# Author: Chulwoo Pack, Michael Shanahan, M Parvez Rashid
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
flags.DEFINE_string('save_dir', '/work/cse496dl/cpack/Assignment_3/models/maxquality/maxquality_encoder_homework_3-0', 'directory where VAE model graph and weights are saved')
flags.DEFINE_integer('batch_size', 250, '')
flags.DEFINE_integer('latent_size', 512, '')
flags.DEFINE_integer('max_epoch', 2000, '')
flags.DEFINE_integer('early_stop', 25, '')
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

################
# VAE Modeling #
################
img_shape = [32, 32, 3]
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, shape=[None] + img_shape, name='encoder_input')
inputs_norm = tf.div(
   tf.subtract(
      inputs, 
      tf.reduce_min(inputs)
   ), 
   tf.subtract(
      tf.reduce_max(inputs), 
      tf.reduce_min(inputs)
   )
)
drop_prob = tf.placeholder_with_default(1.0, shape=())
## ENCODER
means, log_scales = model.gaussian_encoder(inputs, FLAGS.latent_size, drop_prob)  # (?, 4, 4, 8)
codes = model.gaussian_sample(means, log_scales)                 # (?, 4, 4, 8)
tf.identity(codes,name='encoder_output')
## DECODER
outputs = model.decoder(codes, drop_prob)
tf.identity(outputs,name='decoder_output')

# calculate loss with learnable parameter for output log_scale
with tf.name_scope('loss') as scope:
    reconstruction_loss, latent_loss = util.vae_loss(inputs, outputs, means, log_scales, 'bernoulli')
    total_loss = reconstruction_loss + tf.reduce_mean(latent_loss)

    
################
# Training VAE #
################
global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
saver = tf.train.Saver()
best_loss = 1e+10
stop_tol = 0
# setup optimizer
with tf.name_scope('optimizer') as scope:
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss=total_loss, global_step=global_step_tensor)

PSNR = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(FLAGS.max_epoch):
        print("epoch... ", epoch)
        for i in range(train_data.shape[0] // FLAGS.batch_size):
            batch_xs = train_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
            _, gen_loss, lat_loss = sess.run([train_op, reconstruction_loss, latent_loss], {inputs: batch_xs, drop_prob: 0.9})
            tot_loss = np.average(gen_loss) + np.average(lat_loss)
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
                print ("tot_loss: ", tot_loss, " gen_loss: " , np.average(gen_loss), " lat_loss: ", np.average(lat_loss))
    print("Done!")
    
    PSNR = 0
    for i in range(test_data.shape[0]):
        inputs_out, output_out = sess.run([inputs, outputs], {inputs: np.expand_dims(test_data[i], axis=0)})
        mse = np.sum((inputs_out-output_out) ** 2)/(32*32)
        PSNR = PSNR + (20*np.log10(255)-10*np.log10(mse))
    PSNR = PSNR/test_data.shape[0]
    print("Avg PSNR: ",PSNR)
        

