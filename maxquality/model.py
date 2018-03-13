# Author: Chulwoo Pack, Michael Shanahan, M Parvez Rashid
# Credits: Paul Quint
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_util
import math

def lrelu(x, alpha=0.5):
    return tf.maximum(x, tf.multiply(x, alpha))

def downscale_block(inputs, filters, kernel_size=4, downscale=1):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: int
    """
    with tf.name_scope('conv_block') as scope:
        conv = tf.layers.conv2d(inputs, filters, kernel_size=kernel_size, strides=2, padding='same', activation=lrelu)
        #pool = tf.layers.max_pooling2d(conv, pool_size=(2,2), strides=downscale, padding='same')
        return conv    
    
def upscale_block(x, kernel_size=4, scale=2, activation=lrelu):
    """ [Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158) """
    n, w, h, c = x.get_shape().as_list()
    x = tf.layers.conv2d(x, filters=c * scale ** 2, kernel_size=kernel_size, activation=activation, padding='same')
    output = tf.depth_to_space(x, scale)
    return output

def decoder(codes, drop_prob):
    tf.identity(codes,name='decoder_input')
    hidden_decoder = tf.layers.dense(codes, 16*16*3) # (?,  4x4x8)
    decoder_16 = tf.reshape(hidden_decoder, [-1, 16, 16, 3])
    decoder_32 = upscale_block(decoder_16)             # (?,  8,  8,  8)
    decoder_32 = tf.nn.dropout(decoder_32, drop_prob)
    output = decoder_32
    return output

def gaussian_encoder(inputs, latent_size, drop_prob):
    """inputs should be a tensor of images whose height and width are multiples of 4"""
    with tf.name_scope('downscale_1') as scope:
        conv_16 = downscale_block(inputs, 3, downscale=2)      # (?, 16, 16,  4)
        conv_16 = tf.nn.dropout(conv_16, drop_prob)
    conv_16_flat = tf.reshape(conv_16, [-1, 16 * 16 * 3])
    
    mean = tf.layers.dense(conv_16_flat, latent_size, name='mean')              # (?, 4*4*16)
    log_scale = tf.layers.dense(conv_16_flat, latent_size, name='log_scale')    # (?, 4*4*16)
    return mean, log_scale      

def gaussian_sample(mean, log_scale):
    # noise inputsis zero centered and std. dev. 1
    gaussian_noise = tf.random_normal(tf.shape(mean)) # Sampling...
    return mean + tf.exp(tf.clip_by_value(log_scale,-1.0/255.0,1.0/255.0))*gaussian_noise
