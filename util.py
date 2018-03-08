# Author: Chulwoo Pack
# Credits: Paul Quint
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_util
import math

EPS = 1e-10

#imports data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def std_gaussian_KL_divergence(mu, log_sigma):
    """Analytic KL distance between N(mu, e^log_sigma) and N(0, 1)"""
    sigma = tf.exp(log_sigma)
    return -0.5 * tf.reduce_sum(
        1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)

def bernoulli_logp(new_image, orig_image): #(output, input)
    """Calculates log prob of sample under bernoulli distribution.
    
    Note: args must be in range [0,1]
    """
    vae_loss_likelihood = tf.reduce_sum(orig_image * tf.log(EPS + new_image) +
                         ((1 - orig_image) * tf.log(EPS + 1 - new_image)), 1)
    return vae_loss_likelihood

def discretized_logistic_logp(mean, logscale, sample, binsize=1 / 256.0):
    """Calculates log prob of sample under discretized logistic distribution."""
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(
        tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + EPS)

    if logp.shape.ndims == 4:
        logp = tf.reduce_sum(logp, [1, 2, 3])
    elif logp.shape.ndims == 2:
        logp = tf.reduce_sum(logp, 1)
    return logp

def vae_loss(inputs, outputs, latent_mean, latent_log_scale, output_dist, output_log_scale=None):
    """Calculate the VAE loss (aka [ELBO](https://arxiv.org/abs/1312.6114))
    
    Args:
        - inputs: VAE input
        - outputs: VAE output
        - latent_mean: parameter of latent distribution
        - latent_log_scale: log of std. dev. of the latent distribution
        - output_dist: distribution parameterized by VAE output, must be in ['logistic', 'bernoulli']
        - output_log_scale: log scale parameter of the output dist if it's logistic, can be learnable
        
    Note: output_log_scale must be specified if output_dist is logistic
    """
    # Calculate reconstruction loss
    # Equal to minus the log likelihood of the input data under the VAE's output distribution
    if output_dist == 'bernoulli':
        outputs = tf.sigmoid(outputs)
        reconstruction_loss = -bernoulli_logp(outputs, inputs)
    elif output_dist == 'logistic':
        outputs = tf.clip_by_value(outputs, 1 / 512., 1 - 1 / 512.)
        reconstruction_loss = -discretized_logistic_logp(outputs, output_log_scale, inputs)
    else:
        print('Must specify an argument for output_dist in [bernoulli, logistic]')
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        
    # Calculate latent loss
    latent_loss = std_gaussian_KL_divergence(latent_mean, latent_log_scale)
    latent_loss = tf.reduce_mean(latent_loss)
    
    return reconstruction_loss, latent_loss

