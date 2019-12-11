#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import layers
from functools import partial
import numpy as np
import math
from .toolkit import image
from .toolkit import *
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


conv2d = partial(layers.conv2d, stride=2, padding="same")
relu = tf.nn.relu
fc = tf.layers.dense

def gaussian_log_density(samples, mean, log_var):
    pi = tf.constant(math.pi)
    normalization = tf.log(2. * pi)
    inv_sigma = tf.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

class beta_TCVAE:
    def __init__(self, training = True, lr=0.0008, trainable=True,
                 z_dim=8, beta=8, batch_size=32, epochs=2):
        # input
        self.sess = tf.Session(config=config)
        self.training = training
        self.trainable = trainable
        self.beta = beta
        self.experiment_path = '/home/baxter/Documents/kuka_hrl'

        for f in os.listdir('./log'):
            print('Delet! ' + f)
            os.remove('./log/' + f)
        self.next_batch = image(batch_size, epochs=epochs,
                                path=self.experiment_path+'/image')

        self.img_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='image_input_ph')
        # self.img_mean = np.load('img_mean.npy')
        # self.img_std = np.load('img_std.npy')

        # ----------------------- Net Architecture -----------------------
        # encode
        z_mean, z_logvar = self.Enc(self.img_ph, z_dim,
                                    trainable=self.trainable,
                                    is_training=self.training)

        # sample
        if self.training:
            self.z = self.sample_from_latent_distribution(z_mean, z_logvar)
        else:
            self.z = z_mean

        # decode
        self.dec_img = self.Dec(self.z, trainable=self.trainable, is_training=self.training)

        # ----------------------- Loss Definition -----------------------

        reconstruction_loss = self.make_reconstruction_loss(self.img_ph, self.dec_img)

        kl_loss = self.compute_gaussian_kl(z_mean, z_logvar)
        tc = self.tc(z_mean, z_logvar, self.z)

        with tf.control_dependencies([tf.check_numerics(tc, message='NaN!!!!')]):
            regularizer = kl_loss + tc
        self.loss = tf.add(reconstruction_loss, regularizer, name="loss")

        # ----------------------- Optimisation Setting -----------------------

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies( update_ops ):
            self.step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2 = 0.999).minimize(self.loss)

        self.init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        self.sess.run(self.init)
        self.var_list = [var for var in tf.global_variables() if "moving" in var.name]
        self.var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=self.var_list, max_to_keep=1)

        tf.summary.scalar('loss', self.loss, family='loss')
        tf.summary.scalar('reconstruction_loss', reconstruction_loss, family='loss')
        tf.summary.scalar('z_mean', tf.reduce_mean(z_mean), family='loss')
        tf.summary.scalar('z_logvar', tf.reduce_mean(z_logvar), family='loss')
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss), family='loss')
        tf.summary.scalar('tc', tf.reduce_mean(tc), family='loss')

        self.all_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('log/', self.sess.graph)

    def make_reconstruction_loss(self, true_img, dec_img):
        with tf.variable_scope("reconstruction_loss"):
            # reconstruction_loss = tf.reduce_sum(
            #     tf.square(true_img - dec_img))
            dec_img = tf.clip_by_value(
                tf.nn.tanh( dec_img ) / 2 + 0.5, 1e-6, 1 - 1e-6)

            reconstruction_loss = tf.reduce_sum( true_img * tf.log(dec_img) + (1 - true_img) * tf.log(1 - dec_img) ,[1,2,3])
            reconstruction_loss = -tf.reduce_mean( reconstruction_loss )
        return reconstruction_loss

    def compute_gaussian_kl(self, z_mean, z_logvar):
        """Compute KL divergence between input Gaussian and Standard Normal."""
        return tf.reduce_mean(
            0.5 * tf.reduce_sum(
                tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]),
            name="kl_loss")

    def total_correlation(self, z, z_mean, z_logvar):
        """Estimate of total correlation on a batch.
        We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
        log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
        for the minimization. The constant should be equal to (num_latents - 1) *
        log(batch_size * dataset_size)
        Args:
          z: [batch_size, num_latents]-tensor with sampled representation.
          z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
          z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
        Returns:
          Total correlation estimated on a batch.
        """
        # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
        # tensor of size [batch_size, batch_size, num_latents]. In the following
        # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_logvar, 0))
        # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
        # + constant) for each sample in the batch, which is a vector of size
        # [batch_size,].
        log_qz_product = tf.reduce_sum(
            tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
            axis=1,
            keepdims=False)

        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
        # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
        log_qz = tf.reduce_logsumexp(
            tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
            axis=1,
            keepdims=False)
        return tf.reduce_mean(log_qz - log_qz_product)

    def tc(self, z_mean, z_logvar, z_sampled):
        tc = (self.beta - 1.) * self.total_correlation(z_sampled, z_mean, z_logvar)
        return tc

    def sample_from_latent_distribution(self, z_mean, z_logvar):
        """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
        return tf.add(
            z_mean,
            tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1),
            name="sampled_latent_variable")

    def fit(self):
        # print("\n-------------------------Global_Variables:-------------------------\n")
        # for v in self.var_list:
        #     print(v.name)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        step = 0
        try:
            print('\nStart training!\n')
            while not coord.should_stop():
                batch_img = self.sess.run(self.next_batch)

                # normal
                # batch_img_normal = (batch_img - self.img_mean) / self.img_std
                batch_img_normal = normal( batch_img )

                _, s = self.sess.run([self.step, self.all_summary],
                                     {self.img_ph: batch_img_normal})
                self.train_writer.add_summary(s, step)

                if step%10000==0:
                    print('Step: ', step)
                step +=1

        except tf.errors.OutOfRangeError:
            print('Done training! -- epoch limit reached\n')
        finally:
            coord.request_stop()
        coord.join(threads)

    def image_test(self, test_img):
        print('Normaled by 255.')
        # img_normal = (test_img - self.img_mean) / self.img_std
        img_normal = normal( test_img )
        # return reconstruct image
        rec_img = np.squeeze(self.sess.run(self.dec_img, {self.img_ph: img_normal[np.newaxis, :]}))

        print('DeNormaled by 255.')
        # rec_img = rec_img * self.img_std + self.img_mean
        return rec_img.astype(np.uint8)

    def image_disentangle(self, img ):

        print('Normaled by 255.')
        # img_normal = (img - self.img_mean) / self.img_std
        img_normal = normal( img )

        # return latent vector v.
        return self.sess.run(self.z, {self.img_ph: img_normal[np.newaxis, :]})

    def recon_from_v(self, z ):
        # reconstruct image from v.
        rec_img = np.squeeze(self.sess.run(self.dec_img, {self.z: z[np.newaxis, :]}))

        print('DeNormaled by 255.')
        # rec_img = rec_img * self.img_std + self.img_mean
        rec_img = denormal( rec_img )
        return rec_img.astype(np.uint8)

    def Save(self):
        self.saver.save(self.sess, save_path=self.experiment_path+'/model/model.ckpt',
                        write_meta_graph=False)

    def load(self):
        self.saver.restore(self.sess, save_path=self.experiment_path+'/model/model.ckpt')
        print("\n-------------------------Load pretrain model-------------------------\n")

    def Enc(self, img, z_dim, trainable, is_training=True):
        bn = partial(tf.layers.batch_normalization, training=is_training)
        conv2d_ = partial(conv2d, trainable=trainable)
        fc_ = partial(fc, trainable=trainable)
        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            # |inputs| num_outputs | kernel_size
            net = relu(bn(conv2d_(img, 32, 4)))
            net = relu(bn(conv2d_(net, 32, 4)))
            net = relu(bn(conv2d_(net, 32, 4)))
            net = relu(bn(conv2d_(net, 64, 2)))
            net = relu(bn(conv2d_(net, 64, 2)))
            net = layers.flatten(net)

            feature = relu(fc_(net, 256))
            means = fc_(feature, z_dim)
            log_var = fc_(feature, z_dim)
        return means, log_var

    def Dec(self, z, trainable, is_training=True):
        bn = partial(tf.layers.batch_normalization, training=is_training)
        dconv_ = partial(tf.layers.conv2d_transpose, strides=2, trainable=trainable,
                        activation=None, padding="same")
        fc_ = partial(fc, trainable=trainable)
        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            net = relu( bn(fc_(z, 256) ))
            net = relu( bn(fc_(net, 1024) ))
            net = tf.reshape(net, [-1, 4, 4, 64])
            net = relu(bn(dconv_( net, 64, 4)))
            net = relu(bn(dconv_( net, 32, 4)))
            net = relu(bn(dconv_( net, 32, 4)))
            net = relu(bn(dconv_( net, 32, 4)))
            dec_img = dconv_( net, 3, 4 )

        return dec_img
