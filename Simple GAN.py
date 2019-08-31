# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:27:21 2019

@author: adamm
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

plt.close('all')

WIDTH = 28
HEIGHT = 28
CHANNEL = 1
Z_DIM = 100


def load_database():
    return pickle.load(open('datasets/mnist.pkl', 'rb'))


def plot_sample(samples, nrows, ncols):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(6, 6), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            img = (samples[i*3+j]+1)*127.5
            img = np.squeeze(img)
            ax[i][j].imshow(img, cmap='gray')
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
            ax[i][j].set_aspect('equal')
    fig.suptitle('Sample of the Database')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.show()


class GAN():

    def __init__(self, data, data_label):
        self.name = 'GAN'

        tf.compat.v1.reset_default_graph()

        indx = data_label == 8
        self.real_data = data[indx]

        self.X = tf.compat.v1.placeholder(
            tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
        self.Z = tf.compat.v1.placeholder(tf.float32, [None, Z_DIM])

        self.generator = self.build_generator(self.Z)
        real_logits = self.build_discriminator(self.X)
        fake_logits = self.build_discriminator(self.generator, reuse=True)

        #self.disc_acc_real = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_logits),tf.ones_like(real_logits)),tf.float32))
        #self.disc_acc_fake = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_logits),tf.zeros_like(fake_logits)),tf.float32))

        #epsilon = tf.compat.v1.random_uniform([], 0.0, 1.0)
        #x_hat = epsilon * self.X + (1 - epsilon) * self.generator
        #d_hat = self.build_discriminator(x_hat, reuse=True)
        #gradient_penalty = tf.gradients(d_hat, x_hat)[0]
        #gradient_penalty = tf.sqrt(tf.reduce_sum(tf.square(gradient_penalty), axis=np.arange(1, len(gradient_penalty.shape))))
        #gradient_penalty = tf.reduce_mean(tf.square(gradient_penalty - 1.0) * 10)

        #self.disc_loss = tf.reduce_mean(-real_logits+fake_logits)
        #self.gen_loss = tf.reduce_mean(-fake_logits)

        self.disc_loss = - \
            tf.reduce_mean(real_logits) + tf.reduce_mean(fake_logits)
        self.gen_loss = - tf.reduce_mean(fake_logits)

        gen_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        disc_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

        # self.gen_step = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.9).minimize(self.gen_loss, var_list=gen_vars)
        # self.disc_step = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.9).minimize(self.disc_loss, var_list=disc_vars)

        self.gen_step = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=5e-5).minimize(self.gen_loss, var_list=gen_vars)
        self.disc_step = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=5e-5).minimize(self.disc_loss, var_list=disc_vars)

        self.disc_gradients_norm = tf.compat.v1.norm(
            tf.compat.v1.gradients(self.disc_loss, disc_vars)[0])
        self.gen_gradients_norm = tf.compat.v1.norm(
            tf.compat.v1.gradients(self.gen_loss, gen_vars)[2])

        self.disc_clip = [v.assign(tf.clip_by_value(
            v, -0.01, 0.01)) for v in disc_vars]

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(
            config=tf.ConfigProto(gpu_options=gpu_options))

    def build_generator(self, input_noise, reuse=False):
        init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
        with tf.compat.v1.variable_scope("GAN/Generator", reuse=reuse):

            x = tf.keras.layers.Dense(128*7*7)(input_noise)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.Reshape((7, 7, 128))(x)

            x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(
                2, 2), padding='same', kernel_initializer=init)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(
                2, 2), padding='same', kernel_initializer=init)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            out = tf.keras.layers.Conv2D(
                1, (7, 7), padding='same', activation='tanh')(x)
            return out

    def build_discriminator(self, input_image, reuse=False):
        init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
        with tf.compat.v1.variable_scope("GAN/Discriminator", reuse=reuse) as vs:
            if reuse:
                vs.reuse_variables()

            x = tf.keras.layers.Conv2D(64, (4, 4), strides=(
                2, 2), padding='same', kernel_initializer=init)(input_image)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            x = tf.keras.layers.Conv2D(64, (4, 4), strides=(
                2, 2), padding='same', kernel_initializer=init)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            x = tf.keras.layers.Flatten()(x)
            out = tf.keras.layers.Dense(1)(x)
        return out

    def train(self, nb_batches=100, batch_size=64):
        d_losses = []
        g_losses = []
        d_iters = 5
        g_iters = 1
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        print('----------Start Training-----------')
        for batch in range(1, nb_batches+1):
            for d_iter in range(d_iters):
                real_batch = shuffle(self.real_data)[:batch_size]
                noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
                self.sess.run(self.disc_clip)
                self.sess.run(self.disc_step, feed_dict={
                              self.X: real_batch, self.Z: noise_batch})

            for g_iter in range(g_iters):
                noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
                self.sess.run(self.gen_step, feed_dict={self.Z: noise_batch})

            if batch % 1 == 0:
                noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
                #d_loss_real = self.sess.run(self.disc_loss_real, feed_dict={self.X:real_batch})
                #d_loss_fake = self.sess.run(self.disc_loss_fake, feed_dict={self.Z:noise_batch})
                #d_acc_real = int(100*self.sess.run(self.disc_acc_real, feed_dict={self.X:real_batch}))
                #d_acc_fake = int(100*self.sess.run(self.disc_acc_fake, feed_dict={self.Z:noise_batch}))
                #d_loss = d_loss_real + d_loss_fake
                g_loss = self.sess.run(self.gen_loss, feed_dict={
                                       self.Z: noise_batch})
                d_loss = self.sess.run(self.disc_loss, feed_dict={
                                       self.X: real_batch, self.Z: noise_batch})
                g_grad = self.sess.run(self.gen_gradients_norm, feed_dict={
                                       self.Z: noise_batch})
                d_grad = self.sess.run(self.disc_gradients_norm, feed_dict={
                                       self.X: real_batch, self.Z: noise_batch})
                d_losses += [d_loss]
                g_losses += [g_loss]
                #print('b no. %d/%d | dlr = %.3f, dlf = %.3f, dl = %.3f, gl = %.3f | dar = %d%%, daf = %d%%'%(batch, nb_batches, d_loss_real, d_loss_fake, d_loss, g_loss, d_acc_real, d_acc_fake))
                print('b no. %d/%d | dl = %.3f, gl = %.3f | dgrad = %.3f, ggrad = %.3f' %
                      (batch, nb_batches, d_loss, g_loss, d_grad, g_grad))

            # if d_acc_fake > 90 :
            #    d_iters = 1
            #    g_iters = 1
            # else:
            #    d_iters = 1
            #    g_iters = 1
            if batch % int(nb_batches/4) == 0:
                self.sample_images(batch, 5, 5)
        self.plot_losses(d_losses, g_losses)

    def plot_losses(self, d_losses, g_losses):
        plt.figure()
        plt.plot(d_losses)
        plt.plot(g_losses)
        plt.show()

    def sample_images(self, batch, nrows, ncols):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(6, 6), sharex=True, sharey=True)
        noise = np.random.normal(0, 1, (nrows*ncols, Z_DIM))
        gen_imgs = self.sess.run(self.generator, feed_dict={self.Z: noise})
        for i in range(nrows):
            for j in range(ncols):
                img = (gen_imgs[i*nrows+j]+1)*127.5
                img = np.squeeze(img)
                ax[i][j].imshow(img, cmap='gray')
                ax[i][j].get_xaxis().set_visible(False)
                ax[i][j].get_yaxis().set_visible(False)
        fig.suptitle('Sample of the GAN on batch %d' % batch)
        fig.show()


if __name__ == '__main__':
    x_train, y_train, _, _ = load_database()
    gan_test = GAN(x_train, y_train)
    # gan_test.train(nb_batches=1000)
