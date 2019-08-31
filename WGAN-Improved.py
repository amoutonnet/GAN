# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:27:21 2019

@author: adamm
"""

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle
from keras.datasets.mnist import load_data

plt.close('all')


WIDTH = 28
HEIGHT = 28
CHANNEL = 1
Z_DIM = 100


def load_database():
    (x_train, _), (_, _) = load_data()
    X = np.expand_dims(x_train, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return X


def plot_sample(samples, nrows, ncols):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6), sharex=True, sharey=True)
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


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.compat.v1.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


class Generator():

    def __init__(self):
        self.name = 'GAN/Generator'

    def __call__(self, input_noise):
        init = tf.compat.v1.random_normal_initializer(stddev=0.02)
        regul = tf.contrib.layers.l2_regularizer(2.5e-5)
        with tf.compat.v1.variable_scope(self.name):
            bs = tf.shape(input_noise)[0]
            fc1 = tf.contrib.layers.fully_connected(
                input_noise, 1024,
                weights_initializer=init,
                weights_regularizer=regul,
                activation_fn=tf.identity
            )
            fc1 = tf.contrib.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            fc2 = tf.contrib.layers.fully_connected(
                fc1, 7 * 7 * 128,
                weights_initializer=init,
                weights_regularizer=regul,
                activation_fn=tf.identity
            )
            fc2 = tf.reshape(fc2, tf.stack([bs, 7, 7, 128]))
            fc2 = tf.contrib.layers.batch_norm(fc2)
            fc2 = tf.nn.relu(fc2)
            conv1 = tf.contrib.layers.convolution2d_transpose(
                fc2, 64, [4, 4], [2, 2],
                weights_initializer=init,
                weights_regularizer=regul,
                activation_fn=tf.identity
            )
            conv1 = tf.contrib.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2 = tf.contrib.layers.convolution2d_transpose(
                conv1, 1, [4, 4], [2, 2],
                weights_initializer=init,
                weights_regularizer=regul,
                activation_fn=tf.sigmoid
            )
            return conv2

    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]


class Discriminator():
    def __init__(self):
        self.name = 'GAN/Discriminator'

    def __call__(self, input_image, reuse=False):
        init = tf.compat.v1.random_normal_initializer(stddev=0.02)
        with tf.compat.v1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(input_image)[0]
            x = tf.reshape(input_image, [bs, 28, 28, 1])
            conv1 = tf.contrib.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=init,
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tf.contrib.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=init,
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv2 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.contrib.layers.fully_connected(
                conv2, 1024,
                weights_initializer=init,
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            out = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return out

    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]


class WGAN():

    def __init__(self, data, check_grad=False, opt='RMSProp', clip=False, grad_penalty=True):

        tf.compat.v1.reset_default_graph()

        self.real_data = shuffle(data)
        g_net = Generator()
        d_net = Discriminator()
        self.check_grad = check_grad
        self.opt = opt
        self.clip = clip

        self.X = tf.compat.v1.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
        self.Z = tf.compat.v1.placeholder(tf.float32, [None, Z_DIM])

        self.generator = g_net(self.Z)
        real_logits = d_net(self.X)
        fake_logits = d_net(self.generator, reuse=True)

        if grad_penalty:
            epsilon = tf.compat.v1.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.X + (1 - epsilon) * self.generator
            d_hat = d_net(x_hat, reuse=True)
            gradient_penalty = tf.gradients(d_hat, x_hat)[0]
            gradient_penalty = tf.sqrt(tf.reduce_sum(tf.square(gradient_penalty), axis=1))
            gradient_penalty = tf.reduce_mean(tf.square(gradient_penalty - 1.0) * 10)
            self.disc_loss = tf.reduce_mean(real_logits) - tf.reduce_mean(fake_logits) + gradient_penalty
        else:
            self.disc_loss = tf.reduce_mean(real_logits) - tf.reduce_mean(fake_logits)

        self.gen_loss = tf.reduce_mean(fake_logits)

        with tf.compat.v1.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            if self.opt == 'RMSProp':
                self.gen_step = tf.compat.v1.train.RMSPropOptimizer(
                    learning_rate=5e-5).minimize(self.gen_loss, var_list=g_net.vars())
                self.disc_step = tf.compat.v1.train.RMSPropOptimizer(
                    learning_rate=5e-5).minimize(self.disc_loss, var_list=d_net.vars())
            else:
                self.gen_step = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.gen_loss, var_list=g_net.vars())
                self.disc_step = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.disc_loss, var_list=d_net.vars())

            if self.check_grad:
                self.disc_gradients_norm = tf.compat.v1.norm(tf.compat.v1.gradients(self.disc_loss, d_net.vars())[0])
                self.gen_gradients_norm = tf.compat.v1.norm(tf.compat.v1.gradients(self.gen_loss, g_net.vars())[0])

        if self.clip:
            self.disc_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_net.vars()]

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    def train(self, epochs=100, batch_size=64):
        d_losses = []
        g_losses = []
        n_critics = 5
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        saver = tf.train.Saver()
        if 'model.ckpt.meta' in os.listdir('models/'):
            saver.restore(self.sess, "models/model.ckpt")
            print("Model restored...")
        else:
            print("No saved model found!")
        print('Start Training...')
        batch = 0
        epoch = 1
        data_lenght = len(self.real_data)
        total_batches = int(data_lenght//batch_size)
        end = False
        print('--------------------Epoch no. %d---------------------' % epoch)
        while 1:
            for d_iter in range(n_critics):
                if (batch+1)*batch_size <= data_lenght-1:
                    real_batch = self.real_data[batch*batch_size:(batch+1)*batch_size]
                else:
                    epoch += 1
                    if epoch > epochs:
                        end = True
                        break
                    self.sample_images(batch, 5, 5)
                    print('--------------------Epoch no. %d---------------------' % epoch)
                    batch = 0
                    self.real_data = shuffle(self.real_data)
                    real_batch = self.real_data[batch*batch_size:(batch+1)*batch_size]
                batch += 1
                noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
                if self.clip:
                    self.sess.run(self.disc_clip)
                self.sess.run(self.disc_step, feed_dict={self.X: real_batch, self.Z: noise_batch})
            if end:
                break
            noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
            self.sess.run(self.gen_step, feed_dict={self.Z: noise_batch})

            g_loss = self.sess.run(self.gen_loss, feed_dict={self.Z: noise_batch})
            d_loss = self.sess.run(self.disc_loss, feed_dict={self.X: real_batch, self.Z: noise_batch})
            d_losses += [d_loss]
            g_losses += [g_loss]

            if self.check_grad:
                g_grad = self.sess.run(self.gen_gradients_norm, feed_dict={self.Z: noise_batch})
                d_grad = self.sess.run(self.disc_gradients_norm, feed_dict={self.X: real_batch, self.Z: noise_batch})
                print('Batch no. %d/%d | dl = %.3f, gl = %.3f | dgrad = %.3f, ggrad = %.3f' % (batch, total_batches, d_loss, g_loss, d_grad, g_grad))
            else:
                print('Batch no. %d/%d | dl = %.3f, gl = %.3f' % (batch, total_batches, d_loss, g_loss))

        saver.save(self.sess, "models/model.ckpt")
        print('Model saved...')
        self.plot_losses(d_losses, g_losses)

    def plot_losses(self, d_losses, g_losses):
        plt.figure()
        plt.plot(d_losses)
        plt.plot(g_losses)
        plt.show()

    def sample_images(self, batch, nrows, ncols):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6), sharex=True, sharey=True)
        noise = np.random.normal(0, 1, (nrows*ncols, Z_DIM))
        gen_imgs = self.sess.run(self.generator, feed_dict={self.Z: noise})
        for i in range(nrows):
            for j in range(ncols):
                img = (gen_imgs[i*nrows+j]+1)*127.5
                img = np.squeeze(img)
                ax[i][j].imshow(img, cmap='gray')
                ax[i][j].get_xaxis().set_visible(False)
                ax[i][j].get_yaxis().set_visible(False)
        fig.suptitle('Sample generated by the GAN')
        fig.savefig('generated/uptodate.png')


if __name__ == '__main__':
    gan_test = WGAN(load_database(), check_grad=False, opt='Adam')
    gan_test.train(epochs=2, batch_size=256)
