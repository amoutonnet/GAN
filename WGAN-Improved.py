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
import shutil
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


class ClipConstraint(tf.compat.v1.keras.constraints.Constraint):
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
        regul = tf.compat.v2.keras.regularizers.L1L2(l2=2.5e-5)

        with tf.compat.v1.variable_scope(self.name):
            bs = tf.shape(input_noise)[0]
            fc1 = tf.compat.v1.keras.layers.Dense(
                1024,
                kernel_initializer=init,
                kernel_regularizer=regul
            )(input_noise)
            fc1 = tf.compat.v1.keras.layers.BatchNormalization()(fc1)
            fc1 = tf.nn.relu(fc1)
            fc2 = tf.compat.v1.keras.layers.Dense(
                7 * 7 * 128,
                kernel_initializer=init,
                kernel_regularizer=regul
            )(fc1)
            fc2 = tf.reshape(fc2, tf.stack([bs, 7, 7, 128]))
            fc2 = tf.compat.v1.keras.layers.BatchNormalization()(fc2)
            fc2 = tf.nn.relu(fc2)
            conv1 = tf.compat.v1.keras.layers.Conv2DTranspose(
                128, [4, 4], [2, 2],
                kernel_initializer=init,
                kernel_regularizer=regul,
                padding='same'
            )(fc2)
            conv1 = tf.compat.v1.keras.layers.BatchNormalization()(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2 = tf.compat.v1.keras.layers.Conv2DTranspose(
                128, [4, 4], [2, 2],
                kernel_initializer=init,
                kernel_regularizer=regul,
                padding='same'

            )(conv1)
            conv2 = tf.compat.v1.keras.layers.BatchNormalization()(conv2)
            conv2 = tf.nn.relu(conv2)
            conv3 = tf.compat.v1.keras.layers.Conv2D(
                1, [4, 4],
                kernel_initializer=init,
                kernel_regularizer=regul,
                padding='same'

            )(conv2)
            return conv3

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
            conv1 = tf.compat.v1.keras.layers.Conv2D(
                128, [4, 4], [2, 2],
                kernel_initializer=init
            )(input_image)
            conv1 = leaky_relu(conv1)
            conv2 = tf.compat.v1.keras.layers.Conv2D(
                128, [4, 4], [2, 2],
                kernel_initializer=init
            )(conv1)
            conv2 = leaky_relu(conv2)
            conv2 = tf.compat.v1.keras.layers.Flatten()(conv2)
            # fc1 = tf.compat.v1.keras.layers.Dense(
            #     1024,
            #     kernel_initializer=init
            # )(conv2)
            # fc1 = leaky_relu(fc1)
            out = tf.compat.v1.keras.layers.Dense(1)(conv2)
            return out

    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]


class WGAN():

    def __init__(self, data, check_grad=False, opt='RMSProp', clip=False, do_grad_penalty=True, reset_model=False):

        tf.compat.v1.reset_default_graph()

        self.real_data = shuffle(data)
        g_net = Generator()
        d_net = Discriminator()
        self.check_grad = check_grad
        self.do_grad_penalty = do_grad_penalty
        self.opt = opt
        self.clip = clip

        if reset_model:
            list_files = os.listdir('models/')
            if(len(list_files) == 0):
                print('No model found, no reinitialization done...')
            else:
                for i in list_files:
                    os.remove('models/'+i)
                print('Model reinitialized...')

        self.X = tf.compat.v1.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
        self.Z = tf.compat.v1.placeholder(tf.float32, [None, Z_DIM])

        self.generator = g_net(self.Z)
        real_logits = d_net(self.X)
        fake_logits = d_net(self.generator, reuse=True)

        self.disc_loss_real = -tf.reduce_mean(real_logits)
        self.disc_loss_fake = tf.reduce_mean(fake_logits)
        self.gen_loss = -tf.reduce_mean(fake_logits)

        if self.do_grad_penalty:
            epsilon = tf.compat.v1.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.X + (1 - epsilon) * self.generator
            d_hat = d_net(x_hat, reuse=True)
            gradients = tf.gradients(d_hat, x_hat)
            gradient = gradients[0]
            grad_square = tf.compat.v1.reshape(tf.square(gradient), [-1, 28*28])
            grad_sum = tf.reduce_sum(grad_square, axis=1)
            gradient_norm = tf.sqrt(grad_sum)
            self.gradient_penalty = tf.reduce_mean(tf.square(gradient_norm - 1.0) * 10)
            self.disc_loss = self.disc_loss_fake + self.disc_loss_real + self.gradient_penalty
        else:
            self.disc_loss = self.disc_loss_fake + self.disc_loss_real

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
        self.d_losses_real, self.d_losses_fake, self.d_losses, self.g_losses = [], [], [], []
        if self.do_grad_penalty:
            self.grad_penalties = []
        n_critics = 5
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        if 'model.ckpt.meta' in os.listdir('models/'):
            saver.restore(self.sess, "models/model.ckpt")
            print("Model restored...")
        else:
            print("No saved model found!")
        print('Start Training...')
        self.total_batch_of_batches = int(len(self.real_data)//(batch_size*n_critics))
        for epoch in range(epochs):
            print('--------------------Epoch no. %d---------------------' % epoch)
            self.real_data = shuffle(self.real_data)
            for mb in range(self.total_batch_of_batches):
                batch_of_real_batches = self.real_data[mb*batch_size:(mb+n_critics)*batch_size]
                for d_iter in range(n_critics):
                    real_batch = batch_of_real_batches[d_iter*batch_size:(d_iter+1)*batch_size]
                    noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
                    if self.clip:
                        self.sess.run(self.disc_clip)
                    self.sess.run(self.disc_step, feed_dict={self.X: real_batch, self.Z: noise_batch})

                noise_batch = np.random.normal(0, 1, (batch_size, Z_DIM))
                self.sess.run(self.gen_step, feed_dict={self.Z: noise_batch})

                self.follow_evolution(epoch+1, mb+1, real_batch, noise_batch)
            self.sample_images(epoch, 5, 5)

        saver.save(self.sess, "models/model.ckpt")
        print('Model saved...')
        self.plot_losses()

    def follow_evolution(self, epoch, mb, real_batch, noise_batch):
        to_print = 'E.%d | BoB.%d/%d' % (epoch, mb, self.total_batch_of_batches)
        if self.do_grad_penalty:
            d_loss_real, d_loss_fake, g_penalty = self.sess.run([self.disc_loss_real, self.disc_loss_fake,
                                                                 self.gradient_penalty], feed_dict={self.X: real_batch, self.Z: noise_batch})
            d_loss = d_loss_real + d_loss_fake + g_penalty
            self.grad_penalties += [g_penalty]
            to_print += ' | dlr = %.3f, dlf = %.3f, dl = %.3f, gp = %.3f' % (d_loss_real, d_loss_fake, d_loss, g_penalty)
        else:
            d_loss_real, d_loss_fake = self.sess.run([self.disc_loss_real, self.disc_loss_fake],
                                                     feed_dict={self.X: real_batch, self.Z: noise_batch})
            d_loss = d_loss_real + d_loss_fake
            to_print += ' | dlr = %.3f, dlf = %.3f, dl = %.3f' % (d_loss_real, d_loss_fake, d_loss)

        g_loss = self.sess.run(self.gen_loss, feed_dict={self.Z: noise_batch})
        to_print += ' | gl = %.3f' % g_loss

        if self.check_grad:
            g_grad = self.sess.run(self.gen_gradients_norm, feed_dict={self.Z: noise_batch})
            d_grad = self.sess.run(self.disc_gradients_norm, feed_dict={self.X: real_batch, self.Z: noise_batch})
            to_print += ' | dgrad = %.3f, ggrad = %.3f' % (d_grad, g_grad)

        self.d_losses_real += [d_loss_real]
        self.d_losses_fake += [d_loss_fake]
        self.d_losses += [d_loss]
        self.g_losses += [g_loss]

        print(to_print)

    def plot_losses(self):
        plt.figure()
        plt.plot(self.d_losses_real)
        plt.plot(self.d_losses_fake)
        plt.plot(self.g_losses)
        if self.do_grad_penalty:
            plt.plot(self.grad_penalties)
        plt.title('Loss evolution')
        plt.show()

    def sample_images(self, epoch, nrows, ncols):
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
        fig.suptitle('Sample generated by the GAN at epoch %d' % epoch)
        fig.savefig('generated/epoch%d.png' % epoch)


if __name__ == '__main__':
    gan_test = WGAN(load_database(), check_grad=True, clip=False, opt='Adam', reset_model=False, do_grad_penalty=True)
    gan_test.train(epochs=20, batch_size=64)
