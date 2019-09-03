# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:27:21 2019

@author: adamm
"""

import os
import pickle
import shutil

import keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import zoom
from sklearn.utils import shuffle

plt.close('all')


def load_database():
    if(DATASET == 'cifar'):
        (x_train, y_train), (_, _) = datasets.cifar10.load_data()
        indx = y_train == 5
        x_train = x_train[indx.squeeze()]
    else:
        (x_train, y_train), (_, _) = datasets.mnist.load_data()
        # indx = 1-y_train*False
        # x_train = x_train[indx]
        x_train = np.expand_dims(x_train, axis=-1)
    X = x_train.astype('float32')
    X = (X - 127.5) / 127.5
    return X


def plot_sample(samples, nrows, ncols):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            if(DATASET == 'cifar'):
                img = (samples[i*3+j]+1)*127.5/255
                ax[i][j].imshow(img)
            else:
                img = (samples[i*3+j]+1)*127.5
                img = img.squeeze()
                ax[i][j].imshow(img, cmap='gray')
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
            ax[i][j].set_aspect('equal')
    return fig


def create_noise_batch(size):
    return np.random.normal(0., 1., (size, Z_DIM)).astype(np.float32)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def pixel_norm(x, epsilon=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)


def get_weights(shape, fan_in=None, num=1):
    if fan_in is None:
        fan_in = np.sqrt(np.prod(shape[:-1]))
    std = np.sqrt(2) / fan_in
    wscale = tf.constant(np.float32(std))
    return tf.compat.v1.get_variable('weights_%d' % num, shape=shape, initializer=tf.initializers.random_normal()) * wscale


def add_bias(z, num=1):
    b = tf.compat.v1.get_variable('biases_%d' % num, [z.shape[-1]], initializer=tf.constant_initializer(0.))
    if len(z.shape) == 2:
        return z + b
    else:
        return z + tf.reshape(b, [1, 1, 1, -1])


def conv2d(input_, n_filters, k_size, strides=[2, 2], num=1):
    w = get_weights([k_size, k_size, input_.shape[-1].value, n_filters], num=num)
    if strides == [2, 2]:
        return add_bias(tf.nn.conv2d(input_, w, [1, 2, 2, 1], padding='SAME'), num)
    else:
        return add_bias(tf.nn.conv2d(input_, w, [1, 1, 1, 1], padding='SAME'), num)


def conv2dtranspose(input_, n_filters, k_size):
    w = get_weights([k_size, k_size, n_filters, input_.shape[-1].value], k_size*k_size*input_.shape[-1].value)
    os = [tf.shape(input_)[0], input_.shape[1].value * 2, input_.shape[2].value * 2, n_filters]
    return add_bias(tf.nn.conv2d_transpose(input_, w, os, strides=[1, 2, 2, 1], padding='SAME'))


def dense(input_, n_neurons, num=1):
    w = get_weights([input_.shape[1].value, n_neurons], num=num)
    return add_bias(tf.matmul(input_, w), num=num)


def mbstd_layer(input_):
    s = input_.get_shape().as_list()
    y = tf.reshape(input_, [4, -1, s[1], s[2], s[3]])
    y -= tf.reduce_mean(y, axis=0, keepdims=True)
    y = tf.reduce_mean(tf.square(y), axis=0)
    y = tf.sqrt(y + 1e-8)
    y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
    y = tf.tile(y, [4, s[1], s[2], 1])
    return tf.concat([input_, y], axis=3)


def upscale2d(input_):
    s = input_.shape
    x = tf.reshape(input_, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, 2, 1, 2, 1])
    output = tf.reshape(x, [-1, s[1] * 2, s[2] * 2, s[3]])
    return output


def downscale2d(input_):
    ksize = [1, 2, 2, 1]
    return tf.nn.avg_pool(input_, ksize=ksize, strides=ksize, padding='VALID')


def transition(a, b, alpha_trans):
    return (1-alpha_trans) * a + alpha_trans * b


def get_resolutions():
    a = WIDTH
    res = [a]
    while a-int(a) == 0 and a > 2:
        a /= 2
        res = [int(a)] + res
    return res[1:]


def get_filters(res, greatest_number):
    nb_filters = [greatest_number]
    for i in range(len(res)-1):
        nb_filters += [int(nb_filters[-1]/2)]
    return nb_filters


class ClipConstraint(tf.compat.v1.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.compat.v1.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


class Generator():

    def __init__(self, nb_filter_on_greatest_layer, filter_size, res, alphas_transition):
        self.name = 'GAN/Generator'
        self.res = res
        self.lowest_res = self.res[0]
        self.nb_filter = get_filters(self.res, nb_filter_on_greatest_layer)
        self.filter_size = filter_size
        self.outputs = []
        self.summary = 'Generator caracteristics :\n'
        self.alphas_transition = alphas_transition

    def __call__(self, input_noise):
        with tf.compat.v1.variable_scope(self.name):
            self.grow(input_noise, len(self.res)-1)
            return self.outputs

    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if (self.name+'/Gen_conv_block_layer' in var.name or 'To_image_layer_3_2' in var.name)]

    def print_summary(self):
        print(self.summary)

    def block(self, input_, nb_filter, res, num):
        with tf.compat.v1.variable_scope('Gen_conv_block_layer_%d' % num):
            if(res == self.lowest_res):
                x = dense(input_, self.nb_filter[0]*self.lowest_res*self.lowest_res)
                x = tf.reshape(x, [-1, self.lowest_res, self.lowest_res, self.nb_filter[0]])
            else:
                x = conv2dtranspose(input_, nb_filter, self.filter_size)
            x = leaky_relu(x)
            output = pixel_norm(x)
            self.summary += 'Gen_block_layer_%d : %s\n' % (num, str(output))
            return output

    def to_image(self, input_, ind, num):
        with tf.compat.v1.variable_scope('To_image_layer_%d_%d' % (ind, num)):
            output = tf.tanh(conv2d(input_, CHANNEL, self.filter_size, strides=[1, 1]))
            self.summary += 'To_image_layer_%d_%d : %s\n' % (ind, num, str(output))
            return output

    def grow(self, input_, res_ind):
        if res_ind > 0:
            y = self.grow(input_, res_ind-1)
            x = self.block(y, self.nb_filter[res_ind], self.res[res_ind], res_ind+1)
            self.outputs += [transition(self.to_image(upscale2d(y), res_ind+1, 1), self.to_image(x, res_ind+1, 2),
                                        self.alphas_transition[res_ind-1])]
            return x
        else:
            x = self.block(input_, self.nb_filter[res_ind], self.res[res_ind], res_ind+1)
            self.outputs += [self.to_image(x, res_ind, 1)]
            return x


class Critic():
    def __init__(self, nb_filter_on_greatest_layer, filter_size, res, alphas_transition):
        self.name = 'GAN/Critic'
        self.res = res
        self.nb_filter = get_filters(self.res, nb_filter_on_greatest_layer)
        self.filter_size = filter_size
        self.summary = 'Critic caracteristics :\n'
        self.alphas_transition = alphas_transition

    def __call__(self, input_image, reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            x = self.grow(input_image, 0, reuse)
            return self.to_logit(x, reuse)

    def vars(self, layer='all'):

        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]

    def print_summary(self):
        print(self.summary)

    def block(self, input_, nb_filter, num, reuse):
        with tf.compat.v1.variable_scope('Crit_block_layer_%d' % num, reuse=reuse):
            # x = conv2d(input_, nb_filter, self.filter_size, [1, 1], num=1)
            x = conv2d(input_, nb_filter, self.filter_size, [1, 1] if num == 1 else [2, 2], num=2)
            output = leaky_relu(x)
            if not reuse:
                self.summary += 'Crit_block_layer_%d : %s\n' % (num, str(output))
            return output

    def to_logit(self, input_, reuse):
        with tf.compat.v1.variable_scope('To_logit_layer', reuse=reuse):
            x = mbstd_layer(input_)
            x = conv2d(x, input_.shape[3], self.filter_size, [1, 1], num=1)
            x = tf.compat.v1.keras.layers.Flatten()(x)
            output = dense(x, 1, num=2)
            if not reuse:
                self.summary += 'To_logit_layer : %s\n' % str(output)
            return tf.nn.sigmoid(output), output

    def grow(self, input_, res_ind, reuse):
        if res_ind < len(self.res)-1:
            y = self.grow(input_, res_ind+1, reuse)
            x = self.block(y, self.nb_filter[res_ind], len(self.res)-res_ind, reuse)
            return x
        else:
            return self.block(input_, self.nb_filter[res_ind], len(self.res)-res_ind, reuse)


class WGAN():

    def __init__(self, data, check_grad=False, opt='RMSProp', clip=False, do_grad_penalty=True, reset_model=False):

        tf.compat.v1.reset_default_graph()

        self.real_data = shuffle(data)
        self.res = get_resolutions()

        with tf.compat.v1.variable_scope('GAN/Transition'):
            self.alphas_transition = [tf.compat.v1.Variable(1., 'alpha_%d' % i) for i in range(len(self.res)-1)]

        self.g_net = Generator(64, 3, self.res, self.alphas_transition)
        self.c_net = Critic(64, 3, self.res, self.alphas_transition)

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

        self.generator = self.g_net(self.Z)[-1]
        real_output, real_logits = self.c_net(self.X)
        self.g_net.print_summary()
        self.c_net.print_summary()
        fake_output, fake_logits = self.c_net(self.generator, reuse=True)
        self.crit_acc_real = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_output), tf.ones_like(real_output)), tf.float32))
        self.crit_acc_fake = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_output), tf.zeros_like(fake_output)), tf.float32))
        self.crit_loss_real = -tf.reduce_mean(real_logits)
        self.crit_loss_fake = tf.reduce_mean(fake_logits)
        self.gen_loss = -tf.reduce_mean(fake_logits)

        if self.do_grad_penalty:
            epsilon = tf.compat.v1.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.X + (1 - epsilon) * self.generator
            c_hat = self.c_net(x_hat, reuse=True)
            gradients = tf.gradients(c_hat, [x_hat])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
            self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))*10
            self.crit_loss = self.crit_loss_fake + self.crit_loss_real + self.gradient_penalty
        else:
            self.crit_loss = self.crit_loss_fake + self.crit_loss_real

        self.parameters_summary(self.c_net.vars(), self.g_net.vars())

        if self.opt == 'RMSProp':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=5e-5)
        else:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.99)

        self.gen_step = self.optimizer.minimize(self.gen_loss, var_list=self.g_net.vars())
        self.crit_step = self.optimizer.minimize(self.crit_loss, var_list=self.c_net.vars())

        if self.check_grad:
            self.crit_gradients_norm = tf.compat.v1.norm(tf.compat.v1.gradients(self.crit_loss, self.c_net.vars())[0])
            self.gen_gradients_norm = tf.compat.v1.norm(tf.compat.v1.gradients(self.gen_loss, self.g_net.vars())[0])

        if self.clip:
            self.crit_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.c_net.vars()]

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    def parameters_summary(self, c_vars, g_vars):
        print('Critic parameters :')
        nb_params = 0
        for param in c_vars:
            nb_params += np.prod(param.shape.as_list())
            print(param)
        print('Total critic trainable parameters : %d' % int(nb_params))
        nb_params = 0
        print('Generator parameters :')
        for param in g_vars:
            nb_params += np.prod(param.shape.as_list())
            print(param)
        print('Total generator trainable parameters : %d' % int(nb_params))

    def train(self, epochs=100, batch_size=64):
        self.c_losses_real, self.c_losses_fake, self.c_losses, self.g_losses = [], [], [], []
        if self.do_grad_penalty:
            self.grad_penalties = []
        c_iters = 5
        g_iters = 1
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        if 'model.ckpt.meta' in os.listdir('models/'):
            saver.restore(self.sess, "models/model.ckpt")
            print("Model restored...")
        else:
            print("No saved model found...")
        print('Start Training...')
        self.total_batch_of_batches = int(len(self.real_data)//(batch_size*c_iters))
        for epoch in range(epochs):
            print('--------------------Epoch no. %d---------------------' % (epoch+1))
            self.real_data = shuffle(self.real_data)
            for mb in range(self.total_batch_of_batches):
                batch_of_real_batches = self.real_data[mb*batch_size:(mb+c_iters)*batch_size]
                for c_iter in range(c_iters):
                    real_batch = batch_of_real_batches[c_iter*batch_size:(c_iter+1)*batch_size]
                    noise_batch = create_noise_batch(batch_size)
                    if self.clip:
                        self.sess.run(self.crit_clip)
                    self.sess.run(self.crit_step, feed_dict={self.X: real_batch, self.Z: noise_batch})
                for g_iter in range(g_iters):
                    noise_batch = create_noise_batch(batch_size)
                    self.sess.run(self.gen_step, feed_dict={self.Z: noise_batch})

                self.follow_evolution(epoch+1, mb+1, real_batch, noise_batch)
            if epochs > 4:
                if(epoch % int(epochs/4) == 0):
                    self.sample_images(epoch+1, 5, 5)
            else:
                self.sample_images(epoch+1, 5, 5)

        saver.save(self.sess, "models/model.ckpt")
        print('Model saved...')
        self.plot_losses()

    def follow_evolution(self, epoch, mb, real_batch, noise_batch):
        to_print = 'E.%d | BoB.%d/%d' % (epoch, mb, self.total_batch_of_batches)
        c_loss_real, c_loss_fake, c_acc_real, c_acc_fake = self.sess.run([self.crit_loss_real, self.crit_loss_fake, self.crit_acc_real,
                                                                          self.crit_acc_fake], feed_dict={self.X: real_batch, self.Z: noise_batch})
        to_print += ' | car = %d%%, caf = %d%%, clr = %.3f, clf = %.3f' % (int(100*c_acc_real), int(100*c_acc_fake), c_loss_real, c_loss_fake)
        if self.do_grad_penalty:
            g_penalty = self.sess.run(self.gradient_penalty, feed_dict={self.X: real_batch, self.Z: noise_batch})
            c_loss = c_loss_real + c_loss_fake + g_penalty
            self.grad_penalties += [g_penalty]
            to_print += ', cl = %.3f, gp = %.3f' % (c_loss, g_penalty)
        else:
            c_loss = c_loss_real + c_loss_fake
            to_print += ', cl = %.3f' % (c_loss)

        g_loss = self.sess.run(self.gen_loss, feed_dict={self.Z: noise_batch})
        to_print += ' | gl = %.3f' % g_loss

        if self.check_grad:
            g_grad = self.sess.run(self.gen_gradients_norm, feed_dict={self.Z: noise_batch})
            d_grad = self.sess.run(self.crit_gradients_norm, feed_dict={self.X: real_batch, self.Z: noise_batch})
            to_print += ' | cgrad = %.3f, ggrad = %.3f' % (d_grad, g_grad)

        self.c_losses_real += [c_loss_real]
        self.c_losses_fake += [c_loss_fake]
        self.c_losses += [c_loss]
        self.g_losses += [g_loss]

        print(to_print)

    def plot_losses(self):
        plt.figure()
        plt.plot(self.c_losses_real)
        plt.plot(self.c_losses_fake)
        plt.plot(self.g_losses)
        if self.do_grad_penalty:
            plt.plot(self.grad_penalties)
        plt.title('Loss evolution')
        plt.show()

    def sample_images(self, epoch, nrows, ncols):
        noise = create_noise_batch(ncols*nrows)
        gen_imgs = self.sess.run(self.generator, feed_dict={self.Z: noise})
        fig = plot_sample(gen_imgs, nrows, ncols)
        fig.suptitle('Sample generated by the GAN at epoch %d' % epoch)
        fig.savefig('generated/epoch%d.png' % epoch)


if __name__ == '__main__':
    DATASET = 'mnist'
    WIDTH = 28
    HEIGHT = 28
    CHANNEL = 1
    Z_DIM = 10
    gan_test = WGAN(load_database(), check_grad=False, clip=False, opt='Adam', reset_model=True, do_grad_penalty=True)
    gan_test.train(epochs=4, batch_size=64)
