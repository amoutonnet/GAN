# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:27:21 2019

@author: adamm
"""

import os
import pickle
import shutil
import sys
import re

import keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import zoom
from sklearn.utils import shuffle

plt.close('all')
np.random.seed(0)


def get_resolutions(width):
    res = [width]
    while width-int(width) == 0 and width > 2:
        width /= 2
        res = [int(width)] + res
    return res[1:]


def load_database(dataset='mnist'):
    try:
        with open('datasets/'+dataset+'.pkl', 'rb') as f:
            print('Opening the database...')
            return pickle.load(f)
    except FileNotFoundError:
        print('Downloading and processing database...')
        process_data(dataset)
        print('Done, opening it...')
        with open('datasets/'+dataset+'.pkl', 'rb') as f:
            return pickle.load(f)


def process_data(dataset='mnist'):
    training_set = {}
    if dataset == 'mnist':
        (x_train, y_train), (_, _) = datasets.mnist.load_data()
    else:
        (x_train, y_train), (_, _) = datasets.cifar10.load_data()
    if(len(x_train.shape) == 3):
        x_train = np.expand_dims(x_train, axis=-1)
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    image_shape = x_train.shape[1:]
    res = get_resolutions(x_train.shape[1])[::-1]
    training_set['res_%d' % res[0]] = x_train
    for i, j in enumerate(res[1:]):
        training_set['res_%d' % j] = zoom(x_train, [1, 1/(2**(i+1)), 1/(2**(i+1)), 1], mode='nearest')
    with open('datasets/'+dataset+'.pkl', 'wb') as f:
        pickle.dump([training_set, y_train, image_shape], f)


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


def conv2dtranspose(input_, n_filters, k_size, num=1):
    w = get_weights([k_size, k_size, n_filters, input_.shape[-1].value], k_size*k_size*input_.shape[-1].value, num=num)
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
    return tf.nn.avg_pool2d(input_, ksize=ksize, strides=ksize, padding='VALID')


def transition(a, b, alpha_trans):
    return (1-alpha_trans) * a + alpha_trans * b


def get_filters(res, greatest_number):
    nb_filters = [greatest_number]
    for i in range(len(res)-1):
        nb_filters += [int(nb_filters[-1]/2)]
    return nb_filters


def build_schemes(res, nb_filter, alphas):
    schemes = {}
    for i, j in enumerate(nb_filter):
        if i == 0:
            init_filt = j
            schemes['phase_%d' % (i+1)] = [init_filt, [], None, None]
        else:
            schemes['phase_%d' % (2*i)] = [init_filt, schemes['phase_%d' % (2*i-1)][1], j, alphas[i-1]]
            schemes['phase_%d' % (2*i+1)] = [init_filt, schemes['phase_%d' % (2*i-1)][1]+[j], None, None]
    return schemes


def to_name(tensor):
    to_return = str(tensor).replace(', dtype=float32)', '').replace('Tensor("', '').replace('GAN/Critic', '').replace('GAN/Generator', '')
    if(to_return[0] == '/'):
        return to_return[1:]
    m = re.search(r'\d/', to_return)
    if m:
        return to_return[m.start()+2:]
    else:
        return to_return


class Generator():

    def __init__(self, filter_size, lowest_res):
        self.name = 'GAN/Generator'
        self.filter_size = filter_size
        self.lowest_res = lowest_res
        self.summary = {}
        self.vars = {}

    def __call__(self, input_, scheme, phase):
        self.var_infos = []
        self.vars['phase_%d' % phase] = []
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            x = self.add_initial_block(input_, scheme[0], phase)
            for i, j in enumerate(scheme[1]):
                x = self.add_transitional_block(x, j, phase, 2*i+1)
            if scheme[2]:
                y = self.add_transitional_block(x, scheme[2], phase, phase)
                x = upscale2d(x)
                y = self.add_to_image_block(y, phase, phase, 1)
                x = self.add_to_image_block(x, phase, phase, 2)
                output = self.add_residual_block(x, y, scheme[3], phase)
            else:
                if phase == 0:
                    output = self.add_to_image_block(x, phase, phase, 1)
                else:
                    output = self.add_to_image_block(x, phase, phase-1, 1)
            self.do_summary(phase)
            return output

    def do_summary(self, phase):
        self.summary['phase_%d' % phase] = 'Generator caracteristics :\n\n'
        for i in self.var_infos:
            if len(i) == 2:
                self.summary['phase_%d' % phase] += 'Output : %s | Input : %s\n' % (to_name(i[0]), to_name(i[1]))
            else:
                self.summary['phase_%d' % phase] += 'Output : %s | Input_1 : %s | Input_2 : %s\n' % (to_name(i[0]), to_name(i[1]), to_name(i[2]))

    def get_all_vars(self, phase):
        return self.vars['phase_%d' % phase]

    def print_summary(self, phase):
        print(self.summary['phase_%d' % phase])

    def add_initial_block(self, input_, nb_filter, phase):
        with tf.compat.v1.variable_scope('Gen_initial_block', reuse=tf.compat.v1.AUTO_REUSE):
            x = dense(input_, nb_filter*self.lowest_res*self.lowest_res)
            x = tf.reshape(x, [-1, self.lowest_res, self.lowest_res, nb_filter])
            x = leaky_relu(x)
            output = pixel_norm(x)
            self.var_infos += [[output, input_]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Gen_initial_block/' in var.name]
            return output

    def add_transitional_block(self, input_, nb_filter, phase, num):
        with tf.compat.v1.variable_scope('Gen_transitional_block_%d' % num, reuse=tf.compat.v1.AUTO_REUSE):
            x = conv2dtranspose(input_, nb_filter, self.filter_size, num=1)
            x = leaky_relu(x)
            output = pixel_norm(x)
            self.var_infos += [[output, input_]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Gen_transitional_block_%d/' % num in var.name]
            return output

    def add_residual_block(self, input_1, input_2, alpha, phase):
        with tf.compat.v1.variable_scope('Gen_residual_block_%d' % phase, reuse=tf.compat.v1.AUTO_REUSE):
            output = transition(input_1, input_2, alpha)
            self.var_infos += [[output, input_1, input_2]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Gen_residual_block_%d/' % phase in var.name]
            return output

    def add_to_image_block(self, input_, phase, instance, num=1):
        with tf.compat.v1.variable_scope('Gen_to_image_block_%d_%d' % (instance, num), reuse=tf.compat.v1.AUTO_REUSE):
            output = tf.tanh(conv2d(input_, CHANNEL, self.filter_size, strides=[1, 1], num=num))
            self.var_infos += [[output, input_]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Gen_to_image_block_%d_%d' % (instance, num) in var.name]
            return output


class Critic():
    def __init__(self, filter_size):
        self.name = 'GAN/Critic'
        self.filter_size = filter_size
        self.summary = {}
        self.vars = {}

    def __call__(self, input_, scheme, phase, new_instance=True):
        self.var_infos = []
        self.vars['phase_%d' % phase] = []
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if(phase == 0):
                x = self.add_from_image_block(input_, scheme[0], phase, phase)
            else:
                if(phase % 2 == 1):
                    x = self.add_from_image_block(input_, scheme[1][-1] if not scheme[2] else scheme[2], phase, phase)
                else:
                    x = self.add_from_image_block(input_, scheme[1][-1] if not scheme[2] else scheme[2], phase, phase-1)
            if(scheme[2]):
                y = self.add_transitional_block(x, scheme[2]*2, phase, phase)
                x = self.add_residual_block(x, y, scheme[2]*2, scheme[3], phase)
            for i, j in enumerate(scheme[1][::-1]):
                x = self.add_transitional_block(x, j*2, phase, phase-2*i-1)
            output = self.add_initial_block(x, phase)
            if new_instance:
                self.do_summary(phase)
            return tf.nn.sigmoid(output), output

    def do_summary(self, phase):
        self.summary['phase_%d' % phase] = 'Critic caracteristics :\n\n'
        for i in self.var_infos:
            if len(i) == 2:
                self.summary['phase_%d' % phase] += 'Output : %s | Input : %s\n' % (to_name(i[0]), to_name(i[1]))
            else:
                self.summary['phase_%d' % phase] += 'Output : %s | Input_1 : %s | Input_2 : %s\n' % (to_name(i[0]), to_name(i[1]), to_name(i[2]))

    def get_all_vars(self, phase):
        return self.vars['phase_%d' % phase]

    def print_summary(self, phase):
        print(self.summary['phase_%d' % phase])

    def add_initial_block(self, input_, phase):
        with tf.compat.v1.variable_scope('Crit_initial_block', reuse=tf.compat.v1.AUTO_REUSE):
            x = input_
            x = mbstd_layer(x)
            x = conv2d(x, input_.shape[3], self.filter_size, [1, 1], num=1)
            x = leaky_relu(x)
            x = tf.compat.v1.keras.layers.Flatten()(x)
            output = dense(x, 1, num=2)
            self.var_infos += [[output, input_]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Crit_initial_block/' in var.name]
            return output

    def add_transitional_block(self, input_, nb_filter, phase, num):
        with tf.compat.v1.variable_scope('Crit_transitional_block_%d' % num, reuse=tf.compat.v1.AUTO_REUSE):
            x = conv2d(input_, nb_filter, self.filter_size, num=1)
            output = leaky_relu(x)
            self.var_infos += [[output, input_]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Crit_transitional_block_%d/' % num in var.name]
            return output

    def add_residual_block(self, input_1, input_2, nb_filter, alpha, phase):
        with tf.compat.v1.variable_scope('Crit_residual_block_%d' % phase, reuse=tf.compat.v1.AUTO_REUSE):
            x = conv2d(input_2, nb_filter, self.filter_size, [1, 1], num=2)
            y = downscale2d(input_1)
            y = conv2d(y, nb_filter, self.filter_size, [1, 1], num=3)
            output = leaky_relu(transition(y, x, alpha))
            self.var_infos += [[output, input_1, input_2]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Crit_residual_block_%d/' % phase in var.name]
            return output

    def add_from_image_block(self, input_, nb_filter, phase, instance):
        with tf.compat.v1.variable_scope('Crit_from_image_block_%d' % instance, reuse=tf.compat.v1.AUTO_REUSE):
            x = conv2d(input_, nb_filter, self.filter_size, [1, 1], num=1)
            output = leaky_relu(x)
            self.var_infos += [[output, input_]]
            self.vars['phase_%d' % phase] += [var for var in tf.compat.v1.global_variables() if self.name+'/Crit_from_image_block_%d/' % instance in var.name]
            return output


class WGAN():

    def __init__(self, data, opt='RMSProp', reset_model=False):

        tf.compat.v1.reset_default_graph()

        self.real_data = data
        self.res = get_resolutions(WIDTH)
        self.nb_filter = get_filters(self.res, 64)
        with tf.compat.v1.variable_scope('GAN/Transition'):
            self.alphas_transition = [tf.compat.v1.Variable(1., name='alpha_%d' % (i+1)) for i in range(len(self.res)-1)]
        self.schemes = build_schemes(self.res, self.nb_filter, self.alphas_transition)
        self.g_net = Generator(4, self.res[0])
        self.c_net = Critic(4)

        if opt == 'RMSProp':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=5e-5)
        else:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.99)

        if reset_model:
            list_files = os.listdir('models/')
            if(len(list_files) == 0):
                print('No model found, no reinitialization done...')
            else:
                for i in list_files:
                    os.remove('models/'+i)
                print('Model reinitialized...')

        self.X = [tf.compat.v1.placeholder(tf.float32, [None, i, i, CHANNEL]) for i in self.res]
        self.Z = tf.compat.v1.placeholder(tf.float32, [None, Z_DIM])
        self.phase = {}
        print('Processing phases...')
        for p, i in enumerate(self.schemes.values()):
            print('------------Phase %d------------' % (p+1))
            self.phase['phase_%d' % p] = {}
            if(p == 0):
                input_ = self.X[0]
            else:
                input_ = self.X[round(p/2+0.01)]
            self.phase['phase_%d' % p]['input'] = input_
            self.phase['phase_%d' % p]['res'] = input_.shape[2]
            self.phase['phase_%d' % p]['gen_output'] = self.g_net(self.Z, i, p)
            real_output, real_logits = self.c_net(input_, i, p)
            fake_output, fake_logits = self.c_net(self.phase['phase_%d' % p]['gen_output'], i, p, False)
            self.phase['phase_%d' % p]['crit_acc_real'] = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_output), tf.ones_like(real_output)), tf.float32))
            self.phase['phase_%d' % p]['crit_acc_fake'] = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_output), tf.zeros_like(fake_output)), tf.float32))
            self.phase['phase_%d' % p]['crit_loss_real'] = - tf.reduce_mean(real_logits)
            self.phase['phase_%d' % p]['crit_loss_fake'] = tf.reduce_mean(fake_logits)
            self.phase['phase_%d' % p]['grad_penalty'] = self.get_gradient_penalty(input_, self.phase['phase_%d' % p]['gen_output'], self.c_net, i, p)
            self.phase['phase_%d' % p]['crit_loss'] = self.phase['phase_%d' % p]['crit_loss_real'] + self.phase['phase_%d' % p]['crit_loss_fake'] + self.phase['phase_%d' % p]['grad_penalty']
            self.phase['phase_%d' % p]['gen_loss'] = - tf.reduce_mean(fake_logits)
            self.phase['phase_%d' % p]['gen_step'] = self.optimizer.minimize(self.phase['phase_%d' % p]['gen_loss'], var_list=self.g_net.get_all_vars(p))
            self.phase['phase_%d' % p]['crit_step'] = self.optimizer.minimize(self.phase['phase_%d' % p]['crit_loss'], var_list=self.c_net.get_all_vars(p))
            self.g_net.print_summary(p)
            # for i in self.g_net.get_all_vars(p):
            #     print(i)
            self.c_net.print_summary(p)
            # for i in self.c_net.get_all_vars(p):
            #     print(i)

        print('All phases processed...')

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

        print('GPUs OK...')

    def get_gradient_penalty(self, real, fake, critic, scheme, phase, lamda=10):
        epsilon = tf.compat.v1.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * real + (1 - epsilon) * fake
        _, c_hat = critic(x_hat, scheme, phase, False)
        gradients = tf.gradients(c_hat, x_hat)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        return tf.reduce_mean(tf.square(slopes - 1.0))*lamda

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

    def train(self, g_iterations_per_phase=100, batch_size=64, phase='one'):
        self.c_losses_real, self.c_losses_fake, self.c_losses, self.g_losses, self.grad_penalties = [], [], [], [], []
        c_iters = 5
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        for key in self.real_data.keys():
            self.real_data[key] = shuffle(self.real_data[key])

        print('Checking for saved model...')

        saver = tf.compat.v1.train.Saver()
        if 'model.ckpt.meta' in os.listdir('models/'):
            saver.restore(self.sess, "models/model.ckpt")
            print("Model restored...")
        else:
            print("No saved model found...")

        list_files = os.listdir('generated/')
        if len(list_files) > 0:
            for i in list_files:
                os.remove('generated/'+i)

        print('Start Training...')
        for key, value in self.phase.items():
            if phase == 'one' and key == 'phase_1':
                break
            res = value['res']
            training_set = self.real_data['res_%d' % res]
            for ite in range(g_iterations_per_phase):
                phase_num = int(key[-1])
                if phase_num % 2 == 1:
                    print('TODO:alpha transition')
                for c_iter in range(c_iters):
                    indx = np.random.randint(len(training_set), size=batch_size)
                    real_batch = training_set[indx]
                    noise_batch = create_noise_batch(batch_size)
                    self.sess.run(value['crit_step'], feed_dict={value['input']: real_batch,
                                                                 self.Z: noise_batch})
                noise_batch = create_noise_batch(batch_size)
                self.sess.run(value['gen_step'], feed_dict={self.Z: noise_batch})
                self.follow_evolution(ite+1, key, value, real_batch, noise_batch)
                if (ite+1) % 20 == 0:
                    self.sample_images(ite+1, value['gen_output'], 5, 5)

        saver.save(self.sess, "models/model.ckpt")
        print('Model saved...')
        self.plot_losses()

    def follow_evolution(self, ite, phase, value, real_batch, noise_batch):
        to_print = '%s | Ite.%d' % (phase, ite)
        c_loss_real, c_loss_fake, c_acc_real, c_acc_fake = self.sess.run([value['crit_loss_real'],
                                                                          value['crit_loss_fake'],
                                                                          value['crit_acc_real'],
                                                                          value['crit_acc_fake']],
                                                                         feed_dict={value['input']: real_batch,
                                                                                    self.Z: noise_batch})
        to_print += ' | car = %d%%, caf = %d%%, clr = %.3f, clf = %.3f' % (int(100*c_acc_real), int(100*c_acc_fake), c_loss_real, c_loss_fake)
        g_penalty = self.sess.run(value['grad_penalty'], feed_dict={value['input']: real_batch,
                                                                    self.Z: noise_batch})
        c_loss = c_loss_real + c_loss_fake + g_penalty
        self.grad_penalties += [g_penalty]
        to_print += ', cl = %.3f, gp = %.3f' % (c_loss, g_penalty)

        g_loss = self.sess.run(value['gen_loss'], feed_dict={self.Z: noise_batch})
        to_print += ' | gl = %.3f' % g_loss

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
        plt.plot(self.grad_penalties)
        plt.title('Loss evolution')
        plt.show()

    def sample_images(self, ite, gen, nrows, ncols):
        noise = create_noise_batch(ncols*nrows)
        gen_imgs = self.sess.run(gen, feed_dict={self.Z: noise})
        fig = plot_sample(gen_imgs, nrows, ncols)
        fig.suptitle('Sample generated by the GAN at iteration %d' % ite)
        fig.savefig('generated/iteration%d.png' % ite)


if __name__ == '__main__':
    DATASET = 'mnist'  # Choose between mnist & cifar10
    Z_DIM = 100
    training_set, labels, [WIDTH, HEIGHT, CHANNEL] = load_database(DATASET)
    print('Done, building the model...')
    # gan_test = WGAN(load_database()[0], opt='Adam', reset_model=True)
    # gan_test.train(g_iterations_per_phase=100, batch_size=64)
