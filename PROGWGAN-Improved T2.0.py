# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:27:21 2019

@author: adamm
"""

import os
import pickle

import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import zoom
from sklearn.utils import shuffle

plt.close('all')
np.random.seed(10)


def get_resolutions(width):
    """This function returns a list of the resolutions we 
    will use during the learning of the progressive Gan

    e.g. The image we aim to generate is 28x28, we will
    input 28 to this function and it will return [7,14,28]

    Arguments:
        width {int} -- The width of the normal image

    Returns:
        list -- list of resolutions
    """
    res = [width]
    while width-int(width) == 0 and width > 2:
        width /= 2
        res = [int(width)] + res
    return res[1:]


def load_database(dataset='mnist'):
    """This function try to load the database from the files,
    and if it fails, call an other function downloading and 
    processing the database

    Keyword Arguments:
        dataset {str} -- The name of the dataset we want to use
        (default: {'mnist'})

    Returns:
        list -- a list of the form [training_set, labels, image_shape]
        directly loaded from the files, where training_set is a dictionary containing
        the different training set (with different resolution), labels contains
        the associated labels, and image_shape the shape of the normal image (not reduced)
    """
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
    """This function create a file in the system containing the processed
    database of images, creating lower resolution image from the normal ones.
    To understand the content of this file, see the "load_database" function

    Keyword Arguments:
        dataset {str} -- The name of the dataset we want to use
        (default: {'mnist'})

    Raises:
        ValueError: If the argument is not a known database
    """
    training_set = {}
    if dataset == 'mnist':
        (x_train, y_train), (_, _) = datasets.mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (_, _) = datasets.cifar10.load_data()
    else:
        raise ValueError("The dataset must be cifar10 or mnist.")
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
    """This function returns a matplotlib figure with images plotted
    on it

    Arguments:
        samples {list} -- the list of image we want to plot
        nrows {int} -- the number of rows
        ncols {int} -- the number of colums

    Returns:
        matplotlib.figure -- a fig with the plotted images
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            if(DATASET == 'cifar'):
                img = (samples[i*nrows+j]+1)*127.5/255
                ax[i][j].imshow(img)
            else:
                img = (samples[i*nrows+j]+1)*127.5
                ax[i][j].imshow(img[:, :, 0], cmap='gray')
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
            ax[i][j].set_aspect('equal')
    return fig


def create_noise_batch(size):
    """This function create a batch of noise of size "size"
    and returns it

    Arguments:
        size {int} -- the batch size

    Returns:
        tf.tensor -- the tensor of noise
    """
    return tf.random.normal([size, Z_DIM])


def get_filters(res, greatest_number):
    """This function returns a list of number of filter we will
    use in our convolutionnal layers while growing the Gan

    Arguments:
        res {list} -- the list of resolutions used
        greatest_number {int} -- the greatest number of filter used 
        (for the layer with the smallest resolution)

    Returns:
        list -- a list of number of filter we will
        use in our convolutionnal layers while growing the Gan
    """
    nb_filters = [greatest_number]
    for i in range(len(res)-1):
        nb_filters += [int(nb_filters[-1]/2)]
    return nb_filters


def to_name(tensor):
    """This function create an easier name of tensor for debugging purposes

    Arguments:
        tensor {tensor} -- the tensor we want to process

    Returns:
        string -- an easier name of the given tensor
    """
    return str(tensor).replace(', dtype=float32)', '').replace('Tensor("', '').replace('",', '')


class CustomConv2D(tf.keras.layers.Conv2D):
    """A custom convolutional layer managing equalized learning rate (in development)"""

    def __init__(self, gain=np.sqrt(2), **kwargs):
        super().__init__(kernel_initializer=tf.random_normal_initializer(stddev=0.02), use_bias=False, padding='same', **kwargs)
        self.gain = gain

    def build(self, input_shape):
        super().build(input_shape)
        kernel = tf.keras.backend.get_value(self.kernel)
        scale = self.gain/np.sqrt(np.prod(kernel.shape[:-1]))
        # scale = np.sqrt(np.mean(kernel ** 2))
        tf.keras.backend.set_value(self.kernel, kernel*scale)
        self.scale = self.add_weight(name='scale', shape=scale.shape, trainable=False, initializer='zeros')
        tf.keras.backend.set_value(self.scale, scale)

    def call(self, input_):
        return super().call(input_)


class CustomDense(tf.keras.layers.Dense):
    """A custom dense layer managing equalized learning rate (in development)"""

    def __init__(self, gain=np.sqrt(2), **kwargs):
        super().__init__(kernel_initializer=tf.random_normal_initializer(stddev=0.02), use_bias=False, **kwargs)
        self.gain = gain

    def build(self, input_shape):
        super().build(input_shape)
        kernel = tf.keras.backend.get_value(self.kernel)
        scale = self.gain/np.sqrt(kernel.shape[0])
        # scale = np.sqrt(np.mean(kernel ** 2))
        tf.keras.backend.set_value(self.kernel, kernel*scale)
        self.scale = self.add_weight(name='scale', shape=scale.shape, trainable=False, initializer='zeros')
        tf.keras.backend.set_value(self.scale, scale)

    def call(self, input_):
        return super().call(input_)


class CustomConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    """A custom deconvolutional layer managing equalized learning rate (in development)"""

    def __init__(self, gain=np.sqrt(2), **kwargs):
        super().__init__(kernel_initializer=tf.random_normal_initializer(stddev=0.02), use_bias=False, padding='same', **kwargs)
        self.gain = gain

    def build(self, input_shape):
        super().build(input_shape)
        kernel = tf.keras.backend.get_value(self.kernel)
        scale = self.gain/np.sqrt(self.kernel_size[0]*self.kernel_size[0]*kernel.shape[-1])
        # scale = np.sqrt(np.mean(kernel ** 2))
        tf.keras.backend.set_value(self.kernel, kernel*scale)
        self.scale = self.add_weight(name='scale', shape=scale.shape, trainable=False, initializer='zeros')
        tf.keras.backend.set_value(self.scale, scale)

    def call(self, input_):
        return super().call(input_)


class AddBiasLayer(tf.keras.layers.Layer):
    """A custom layer adding biases"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        super().build(input_shape)

    def call(self, input_, **kwargs):
        input_ = tf.nn.bias_add(input_, self.bias)
        return input_


class PixelNorm(tf.keras.layers.Layer):
    """A custom layer for the pixelwise feature vector normalization"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_):
        return input_ * tf.math.rsqrt(tf.reduce_mean(tf.square(input_), axis=3, keepdims=True) + 1e-8)


class MbstdLayer(tf.keras.layers.Layer):
    """A custom layer for the mini batch standard deviation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_):
        s = input_.get_shape().as_list()
        y = tf.reshape(input_, [4, -1, s[1], s[2], s[3]])
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        y = tf.tile(y, [4, s[1], s[2], 1])
        return tf.concat([input_, y], axis=3)


class TransitionLayer(tf.keras.layers.Add):
    """A custom layer to add the output of two layers"""

    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _merge_function(self, inputs):
        return (1-self.alpha) * inputs[0] + self.alpha * inputs[1]


class UpScale2D(tf.keras.layers.Layer):
    """A custom layer to upscale (resolution*2) an image"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_):
        s = input_.shape
        x = tf.reshape(input_, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, 2, 1, 2, 1])
        return tf.reshape(x, [-1, s[1] * 2, s[2] * 2, s[3]])


class DownScale2D(tf.keras.layers.Layer):
    """A custom layer to downscale (resolution/2) an image"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_):
        ksize = [1, 2, 2, 1]
        return tf.nn.avg_pool2d(input_, ksize=ksize, strides=ksize, padding='VALID')


class Block():
    """Block class from which all block classes will inherit"""

    def __init__(self, biases=False):
        self.infos = None
        self.biases = biases

    def get_infos(self):
        return self.infos


class GenInitialBlock(Block):
    """The initial block of the generator, the one taking the noise as input and processing it
    with dense and reshape layer, and finally with a convolution layer"""

    def __init__(self, nb_filter, lowest_res, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = CustomDense(gain=np.sqrt(2)/4, units=nb_filter*lowest_res*lowest_res)
        self.addbias_1 = AddBiasLayer()
        self.reshape_1 = tf.keras.layers.Reshape((lowest_res, lowest_res, nb_filter))
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.pixel_norm_1 = PixelNorm()

    def __call__(self, input_):
        x = self.dense_1(input_)
        x = self.addbias_1(x)
        x = self.reshape_1(x)
        x = self.act_1(x)
        output = self.pixel_norm_1(x)
        if not self.infos:
            self.infos = [output, input_]
        return output


class GenBlock(Block):
    """A block of the generator, with a deconvolution, a bias, an activation and a pixel normalization"""

    def __init__(self, nb_filter, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.conv2dtranspose_1 = CustomConv2DTranspose(filters=nb_filter, kernel_size=filter_size, strides=2)
        self.addbias_1 = AddBiasLayer()
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.pixel_norm_1 = PixelNorm()

    def __call__(self, input_):
        x = self.conv2dtranspose_1(input_)
        x = self.addbias_1(x)
        x = self.act_1(x)
        output = self.pixel_norm_1(x)
        if not self.infos:
            self.infos = [output, input_]
        return output


class GenResidualBlock(Block):
    """A residual block of the generator, the one used for introducing progressively a new block 
    when growing the Gan"""

    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.transition_1 = TransitionLayer(alpha)

    def __call__(self, input_1, input_2):
        output = self.transition_1([input_1, input_2])
        if not self.infos:
            self.infos = [output, input_1, input_2]
        return output


class ToRGB(Block):
    """The ToRGB block of the paper"""

    def __init__(self, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = CustomConv2D(gain=1, filters=CHANNEL, kernel_size=filter_size, strides=1)
        self.addbias_1 = AddBiasLayer()
        self.act_1 = tf.keras.layers.Activation('tanh')

    def __call__(self, input_):
        x = self.conv_1(input_)
        x = self.addbias_1(x)
        output = self.act_1(x)
        if not self.infos:
            self.infos = [output, input_]
        return output


class FromRGB(Block):
    """The FromRGB block of the paper"""

    def __init__(self, nb_filter, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = CustomConv2D(filters=nb_filter, kernel_size=filter_size, strides=1)
        self.addbias_1 = AddBiasLayer()
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def __call__(self, input_):
        x = self.conv_1(input_)
        x = self.addbias_1(x)
        output = self.act_1(x)
        if not self.infos:
            self.infos = [output, input_]
        return output


class CritBlock(Block):
    """A block of the critic, with a convolution, a bias and an activation"""

    def __init__(self, nb_filter, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = CustomConv2D(filters=nb_filter, kernel_size=filter_size, strides=2)
        self.addbias_1 = AddBiasLayer()
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def __call__(self, input_):
        x = self.conv_1(input_)
        x = self.addbias_1(x)
        output = self.act_1(x)
        if not self.infos:
            self.infos = [output, input_]
        return output


class CritInitialBlock(Block):
    """The initial block of the critic, taking the output of the fromRGB block or 
    the output of the previous layer and returning a logit. It uses the mini batch
    standard deviation layer."""

    def __init__(self, nb_filter, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.mbstd = MbstdLayer()
        self.conv_1 = CustomConv2D(filters=nb_filter, kernel_size=filter_size, strides=1)
        self.addbias_1 = AddBiasLayer()
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = CustomDense(gain=1, units=1)
        self.addbias_2 = AddBiasLayer()

    def __call__(self, input_):
        x = self.mbstd(input_)
        x = self.conv_1(x)
        x = self.addbias_1(x)
        x = self.act_1(x)
        x = self.flat(x)
        x = self.dense_1(x)
        output = self.addbias_2(x)
        if not self.infos:
            self.infos = [output, input_]
        return output


class CritResidualBlock(Block):
    """A residual block of the critic, the one used for introducing progressively a new block 
    when growing the Gan"""

    def __init__(self, nb_filter, filter_size, alpha, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = CustomConv2D(filters=nb_filter, kernel_size=filter_size, strides=1)
        self.addbias_1 = AddBiasLayer()
        self.downscale_1 = DownScale2D()
        self.conv_2 = CustomConv2D(filters=nb_filter, kernel_size=filter_size, strides=1)
        self.addbias_2 = AddBiasLayer()
        self.transition_1 = TransitionLayer(alpha)
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def __call__(self, input_1, input_2):
        y = self.conv_1(input_2)
        y = self.addbias_1(y)
        x = self.downscale_1(input_1)
        x = self.conv_2(x)
        x = self.addbias_2(x)
        x = self.transition_1([x, y])
        output = self.act_1(x)
        if not self.infos:
            self.infos = [output, input_1, input_2]
        return output


class Generator():
    """The generator class, used to build the generator, assigning each block to the right place
    and managing the residual blocks."""

    def __init__(self, filter_size, filters, res, alphas, **kwargs):
        super().__init__(**kwargs)
        self.summary = {}
        self.init_block = GenInitialBlock(filters[0], res[0])
        self.torgb_blocks = [ToRGB(filter_size) for i in range(2*len(res)-1)]
        self.blocks = [GenBlock(i, filter_size) for i in filters[1:]]
        self.res_blocks = [GenResidualBlock(i) for i in alphas]
        self.models = [self.init_block] + self.torgb_blocks + self.blocks + self.res_blocks

    def __call__(self, input_, phase):
        self.var_infos = []
        x = self.init_block(input_)
        self.var_infos += [self.init_block.get_infos()]
        if phase == 0:
            output = self.torgb_blocks[0](x)
            self.var_infos += [self.torgb_blocks[0].get_infos()]
        else:
            if(phase % 2 == 1):
                for i in range(int((phase-1)/2)):
                    x = self.blocks[i](x)
                    self.var_infos += [self.blocks[i].get_infos()]
                y = self.blocks[int((phase-1)/2)](x)
                self.var_infos += [self.blocks[int((phase-1)/2)].get_infos()]
                y = self.torgb_blocks[phase+1](y)
                self.var_infos += [self.torgb_blocks[phase+1].get_infos()]
                x = UpScale2D()(x)
                x = self.torgb_blocks[phase](x)
                self.var_infos += [self.torgb_blocks[phase].get_infos()]
                output = self.res_blocks[int((phase-1)/2)](x, y)
                self.var_infos += [self.res_blocks[int((phase-1)/2)].get_infos()]
            else:
                for i in range(int(phase/2)):
                    x = self.blocks[i](x)
                    self.var_infos += [self.blocks[i].get_infos()]
                output = self.torgb_blocks[phase](x)
                self.var_infos += [self.torgb_blocks[phase].get_infos()]
        self.do_summary(phase)
        return tf.keras.Model(inputs=input_, outputs=output)

    def do_summary(self, phase):
        self.summary['phase_%d' % phase] = '\nGenerator caracteristics :\n\n'
        for i in self.var_infos:
            if len(i) == 2:
                self.summary['phase_%d' % phase] += 'Output : %s | Input : %s\n' % (to_name(i[0]), to_name(i[1]))
            else:
                self.summary['phase_%d' % phase] += 'Output : %s | Input_1 : %s | Input_2 : %s\n' % (to_name(i[0]), to_name(i[1]), to_name(i[2]))

    def print_summary(self, phase):
        print(self.summary['phase_%d' % phase])


class Critic():
    """The critic class, used to build the critic, assigning each block to the right place
    and managing the residual blocks."""

    def __init__(self, filter_size, filters, res, alphas, **kwargs):
        super().__init__(**kwargs)
        self.summary = {}
        self.init_block = CritInitialBlock(filters[0], res[0])
        self.fromrgb_blocks = [FromRGB(i, filter_size) for i in filters]
        self.blocks = [CritBlock(i, filter_size) for i in filters[:-1]]
        self.res_blocks = [CritResidualBlock(j, filter_size, i) for i, j in zip(alphas, filters[:-1])]
        self.models = [self.init_block] + self.fromrgb_blocks + self.blocks + self.res_blocks

    def __call__(self, input_, phase, new_instance=True):
        self.var_infos = []
        if phase == 0:
            x = self.fromrgb_blocks[0](input_)
            self.var_infos += [self.fromrgb_blocks[0].get_infos()]
        else:
            if(phase % 2 == 1):
                x = self.fromrgb_blocks[int((phase+1)/2)](input_)
                self.var_infos += [self.fromrgb_blocks[int((phase+1)/2)].get_infos()]
                y = self.blocks[int((phase-1)/2)](x)
                self.var_infos += [self.blocks[int((phase-1)/2)].get_infos()]
                x = self.res_blocks[int((phase-1)/2)](x, y)
                self.var_infos += [self.res_blocks[int((phase-1)/2)].get_infos()]
                for i in range(int((phase-1)/2)):
                    x = self.blocks[int((phase-1)/2)-i-1](x)
                    self.var_infos += [self.blocks[int((phase-1)/2)-i-1].get_infos()]
            else:
                x = self.fromrgb_blocks[int(phase/2)](input_)
                self.var_infos += [self.fromrgb_blocks[int(phase/2)].get_infos()]
                for i in range(int(phase/2)):
                    x = self.blocks[int(phase/2)-i-1](x)
                    self.var_infos += [self.blocks[int(phase/2)-i-1].get_infos()]
        output = self.init_block(x)
        self.var_infos += [self.init_block.get_infos()]
        if new_instance:
            self.do_summary(phase)
        return tf.keras.Model(inputs=input_, outputs=[tf.nn.sigmoid(output), output])

    def do_summary(self, phase):
        self.summary['phase_%d' % phase] = '\nCritic caracteristics :\n\n'
        for i in self.var_infos:
            if len(i) == 2:
                self.summary['phase_%d' % phase] += 'Output : %s | Input : %s\n' % (to_name(i[0]), to_name(i[1]))
            else:
                self.summary['phase_%d' % phase] += 'Output : %s | Input_1 : %s | Input_2 : %s\n' % (to_name(i[0]), to_name(i[1]), to_name(i[2]))

    def print_summary(self, phase):
        print(self.summary['phase_%d' % phase])


class ProgWGAN():
    """ The class building the Gan, managing the growing process splitting it
    into phases, and each phase will teach only some parts of the generator
    and the critic, with some phases during which the Gan will grow, making the
    alpha shift progressively from 0 to 1"""

    def __init__(self, data, opt='RMSProp', reset_model=False, summary=True):

        self.real_data = data
        self.res = get_resolutions(WIDTH)
        self.nb_filter = get_filters(self.res, 64)
        self.alphas_transition = [tf.Variable(1., name='alpha_%d' % (i+1)) for i in range(len(self.res)-1)]
        self.g_net = Generator(4, self.nb_filter, self.res, self.alphas_transition)
        self.c_net = Critic(4, self.nb_filter, self.res, self.alphas_transition)

        if opt == 'RMSProp':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
        else:
            self.optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)

        if reset_model:
            list_files = os.listdir('models/')
            if(len(list_files) == 0):
                print('No model found, no reinitialization done...')
            else:
                for i in list_files:
                    os.remove('models/'+i)
                print('Model reinitialized...')

        self.X = [tf.keras.layers.Input(shape=(i, i, CHANNEL)) for i in self.res]
        self.Z = tf.keras.layers.Input(shape=(Z_DIM,))
        self.phase = {}
        print('Processing phases...')
        for p in range(len(self.res)*2-1):
            self.phase['phase_%d' % p] = {}
            if(p == 0):
                input_ = self.X[0]
            else:
                input_ = self.X[round(p/2+0.01)]
            self.phase['phase_%d' % p]['input'] = input_
            self.phase['phase_%d' % p]['phase'] = p
            self.phase['phase_%d' % p]['res'] = input_.shape[1]
            self.phase['phase_%d' % p]['gen_model'] = self.g_net(self.Z, p)
            self.phase['phase_%d' % p]['gen_trainable_variables'] = [i for i in self.phase['phase_%d' % p]['gen_model'].trainable_variables if 'alpha' not in i.name]
            self.phase['phase_%d' % p]['crit_model'] = self.c_net(input_, p)
            self.phase['phase_%d' % p]['crit_trainable_variables'] = [i for i in self.phase['phase_%d' % p]['crit_model'].trainable_variables if 'alpha' not in i.name]

            if summary:
                print('\n------------Phase %d------------' % (p+1))
                self.g_net.print_summary(p)
                print('Generator trainable parameters\n')
                for i in self.phase['phase_%d' % p]['gen_trainable_variables']:
                    print(i.name)
                self.c_net.print_summary(p)
                print('Critic trainable parameters\n')
                for i in self.phase['phase_%d' % p]['crit_trainable_variables']:
                    print(i.name)
        print('\nAll phases processed...')

    def gen_train_step(self, noise_batch, phase):
        with tf.GradientTape() as t:
            fake_batch = self.phase[phase]['gen_model'](noise_batch)
            _, fake_logits = self.phase[phase]['crit_model'](fake_batch)
            gen_loss = - tf.reduce_mean(fake_logits)
        gen_gradients = t.gradient(gen_loss, self.phase[phase]['gen_trainable_variables'])
        self.optimizer.apply_gradients(zip(gen_gradients, self.phase[phase]['gen_trainable_variables']))
        return gen_loss

    def crit_train_step(self, real_batch, noise_batch, phase):
        with tf.GradientTape() as t:
            fake_batch = self.phase[phase]['gen_model'](noise_batch)
            real_output, real_logits = self.phase[phase]['crit_model'](real_batch)
            fake_output, fake_logits = self.phase[phase]['crit_model'](fake_batch)

            def _interpolate(a, b):
                alpha = tf.random.uniform(shape=[], minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.shape)
                return inter

            x_hat = _interpolate(real_batch, fake_batch)
            with tf.GradientTape() as g_pen_tape:
                g_pen_tape.watch(x_hat)
                _, c_hat = self.phase[phase]['crit_model'](x_hat)
            grad = g_pen_tape.gradient(c_hat, x_hat)
            norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
            grad_penalty = tf.reduce_mean((norm - 1.)**2)*10

            crit_loss_real = - tf.reduce_mean(real_logits)
            crit_loss_fake = tf.reduce_mean(fake_logits)
            crit_loss = crit_loss_real + crit_loss_fake + grad_penalty

            crit_acc_real = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_output), tf.ones_like(real_output)), tf.float32))
            crit_acc_fake = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_output), tf.zeros_like(fake_output)), tf.float32))

        disc_gradients = t.gradient(crit_loss, self.phase[phase]['crit_trainable_variables'])
        self.optimizer.apply_gradients(zip(disc_gradients, self.phase[phase]['crit_trainable_variables']))

        return crit_loss_real, crit_loss_fake, crit_loss, grad_penalty, int(100*crit_acc_real), int(100*crit_acc_fake)

    def train(self, g_iterations_per_phase=100, batch_size=64, phase='one'):
        self.c_losses_real, self.c_losses_fake, self.c_losses, self.g_losses, self.grad_penalties = [], [], [], [], []
        c_iters = 5
        for key in self.real_data.keys():
            self.real_data[key] = shuffle(self.real_data[key])

        list_files = os.listdir('generated/')
        if len(list_files) > 0:
            for i in list_files:
                os.remove('generated/'+i)

        print('Start Training...')
        for key, value in self.phase.items():
            phase_num = value['phase']
            if phase_num == phase:
                break
            res = value['res']
            training_set = self.real_data['res_%d' % res]
            for ite in range(g_iterations_per_phase*c_iters):
                gen_ite = ite//c_iters+1
                indx = np.random.randint(len(training_set), size=batch_size)
                real_batch = training_set[indx]
                if phase_num % 2 == 1:
                    self.alphas_transition[int((phase_num-1)/2)]+1/(g_iterations_per_phase*c_iters)
                noise_batch = create_noise_batch(batch_size)
                c_loss_real, c_loss_fake, c_loss, g_penalty, c_acc_real, c_acc_fake = self.crit_train_step(real_batch, noise_batch, key)
                if ite % c_iters == 0:
                    g_loss = self.gen_train_step(noise_batch, key)
                    to_print = '%s | Ite.%d' % (key, gen_ite)
                    self.c_losses_real += [c_loss_real]
                    self.c_losses_fake += [c_loss_fake]
                    self.c_losses += [c_loss]
                    self.grad_penalties += [g_penalty]
                    self.g_losses += [g_loss]
                    to_print += ' | car = %d%%, caf = %d%%, clr = %.3f, clf = %.3f' % (c_acc_real, c_acc_fake, c_loss_real, c_loss_fake)
                    to_print += ', cl = %.3f, gp = %.3f' % (c_loss, g_penalty)
                    to_print += ' | gl = %.3f' % g_loss
                    print(to_print)
                if (gen_ite) % 100 == 0:
                    self.sample_images(gen_ite, phase_num, value['gen_model'], 5, 5)

        print('Model saved...')
        self.plot_losses()

    def plot_losses(self):
        plt.figure()
        plt.plot(self.c_losses_real)
        plt.plot(self.c_losses_fake)
        plt.plot(self.g_losses)
        plt.plot(self.grad_penalties)
        plt.title('Loss evolution')
        plt.savefig('generated/loss.png')

    def sample_images(self, ite, phase_num, gen, nrows, ncols):
        noise = create_noise_batch(ncols*nrows)
        gen_imgs = gen(noise)
        fig = plot_sample(gen_imgs, nrows, ncols)
        fig.suptitle('Sample generated by the GAN at iteration %d' % ite)
        fig.savefig('generated/iteration%d_phase%d.png' % (ite, phase_num))


if __name__ == '__main__':
    DATASET = 'mnist'  # Choose between mnist & cifar10
    Z_DIM = 100  # Size of the noise
    training_set, labels, [WIDTH, HEIGHT, CHANNEL] = load_database(DATASET)  # Loading the dataset
    print('Done, building the model...')
    gan_test = ProgWGAN(training_set, opt='Adam', reset_model=True, summary=False)  # Creating the progressive Gan
    gan_test.train(g_iterations_per_phase=200, batch_size=64, phase=3)  # Training it
