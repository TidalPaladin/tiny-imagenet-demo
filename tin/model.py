"""
This module provides a Tensorflow 2.0 implementation of a vision
classification network for Tiny ImageNet.
"""
from functools import wraps
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class Tail(tf.keras.Model):
    """
    Convolutional tail layer consisting of:
        1. 7x7/2 conv2d + batch norm + ReLU
        2. 3x3/2 max pool
    """

    # Default args for convolution layer
    CONV_ARGS = {
            'kernel_size': (7, 7),
            'strides': 2,
            'use_bias': False,
            'name': 'tail_conv',
    }

    # Default args for max pool layer
    POOL_ARGS = {
            'pool_size': (2, 2),
            'strides': 2,
            'name': 'tail_pool'
    }

    def __init__(self, filters, *args, **kwargs):
        """
        Arguments:
            filters: Number of input feature maps

        Keyword Arguments:
            conv_kwargs: keyword args forwarded to the Conv2D layer
            bn_args: keyword args forwarded to the batch norm layer
            relu_args: keyword args forwarded to the ReLU layer
            pool_args: keyword args forwarded to the MaxPool layer
        """
        # Override default args with those given in __init__ call
        conv_args = Tail.CONV_ARGS
        conv_args.update(kwargs.pop('conv_kwargs', {}))
        conv_args['filters'] = filters

        pool_args = Tail.POOL_ARGS
        pool_args.update(kwargs.pop('pool_kwargs', {}))

        bn_args = kwargs.pop('bn_kwargs', {})
        relu_args = kwargs.pop('relu_kwargs', {})

        # Call parent constructor with *args, **kwargs
        super().__init__(*args, **kwargs)

        self.conv = layers.Conv2D(**conv_args)
        self.bn = layers.BatchNormalization(**bn_args)
        self.relu = layers.ReLU(**relu_args)
        self.pool = layers.MaxPool2D(**pool_args)

    def call(self, input, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input to this layer

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass: Conv2D -> BatchNorm -> ReLU -> MaxPool
        """
        _ = self.conv(inputs, **kwargs)
        _ = self.bn(_, **kwargs)
        _ = self.relu(_, **kwargs)
        return self.pool(_, **kwargs)

class ResnetBasic(tf.keras.Model):
    """
    Base class for a generic residual layer consisting of:
        1. Batch norm + ReLU
        2. Conv2D (with parameterized kernel/filters)
    """

    # Default args for convolution layer
    CONV_ARGS = {
            'kernel_size': (3, 3),
            'strides': 1,
            'padding': 'same',
            'activation': None,
            'use_bias': False,
    }

    def __init__(self, filters, *args, **kwargs):
        """
        Arguments:
            filters: Number of input feature maps

        Keyword Arguments:
            conv_kwargs: keyword args forwarded to the Conv2D layer
            bn_args: keyword args forwarded to the batch norm layer
            relu_args: keyword args forwarded to the ReLU layer
        """
        # Override default args with those given in __init__ call
        conv_args = ResnetBasic.CONV_ARGS
        conv_args.update(kwargs.pop('conv_kwargs', {}))
        conv_args['filters'] = filters

        bn_args = kwargs.pop('bn_kwargs', {})
        relu_args = kwargs.pop('relu_kwargs', {})

        # Call parent constructor with *args, **kwargs
        super().__init__(*args, **kwargs)

        self.batch_norm = layers.BatchNormalization(**bn_args)
        self.relu = layers.ReLU(**relu_args)
        self.conv2d = layers.Conv2D(**conv_args)

    def call(self, input, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input to this layer

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass: Conv2D -> BatchNorm -> ReLU
        """
        _ = self.batch_norm(input, **kwargs)
        _ = self.relu(x, **kwargs)
        return self.conv2d(_, **kwargs)

class Bottleneck(tf.keras.Model):
    """
    Residual bottleneck layer consisting of:

                          |
        +-----------------+
        |                 |
        |                 v
        |   +-------------+------------+
        |   |             Conv2D 1x1/1 |
        |   | BN + ReLU + padding=same |
        |   |             No=Ni/4      |
        |   +-------------+------------+
        |                 |
        |                 v
        |   +-------------+------------+
        |   |             Conv2D 3x3/1 |
        |   | BN + ReLU + padding=same |
        |   |             No=Ni        |
        |   +-------------+------------+
        |                 |
        |                 v
        |   +-------------+------------+
        |   |             Conv2D 1x1/1 |
        |   | BN + ReLU + padding=same |
        |   |             No=4Ni       |
        |   +-------------+------------+
        |                 |
        |                 v
        |              +--+--+
        +------------->+ Add |
                       +--+--+
                          |
                          v
    """
    def __init__(self, Ni, *args, **kwargs):
        """
        Arguments:
            Ni: Number of input feature maps

        Keyword Arguments:
            Forwarded to tf.keras.Model
        """
        super().__init__(*args, **kwargs)

        # Three residual convolution blocks
        kernels = [(1, 1), (3, 3), (1, 1)]
        feature_maps = [Ni // 4, Ni // 4, Ni]
        self.residual_filters = [
            ResnetBasic(N, K)
            for N, K in zip(feature_maps, kernels)
        ]

        # Merge operation
        self.merge = layers.Add()

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res, **kwargs)

        # Combine residual pass with identity
        return self.merge([inputs, res], **kwargs)

class SpecialBottleneck(Bottleneck):
    """
    Special bottleneck layer consisting of:

                                  +
                                  |
                                  v
                 +----------------+----------------+
                 |                                 |
                 v                                 v
    +------------+-------------+     +-------------+------------+
    |             Conv2D 1x1/1 |     |             Conv2D 1x1/1 |
    | BN + ReLU + padding=same |     | BN + ReLU + padding=same |
    |             No=Ni        |     |             No=Ni/4      |
    +------------+-------------+     +-------------+------------+
                 |                                 |
                 |                                 v
                 |                   +-------------+------------+
                 |                   |             Conv2D 3x3/1 |
                 |                   | BN + ReLU + padding=same |
                 |                   |             No=Ni        |
                 |                   +-------------+------------+
                 |                                 |
                 |                                 v
                 |                   +-------------+------------+
                 |                   |             Conv2D 1x1/1 |
                 |                   | BN + ReLU + padding=same |
                 |                   |             No=4Ni       |
                 |                   +-------------+------------+
                 |                                 |
                 |                                 v
                 |                              +--+--+
                 +----------------------------->+ Add |
                                                +--+--+
                                                   |
                                                   v
    """
    def __init__(self, Ni, *args, **kwargs):

        # Layers that also appear in standard bottleneck
        super(SpecialBottleneck, self).__init__(Ni, *args, **kwargs)

        # Add convolution layer along main path
        self.main = layers.Conv2D(
                Ni,
                (1, 1),
                padding='same',
                activation=None,
                use_bias=False)

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res, **kwargs)

        # Convolution on main forward pass
        main = self.main(inputs, **kwargs)

        # Merge residual and main
        return self.merge([main, res])

class Downsample(tf.keras.Model):
    """
    Residual downsampling layer consisting of:

                                  +
                                  |
                                  v
                 +----------------+----------------+
                 |                                 |
                 v                                 v
    +------------+-------------+     +-------------+------------+
    |             Conv2D 1x1/2 |     |             Conv2D 1x1/2 |
    | BN + ReLU +              |     | BN + ReLU +              |
    |             No=Ni/2      |     |             No=Ni/2      |
    +------------+-------------+     +-------------+------------+
                 |                                 |
                 |                                 v
                 |                   +-------------+------------+
                 |                   |             Conv2D 3x3/1 |
                 |                   | BN + ReLU + padding=same |
                 |                   |             No=Ni        |
                 |                   +-------------+------------+
                 |                                 |
                 |                                 v
                 |                   +-------------+------------+
                 |                   |             Conv2D 1x1/1 |
                 |                   | BN + ReLU + padding=same |
                 |                   |             No=2Ni       |
                 |                   +-------------+------------+
                 |                                 |
                 |                                 v
                 |                              +--+--+
                 +----------------------------->+ Add |
                                                +--+--+
                                                   |
                                                   v
    """
    def __init__(self, Ni, *args, **kwargs):
        super(Downsample, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        kernels = [(1, 1), (3, 3), (1, 1)]
        strides = [(2, 2), (1, 1), (1, 1)]
        feature_maps = [Ni // 2, Ni // 2, 2*Ni]

        self.residual_filters = [
            ResnetBasic(N, K, strides=S)
            for N, K, S in zip(feature_maps, kernels, strides)
        ]

        # Convolution on main path
        self.main = ResnetBasic(2*Ni, (1,1), strides=(2,2))

        # Merge operation for residual and main
        self.merge = layers.Add()

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res,**kwargs)

        # Main forward pass
        main = self.main(inputs, **kwargs)

        # Merge residual and main
        return self.merge([main, res])

class Resnet(tf.keras.Model):
    """
    Full network model consisting of the following layers, each
    of which is paramterized in width and kernel size.

        1. Tail
        2. SpecialBottleneck
        3. Some number of bottleneck + downsampling blocks
        4. Global average pooling
        5. Fully connected layer + ReLU
    """

    def __init__(self, classes, filters, levels, *args, **kwargs):
        super(Resnet, self).__init__(*args, **kwargs)

        # Tail
        self.tail = Tail(filters)

        # Special bottleneck layer with convolution on main path
        self.level_0_special = SpecialBottleneck(filters)

        # Lists to hold various layers
        # Note: declare layer lists immediately before filling the list
        # If self.blocks was declared before tail, the tail would appear
        # after all layers in the list when using model.summary()
        self.blocks = list()

        # Loop through levels and their parameterized repeat counts
        for level, repeats in enumerate(levels):
            for block in range(repeats):
                # Append a bottleneck block for each repeat
                name = 'bottleneck_%i_%i' % (level, block)
                layer = Bottleneck(filters, name=name)
                self.blocks.append(layer)

            # Downsample and double feature maps at end of level
            name = 'downsample_%i' % (level)
            layer = Downsample(filters, name=name)
            self.blocks.append(layer)
            filters *= 2

        self.level2_batch_norm = layers.BatchNormalization(name='final_bn')
        self.level2_relu = layers.ReLU(name='final_relu')

        # Decoder - global average pool and fully connected
        self.global_avg = layers.GlobalAveragePooling2D(
                name='GAP'
        )

        # Dense with regularizer, just as a test
        self.dense = layers.Dense(
                classes,
                name='dense',
                # Just for fun, show a regularized layer
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                use_bias=True
        )


    def call(self, inputs, **kwargs):
        x = self.tail(inputs, **kwargs)
        x = self.level_0_special(x)

        # Loop over layers by level
        for layer in self.blocks:
            x = layer(x, **kwargs)

        # Finish up specials in level 2
        x = self.level2_batch_norm(x, **kwargs)
        x = self.level2_relu(x)

        # Decoder
        x = self.global_avg(x)
        return self.dense(x, **kwargs)
