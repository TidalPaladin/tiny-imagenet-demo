"""
This module provides a Tensorflow 2.0 implementation of a vision
classification network for Tiny ImageNet based on Inception-Resnet.

NOTES / TODO:

1.  In the Inception Resnet paper, batch norm was omitted after the
    residual + main path additions to reduce memory requirements.
    This may benefit this model by allowing for larger batch sizes at
    a similar memory footprint, which in turn would allow for higher
    learning rates.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tin.resnet import TinyImageNetHead, TinyImageNet

from absl import logging


class InceptionMiniConv(Model):
    """
    A fundamental building block for separable Inception convolutions
    consisting of the following:

        1. 1x1/1 Conv2D
        2. Some repetitions of:
            a. Batch norm
            b. Rx1/1 DepthwiseConv2D (spatial mixing of rows)
            c. 1xC/1 DepthwiseConv2D (spatial mixing of columns)

    Separating row and column convolutions reduces the number of filter
    weights while still covering the same spatial region. For a square
    kernel of dimension F, we will have:

        (standard) -> N_o * N_i * F^2
        (separate) -> 2 * F * N_o * N_i

    Thus for kernels of size > 2 we will reduce parameter counts.

    Similarly, we can achieve the same receptive field size as a NxN
    convolution kernel using a chain of 3x3 convolutions while further
    reducing the number of filter weights.

        (separate) -> 2 * F * N_o * N_i
        (separate decomp) -> (floor(F / 2)+1) * 2 * 3 * N_o * N_i

    Using a 5x5 convolution for F:

        (separate) -> 10 * N_o * N_i
        (separate decomp) -> 2 * 2 * 3 * N_o * N_i
    """
    def __init__(self, out_width, num_convs, kernel=3, **kwargs):
        """
        Arguments:
            out_width:  Positive integer, number of output feature maps.
                        Typically this will be smaller than the input feature
                        map count, representing a bottlneck entry. A 1x1
                        pointwise convolution will follow to exit the bottleneck.


            num_convs:  Positive integer, number of times to repeat the spatial conv

            kernel:  Positive integer, size of the spatial convolution kernel. Default 3.

        """
        super().__init__(**kwargs)

        self.bottleneck = layers.Conv2D(
            filters=out_width,
            kernel_size=1,
            strides=1,
            use_bias=False,
            activation=None,
            padding='same'
        )

        self.spatial_convs = list()

        for i in range(num_convs):

            bn = layers.BatchNormalization()

            row_conv = layers.DepthwiseConv2D(
                kernel_size=(kernel, 1),
                strides=1,
                use_bias=False,
                activation=None,
                padding='same'
            )

            col_conv = layers.DepthwiseConv2D(
                kernel_size=(1, kernel),
                strides=1,
                use_bias=False,
                activation=None,
                padding='same'
            )

            self.spatial_convs.append(bn)
            self.spatial_convs.append(row_conv)
            self.spatial_convs.append(col_conv)

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """

        # Enter bottleneck, depthwise convolution
        _ = self.bottleneck(inputs)

        for l in self.spatial_convs:
            _ = l(_, training=training)
        return _


class InceptionResnetA(Model):
    def __init__(self, out_width, bottleneck=8, **kwargs):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            bottleneck:  Positive integer, bottleneck factor along residual path.
                         Feature map width will be reduced by 1 / bottleneck
        """
        super().__init__(**kwargs)

        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.path1 = [
            InceptionMiniConv(
                out_width // bottleneck,
                num_convs=1,
                name=self.name + '_path1',
            ),
        ]

        self.path2 = [
            InceptionMiniConv(
                out_width // bottleneck,
                num_convs=2,
                name=self.name + '_path2',
            ),
        ]

        self.path3 = [
            layers.Conv2D(
                filters=out_width // bottleneck,
                kernel_size=1,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path3',
            ),
        ]

        self.bn2 = layers.BatchNormalization()
        self.upsample = layers.Conv2D(
            filters=out_width,
            kernel_size=1,
            use_bias=False,
            activation=None,
            padding='same',
            name=self.name + '_up',
        )

        self.paths = [self.path1, self.path2, self.path3]

        self.concat = layers.Concatenate(name=self.name + '_concat')
        self.add = layers.Add(name=self.name + '_add')

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """

        # Enter bottleneck, depthwise convolution
        _ = self.bn1(inputs, training=training)
        _ = self.relu1(_)

        pre_concat = list()
        for path in self.paths:
            out = _
            for l in path:
                out = l(out, training=training)
            pre_concat.append(out)

        concat = self.concat([out for out in pre_concat])
        concat = self.bn2(concat, training=training)
        upsample = self.upsample(concat)

        # Combine residual and main paths
        return self.add([_, upsample])


class InceptionResnetB(InceptionResnetA):
    def __init__(self, out_width, bottleneck=8, **kwargs):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            bottleneck:  Positive integer, bottleneck factor along residual path.
                         Feature map width will be reduced by 1 / bottleneck
        """
        super().__init__(out_width, bottleneck, **kwargs)

        self.path1 = [
            InceptionMiniConv(
                out_width // bottleneck,
                kernel=7,
                num_convs=1,
                name=self.name + '_path1'
            )
        ]

        self.path2 = [
            layers.Conv2D(
                filters=out_width // bottleneck,
                kernel_size=1,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path2'
            ),
        ]

        self.paths = [self.path1, self.path2]


class InceptionResnetC(InceptionResnetB):
    def __init__(self, out_width, bottleneck=8, **kwargs):
        """
        Resnet style residual bottleneck block consisting of:

        Path 1: 1x1 -> 3x1 depthwise -> 1x3 depthwise (BN + ReLU)
        Path 2: 1x1 -> 3x3 separable -> 3x1 depthwise -> 1x3 depthwise (BN + ReLU)
        Path 3: 3x3 max pool -> 1x1
        Path 4: 1x1 depthwise (main path)

        Output is a concatenation of these paths
        """
        super().__init__(out_width, bottleneck, **kwargs)

        self.path1 = [
            InceptionMiniConv(
                out_width // bottleneck,
                num_convs=1,
                name=self.name + '_path1'
            ),
        ]

        self.path2 = [
            layers.Conv2D(
                filters=out_width // bottleneck,
                kernel_size=1,
                use_bias=False,
                activation=None,
                name=self.name + '_path2'
            ),
        ]
        self.paths = [self.path1, self.path2]


class InceptionReductionA(Model):

    # Output of this layer will reduce width by this factor
    WIDTH_FACTOR = 3

    def __init__(self, out_width, **kwargs):
        """
        Arguments:
            out_width:  Positive integer, number of output feature maps.
        """
        super().__init__(**kwargs)

        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.path1 = [
            layers.SeparableConv2D(
                filters=out_width,
                kernel_size=3,
                strides=2,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path1'
            )
        ]

        self.path2 = [
            layers.Conv2D(
                filters=out_width // 2,
                kernel_size=1,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path2_conv1'
            ),
            layers.DepthwiseConv2D(
                kernel_size=3,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path2_conv2'
            ),
            layers.SeparableConv2D(
                filters=out_width,
                kernel_size=3,
                strides=2,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path2_conv3'
            )
        ]

        self.path3 = [
            layers.MaxPooling2D(
                pool_size=3,
                padding='same',
                strides=2,
                name=self.name + '_path3'
            ),
        ]

        self.paths = [self.path1, self.path2, self.path3]
        self.merge = layers.Concatenate(name=self.name + '_concat')

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """

        # Enter bottleneck, depthwise convolution
        _ = self.bn1(inputs, training=training)
        _ = self.relu1(_)

        pre_concat = list()
        for path in self.paths:
            out = _
            for l in path:
                out = l(out, training=training)
            pre_concat.append(out)

        # Combine residual and main paths
        return self.merge([out for out in pre_concat], **kwargs)


class InceptionReductionB(InceptionReductionA):

    # Output of this layer will reduce width by this factor
    WIDTH_FACTOR = 4

    def __init__(self, out_width, **kwargs):
        """
        Arguments:
            out_width:  Positive integer, number of output feature maps.
        """
        super().__init__(out_width, **kwargs)

        self.path1 = [
            layers.SeparableConv2D(
                filters=out_width // 2,
                kernel_size=3,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path1_conv1'
            ),
            layers.SeparableConv2D(
                filters=out_width,
                kernel_size=3,
                strides=2,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path1_conv2'
            )
        ]

        self.path4 = [
            layers.SeparableConv2D(
                filters=out_width // 2,
                kernel_size=3,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path4_conv1'
            ),
            layers.SeparableConv2D(
                filters=out_width,
                kernel_size=3,
                strides=2,
                use_bias=False,
                activation=None,
                padding='same',
                name=self.name + '_path4_conv2'
            )
        ]

        self.paths = [self.path1, self.path2, self.path3, self.path4]


class InceptionTail(Model):
    """
    A slightly simplified version of the Inception stem.
    """
    def __init__(self, out_width=32, **kwargs):
        """
        Arguments:
            out_width:  Number of output feature maps. Default 32.
        """
        super().__init__(**kwargs)

        self.conv1 = layers.SeparableConv2D(
            filters=out_width // 4,
            name=self.name + '_conv1',
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            activation=None,
        )

        self.conv2 = layers.SeparableConv2D(
            filters=out_width // 4,
            name=self.name + '_conv2',
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            activation=None,
        )
        self.pool1 = layers.MaxPooling2D(
            pool_size=3, strides=2, name=self.name + '_pool1', padding='same'
        )
        self.concat1 = layers.Concatenate(name=self.name + '_concat1')

        self.conv3 = InceptionMiniConv(
            out_width // 4,
            num_convs=1,
            kernel=7,
            name=self.name + '_conv3',
        )
        self.conv4 = layers.SeparableConv2D(
            filters=out_width // 4,
            name=self.name + '_conv4',
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            activation=None,
        )
        self.concat2 = layers.Concatenate(name=self.name + '_concat2')

        self.conv5 = layers.SeparableConv2D(
            filters=out_width // 2,
            name=self.name + '_conv5',
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            activation=None,
        )
        self.pool2 = layers.MaxPooling2D(
            pool_size=3, strides=1, name=self.name + '_pool2', padding='same'
        )
        self.concat3 = layers.Concatenate(name=self.name + '_concat3')

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """

        conv1 = self.conv1(inputs)

        conv2 = self.conv2(conv1)
        pool1 = self.pool1(conv1)
        out1 = self.concat1([conv2, pool1])

        conv3 = self.conv3(out1, training=training)
        conv4 = self.conv4(out1)
        out2 = self.concat2([conv3, conv4])

        conv5 = self.conv5(out2)
        pool2 = self.pool2(out2)
        return self.concat3([conv5, pool2])


class TinyInceptionNet(tf.keras.Model):

    NUM_CLASSES = 61

    EXTRACTOR_LEVELS = [InceptionResnetA, InceptionResnetB, InceptionResnetC]

    REDUCTION_LEVELS = [InceptionReductionA, InceptionReductionB, None]

    def __init__(
        self, levels, use_head=True, use_tail=True, width=32, **kwargs
    ):
        """
        Arguments:
            levels: List of positive integers. Each list entry denotes a level of
                    downsampling, with the value of the i'th entry giving the number
                    of times the bottleneck layer is repeated at the i;th level

            use_head: boolean, if true include a default network head

            use_tail: boolean, if true include a default network tail

            width: int, expected number of feature maps at tail output

        Keyword Arguments:
            Forwarded to tf.keras.Model
        """
        super().__init__(**kwargs)

        if len(levels) > 3 or len(levels) <= 0:
            logging.error(
                "Inception must have between 1 and 3 levels: got %i",
                len(levels)
            )

        logging.info(
            "Building InceptionNet model: levels=%s, width=%i", levels, width
        )

        # Use default / custom / no tail based on `use_tail`
        if use_tail == True:
            self.tail = InceptionTail(out_width=width)
        elif not use_tail == None:
            self.tail = use_tail
        else:
            self.tail = None

        # Loop through levels and their parameterized repeat counts
        self.blocks = list()
        for level, repeats in enumerate(levels):

            extractor_type = TinyInceptionNet.EXTRACTOR_LEVELS[level]
            reduction_type = TinyInceptionNet.REDUCTION_LEVELS[level]

            # Create `repeats` Bottleneck blocks and add to the block list
            ident = extractor_type.__name__[-1]
            for block in range(repeats):
                bottleneck = extractor_type(
                    out_width=width, name='incept_%s_%i' % (ident, block + 1)
                )
                self.blocks.append(bottleneck)

            if level <= 1:
                ident = reduction_type.__name__[-1]
                downsample = reduction_type(
                    out_width=width, name='reduct_%s_%i' % (ident, level + 1)
                )
                self.blocks.append(downsample)

                width *= reduction_type.WIDTH_FACTOR

        self.final_bn = layers.BatchNormalization(name='bn')
        self.final_relu = layers.ReLU(name='relu')

        # Use default / custom / no head based on `use_head`
        if use_head == True:
            self.head = TinyImageNetHead(num_classes=TinyImageNet.NUM_CLASSES)
        elif not use_head == None:
            self.head = use_head
        else:
            self.head = None

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """
        _ = self.tail(inputs, training=training) if self.tail else inputs

        # Loop over encoder layers by level
        for layer in self.blocks:
            _ = layer(_, training=training)

        # Finish up BN + ReLU on last level
        _ = self.final_bn(_, training=training)
        _ = self.final_relu(_)

        return self.head(_, training=training, **kwargs) if self.head else _
