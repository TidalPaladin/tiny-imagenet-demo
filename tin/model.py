"""
This module provides a Tensorflow 2.0 implementation of a vision
classification network for Tiny ImageNet.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from absl import logging

# Some default level args for Resnet
LEVELS_RESNET_18 = [2, 2, 2, 2]
LEVELS_RESNET_50 = [3, 4, 6, 3]
LEVELS_RESNET_101 = [3, 4, 23, 3]
LEVELS_RESNET_152 = [3, 8, 36, 3]

class Tail(layers.Layer):
    """ Basic tail consisting of a 7x7/1 separable convolution """

    def __init__(self, out_width=32, depth_multiplier=3):
        """
        Arguments:
            out_width:  Number of output feature maps. Default 32.

            depth_multiplier:
                        Multiply channels_in by depth_multiplier at the
                        depthwise stage. Default 3.
        """
        super().__init__()
        logging.debug("Created tail, width=%i, multiplier=%i", out_width, depth_multiplier)

        # 7x7/1 separable conv
        # Step 1 - depthwise conv; increase depth by depth_multiplier
        # Step 2 - pointwise conv; further increase depth to out_width
        # No BN / ReLU, handled in later blocks
        # NOTE SeparableConv2D may not respect depth multipier (bug)
        self.conv = layers.SeparableConv2D(
                filters=out_width,
                #depth_multiplier=depth_multiplier,
                name='Tail_conv',
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
        )


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
        _ = self.conv(inputs, **kwargs)
        return _

class Bottleneck(layers.Layer):
    """
    Resnet style residual bottleneck block consisting of:
        1. 1x1/1 pointwise bottleneck convolution (+BN + ReLU)
        2. 3x3/1 separable conv to exit bottleneck (+BN + ReLU)
    """

    def __init__(self, out_width, bottleneck=4):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            bottleneck:
                        Positive integer, factor by which to bottleneck
                        relative to `out_width`. Default 4.
        """
        super().__init__()

        # Width must be integer multiple of bottlneck factor
        assert(out_width / bottleneck == out_width // bottleneck)

        # Pointwise conv, enter bottleneck
        self.conv1 = layers.Conv2D(
                filters=out_width // bottleneck,
                name='Bottleneck_enter',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # 3x3 separable conv
        # Step 1 - depthwise conv; spatial mixing in bottleneck
        # Step 2 - pointwise conv; exit bottleneck
        #
        # Note: Can specify `bottleneck` arg to exit bottleneck
        #       at the depthwise step (rather than pointwise)
        self.conv2 = layers.SeparableConv2D(
                filters=out_width,
                name='Bottleneck_exit',
                kernel_size=3,
                strides=1,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # Merge operation to join residual + main paths
        self.merge = layers.Add()

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
        _ = self.conv1(_)

        # Spatial convolution, depthwise separable
        _ = self.bn2(_, training=training)
        _ = self.relu2(_)
        _ = self.conv2(_)

        # Combine residual and main paths
        return self.merge([inputs, _], **kwargs)

class Downsample(layers.Layer):
    """
    Resnet style residual downsampling block consisting of:
        1. 1x1/1 pointwise bottleneck convolution (+BN + ReLU)
        2. 3x3/2 separable conv to exit bottleneck (with downsampling)
        3. 3x3/2 separable conv along main path (no bottlenecking)
    """

    def __init__(self, out_width, bottleneck=4, stride=2):
        """
        Constructs a downsample block with the final number of output
        feature maps given by `out_width`. Stride of the spatial convolution
        layer is given by `stride`. Take care to increase width appropriately
        for a given spatial downsample.

        The first two convolutions are bottlenecked according to `bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            bottleneck:
                        Positive integer, factor by which to bottleneck
                        relative to `out_width`. Default 4.

            stride:     Positive integer or tuple of positive integers giving
                        the stride of the depthwise separable convolution layer.
                        If a single value, row and col stride will be
                        set to the given value. If a tuple, assign row and
                        col stride from the tuple as (row, col).  Default 2.

        """
        super().__init__()

        # Width must be integer multiple of bottlneck factor
        assert(out_width / bottleneck == out_width // bottleneck)

        # Pointwise conv, enter bottleneck (residual)
        self.channel_conv_1 = layers.Conv2D(
                filters=out_width // bottleneck,
                name='Downsample_enter',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # 3x3 separable conv (residual)
        # Step 1 - depthwise conv; spatial mixing in bottleneck
        # Step 2 - pointwise conv; exit bottleneck
        self.spatial_conv = layers.SeparableConv2D(
                filters=out_width,
                name='Downsample_conv',
                kernel_size=3,
                strides=stride,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # 3x3 separable conv (main)
        # Step 1 - depthwise conv; spatial mixing in bottleneck
        # Step 2 - pointwise conv; exit bottleneck
        self.main = layers.SeparableConv2D(
                filters=out_width,
                name='Downsample_main',
                kernel_size=3,
                strides=stride,
                use_bias=False,
                activation=None,
                padding='same'
        )

        # Merge operation to join residual + main paths
        self.merge = layers.Add()

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

        # BN + ReLU prior to main / residual split
        inputs = self.bn1(inputs, training=training)
        inputs = self.relu1(inputs)

        # Enter bottleneck
        _ = self.channel_conv_1(inputs)

        # Spatial convolution
        _ = self.bn2(_, training=training)
        _ = self.relu2(_)
        _ = self.spatial_conv(_)

        # Main path with convolution
        main = self.main(inputs)

        # Combine residual and main paths
        return self.merge([main, _])

class TinyImageNetHead(layers.Layer):
    """
    Basic vision classification network head consisting of:
        1. 2D Global average pooling
        2. Fully connected layer + bias (no activation)
    """

    @staticmethod
    def get_regularizers(l1, l2):

        # TODO this should all fall under L1L2, but this was giving nan loss
        if l1 and l2:
            regularize = tf.keras.regularizers.L1L2(l1, l2)
            logging.debug("FC l1 regularization=%f", l1)
        elif l1 and not l2:
            regularize = tf.keras.regularizers.l1(l1)
            logging.debug("FC l2 regularization=%f", l1)
        elif l2 and not l1:
            regularize = tf.keras.regularizers.l2(l2)
            logging.debug("FC regularization l1=%f l2=%f", l1, l2)
        else:
            regularize = None
            logging.debug("FC no regularization")

        return regularize

    def __init__(self, num_classes, l1=0.0, l2=0.0, dropout=None, **kwargs):
        """
        Arguments:
            num_classes: Positive integer, number of classes in the output of the
                         fully connected layer.

            l1: Positive float, l1 regularization lambda
            l2: Positive float, l2 regularization lambda
            dropout: Positive float, dropout ratio between GAP and FC layers

        Keyword Arguments:
            Forwarded to the dense layer.
        """
        super(TinyImageNetHead, self).__init__(**kwargs)
        logging.debug("Building head with %i output classes", num_classes)

        self.global_avg = layers.GlobalAveragePooling2D()

        regularize = TinyImageNetHead.get_regularizers(l1, l2)

        if dropout:
            logging.info("Using dropout=%f", dropout)
            self.dropout = layers.Dropout(dropout)
        else:
            logging.info("Not using dropout")
            self.dropout = None

        self.dense = layers.Dense(
                units=num_classes,
                use_bias=True,
                activation=None,
                name='Head_dense',
                kernel_regularizer=regularize,
                bias_regularizer=regularize,
        )

        self.softmax = layers.Softmax()


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
        _ = self.global_avg(inputs)
        _ = self.dropout(_, training=training) if self.dropout else _
        _ = self.dense(_)

        # Apply softmax if not training, otherwise return unscaled logits
        # In training, use softmax + cross entropy with `from_logits=True`
        #   in order to exploit numerically stable result of combined op
        _ = self.softmax(_) if not training else _
        return _

class TinyImageNetMultiStageHead(TinyImageNetHead):
    """
    Similar to TinyImageNetHead, but uses two FC layers. Probably only
    useful where num_features >> num_classes
    """

    def __init__(self, num_classes, l1=0.0, l2=0.0, dropout=None, **kwargs):
        """
        Arguments:
            num_classes: Positive integer, number of classes in the output of the
                         fully connected layer.

            l1: Positive float, l1 regularization lambda
            l2: Positive float, l2 regularization lambda
            dropout: Positive float, dropout ratio between GAP and FC layers

        Keyword Arguments:
            Forwarded to the dense layer.
        """
        super().__init__(num_classes, l1, l2, dropout, **kwargs)

        regularize = TinyImageNetHead.get_regularizers(l1, l2)

        self.dense_pre = layers.Dense(
                units=num_classes*2,
                use_bias=True,
                activation='relu',
                name='Head_dense',
                kernel_regularizer=regularize,
                bias_regularizer=regularize,
        )

        self.dropout2 = layers.Dropout(dropout) if dropout else None

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
        _ = self.global_avg(inputs)
        _ = self.dropout(_, training=training) if self.dropout else _
        _ = self.dense_pre(_)

        _ = self.dropout2(_, training=training) if self.dropout2 else _
        _ = self.dense(_)

        # Apply softmax if not training, otherwise return unscaled logits
        # In training, use softmax + cross entropy with `from_logits=True`
        #   in order to exploit numerically stable result of combined op
        _ = self.softmax(_) if not training else _
        return _

class TinyImageNet(tf.keras.Model):
    """
    Model subclass implementing dominant object detection for Tiny ImageNet.

    Information:
     - Inputs are expected to follow Tensorflow's default channel ordering (`channels_last`).
     - Default parameterizations are aimed at 61 class classification with 64x64x3 inputs.
     - Levels of downsampling and bottleneck repeats per level is parameterized
     - Inclusion of default, custom or no head / tail is parameterized
    """

    NUM_CLASSES = 61

    def __init__(self, levels, use_head=True, use_tail=True, width=32):
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
        super().__init__()

        logging.info("Building TinyImagnet model: levels=%s, width=%i", levels, width)

        # Use default / custom / no tail based on `use_tail`
        if use_tail == True:
            self.tail = Tail(out_width=width)
        elif not use_tail == None:
            self.tail = use_tail
        else:
            self.tail = None

        # Loop through levels and their parameterized repeat counts
        self.blocks = list()
        for level, repeats in enumerate(levels):

            # Create `repeats` Bottleneck blocks and add to the block list
            for block in range(repeats):
                bottleneck_layer = Bottleneck(out_width=width)
                self.blocks.append(bottleneck_layer)

            # Create a downsample layer that doubles width
            # Default stride=2
            downsample_layer = Downsample(out_width=2*width)
            self.blocks.append(downsample_layer)

            # Update `width` for the next iteration
            width *= 2

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


class InceptionMiniConv(layers.Layer):

    def __init__(self, out_width, num_convs):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.
        """
        super().__init__()

        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

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
            relu = layers.ReLU()
            row_conv = layers.DepthwiseConv2D(
                    kernel_size=(3, 1),
                    strides=1,
                    use_bias=False,
                    activation=None,
                    padding='same'
            )
            self.spatial_convs.append(bn)
            self.spatial_convs.append(relu)
            self.spatial_convs.append(row_conv)

            bn = layers.BatchNormalization()
            relu = layers.ReLU()
            col_conv = layers.DepthwiseConv2D(
                    kernel_size=(1, 3),
                    strides=1,
                    use_bias=False,
                    activation=None,
                    padding='same'
            )
            self.spatial_convs.append(bn)
            self.spatial_convs.append(relu)
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
        _ = self.bn1(inputs, training=training)
        _ = self.relu1(_)
        _ = self.bottleneck(_)

        for l in self.spatial_convs:
            if type(l) == layers.BatchNormalization():
                _ = l(_, training=training)
            else:
                _ = l(_)
        return _


class InceptionModule(layers.Layer):
    """
    Resnet style residual bottleneck block consisting of:

    Path 1: 1x1 -> 3x1 depthwise -> 1x3 depthwise (BN + ReLU)
    Path 2: 1x1 -> 3x3 separable -> 3x1 depthwise -> 1x3 depthwise (BN + ReLU)
    Path 3: 3x3 max pool -> 1x1
    Path 4: 1x1 depthwise (main path)

    Output is a concatenation of these paths
    """

    def __init__(self, out_width):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.
        """
        super().__init__()

        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.path1 = [
            InceptionMiniConv(out_width // 8, num_convs=1),
        ]

        self.path2 = [
            InceptionMiniConv(out_width // 8, num_convs=2),
        ]

        self.path3 = [
            layers.Conv2D(
                filters=out_width // 4,
                kernel_size=1,
                use_bias=False,
                activation=None,
                padding='same'
            ),
            layers.MaxPooling2D(pool_size=3, padding='same', strides=1),
        ]

        self.path4 = [
            layers.DepthwiseConv2D(kernel_size=1, use_bias=False, activation=None, padding='same'),
        ]

        self.paths = [self.path1, self.path2, self.path3, self.path4]

        self.merge = layers.Concatenate()

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

        assert(len(pre_concat) == 4)

        # Combine residual and main paths
        return self.merge([out for out in pre_concat], **kwargs)

class TinyInceptionNet(tf.keras.Model):

    NUM_CLASSES = 61

    def __init__(self, levels, use_head=True, use_tail=True, width=32):
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
        super().__init__()

        logging.info("Building TinyImagnet model: levels=%s, width=%i", levels, width)

        # Use default / custom / no tail based on `use_tail`
        if use_tail == True:
            self.tail = Tail(out_width=width)
        elif not use_tail == None:
            self.tail = use_tail
        else:
            self.tail = None

        # Loop through levels and their parameterized repeat counts
        self.blocks = list()
        for level, repeats in enumerate(levels):

            # Create `repeats` Bottleneck blocks and add to the block list
            for block in range(repeats):
                width *= 2
                bottleneck_layer = InceptionModule(out_width=width)
                self.blocks.append(bottleneck_layer)

            downsample_layer = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
            self.blocks.append(downsample_layer)

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
