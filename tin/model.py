"""
This module provides a Tensorflow 2.0 implementation of a vision
classification network for Tiny ImageNet.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

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

        # 7x7/1 separable conv
        # Step 1 - depthwise conv; increase depth by depth_multiplier
        # Step 2 - pointwise conv; further increase depth to out_width
        # No BN / ReLU, handled in later blocks
        self.conv = layers.SeparableConv2D(
                filters=out_width,
                depth_multiplier=depth_multiplier,
                name='Tail_conv',
                kernel_size=7,
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

    def __init__(self, out_width, depth_multiplier=4):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // depth_multiplier`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            depth_multiplier:
                        Positive integer, factor by which to bottleneck
                        relative to `out_width`. Default 4.
        """
        super().__init__()

        # Pointwise conv, enter bottleneck
        self.conv1 = layers.SeparableConv2D(
                filters=out_width // depth_multiplier,
                name='Bottleneck_enter',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # 3x3 separable conv
        # Step 1 - depthwise conv; spatial mixing in bottleneck
        # Step 2 - pointwise conv; exit bottleneck
        #
        # Note: Can specify `depth_multiplier` arg to exit bottleneck
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

    def __init__(self, out_width, depth_multiplier=4, stride=2):
        """
        Constructs a downsample block with the final number of output
        feature maps given by `out_width`. Stride of the spatial convolution
        layer is given by `stride`. Take care to increase width appropriately
        for a given spatial downsample.

        The first two convolutions are bottlenecked according to `bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            depth_multiplier:
                        Positive integer, factor by which to bottleneck
                        relative to `out_width`. Default 4.

            stride:     Positive integer or tuple of positive integers giving
                        the stride of the depthwise separable convolution layer.
                        If a single value, row and col stride will be
                        set to the given value. If a tuple, assign row and
                        col stride from the tuple as (row, col).  Default 2.

        """
        super().__init__()

        # Pointwise conv, enter bottleneck (residual)
        self.channel_conv_1 = layers.SeparableConv2D(
                filters=out_width // depth_multiplier,
                name='Downsample_enter',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
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
        self.main = layers.Conv2D(
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

    def __init__(self, num_classes, **kwargs):
        """
        Arguments:
            num_classes: Positive integer, number of classes in the output of the
                         fully connected layer.

        Keyword Arguments:
            Forwarded to the dense layer.
        """
        super(TinyImageNetHead, self).__init__(**kwargs)

        self.global_avg = layers.GlobalAveragePooling2D()

        self.dense = layers.Dense(
                units=num_classes,
                use_bias=True,
                activation=None,
                name='Head_dense',
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

    def __init__(self, levels, use_head=True, use_tail=True):
        """
        Arguments:
            levels: List of positive integers. Each list entry denotes a level of
                    downsampling, with the value of the i'th entry giving the number
                    of times the bottleneck layer is repeated at the i;th level

            use_head: boolean, if true include a default network head

            use_tail: boolean, if true include a default network tail

        Keyword Arguments:
            Forwarded to tf.keras.Model
        """
        super().__init__()
        width = 32

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
            self.head = TinyImageNetHead(num_classes=100)
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
