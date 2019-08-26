#!python3
import tensorflow as tf
from tin.model import TinyImageNet

levels = [4, 6, 2]
model = TinyImageNet(levels=levels, use_head=True, use_tail=True)

# tf.keras.InputLayer in TF 2.0
inputs = tf.keras.layers.InputLayer(
        input_shape=(64, 64, 3),
        batch_size=32,
        dtype=tf.float32
)
outputs = model(inputs.input)


if __name__ == '__main__':
    model.summary()
