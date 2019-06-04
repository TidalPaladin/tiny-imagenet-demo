import pytest
import logging
import tensorflow as tf
import tensorflow.keras.layers as layers

LAYERS = [
        layers.ReLU,
        layers.BatchNormalization,
        layers.MaxPool2D,
        layers.Conv2D,
        layers.GlobalAveragePooling2D,
        layers.Dense
]
