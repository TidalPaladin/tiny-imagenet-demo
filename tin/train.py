#!python3
import os
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger
from tensorflow.data.experimental import AUTOTUNE

from pathlib import Path
from glob import glob as glob_func
from util import *
from model import *

from tensorboard.plugins.hparams import api as hp
import tensorflow.feature_column as fc
from tensorflow.keras.layers import *

from absl import app, logging
from flags import FLAGS

from tensorflow.keras.preprocessing.image import ImageDataGenerator

callbacks = list()

def preprocess():

    logging.info("Reading images from: %s", FLAGS.src)

    # TODO get all these params from FLAGS
    # Create generator to read training images w/ preprocessing
    datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=10,
            rescale=1./255,
            width_shift_range=10,
            height_shift_range=10,
            brightness_range=None,
            horizontal_flip=True,
            data_format=None,
            validation_split=0.01,
    )

    train_generator = datagen.flow_from_directory(
            FLAGS.src,
            subset='training',
            target_size=(64, 64),
            class_mode='sparse',
            batch_size=FLAGS.batch_size
    )
    val_generator = datagen.flow_from_directory(
            FLAGS.src,
            subset='validation',
            target_size=(64, 64),
            class_mode='sparse',
            batch_size=FLAGS.batch_size
    )

    return train_generator, val_generator


def construct_model():

    # Model
    logging.debug("Running with num classes: %i", FLAGS.classes)
    head = TinyImageNetHead(num_classes=FLAGS.classes, name='head')
    model = TinyImageNet(levels=FLAGS.levels, use_head=head, use_tail=True)

    return model

def train_model(model, train, validate, initial_epoch):
    """
    Compiles `model` with metrics / loss / optimizer and
    begins training on `train` / `validate` Datasets starting
    at `initial_epoch`
    """

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.lr,
            epsilon=0.1
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    fit_args = {
        'generator': train,
        'epochs': FLAGS.epochs,
        'validation_data': validate,
        'callbacks': callbacks,
        'initial_epoch': initial_epoch
    }
    pretty_args = json.dumps({k: str(v) for k, v in fit_args.items()}, indent=2)

    logging.info("Fitting model with args: \n%s", pretty_args)
    history = model.fit_generator(**fit_args)

def main(argv):

    image_shape = (FLAGS.image_dim, FLAGS.image_dim, 3)
    inputs = layers.Input(shape=image_shape, dtype=tf.float32)

    FLAGS.levels = [int(x) for x in FLAGS.levels]
    model = construct_model()
    outputs = model(inputs)

    if FLAGS.summary:
        out_path = os.path.join(FLAGS.artifacts_dir, 'summary.txt')
        model.summary()
        save_summary(model, out_path)
        logging.info("Exiting after printing model summary")
        sys.exit(0)

    checkpoint_dir = os.path.join(FLAGS.artifacts_dir, 'checkpoint')
    clean_empty_dirs(checkpoint_dir)

    initial_epoch = 0
    resume_file = FLAGS.resume if FLAGS.resume else None

    if FLAGS.resume_last:
        # Find directory of latest run
        chkpt_path = Path(FLAGS.artifacts_dir, 'checkpoint')
        latest_path = sorted(list(chkpt_path.glob('20*')))[-1]
        logging.info("Using latest run - %s", latest_path)

        # Find latest checkpoint file
        latest_checkpoint = sorted(list(latest_path.glob('*.hdf5')))[-1]
        FLAGS.resume = str(latest_checkpoint.resolve())

    if FLAGS.resume:
        logging.info("Loading weights from file %s", FLAGS.resume)
        model.load_weights(FLAGS.resume)
        initial_epoch = int(re.search('([0-9]*)\.hdf5', FLAGS.resume).group(1))
        logging.info("Starting from epoch %i", initial_epoch+1)

    global callbacks
    callbacks = get_callbacks(FLAGS) if not FLAGS.dry else []

    train, validate = preprocess()
    train_model(model, train, validate, initial_epoch)

if __name__ == '__main__':
  app.run(main)
