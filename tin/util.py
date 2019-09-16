#!python3
"""
Provides utility functions for Tiny ImageNet training pipeline
"""

import os
import sys
import itertools
from pathlib import Path

import os
from enum import Enum
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger
import tensorflow.feature_column as fc
from tensorboard.plugins.hparams import api as hp
from absl import logging

# Used for directory / file names
DATE = datetime.now().strftime("%Y%m%d-%H%M%S")

def get_callbacks(FLAGS):
    """ Gets model callbacks based on CLI flags """
    callbacks = list()

    # Join checkpoint / Tensorboard log directories
    checkpoint_dir = os.path.join(FLAGS.artifacts_dir, 'checkpoint', DATE)
    logging.info("Model checkpoint dir: %s", checkpoint_dir)
    tb_dir = os.path.join(FLAGS.artifacts_dir, 'tblogs')
    logging.info("Tensorboard log dir: %s", tb_dir)


    # Reduce learning rate if loss isn't improving
    learnrate_args = {
            'monitor':'loss',
            'factor': 0.5,
            'patience': 3,
            'min_lr': 0.0001
    }
    logging.info("ReduceLROnPlateau: %s", learnrate_args)
    learnrate_cb = tf.keras.callbacks.ReduceLROnPlateau(**learnrate_args)
    callbacks.append(learnrate_cb)

    # Stop early if loss isn't improving
    stopping_args = {
            'monitor':'loss',
            'min_delta': 0.001,
            'patience': 5,
    }
    logging.info("EarlyStopping: %s", stopping_args)
    stopping_cb = tf.keras.callbacks.EarlyStopping(**stopping_args)
    callbacks.append(stopping_cb)

    # Decay LR exponentially over epochs
    def scheduler(epoch):
        if epoch < FLAGS.lr_decay_freq:
            result = FLAGS.lr
        else:
            result = FLAGS.lr * (FLAGS.lr_decay_coeff ** (epoch - FLAGS.lr_decay_freq))
        logging.info("Scheduled LR: %0.4f", result)
        return float(result)

    if FLAGS.lr_decay_coeff:
        lr_decay_args = {
                'interval': FLAGS.lr_decay_freq,
                'coeff': FLAGS.lr_decay_coeff,
        }
        logging.info("LRDecay: %s", lr_decay_args)
        lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(lr_decay_cb)


    # Skip IO callbacks if requested, return callback list
    if FLAGS.dry: return callbacks

    # Make TB / checkpoint dirs
    # (after FLAGS.dry early return to avoid empty dir buildup)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Periodic model weight checkpointing
    chkpt_fmt = os.path.join(checkpoint_dir, FLAGS.checkpoint_fmt)
    chkpt_cb = ModelCheckpoint(
        filepath=chkpt_fmt,
        save_freq='epoch',
        save_weights_only=True
    )
    callbacks.append(chkpt_cb)

    # Log to Tensorboard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(tb_dir, DATE),
        write_graph=True,
        histogram_freq=1,
        embeddings_freq=3,
        update_freq='epoch'
    )
    file_writer = tf.summary.create_file_writer(tb_dir + "/metrics")
    file_writer.set_as_default()
    callbacks.append(tensorboard_cb)

    # Return callback list
    return callbacks

def save_summary(model, filepath, line_length=80):
    """Writes the output of `model.summary()` to `filepath`"""
    logging.info("Saving model summary to file: %s", filepath)
    with open(filepath, 'w') as fh:
        model.summary(
                print_fn=lambda x: fh.write(x + '\n'),
                line_length=line_length
        )

def clean_empty_dirs(path):
    """Deletes empty subdirectories of `path`"""
    logging.debug("Checking dir for cleaning: %s", path)
    for p in Path(path).glob('2*/'):
        if not list(p.glob('*')):
            logging.info('Removing empty directory: %s', p)
            p.rmdir()
        logging.debug('Ignoring non-empty checkpoint directory: %s', p)

def get_last_checkpoint(checkpoint_dir):
    """
    Gets the filepath for the most recent checkpoint file
    in `checkpoint_dir`. Assumes checkpoint filenames and
    subdirectories were created using defaults given in `flags.py`
    """

    # Find directory of latest run
    path = Path(checkpoint_dir)
    latest_path = sorted(list(path.glob('20*')))[-1]

    # Find latest checkpoint file
    latest_checkpoint = sorted(list(latest_path.glob('*.hdf5')))[-1]

    return str(latest_checkpoint.resolve())
