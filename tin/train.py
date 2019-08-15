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

callbacks = list()

TFREC_SPEC = {
    'features': tf.io.FixedLenSequenceFeature([TFREC_FEATURES], tf.float32),
    'change': tf.io.FixedLenFeature([], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def preprocess():

    logging.info("Reading TFRecords from: %s/%s", FLAGS.src, FLAGS.glob)

    def _parse_tfrec(example_proto):
        # Sequence features (feature matrix)
        seq_f = {'features': TFREC_SPEC['features']}

        # Context features (FixedLenFeature scalars)
        context_f = {x: TFREC_SPEC[x] for x in ['change', 'label']}

        # Read example proto into dicts
        con, seq, dense = tf.io.parse_sequence_example(example_proto, context_f, seq_f)

        # Return (features, label) tuple of tensors
        return seq['features'], con[FLAGS.label]

    # Build a list of input TFRecord files
    target = Path(FLAGS.src).glob(FLAGS.glob)
    target = [x.as_posix() for x in target]

    # Read files to dataset and apply parsing function
    raw_tfrecs = tf.data.TFRecordDataset(list(target))

    # Training/test split
    if FLAGS.speedrun:
        logging.info("Taking %s examples for validation", FLAGS.batch_size * 10)
        train = raw_tfrecs.skip(FLAGS.validation_size).take(FLAGS.batch_size * 100)
        validate = raw_tfrecs.take(FLAGS.batch_size * 10)
    else:
        logging.info("Taking %s examples for validation", FLAGS.validation_size)
        train = raw_tfrecs.skip(FLAGS.validation_size)
        validate = raw_tfrecs.take(FLAGS.validation_size)

    # TODO fix caching, seems to cause memory leak
    #train = train.cache()
    #validate = validate.cache()

    # Prefetch if requested
    if FLAGS.prefetch:
        logging.debug("Prefetching data")
        train = train.prefetch(buffer_size=128)
        validate = validate.prefetch(buffer_size=128)

    # Repeat if requested
    if FLAGS.repeat:
        logging.debug("Repeating dataset")
        train = train.repeat()
        validate = validate.repeat()

    # Shuffle if reqested
    if FLAGS.shuffle_size > 0:
        logging.debug("Shuffling with buffer size %i", FLAGS.shuffle_size)
        train = train.shuffle(FLAGS.shuffle_size)
        validate = validate.shuffle(FLAGS.shuffle_size)

    # Batch
    logging.debug("Applying batch size %i", FLAGS.batch_size)
    validate = validate.batch(FLAGS.batch_size, drop_remainder=True)
    train = train.batch(FLAGS.batch_size, drop_remainder=True)

    # Parse serialized example Protos before handoff to training pipeline
    train = train.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    validate = validate.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    return train, validate



def construct_model():

    # Model
    if FLAGS.mode == 'regression':
        logging.debug("Running in regresison mode with label: %s", FLAGS.label)
        head = RegressionHead(name='head')
        model = TraderNet(levels=FLAGS.levels, use_head=head, use_tail=True, use_attn=FLAGS.attention)
    else:
        logging.debug("Running in classification mode with num classes: %i", FLAGS.classes)
        head = ClassificationHead(classes=FLAGS.classes, name='head')
        model = TraderNet(levels=FLAGS.levels, use_head=head, use_tail=True, use_attn=FLAGS.attention)

    return model

def train_model(model, train, validate, initial_epoch):
    """
    Compiles `model` with metrics / loss / optimizer and
    begins training on `train` / `validate` Datasets starting
    at `initial_epoch`
    """

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2),
    ]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.lr,
            epsilon=0.1
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    validation_steps=FLAGS.validation_size // FLAGS.batch_size
    model_callbacks = callbacks + [hp.KerasCallback(hparam_dir, hparams)]
    steps_per_epoch=FLAGS.steps_per_epoch


    fit_args = {
        'x': train,
        'epochs': FLAGS.epochs,
        'steps_per_epoch': steps_per_epoch,
        'validation_data': validate,
        'validation_steps': validation_steps,
        'callbacks': model_callbacks,
        'initial_epoch': initial_epoch
    }
    pretty_args = json.dumps({k: str(v) for k, v in fit_args.items()}, indent=2)
    logging.info("Fitting model with args: \n%s", pretty_args)
    history = model.fit(**fit_args)

def main(argv):

    image_shape = (FLAGS.image_dim, FLAGS.image_dim)
    inputs = layers.Input(shape=image_shape, dtype=tf.float32)

    FLAGS.levels = [int(x) for x in FLAGS.levels]
    model = construct_model()
    outputs = model(inputs)

    if FLAGS.summary:
        out_path = os.path.join(FLAGS.artifacts_dir, 'summary.txt')
        model.summary()
        save_summary(model, out_path)

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
    callbacks = get_callbacks(FLAGS) if not FLAGS.speedrun else []

    train, validate = preprocess()

    # Print brief summary of training dataset
    with np.printoptions(precision=3):
        for x in train.take(1):
            f, l = x
            print("Dataset feature tensor shape: %s" % f.shape)
            print("Dataset label tensor shape: %s" % l.shape)
            print("First batch labels: %s" % l.numpy())
            print("First batch element features (truncated):")
            print(f.numpy()[0][:10])
        return

    train_model(model, train, validate, initial_epoch)

if __name__ == '__main__':
  app.run(main)
