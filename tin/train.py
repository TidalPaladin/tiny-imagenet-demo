#!python3
"""
Training pipeline for Tiny ImageNet. Some code cleanup is still needed here
"""

# TODO which of these imports can be removed
import os
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger

from pathlib import Path
from glob import glob as glob_func
from tin.util import *
from tin.inception import *
from tin.resnet import *

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import *

from absl import app, logging
from tin.flags import FLAGS

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess(args):
    """
    Returns a tuple of the form (training generator, validation generator).
    Image preprocessing is handled here via Keras ImageDataGenerator
    """

    logging.info("Reading images from: %s", FLAGS.src)

    # Create object to read training images w/ preprocessing
    #
    # Standardize each image to zero mean unit variance with
    #   with random brightness perturbartions and horizontal flips
    #
    # Training / validation split is specified here
    # TODO get all these params from FLAGS
    datagen = ImageDataGenerator(
    #samplewise_center=True,
    #samplewise_std_normalization=True,
        horizontal_flip=True,
        data_format='channels_last',
        validation_split=args.validation_split,
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        rescale=1. / 255
    )

    # Create generator to yield a training set from directory
    train_generator = datagen.flow_from_directory(
        FLAGS.src,
        subset='training',
        target_size=(64, 64),
        class_mode='sparse',
        batch_size=args.batch_size,
        seed=args.seed
    )

    # Create generator to yield a validation set from directory
    val_generator = datagen.flow_from_directory(
        args.src,
        subset='validation',
        target_size=(64, 64),
        class_mode='sparse',
        batch_size=args.batch_size,
        seed=args.seed
    )

    def generator_to_ds(x):
        gen = lambda: x
        return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

    train_ds = generator_to_ds(train_generator)
    val_ds = generator_to_ds(val_generator)

    def pipeline(x):
        return x.prefetch(10)

    train_ds = pipeline(train_ds)
    val_ds = pipeline(val_ds)

    return train_ds, val_ds


def construct_model(args):
    """Returns a TinyImageNet model. Place custom head/tail layer inclusions here"""

    # Here TinyImageNetHead is explicitly constructed for clarity and passed
    #   as the use_head arg to TinyImageNet. Can also use `use_head=True`.
    head = TinyImageNetHead(
        num_classes=args.num_classes,
        l1=args.l1,
        l2=args.l2,
        dropout=args.dropout,
        seed=args.seed,
        name='head'
    )

    if args.inception:
        logging.info(
            "Using Inception network with %i classes", args.num_classes
        )
        model = TinyInceptionNet(
            levels=args.levels, width=args.width, use_head=head, use_tail=True
        )

    elif FLAGS.resnet:
        logging.info("Using Resnet network with %i classes", args.num_classes)
        model = TinyImageNet(
            levels=args.levels, width=args.width, use_head=head, use_tail=True
        )

    return model


def train_model(args, model, train, validate, initial_epoch, callbacks=[]):
    """
    Compiles `model` with metrics / loss / optimizer and
    begins training on `train` / `validate` Datasets starting
    at `initial_epoch`.

    `initial_epoch` is needed when resuming from checkpoints
    """

    # Top 1 and top 5 categorical accuracy
    k = 5
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            name='top_%i_acc' % k, k=k
        ),
    ]

    # Use softmax + cross entropy loss
    #   `from_logits=True` means loss expects unscaled logits and will apply
    #   softmax internally for a more stable backward pass
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if args.adam:
        logging.info(
            "Adam: e=%f, b1=%f, b2=%f", args.epsilon, args.beta1, args.beta2
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr,
            epsilon=args.epsilon,
            beta_1=args.beta1,
            beta_2=args.beta2
        )
    elif args.rmsprop:
        logging.info("RMSProp: e=%f, rho=%f", args.epsilon, args.rho)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=args.lr, epsilon=args.epsilon, rho=args.rho
        )

    # Compile model with given parameters prior to training
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Gather model.fit_generator args as a dict for printing to user
    fit_args = {
        'x': train,
        'epochs': args.epochs,
        'validation_data': validate,
        'callbacks': callbacks,
        'initial_epoch': initial_epoch
    }
    pretty_args = json.dumps(
        {k: str(v)
         for k, v in fit_args.items()}, indent=2
    )

    logging.info("Writing training data images")
    plot_inputs(train)

    logging.info("Fitting model with args: \n%s", pretty_args)
    history = model.fit(**fit_args)
    return history


def main(argv):
    """
    Main method, must call this method with app.main() for absl-py libraries to
    work as intended
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Specify input shape for model.summary(). Batch dimension always excluded
    image_shape = (FLAGS.image_dim, FLAGS.image_dim, 3)
    inputs = layers.Input(shape=image_shape, dtype=tf.float32)

    # Assemble model
    FLAGS.levels = [int(x) for x in FLAGS.levels]
    model = construct_model(FLAGS)

    # Call model on inputs to generate shape info for model.summary()
    outputs = model(inputs, training=True)

    # Print / save summary and exit if --summary flag given
    if FLAGS.summary:
        out_path = os.path.join(FLAGS.artifacts_dir, 'summary.txt')
        model.summary()
        save_summary(model, out_path)
        logging.info("Exiting after printing model summary")
        sys.exit(0)

    # Build checkpoint dir path and clean empty dirs
    checkpoint_dir = os.path.join(FLAGS.artifacts_dir, 'checkpoint')
    clean_empty_dirs(checkpoint_dir)

    initial_epoch = 0
    resume_file = FLAGS.resume if FLAGS.resume else None

    # If --resume_last given, try to resume from last checkpoint by filename
    if FLAGS.resume_last:
        # Find directory of latest run
        chkpt_path = Path(FLAGS.artifacts_dir, 'checkpoint')
        latest_path = sorted(list(chkpt_path.glob('20*')))[-1]
        logging.info("Using latest run - %s", latest_path)

        # Find latest checkpoint file
        latest_checkpoint = sorted(list(latest_path.glob('*.hdf5')))[-1]
        FLAGS.resume = str(latest_checkpoint.resolve())

    # If --resume, try to resume from the given checkpoint file
    if FLAGS.resume:
        logging.info("Loading weights from file %s", FLAGS.resume)
        model.load_weights(FLAGS.resume)
        initial_epoch = int(re.search('([0-9]*)\.hdf5', FLAGS.resume).group(1))
        logging.info("Starting from epoch %i", initial_epoch + 1)

    callbacks = get_callbacks(FLAGS) if not FLAGS.dry else []

    train, validate = preprocess(FLAGS)
    train_model(FLAGS, model, train, validate, initial_epoch, callbacks)


if __name__ == '__main__':
    app.run(main)
