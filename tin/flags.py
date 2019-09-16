#!python
"""Provides command line flags to customize the training pipleine"""

import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'src',
    os.environ.get('SRC_DIR', ''),
    'Dataset source directory. Target of ImageDataGenerator.flow_from_directory'
)

flags.DEFINE_string(
    'artifacts_dir',
    os.environ.get('ARTIFACTS_DIR', ''),
    'Destination directory for checkpoints / Tensorboard logs'
)

flags.DEFINE_bool(
    'dry',
    False,
    ('If true, dont write Tensorboard logs or checkpoint files. '
    'Use this to experiment without worrying about writing artifacts')
)

flags.DEFINE_integer(
    'batch_size',
    32,
    'Batch size for training'
)

flags.DEFINE_integer(
    'image_dim',
    64,
    'Dimension in pixels of the square training images'
)

flags.DEFINE_bool(
    'summary',
    False,
    ('Print/save a model layer summary and exit. '
    'Model summary will be saved in artifact directory')
)

flags.DEFINE_bool(
    'tune',
    False,
    'Reserved for future use. Hyperparameter tuning'
)

flags.DEFINE_list(
    'levels',
    [3, 6, 4],
    ('Levels to use in the TraderNet encoder architecture. '
    'ie. --levels=3,6,4 for 3 levels of downsampling with 3,6,4'
    'bottleneck blocks for the respective levels')
)

flags.DEFINE_integer(
    'classes',
    61,
    'Number of output classes if running in classification mode.'
)

flags.DEFINE_integer(
    'epochs',
    100,
    'Number of training epochs'
)

flags.DEFINE_float(
    'validation_split',
    0.1,
    'Fraction of dataset to reserve for validation'
)

flags.DEFINE_float(
    'lr',
    0.001,
    'Initial learning rate'
)

flags.DEFINE_float(
    'l1',
    None,
    'L1 norm for head'
)

flags.DEFINE_float(
    'l2',
    None,
    'L2 norm for head'
)

flags.DEFINE_integer(
    'seed',
    42,
    'If set, the integer to seed all random generators with'
)

flags.DEFINE_string(
    'checkpoint_fmt',
    'tin_{epoch:03d}.hdf5',
    'Filename format to use when writing checkpoints'
)

flags.DEFINE_string(
    'checkpoint_freq',
    'epoch',
    'Checkpoint frequency passed to tf.keras.callbacks.ModelCheckpoint'
)

flags.DEFINE_string(
    'tb_update_freq',
    'epoch',
    'Update frequency passed to tf.keras.callbacks.TensorBoard'
)

flags.DEFINE_string(
    'resume',
    None,
    'Resume from the specified model checkpoint filepath'
)

flags.DEFINE_bool(
    'resume_last',
    None,
    'Attempt to resume from the most recent checkpoint'
)

flags.register_validator(
    'src',
    lambda v: os.path.isdir(v) and os.access(v, os.R_OK),
    message='--src must point to an existing directory'
)

flags.register_validator(
    'artifacts_dir',
    lambda v: os.path.isdir(v) and os.access(v, os.W_OK),
    message='--artifacts_dir must point to an existing directory'
)

flags.register_validator(
    'batch_size',
    lambda v: v > 0,
    message='--batch_size must be an int > 0'
)

flags.register_validator(
    'levels',
    lambda v: len(v) > 0,
    message='--levels must be a non-empty list of integers'
)

flags.register_validator(
    'classes',
    lambda v: v > 0,
    message='--classes must be an integer > 0'
)

flags.register_validator(
    'epochs',
    lambda v: v > 0,
    message='--epochs must be an integer > 0'
)

flags.register_validator(
    'validation_split',
    lambda v: v >= 0,
    message='--validation_split must be a float on interval (0, 1)'
)

flags.register_validator(
    'validation_split',
    lambda v: v > 0,
    message='--image_dim must be an int > 0'
)

flags.register_validator(
    'lr',
    lambda v: 0 < v < 1.0,
    message='--lr must be an float on interval (0.0, 1.0)'
)

flags.register_validator(
flags.register_validator(
    'l1',
    lambda v: v == None or v > 0,
    message='--l1 must be a float > 0. Use l1=None for no regularization'
)

flags.register_validator(
    'l2',
    lambda v: v == None or v > 0,
    message='--l2 must be a float > 0. Use l2=None for no regularization'
)

flags.register_validator(
    'seed',
    lambda v: v == None or int(v) == v,
    message='--seed must be None or an integer'
)

flags.register_validator(
    'checkpoint_fmt',
    lambda v: len(v) > 0,
    message='--checkpoint_fmt must be a non-empty string'
)

flags.register_validator(
    'resume',
    lambda v: v == None or os.path.isfile(v),
    message='--resume must point to an existing checkpoint file'
)

flags.register_validator(
    'image_dim',
    lambda v: v > 0,
    message='--image_dim must be an int > 0'
)
