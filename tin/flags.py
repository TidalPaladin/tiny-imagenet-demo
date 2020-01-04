#!python
"""Provides command line flags to customize the training pipleine"""

import argparse
import os
from distutils.util import strtobool


def str2bool(v):
    """Parses bool args given to argparse"""
    try:
        return strtobool(v)
    except ValueError:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class NumericValidator(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        low=None,
        high=None,
        inclusive=True,
        **kwargs
    ):
        super(NumericValidator, self).__init__(option_strings, dest, **kwargs)
        if self.type not in [int, float]:
            raise TypeError('dtype must be one of int, float')
        if low is None and high is None:
            raise ValueError('low and high cannot both be None')
        if low is not None and high is not None and low >= high:
            raise ValueError('must have low < high')

        self.low = self.type(low) if low is not None else float('-inf')
        self.high = self.type(high) if high is not None else float('inf')

        if not isinstance(inclusive, tuple):
            inclusive = (inclusive, ) * 2
        if not all([isinstance(x, bool) for x in inclusive]):
            raise TypeError(
                'inclusive must be a single bool or 2-tuple of bools'
            )
        self.inclusive = inclusive

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            values = [self._validate(v) for v in values]
        else:
            values = self._validate(values)
        setattr(namespace, self.dest, values)

    def _validate(self, v):
        try:
            v = self.type(v)
        except ValueError:
            raise argparse.ArgumentTypeError('could not parse numeric range')

        if not self._check_low(v):
            raise argparse.ArgumentError(
                self, 'value {} exceeded minimum {}'.format(v, self.low)
            )
        if not self._check_high(v):
            raise argparse.ArgumentError(
                self, 'value {} exceeded maximum {}'.format(v, self.high)
            )
        return v

    def _check_low(self, v):
        return v >= self.low if self.inclusive[0] else v > self.low

    def _check_high(self, v):
        return v <= self.high if self.inclusive[0] else v < self.high


# Read env vars for some defaults
def parse_args(args):
    SRC_DIR = os.environ.get('SRC_DIR', '')
    ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', '')
    default_result_path = os.path.join(ARTIFACT_DIR, 'results')
    default_log_path = os.path.join(ARTIFACT_DIR, 'logs/cnndm.log')
    default_model_path = os.path.join(ARTIFACT_DIR, 'models')

    parser = argparse.ArgumentParser()

    # File path selection
    file_group = parser.add_argument_group('file paths')
    file_group.add_argument(
        "--src", default=SRC_DIR, help='Filepath of dataset'
    )
    file_group.add_argument(
        "--model_path",
        default=default_model_path,
        help='Output filepath of model checkpoints'
    )
    file_group.add_argument(
        "--result_path", type=str, default=default_result_path
    )
    file_group.add_argument(
        "--bert_config_path", default='/app/bert_config_uncased_base.json'
    )
    file_group.add_argument('--log_file', default=default_log_path)
    file_group.add_argument('--dataset', default='')

    # Model properties
    model_group = parser.add_argument_group('model properties')
    model_group.add_argument(
        "--model", default='resnet', type=str, choices=['resnet', 'inception']
    )
    model_group.add_argument(
        "--width",
        default=32,
        type=int,
        low=1,
        action=NumericValidator,
        help='model width after tail'
    )
    model_group.add_argument(
        "--levels",
        default=[3, 6, 4],
        type=int,
        low=1,
        action=NumericValidator,
        nargs='+',
        help='bottleneck repeats per levels'
    )
    model_group.add_argument(
        "--l1",
        default=0,
        type=float,
        low=0,
        high=1.0,
        inclusive=(True, False),
        action=NumericValidator,
        help='l1 regularization value for head'
    )
    model_group.add_argument(
        "--l2",
        default=0,
        type=float,
        low=0,
        high=1.0,
        inclusive=(True, False),
        action=NumericValidator,
        help='l2 regularization value for head'
    )
    model_group.add_argument(
        "--num_classes",
        default=64,
        type=int,
        low=1,
        action=NumericValidator,
    )
    model_group.add_argument(
        "--dropout",
        default=0,
        type=float,
        low=0,
        high=1.0,
        inclusive=(True, False),
        action=NumericValidator,
    )

    # runtime properties
    runtime_group = parser.add_argument_group('runtime')
    runtime_group.add_argument(
        "--mode", default='train', type=str, choices=['train', 'test']
    )
    runtime_group.add_argument(
        "--validation_split",
        default=0.1,
        type=float,
        low=0,
        high=1,
        inclusive=(True, False),
        action=NumericValidator,
        help='fraction of data for validation'
    )
    runtime_group.add_argument(
        "--batch_size", default=32, type=int, low=1, action=NumericValidator
    )
    runtime_group.add_argument(
        '--visible_gpus',
        default='1',
        type=str,
        help='gpus visible for execution'
    )
    runtime_group.add_argument(
        '--seed', default=42, type=int, help='random seed'
    )
    runtime_group.add_argument(
        "--dry",
        type=str2bool,
        nargs='?',
        default=False,
        help='run without generating artifacts'
    )
    runtime_group.add_argument(
        "--summary",
        type=str2bool,
        nargs='?',
        default=False,
        help='print model.summary() and exit'
    )
    runtime_group.add_argument(
        "--tune",
        type=str2bool,
        nargs='?',
        default=False,
        help='reserved for future use'
    )
    runtime_group.add_argument(
        "--image_dim",
        type=int,
        low=1,
        action=NumericValidator,
        nargs=2,
        default=[64, 64]
    )

    # training properties
    training_group = parser.add_argument_group('training')
    training_group.add_argument(
        "--optim",
        default='adam',
        type=str,
        choices=['adam', 'rmsprop'],
        help='optimizer selection'
    )
    training_group.add_argument(
        "--lr",
        default=0.001,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        action=NumericValidator,
        help='learning rate'
    )
    training_group.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        action=NumericValidator,
        help='adam beta1'
    )
    training_group.add_argument(
        "--beta2",
        default=0.999,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        action=NumericValidator,
        help='adam beta2'
    )
    training_group.add_argument(
        "--epsilon",
        default=1e-6,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        action=NumericValidator,
        help='optimizer epsilon value'
    )
    training_group.add_argument(
        "--lr_decay_coeff",
        default=None,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        action=NumericValidator,
        help='coefficient of exponential learning rate decay'
    )
    training_group.add_argument(
        "--lr_decay_freq",
        default=10,
        type=int,
        low=1,
        action=NumericValidator,
        help='epoch frequency of decay step function'
    )

    training_group.add_argument(
        "--epochs",
        default=100,
        type=int,
        low=1,
        action=NumericValidator,
        help='num epochs to train for'
    )
    training_group.add_argument(
        "--early_stopping",
        type=str2bool,
        nargs='?',
        default=False,
    )
    training_group.add_argument(
        "--resume_last", type=str2bool, nargs='?', default=False
    )

    checkpoint_group = parser.add_argument_group('checkpoint')
    checkpoint_group.add_argument(
        '--load_model',
        default=None,
        type=str,
        help='path to model checkpoint'
    )
    checkpoint_group.add_argument(
        "--save_checkpoint_steps", default=5, type=int
    )
    checkpoint_group.add_argument(
        "--checkpoint_fmt",
        default='tin_{epoch:03d}.hdf5',
        type=str,
        help='format for model checkpoint files'
    )
    checkpoint_group.add_argument(
        "--checkpoint_freq",
        type=int,
        low=1,
        action=NumericValidator,
        help='frequency of model checkpointing'
    )

    return parser.parse_args(args)
