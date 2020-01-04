#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tin.flags import *
from argparse import ArgumentTypeError
import argparse
import pytest
import os


@pytest.fixture(scope='class')
def flag():
    raise pytest.UsageError('base flag fixture not overridden')


@pytest.fixture
def nargs(request):
    val = [x.args[0] for x in request.node.iter_markers(name='nargs')]
    if not val:
        return 1
    elif len(val) > 1:
        raise pytest.UsageError('nargs mark takes 1 arg')
    else:
        return val[0]


@pytest.fixture(scope='class')
def const(request):
    val = [x for x in request.node.iter_markers(name='const')]
    if not val:
        return False
    elif len(val) > 1:
        raise pytest.UsageError('const mark takes 1 arg')
    return bool(val)


@pytest.fixture(scope='class')
def dtype(request):
    val = [x.args[0] for x in request.node.iter_markers(name='dtype')]
    if not val:
        pytest.skip('dtype not used with this arg')
    elif len(val) > 1:
        raise pytest.UsageError('dtype mark takes 1 arg')
    else:
        return eval(val[0])


@pytest.fixture(scope='class')
def default(request):
    val = [x.args[0] for x in request.node.iter_markers(name='default')]
    if not val:
        pytest.skip('default not used with this arg')
    elif len(val) > 1:
        raise pytest.UsageError('default mark takes 1 arg')
    else:
        return val[0]


@pytest.fixture(scope='class')
def required(request):
    val = [x for x in request.node.iter_markers(name='required')]
    return bool(val)


@pytest.fixture(scope='class')
def positional(request):
    val = [x.args[0] for x in request.node.iter_markers(name='positional')]
    if not val:
        pytest.skip('arg is not a positional arg')
    elif len(val) > 1:
        raise pytest.UsageError('positional mark takes 1 arg')
    else:
        return val[0]


@pytest.fixture(scope='class')
def action(request):
    val = [
        (x.args[0], x.kwargs)
        for x in request.node.iter_markers(name='action')
    ]
    if not val:
        return ('store', {})
    elif len(val) > 1:
        raise pytest.UsageError('action mark takes 1 arg')
    else:
        return val[0]


@pytest.fixture(scope='class')
def group(request):
    val = [x.args[0] for x in request.node.iter_markers(name='group')]
    if not val:
        return None
    elif len(val) > 1:
        raise pytest.UsageError('group mark takes 1 arg')
    else:
        return val[0]


@pytest.fixture(scope='class')
def key(flag):
    return flag.replace('-', '')


class BaseTestFlag:
    def test_arg_added_to_parser(self, flag, key, mock_parser):
        args = parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        assert call != None

    def test_default_value(self, flag, key, default, mock_parser):
        args = parse_args([])
        call = self.get_mock_add_arg(mock_parser, flag)
        assert 'default' in call.kwargs
        assert call.kwargs['default'] == default

    def test_nargs(self, flag, nargs, mock_parser):
        parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        if nargs != 1:
            assert 'nargs' in call.kwargs
            assert call.kwargs['nargs'] == nargs
        else:
            assert 'nargs' not in call.kwargs or call.kwargs['nargs'] == nargs

    def test_required(self, flag, required, mock_parser):
        parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        if required:
            assert 'required' in call.kwargs
            assert call.kwargs['required'] == required
        else:
            assert (
                'required' not in call.kwargs
                or call.kwargs['required'] == False
            )

    def test_positional(self, flag, positional):
        assert False

    def test_dtype(self, flag, dtype, mock_parser):
        parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        if dtype != str:
            assert 'type' in call.kwargs
            assert call.kwargs['type'] == dtype
        else:
            assert 'type' not in call.kwargs or call.kwargs['type'] == dtype

    def test_const(self, flag, const, mock_parser):
        parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        if const:
            assert 'const' in call.kwargs
            assert call.kwargs['const'] == const
        else:
            assert 'const' not in call.kwargs or call.kwargs['const'] == False

    def test_action(self, flag, action, mock_parser):
        action, kwargs = action
        parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        if action in [None, 'store']:
            assert 'action' not in call.kwargs or call.kwargs['action'] == None
        else:
            assert 'action' in call.kwargs
            for kwarg, value in kwargs.items():
                assert call.kwargs[kwarg] == value

    @pytest.fixture
    def group_mocker(self, mocker, group, mock_parser):
        target_mock = mocker.MagicMock(name='target_group')
        other_mock = mocker.MagicMock(name='other_group')

        def group_side_effect(name, *args, **kwargs):
            if name == group:
                return target_mock
            else:
                return other_mock

        mock_parser.add_argument_group.side_effect = group_side_effect
        return target_mock, other_mock

    def test_group(self, flag, group, mock_parser, group_mocker):
        target_group, other_group = group_mocker
        parse_args('')

        other_calls = [
            args for _, args, _ in other_group.add_argument.mock_calls
            if flag in args
        ]
        for args in other_calls:
            if flag in args:
                pytest.fail('target flag was called as part of other group')

        if group is None:
            return

        target_group.add_argument.assert_called()
        target_calls = [
            args for _, args, _ in target_group.add_argument.mock_calls
            if flag in args
        ]

        for args in target_calls:
            if flag in args:
                return
        pytest.fail('target flag was not called as part of target group')

    def get_mock_add_arg(self, mock_parser, flag):
        for call in mock_parser.mock_calls:
            name, args, kwargs = call
            if 'add_argument' in name and flag in args:
                return call
        return None


@pytest.mark.nargs('?')
@pytest.mark.dtype('str2bool')
class BaseTestBoolFlag(BaseTestFlag):
    pass


@pytest.mark.nargs(1)
@pytest.mark.dtype('str')
class BaseTestFilePathFlag(BaseTestFlag):
    pass


@pytest.mark.nargs(1)
class BaseTestChoiceFlag(BaseTestFlag):
    @pytest.fixture(scope='class')
    def choices(self, request):
        val = [x.args[0] for x in request.node.iter_markers(name='choices')]
        if not val:
            pytest.skip('arg is not a positional arg')
        else:
            return val

    def test_choices(self, flag, choices, mock_parser):
        parse_args('')
        call = self.get_mock_add_arg(mock_parser, flag)
        assert 'choices' in call.kwargs
        assert call.kwargs['choices'] == choices


@pytest.mark.default(False)
@pytest.mark.group('runtime')
class TestDry(BaseTestBoolFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--dry'


@pytest.mark.default(False)
@pytest.mark.group('runtime')
class TestSummary(BaseTestBoolFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--summary'


@pytest.mark.default(False)
@pytest.mark.group('runtime')
class TestTune(BaseTestBoolFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--tune'


@pytest.mark.default(False)
@pytest.mark.group('training')
class TestEarlyStopping(BaseTestBoolFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--early_stopping'


@pytest.mark.nargs(2)
@pytest.mark.default([64, 64])
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('runtime')
class TestImageDim(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--image_dim'


@pytest.mark.group('file paths')
class TestSrc(BaseTestFilePathFlag):
    @pytest.fixture(scope='class')
    def default(self):
        os.environ['SRC_DIR'] = 'SRC_DIR'
        return 'SRC_DIR'

    @pytest.fixture(scope='class')
    def flag(self):
        return '--src'


@pytest.mark.default('resnet')
@pytest.mark.choices('resnet', 'inception')
@pytest.mark.dtype('str')
@pytest.mark.group('model properties')
class TestModel(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--model'


@pytest.mark.default('adam')
@pytest.mark.choices('adam', 'rmsprop')
@pytest.mark.dtype('str')
@pytest.mark.group('training')
class TestOptimizer(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--optim'


@pytest.mark.default(32)
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('runtime')
class TestBatchSize(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--batch_size'


@pytest.mark.default(32)
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('model properties')
class TestWidth(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--width'


@pytest.mark.default([3, 6, 4])
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.nargs('+')
@pytest.mark.group('model properties')
class TestLevels(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--levels'


@pytest.mark.default(64)
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('model properties')
class TestNumClasses(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--num_classes'


@pytest.mark.default(0.1)
@pytest.mark.action(NumericValidator, low=0, high=1.0, inclusive=(True, False))
@pytest.mark.dtype('float')
@pytest.mark.group('runtime')
class TestValidationSplit(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--validation_split'


@pytest.mark.default(100)
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('training')
class TestEpochs(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--epochs'


@pytest.mark.default(0.001)
@pytest.mark.action(
    NumericValidator, low=0, high=1.0, inclusive=(False, False)
)
@pytest.mark.dtype('float')
@pytest.mark.group('training')
class TestLr(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--lr'


@pytest.mark.default(None)
@pytest.mark.action(
    NumericValidator, low=0, high=1.0, inclusive=(False, False)
)
@pytest.mark.dtype('float')
@pytest.mark.group('training')
class TestLrDecayCoeff(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--lr_decay_coeff'


@pytest.mark.default(10)
@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('training')
class TestLrDecayFreq(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--lr_decay_freq'


@pytest.mark.default(0)
@pytest.mark.action(NumericValidator, low=0, high=1.0, inclusive=(True, False))
@pytest.mark.dtype('float')
@pytest.mark.group('model properties')
class TestL1(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--l1'


@pytest.mark.default(0)
@pytest.mark.action(NumericValidator, low=0, high=1.0, inclusive=(True, False))
@pytest.mark.dtype('float')
@pytest.mark.group('model properties')
class TestL2(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--l2'


@pytest.mark.default(0.9)
@pytest.mark.action(
    NumericValidator, low=0, high=1.0, inclusive=(False, False)
)
@pytest.mark.group('training')
@pytest.mark.dtype('float')
class TestBeta1(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--beta1'


@pytest.mark.default(0.999)
@pytest.mark.dtype('float')
@pytest.mark.action(
    NumericValidator, low=0, high=1.0, inclusive=(False, False)
)
@pytest.mark.group('training')
class TestBeta2(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--beta2'


@pytest.mark.default(1e-6)
@pytest.mark.action(
    NumericValidator, low=0, high=1.0, inclusive=(False, False)
)
@pytest.mark.dtype('float')
@pytest.mark.group('training')
class TestEpsilon(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--epsilon'


@pytest.mark.default(0)
@pytest.mark.action(NumericValidator, low=0, high=1.0, inclusive=(True, False))
@pytest.mark.dtype('float')
@pytest.mark.group('model properties')
class TestDropout(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--dropout'


@pytest.mark.default(42)
@pytest.mark.dtype('int')
@pytest.mark.group('runtime')
class TestSeed(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--seed'


@pytest.mark.default('tin_{epoch:03d}.hdf5')
@pytest.mark.group('checkpoint')
class TestCheckpointFmt(BaseTestFilePathFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--checkpoint_fmt'


@pytest.mark.action(NumericValidator, low=1)
@pytest.mark.dtype('int')
@pytest.mark.group('checkpoint')
class TestCheckpointFreq(BaseTestFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--checkpoint_freq'


@pytest.mark.default(None)
@pytest.mark.group('checkpoint')
class TestLoadModel(BaseTestFilePathFlag):
    @pytest.fixture(scope='class')
    def flag(self):
        return '--load_model'
