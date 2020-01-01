#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tin
from tin.train import *
import pytest
import os
import tensorflow
from PIL import Image
import numpy
import logging


@pytest.fixture
def tf(mocker):
    m = mocker.MagicMock()
    mocker.patch('tin.train.tf', m)
    return m


def assert_kwarg_call(call, key, value):
    assert key in call.kwargs, ('{} not in call kwargs'.format(key))
    assert call.kwargs[key] == value, (
        'expected kwarg {}={}, got {}={}'.format(
            key, value, key, call.kwargs[key]
        )
    )


class TestPreprocess:
    @pytest.fixture
    def mock_datagen(self, mocker):
        m = mocker.MagicMock(spec_set=ImageDataGenerator)
        mocker.patch('tin.train.ImageDataGenerator', m)
        return m

    @pytest.mark.parametrize(
        'kwarg', [
            pytest.param('validation_split', id='validation_split'),
        ]
    )
    def testImageDataGenFlagArgs(self, kwarg, mock_flags, mock_datagen):
        _, _ = preprocess(mock_flags)
        mock_datagen.assert_called_once()
        call = mock_datagen.call_args
        value = getattr(mock_flags, kwarg)
        assert_kwarg_call(call, kwarg, value)

    def testImageDataGenSrc(self, mock_flags, mock_datagen):
        _, _ = preprocess(mock_flags)
        mock_datagen().flow_from_directory.assert_called()
        args = mock_datagen().flow_from_directory.call_args
        assert mock_flags.src in args.args, '--src not passed to ImageDataGenerator'

    @pytest.mark.parametrize(
        'kwarg,value', [
            pytest.param('target_size', (64, 64), id='target_size'),
            pytest.param('class_mode', 'sparse', id='class_mode'),
        ]
    )
    def testDatasetFixedFlowArgs(self, kwarg, value, mock_flags, mock_datagen):
        _, _ = preprocess(mock_flags)
        flow = mock_datagen().flow_from_directory
        flow.assert_called()
        call_list = flow.call_args_list
        for call in call_list:
            assert_kwarg_call(call, kwarg, value)

    @pytest.mark.parametrize(
        'kwarg', [
            pytest.param('batch_size', id='batch_size'),
            pytest.param('seed', id='seed'),
        ]
    )
    def testDatasetFlagFlowArgs(self, kwarg, mock_flags, mock_datagen):
        _, _ = preprocess(mock_flags)
        flow = mock_datagen().flow_from_directory
        flow.assert_called()
        call_list = flow.call_args_list
        value = getattr(mock_flags, kwarg)
        for call in call_list:
            assert_kwarg_call(call, kwarg, value)

    def testGetsTrainDataset(self, mock_flags, mock_datagen):
        train_gen, _ = preprocess(mock_flags)
        datagen_call = mock_datagen.call_args
        flow = mock_datagen().flow_from_directory
        flow.assert_called()
        assert train_gen == flow()
        assert_kwarg_call(datagen_call, 'data_format', 'channels_last')


class TestConstructModel:
    @pytest.fixture
    def mock_head(self, mocker):
        m = mocker.MagicMock(name='head', spec_set=TinyImageNetHead)
        mocker.patch('tin.resnet.TinyImageNetHead', m)
        mocker.patch('tin.train.TinyImageNetHead', m)
        return m

    @pytest.fixture
    def mock_resnet(self, mocker):
        m = mocker.MagicMock(name='resnet', spec_set=TinyImageNet)
        mocker.patch('tin.resnet.TinyImageNet', m)
        mocker.patch('tin.train.TinyImageNet', m)
        return m

    @pytest.fixture
    def mock_inception(self, mocker):
        m = mocker.MagicMock(name='inception', spec_set=TinyInceptionNet)
        mocker.patch('tin.inception.TinyInceptionNet', m)
        mocker.patch('tin.train.TinyInceptionNet', m)
        return m

    @pytest.fixture(params=['resnet', 'inception'])
    def mock_model(
        self, request, mock_flags, mock_head, mock_resnet, mock_inception
    ):
        if request.param == 'resnet':
            mock_flags.inception = False
            mock_flags.resnet = True
            return mock_resnet
        else:
            mock_flags.inception = True
            mock_flags.resnet = False
            return mock_inception

    @pytest.mark.parametrize(
        'kwarg', [
            pytest.param('num_classes', id='num_classes'),
            pytest.param('l1', id='l1'),
            pytest.param('l2', id='l2'),
            pytest.param('dropout', id='dropout'),
            pytest.param('seed', id='seed'),
        ]
    )
    @pytest.mark.usefixtures('mock_resnet', 'mock_inception')
    def testHeadConstructorFromFlags(self, kwarg, mock_head, mock_flags):
        value = getattr(mock_flags, kwarg)
        model = construct_model(mock_flags)
        mock_head.assert_called()
        assert_kwarg_call(mock_head.call_args, kwarg, value)

    @pytest.mark.usefixtures('mock_head')
    def testModelTypeFromFlags(self, mock_model, mock_flags):
        model = construct_model(mock_flags)
        mock_model.assert_called()
        assert model == mock_model()

    @pytest.mark.parametrize(
        'kwarg', [
            pytest.param('levels', id='levels'),
            pytest.param('width', id='width'),
        ]
    )
    @pytest.mark.usefixtures('mock_head')
    def testModelConstructorFromFlags(
        self, kwarg, mock_head, mock_model, mock_flags
    ):
        value = getattr(mock_flags, kwarg)
        model = construct_model(mock_flags)
        mock_model.assert_called()
        assert_kwarg_call(mock_model.call_args, kwarg, value)


class TestMockTrain:
    @pytest.fixture(autouse=True)
    def data(self, tf):
        return tf.data.Dataset(), tf.data.Dataset()

    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        m = mocker.MagicMock(name='model', spec_set=TinyImageNet)
        mocker.patch('tin.resnet.TinyImageNet', m)
        mocker.patch('tin.train.TinyImageNet', m)
        return m

    def test_uses_SCCE_loss(self, tf, mock_model, data, mock_flags):
        train_model(mock_flags, mock_model, *data, 1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy
        loss.assert_called()

    def test_loss_from_logits(self, tf, mock_model, data, mock_flags):
        train_model(mock_flags, mock_model, *data, 1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy
        loss.assert_called()
        assert_kwarg_call(loss.call_args, 'from_logits', True)

    @pytest.fixture(params=['adam', 'rmsprop'])
    def optimizer(self, request, tf, mock_flags):
        if request.param == 'adam':
            mock_flags.adam = True
            mock_flags.rmsprop = False
            return tf.keras.optimizers.Adam
        elif request.param == 'rmsprop':
            mock_flags.adam = False
            mock_flags.rmsprop = True
            return tf.keras.optimizers.RMSprop

    def test_get_optimizer_from_flag(
        self, tf, mock_model, data, optimizer, mock_flags
    ):
        train_model(mock_flags, mock_model, *data, 1)
        optimizer.assert_called()

    @pytest.mark.parametrize(
        'kwarg,flag', [
            pytest.param('learning_rate', 'lr', id='learning_rate'),
            pytest.param('epsilon', 'epsilon', id='epsilon'),
            pytest.param('beta_1', 'beta1', id='beta1'),
            pytest.param('beta_2', 'beta2', id='beta2'),
            pytest.param('rho', 'rho', id='rho'),
        ]
    )
    def test_get_optimizer_params_from_flag(
        self, tf, mock_model, data, optimizer, kwarg, flag, mock_flags, request
    ):
        if optimizer == tf.keras.optimizers.Adam:
            if kwarg == 'rho':
                pytest.xfail()
        else:
            if kwarg in ['beta_1', 'beta_2']:
                pytest.xfail()
        value = getattr(mock_flags, flag)
        train_model(mock_flags, mock_model, *data, 1)
        optimizer.assert_called()
        assert_kwarg_call(optimizer.call_args, kwarg, value)

    def test_model_compiled(self, tf, mock_model, data, mock_flags):
        train_model(mock_flags, mock_model, *data, 1)
        mock_model.compile.assert_called()

    @pytest.fixture(params=['loss', 'optimizer', 'metrics'])
    def compile_arg(self, request, tf, optimizer):
        if request.param == 'loss':
            return 'loss', tf.keras.losses.SparseCategoricalCrossentropy()
        elif request.param == 'metrics':
            return 'metrics', [
                tf.keras.metrics.SparseCategoricalAccuracy(),
                tf.keras.metrics.SparseTopKCategoricalAccuracy()
            ]
        else:
            return 'optimizer', optimizer()

    def test_model_compile_args(
        self, tf, mock_model, data, compile_arg, mock_flags
    ):
        key, value = compile_arg
        train_model(mock_flags, mock_model, *data, 1)
        mock_model.compile.assert_called()
        assert_kwarg_call(mock_model.compile.call_args, key, value)

    def test_model_fit_called(self, tf, mock_model, data, mock_flags):
        train_model(mock_flags, mock_model, *data, 1)
        mock_model.fit.assert_called()

    @pytest.fixture(
        params=[
            'x',
            'epochs',
            'validation_data',
            'initial_epoch',
            'callbacks',
        ]
    )
    def fit_arg(self, mocker, request, tf, data, mock_flags):
        if request.param == 'x':
            val = data[0]
        elif request.param == 'epochs':
            val = mock_flags.epochs
        elif request.param == 'validation_data':
            val = data[1]
        elif request.param == 'initial_epoch':
            val = mocker.MagicMock(name='initial_epoch')
        elif request.param == 'callbacks':
            val = mocker.MagicMock(name='callbacks', spec_set=list)
        return request.param, val

    def test_model_fit_args(self, tf, mock_model, data, fit_arg, mock_flags):
        key, value = fit_arg
        if key == 'initial_epoch':
            train_model(mock_flags, mock_model, *data, value)
        elif key == 'callbacks':
            train_model(mock_flags, mock_model, *data, 1, callbacks=value)
        else:
            train_model(mock_flags, mock_model, *data, 1)
        mock_model.fit.assert_called()
        call = mock_model.fit.call_args
        assert_kwarg_call(call, key, value)


class TestTrain:
    @pytest.fixture(autouse=True)
    def mock_flags(self, mocker):
        m = mocker.MagicMock(name='FLAGS')
        return m

    @pytest.fixture(scope='class')
    def src_dir(self, tmpdir_factory):
        dir = tmpdir_factory.mktemp('data')
        for cls in range(3):
            subdir = 'n000%d' % cls
            os.makedirs(os.path.join(dir, subdir))
            for n in range(3):
                a = numpy.random.rand(64, 64, 3) * 255
                im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
                im_path = os.path.join(
                    dir, subdir, '%s_00%d.JPEG' % (subdir, n)
                )
                print(im_path)
                im_out.save(im_path)
        print(os.listdir(dir))
        return dir

    @pytest.fixture(scope='class')
    def artifacts_dir(self, tmpdir_factory):
        return tmpdir_factory.mktemp('artif')

    def test_main_call(self, src_dir, artifacts_dir, caplog):
        sys.argv = [
            'train.py',
            '--src=%s' % src_dir,
            '--artifacts_dir=%s' % artifacts_dir,
            '--validation_split=0.5',
            '--num_classes=3',
            '--resnet',
            '--epochs=1',
            '--levels=1,1',
        ]
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                app.run(main)
