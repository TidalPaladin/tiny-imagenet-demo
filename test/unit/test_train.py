#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tin
from tin.train import *
import pytest
import os
import tensorflow
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def assert_kwarg_call(call, key, value):
    assert key in call.kwargs, ('{} not in call kwargs'.format(key))
    assert call.kwargs[key] == value, (
        'expected kwarg {}={}, got {}={}'.format(
            key, value, key, call.kwargs[key]
        )
    )


class TestPreprocessMock:
    @pytest.fixture(autouse=True)
    def datagen(self, mocker, batch_gen, flags):
        m = mocker.MagicMock(name='datagen', spec_set=ImageDataGenerator)
        m().flow_from_directory.return_value = batch_gen(flags.batch_size)
        m.reset_mock()
        mocker.patch(
            'tensorflow.keras.preprocessing.image.ImageDataGenerator', m
        )
        mocker.patch('tin.train.ImageDataGenerator', m)
        return m

    @pytest.fixture(
        params=[
            pytest.param(0, id='flags1'),
            pytest.param(1, id='flags2'),
        ]
    )
    def flags(self, mocker, mock_flags, request):
        if request.param == 0:
            mock_flags.validation_split = 0.5
            mock_flags.batch_size = 1
            mock_flags.seed = 42
        else:
            mock_flags.validation_split = 0.4
            mock_flags.batch_size = 2
            mock_flags.seed = 21
        return mock_flags

    @pytest.fixture(
        scope='class',
        params=[
            pytest.param(0, id='train'),
            pytest.param(1, id='val'),
        ]
    )
    def split(self, request):
        return request.param

    def test_datagen_called(self, flags, datagen):
        preprocess(flags)
        datagen.assert_called_once()

    def test_flow_called(self, flags, datagen):
        preprocess(flags)
        datagen.assert_called_once()

    def test_train_val_tuple_returned(self, flags):
        ret = preprocess(flags)
        assert isinstance(ret, tuple)
        assert len(ret) == 2

    def test_image_ds_yields_img_label_tuples(self, flags, split):
        example = self.call(flags, split)
        assert isinstance(example, tuple)
        assert len(example) == 2

    def test_image_shape(self, flags, split):
        img, label = self.call(flags, split)
        assert img.shape == (flags.batch_size, 64, 64, 3)

    def test_label_shape(self, flags, split):
        img, label = self.call(flags, split)
        assert label.shape == (flags.batch_size, )

    def test_deterministic_output(self, flags, split):
        img1, lab1 = self.call(flags, split)
        img2, lab2 = self.call(flags, split)
        assert tensorflow.reduce_all(tensorflow.math.equal(img1, img2))
        assert tensorflow.reduce_all(tensorflow.math.equal(lab1, lab2))

    def call(self, flags, split):
        print(preprocess(flags))
        print(preprocess(flags)[split])
        ds = list(preprocess(flags)[split].take(1))
        print(ds)
        return ds[0]

    @pytest.mark.parametrize('split', [
        pytest.param(0.5),
    ])
    def test_valdation_split_from_flag(self, datagen, flags, split):
        flags.validation_split = split
        preprocess(flags)
        datagen.assert_called_once()
        assert datagen.call_args.kwargs['validation_split'] == split


class TestConstructModel:
    @pytest.mark.parametrize(
        'kwarg,flag', [
            pytest.param('num_classes', 'num_classes', id='num_classes'),
            pytest.param('l1', 'l1', id='l1'),
            pytest.param('l2', 'l2', id='l2'),
            pytest.param('dropout', 'dropout', id='dropout'),
            pytest.param('seed', 'seed', id='seed'),
        ]
    )
    def test_head_constructor_args(self, head, mock_flags, kwarg, flag):
        value = getattr(mock_flags, flag)
        construct_model(mock_flags)
        head.assert_called_once()
        assert_kwarg_call(head.call_args, kwarg, value)

    @pytest.mark.parametrize(
        'kwarg,flag', [
            pytest.param('levels', 'levels', id='levels'),
            pytest.param('width', 'width', id='width'),
        ]
    )
    @pytest.mark.usefixtures('head')
    def test_model_constructor_flag_args(self, model, mock_flags, kwarg, flag):
        value = getattr(mock_flags, flag)
        construct_model(mock_flags)
        model.assert_called_once()
        assert_kwarg_call(model.call_args, kwarg, value)

    @pytest.mark.parametrize(
        'kwarg,val', [
            pytest.param('use_tail', True, id='use_tail'),
        ]
    )
    @pytest.mark.usefixtures('head')
    def test_model_constructor_fixed_args(self, model, mock_flags, kwarg, val):
        construct_model(mock_flags)
        model.assert_called_once()
        assert_kwarg_call(model.call_args, kwarg, val)

    def test_model_use_head(self, model, head, mock_flags):
        construct_model(mock_flags)
        model.assert_called_once()
        assert_kwarg_call(model.call_args, 'use_head', head())


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
