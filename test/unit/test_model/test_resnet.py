#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from tin.resnet import *
import pytest
from test.test_base import BaseModelTest


class TestTail(BaseModelTest):

    @pytest.fixture(
        params=[
            pytest.param((2,12,12,1), id='2x12x12x1'),
            pytest.param((1,8,10,3), id='1x8x10x3'),
        ]
    )
    def input_shape(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pytest.param(8, id='c=8'),
            pytest.param(12, id='c=12'),
        ]
    )
    def output_shape(self, request, input_shape):
        return *input_shape[:-1], request.param

    @pytest.fixture
    def model(self, output_shape):
        return Tail(output_shape[-1])

class TestBottleneck(BaseModelTest):

    @pytest.fixture(
        params=[
            pytest.param((2,12,12,24), id='2x12x12x24'),
            pytest.param((1,8,10,16), id='1x8x10x16'),
        ]
    )
    def input_shape(self, request):
        return request.param

    @pytest.fixture
    def output_shape(self, request, input_shape):
        return input_shape

    @pytest.fixture
    def model(self, output_shape):
        return Bottleneck(output_shape[-1])

class TestDownsample(BaseModelTest):

    @pytest.fixture(
        params=[
            pytest.param((2,12,12,24), id='2x12x12x24'),
            pytest.param((1,8,10,16), id='1x8x10x16'),
        ]
    )
    def input_shape(self, request):
        return request.param

    @pytest.fixture
    def output_shape(self, request, input_shape):
        r, c = input_shape[1] // 2, input_shape[2] // 2
        return input_shape[0], r, c, input_shape[-1] * 2

    @pytest.fixture
    def model(self, output_shape):
        return Downsample(output_shape[-1])

class TestHead(BaseModelTest):

    @pytest.fixture(
        params=[
            pytest.param((2,12,12,1), id='2x12x12x1'),
            pytest.param((1,8,10,3), id='1x8x10x3'),
        ]
    )
    def input_shape(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pytest.param(8, id='num_classes=8'),
            pytest.param(12, id='num_classes=12'),
        ]
    )
    def output_shape(self, request, input_shape):
        return input_shape[0], request.param

    @pytest.fixture
    def model(self, output_shape):
        return TinyImageNetHead(output_shape[-1], dropout=0.1)

class TestMultiStageHead(TestHead):
    pass

class TestTinyImageNet(BaseModelTest):

    @pytest.fixture(
        params=[
            pytest.param((2, 64, 64, 3), id='2x64x64x3'),
            pytest.param((3, 48, 48, 1), id='3x48x48x1'),
        ]
    )
    def input_shape(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pytest.param(8, id='num_classes=8'),
            pytest.param(12, id='num_classes=12'),
        ]
    )
    def output_shape(self, request, input_shape):
        return (input_shape[0], request.param)

    @pytest.fixture
    def model(self, input_shape, output_shape):
        levels = [1, 1, 1]
        head = TinyImageNetHead(output_shape[-1])
        return TinyImageNet(levels=levels, use_head=head, use_tail=True)

    @pytest.fixture
    def mock_model(self, mocker, model):
        for attr, obj in list(vars(model).items()):
            if isinstance(obj, tf.keras.layers.Layer):
                m = mocker.MagicMock(name=obj.name, spec=obj, spec_set=obj)
                setattr(model, attr, m)

        for i, obj in enumerate(model.blocks):
            m = mocker.MagicMock(name=obj.name, spec=obj, spec_set=obj)
            model.blocks[i] = m

        return model
