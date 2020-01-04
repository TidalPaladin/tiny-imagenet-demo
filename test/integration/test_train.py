#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tin
from tin.train import *
from tin.flags import parse_args
import pytest
import os
import tensorflow
import logging


def assert_kwarg_call(call, key, value):
    assert key in call.kwargs, ('{} not in call kwargs'.format(key))
    assert call.kwargs[key] == value, (
        'expected kwarg {}={}, got {}={}'.format(
            key, value, key, call.kwargs[key]
        )
    )


@pytest.fixture
def flags(src_dir, artifacts_dir):
    flags = [
        '--src=%s' % src_dir,
        '--validation_split=0.5',
        '--num_classes=3',
        '--model=resnet',
        '--epochs=1',
        '--batch_size=3',
        '--levels',
        '1',
        '1',
    ]
    return parse_args(flags)


class TestPreprocess:
    @pytest.fixture(
        scope='class',
        params=[
            pytest.param(0, id='train'),
            pytest.param(1, id='val'),
        ]
    )
    def split(self, request):
        return request.param

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
        return list(preprocess(flags)[split].take(1))[0]

    @pytest.mark.parametrize(
        'split', [
            pytest.param(0.5),
            pytest.param(1.0, marks=pytest.mark.xfail(raises=ValueError)),
        ]
    )
    def test_val_split(self, flags, split):
        flags.validation_split = split
        train, val = preprocess(flags)
        train1, _ = list(train.take(1))[0]
        val1, _ = list(val.take(1))[0]
        assert not tensorflow.reduce_all(tensorflow.math.equal(train1, val1))


class TestTrain:
    def test_main_call(self, flags, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                main(flags)
