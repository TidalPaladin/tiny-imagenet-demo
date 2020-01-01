#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import tensorflow
import numpy as np
import inspect
import os
import unittest.mock as mock
from PIL import Image


def pytest_report_header(config):
    return "tensorflow version: {}".format(tensorflow.__version__)


@pytest.fixture(scope='session')
def src_dir(tmpdir_factory):
    dir = tmpdir_factory.mktemp('data')
    for cls in range(3):
        subdir = 'n000%d' % cls
        os.makedirs(os.path.join(dir, subdir))
        for n in range(3):
            a = np.random.rand(64, 64, 3) * 255
            im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
            im_path = os.path.join(dir, subdir, '%s_00%d.JPEG' % (subdir, n))
            print(im_path)
            im_out.save(im_path)
    print(os.listdir(dir))
    return dir


@pytest.fixture
def artifacts_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('artif')


@pytest.fixture(autouse=True)
def mock_flags(mocker, src_dir, artifacts_dir):
    m = mocker.MagicMock(name='FLAGS')
    m.src = src_dir
    m.artifacts_dir = artifacts_dir
    mocker.patch('tin.train.FLAGS', m)
    return m


@pytest.fixture
def tf(mocker):
    m = mocker.MagicMock()
    mocker.patch('tin.train.tf', m)
    return m
