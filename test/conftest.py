#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import pytest
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import inspect
import unittest.mock as mock


def pytest_report_header(config):
    return "tf version: {}".format(tf.__version__)


@pytest.fixture(autouse=True)
def mock_flags(mocker):
    m = mocker.MagicMock(name='FLAGS')
    mocker.patch('tin.train.FLAGS', m)
    return m
