#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import tensorflow
import tin


def pytest_report_header(config):
    return "tensorflow version: {}".format(tensorflow.__version__)


@pytest.fixture
def head(mocker, request, mock_flags):
    m = mocker.MagicMock(
        name='TinyImageNetHead', spec_set=tin.resnet.TinyImageNetHead
    )
    mocker.patch('tin.resnet.TinyImageNetHead', m)
    mocker.patch('tin.train.TinyImageNetHead', m)
    return m


@pytest.fixture(params=['TinyImageNet', 'TinyInceptionNet'])
def model(mocker, request, mock_flags):
    if request.param == 'TinyImageNet':
        spec = tin.resnet.TinyImageNet
        m = mocker.MagicMock(name='model', spec_set=spec)
        mocker.patch('tin.resnet.TinyImageNet', m)
        mocker.patch('tin.train.TinyImageNet', m)
        mock_flags.inception = False
        mock_flags.resnet = True
    else:
        spec = tin.inception.TinyInceptionNet
        m = mocker.MagicMock(name='model', spec_set=spec)
        mocker.patch('tin.inception.TinyInceptionNet', m)
        mocker.patch('tin.train.TinyInceptionNet', m)
        mock_flags.inception = True
        mock_flags.resnet = False
    return m
