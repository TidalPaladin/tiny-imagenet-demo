#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import pytest
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import inspect


def pytest_report_header(config):
    return "tf version: {}".format(tf.__version__)


class BaseModelTest:
    @pytest.fixture
    def model(self):
        raise NotImplementedError('model fixture not implemented')

    @pytest.fixture
    def input_shape(self):
        raise NotImplementedError('model fixture not implemented')

    @pytest.fixture
    def output_shape(self):
        raise NotImplementedError('model fixture not implemented')

    @pytest.fixture
    def inputs(self, input_shape):
        return tf.ones(input_shape)

    @pytest.fixture
    def targets(self, output_shape):
        return tf.zeros(output_shape)

    @pytest.fixture
    def mock_model(self, mocker, model):
        for attr, obj in list(vars(model).items()):
            if isinstance(obj, tf.keras.layers.Layer):
                m = mocker.MagicMock(name=obj.name, spec=obj, spec_set=obj)
                setattr(model, attr, m)
        return model

    def testOutputShape(self, model, output_shape, inputs):
        outputs = model(inputs)
        assert outputs.shape == output_shape, (
            'output shape mismatch, expected {} got {}'.format(
                outputs.shape, output_shape
            )
        )

    def testTrainableWeightsUpdated(self, model, output_shape, inputs):
        targets = tf.zeros(output_shape)
        grads, _, _ = self.forward_pass(model, inputs, targets)
        for grad in grads:
            assert grad != None, (
                'got empty grad for {}, was layer called in call()?'.format(
                    grad.name
                )
            )

    @pytest.mark.parametrize('training', [True, False])
    def testTrainingStatePassedToLayerCall(self, mock_model, inputs, training):
        mock_model(inputs, training=training)
        for layer in mock_model.layers:
            sig = inspect.signature(layer.__class__.call)
            if 'training' in sig.parameters and layer.called:
                for call in layer.call_args_list:
                    assert 'training' in call.kwargs, (
                        'train kwarg not passed to call in layer {}'.format(
                            layer._extract_mock_name()
                        )
                    )
                    actual_state = call.kwargs.get('training')
                    assert actual_state == training, (
                        'incorrect training val, expected {} got {}'.format(
                            training, actual_state
                        )
                    )

    def forward_pass(self, model, inputs, targets):
        opt, loss = Adam(), BinaryCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss_value = loss(targets, outputs)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return grads, loss, outputs
