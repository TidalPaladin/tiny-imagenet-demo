import pytest
from tin.model import ResnetBasic

import tensorflow.keras.layers as layers

@pytest.fixture
def inpt_maps():
    return 10

class TestConv2D():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'Conv2D')

    @pytest.fixture
    def model(self, spy, inpt_maps):
        return ResnetBasic(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, model):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, model, spy):
        spy.assert_called_once()

    @pytest.mark.parametrize('arg_name,arg_val', [
        ('strides', 1),
        ('kernel_size', (3, 3)),
        ('use_bias', False),
    ])
    def test_default_arg(self, arg_name, arg_val, model, spy_args):
        assert(spy_args.get(arg_name) == arg_val)

    def test_inpt_map_arg(self, model, inpt_maps, spy):
        args, kwargs = spy.call_args
        assert(kwargs.get('filters') == inpt_maps)

    @pytest.mark.parametrize('arg_name,arg_val', [
        ('strides', 2),
        ('kernel_size', (1, 1)),
        ('use_bias', True),
        ('name', 'model_conv'),
    ])
    def test_kwarg_assignment(self, inpt_maps, arg_name, arg_val, spy):
        kwargs = {arg_name: arg_val}
        ResnetBasic(inpt_maps, conv_kwargs = kwargs)
        _, spy_args = spy.call_args
        assert(spy_args.get(arg_name) == arg_val)

class TestBatchNorm():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'BatchNormalization')

    @pytest.fixture
    def model(self, spy, inpt_maps):
        return ResnetBasic(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, model):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, model, spy):
        spy.assert_called_once()

class TestReLU():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'ReLU')

    @pytest.fixture
    def model(self, spy, inpt_maps):
        return ResnetBasic(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, model):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, model, spy):
        spy.assert_called_once()
