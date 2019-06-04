import pytest
from tin.model import Tail

import tensorflow.keras.layers as layers

@pytest.fixture
def inpt_maps():
    return 10

class TestConv2D():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'Conv2D')

    @pytest.fixture
    def tail(self, spy, inpt_maps):
        return Tail(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, tail):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, tail, spy):
        spy.assert_called_once()

    @pytest.mark.parametrize('arg_name,arg_val', [
        ('strides', 2),
        ('kernel_size', (7, 7)),
        ('use_bias', False),
        ('name', 'tail_conv'),
    ])
    def test_default_arg(self, arg_name, arg_val, tail, spy_args):
        assert(spy_args.get(arg_name) == arg_val)

    def test_inpt_map_arg(self, tail, inpt_maps, spy):
        args, kwargs = spy.call_args
        assert(kwargs.get('filters') == inpt_maps)

    @pytest.mark.parametrize('arg_name,arg_val', [
        ('strides', 3),
        ('kernel_size', (3, 3)),
        ('use_bias', True),
        ('name', 'name'),
    ])
    def test_kwarg_assignment(self, inpt_maps, arg_name, arg_val, spy):
        kwargs = {arg_name: arg_val}
        Tail(inpt_maps, conv_kwargs = kwargs)
        _, spy_args = spy.call_args
        assert(spy_args.get(arg_name) == arg_val)

class TestBatchNorm():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'BatchNormalization')

    @pytest.fixture
    def tail(self, spy, inpt_maps):
        return Tail(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, tail):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, tail, spy):
        spy.assert_called_once()

class TestReLU():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'ReLU')

    @pytest.fixture
    def tail(self, spy, inpt_maps):
        return Tail(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, tail):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, tail, spy):
        spy.assert_called_once()

class TestMaxPool():

    @pytest.fixture
    def spy(self, mocker):
        return mocker.spy(layers, 'MaxPool2D')

    @pytest.fixture
    def tail(self, spy, inpt_maps):
        return Tail(inpt_maps)

    @pytest.fixture
    def spy_args(self, spy, tail):
        args, kwargs = spy.call_args
        return kwargs

    def test_called(self, tail, spy):
        spy.assert_called_once()

    @pytest.mark.parametrize('arg_name,arg_val', [
            ('pool_size', (2, 2)),
            ('strides', 2),
            ('name', 'tail_pool')
    ])
    def test_default_arg(self, arg_name, arg_val, tail, spy_args):
        assert(spy_args.get(arg_name) == arg_val)

    @pytest.mark.parametrize('arg_name,arg_val', [

            ('pool_size', (1, 1)),
            ('strides', 1),
            ('name', 'name')
    ])
    def test_kwarg_assignment(self, inpt_maps, arg_name, arg_val, spy):
        kwargs = {arg_name: arg_val}
        Tail(inpt_maps, pool_kwargs = kwargs)
        _, spy_args = spy.call_args
        assert(spy_args.get(arg_name, None) == arg_val)
