import numpy
import random
import numpy as np
import torch
from pprint import pprint, pformat
import time
from torch import nn, Tensor
from enum import Enum
import torch.nn.functional as F
from loguru import logger as logging
from collections import defaultdict
from typing import Dict, NamedTuple, List, Optional, Tuple, Callable, Union
import itertools


class ParamData(object):

    def __init__(
        self,
        name: str,
        shape: Optional[tuple] = None,
        init_function: Callable[[torch.Size], torch.Tensor] = torch.zeros,
        init_tensor: Union[torch.Tensor, None] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        An object that contains a tensor with methods to expand it along
        a single dimension.

        :param name: data tensor name as a string
        :param shape: data tensor shape. Will be set to the `init_tensor`
            shape, if provided.
        :param init_function: function used to initialize the data tensor.
            If `init_tensor` is provided, `init_function` will only be used
            on subsequent calls of `reset_like` method.
        :param init_tensor: value to be used when creating the object. If None,
            `init_function` will be used.
        :param device: pytorch like device specification as a string or
            `torch.device`.
        """
        assert isinstance(name, str)
        assert (init_tensor is not None) or (shape is not None)
        if init_tensor is not None and shape is not None:
            assert init_tensor.shape == shape

        self.init_function = init_function
        self.name = name
        if shape is not None:
            self.shape = torch.Size(shape)
        else:
            assert init_tensor is not None
            self.shape = init_tensor.size()

        self.device = torch.device(device)
        if init_tensor is not None:
            self._data: torch.Tensor = init_tensor
        else:
            self.reset_like()

    def reset_like(self, shape=None, init_function=None):
        """
        Reset the tensor with the shape provided or, otherwise, by
        using the one most recently provided. The `init_function`,
        if provided, does not override the default one.

        :param shape: the new shape or None to use the current one
        :param init_function: init function to use or None to use
            the default one.
        """
        if shape is not None:
            self.shape = torch.Size(shape)
        if init_function is None:
            init_function = self.init_function
        self._data = init_function(self.shape).to(self.device)

    def expand(self, new_shape, padding_fn=torch.zeros):
        """
        Expand the data tensor along one dimension.
        The shape cannot shrink. It cannot add new dimensions, either.
        If the shape does not change, this method does nothing.

        :param new_shape: expanded shape
        :param padding_fn: function used to create the padding
            around the expanded tensor.

        :return the expanded tensor or the previous tensor
        """
        assert len(new_shape) == len(
            self.shape), "Expansion cannot add new dimensions"
        expanded = False
        for i, (snew, sold) in enumerate(zip(new_shape, self.shape)):
            assert snew >= sold, "Shape cannot decrease."
            if snew > sold:
                assert (not expanded
                        ), "Expansion cannot occur in more than one dimension."
                expanded = True
                exp_idx = i

        if expanded:
            old_data = self._data.clone()
            old_shape_len = self._data.shape[exp_idx]
            self.reset_like(new_shape, init_function=padding_fn)
            idx = [
                slice(el) if i != exp_idx else slice(old_shape_len)
                for i, el in enumerate(new_shape)
            ]
            self._data[idx] = old_data
        return self.data

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value):
        assert value.shape == self._data.shape, (
            "Shape of new value should be the same of old value. "
            "Use `expand` method to expand one dimension. "
            "Use `reset_like` to reset with a different shape.")
        self._data = value

    def __str__(self):
        return f"ParamData_{self.name}:{self.shape}:{self._data}"


def zerolike_params_dict(model: nn.Module) -> Dict[str, "ParamData"]:
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    """

    return dict([(k, ParamData(k, p.shape, device=p.device))
                 for k, p in model.named_parameters()])


def copy_params_dict(model, copy_grad=False) -> Dict[str, "ParamData"]:
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """
    out: Dict[str, ParamData] = {}
    for k, p in model.named_parameters():
        if copy_grad and p.grad is None:
            continue
        init = p.grad.data.clone() if copy_grad else p.data.clone()
        out[k] = ParamData(k, p.shape, device=p.device, init_tensor=init)
    return out
