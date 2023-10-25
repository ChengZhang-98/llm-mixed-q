# coding=utf-8
# Copyright 2023, Cheng Zhang, PhD student at Imperial College London, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from math import log2
from typing import Union

import torch
from numpy import ndarray
from torch import Tensor

from .utils import my_clamp, my_round


def _integer_quantize(
    x: Union[Tensor, ndarray], width: int, frac_width: int, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), int_min, int_max) / scale


class IntegerQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width, frac_width, is_signed):
        return _integer_quantize(
            x, width=width, frac_width=frac_width, is_signed=is_signed
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        STE shortcut for saving GPU memory
        """
        return grad_output, None, None, None


def integer_quantizer(
    x: Union[Tensor, ndarray], width: int, frac_width: int, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    return IntegerQuantize.apply(x, width, frac_width, is_signed)


def integer_fraction(
    width: int, frac_choices: list, min_value: float, max_value: float
):
    max_half_range = max(abs(min_value), abs(max_value))
    int_width = int(log2(max(0.5, max_half_range))) + 2
    frac_width = max(0, width - int_width)
    frac_width = max(filter(lambda x: x <= frac_width, frac_choices))
    return frac_width
