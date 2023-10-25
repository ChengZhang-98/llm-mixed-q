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
import logging
from functools import partial
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantizers import (block_fp_quantizer, block_log_quantizer,
                          block_minifloat_quantizer, integer_quantizer,
                          log_quantizer, minifloat_denorm_quantizer,
                          minifloat_ieee_quantizer)

logger = logging.getLogger(__name__)


class _LinearBase(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config: dict = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.config = config
        self.bypass = config.get("bypass", False)
        self.is_ptq = config.get("is_ptq", False)
        self.weight_requires_quantisation = True if self.is_ptq else False
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None

        if not self.bypass:
            self._setup_quantizers(config)

    def _setup_quantizers(self, config: dict):
        """
        Setup quantizers for input, weight and bias
        """
        raise NotImplementedError

    def forward(self, x):
        if self.bypass:
            # if bypss, there is no quantization
            return F.linear(x, self.weight, self.bias)
        elif self.is_ptq:
            with torch.no_grad():
                x = self.x_quantizer(x)
                if self.weight_requires_quantisation:
                    self.weight.copy_(self.w_quantizer(self.weight.data))
                    if self.bias is not None:
                        self.bias.copy_(self.b_quantizer(self.bias.data))
                    self.weight_requires_quantisation = False
            return F.linear(x, self.weight, self.bias)
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            bias = self.b_quantizer(self.bias) if self.bias is not None else None
            return F.linear(x, w, bias)

    @classmethod
    def from_float(cls, linear_fp32: nn.Linear, config: dict):
        linear = cls(
            linear_fp32.in_features,
            linear_fp32.out_features,
            bias=linear_fp32.bias is not None,
            config=config,
        )
        _copy_weight(linear, linear_fp32)
        return linear

    def __repr__(self):
        txt = "{}(in_features={}, out_features={}, bias={}, bypass={}, is_ptq={}, x/w/b-width={}/{}/{})".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.bypass,
            self.is_ptq,
            self.config["data_in_width"],
            self.config["weight_width"],
            self.config.get("bias_width", "NA"),
        )
        return txt


@torch.no_grad()
def _copy_weight(linear: _LinearBase, linear_fp32: nn.Linear):
    linear.weight.copy_(linear_fp32.weight)
    if linear.bias is not None:
        linear.bias.copy_(linear_fp32.bias)

    return linear


class LinearBlockFP(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            block_fp_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
            block_size=config["data_in_block_size"],
            skip_first_dim=True,
        )
        self.w_quantizer = partial(
            block_fp_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
            block_size=config["weight_block_size"],
            skip_first_dim=False,
        )
        self.b_quantizer = (
            partial(
                block_fp_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
                block_size=config["bias_block_size"],
                skip_first_dim=False,
            )
            if self.bias is not None
            else None
        )


class LinearBlockLog(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            block_log_quantizer,
            width=config["data_in_width"],
            exponent_bias_width=config["data_in_exponent_bias_width"],
            block_size=config["data_in_block_size"],
            skip_first_dim=True,
        )
        self.w_quantizer = partial(
            block_log_quantizer,
            width=config["weight_width"],
            exponent_bias_width=config["weight_exponent_bias_width"],
            block_size=config["weight_block_size"],
            skip_first_dim=False,
        )
        self.b_quantizer = (
            partial(
                block_log_quantizer,
                width=config["bias_width"],
                exponent_bias_width=config["bias_exponent_bias_width"],
                block_size=config["bias_block_size"],
                skip_first_dim=False,
            )
            if self.bias is not None
            else None
        )


class LinearBlockMinifloat(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            block_minifloat_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias_width=config["data_in_exponent_bias_width"],
            block_size=config["data_in_block_size"],
            skip_first_dim=True,
        )
        self.w_quantizer = partial(
            block_minifloat_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias_width=config["weight_exponent_bias_width"],
            block_size=config["weight_block_size"],
            skip_first_dim=False,
        )
        self.b_quantizer = (
            partial(
                block_minifloat_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias_width=config["bias_exponent_bias_width"],
                block_size=config["bias_block_size"],
                skip_first_dim=False,
            )
            if self.bias is not None
            else None
        )


class LinearInteger(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            integer_quantizer,
            width=config["data_in_width"],
            frac_width=config["data_in_frac_width"],
            is_signed=True,
        )
        self.w_quantizer = partial(
            integer_quantizer,
            width=config["weight_width"],
            frac_width=config["weight_frac_width"],
            is_signed=True,
        )
        self.b_quantizer = (
            partial(
                integer_quantizer,
                width=config["bias_width"],
                frac_width=config["bias_frac_width"],
                is_signed=True,
            )
            if self.bias is not None
            else None
        )


class LinearLog(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            log_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
        )
        self.w_quantizer = partial(
            log_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
        )
        self.b_quantizer = (
            partial(
                log_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
            )
            if self.bias is not None
            else None
        )


class LinearMinifloatDenorm(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            minifloat_denorm_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
        )
        self.w_quantizer = partial(
            minifloat_denorm_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
        )
        self.b_quantizer = (
            partial(
                minifloat_denorm_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
            )
            if self.bias is not None
            else None
        )


class LinearMinifloatIEEE(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
        )
        self.w_quantizer = partial(
            minifloat_ieee_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
        )
        self.b_quantizer = (
            partial(
                minifloat_ieee_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
            )
            if self.bias is not None
            else None
        )
