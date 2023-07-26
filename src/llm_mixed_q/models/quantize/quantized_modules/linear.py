from typing import Any, Mapping
import torch
from functools import partial
import torch.nn.functional as F
import torch.nn as nn

from ..quantizers import (
    integer_quantizer,
    block_fp_quantizer,
    block_minifloat_quantizer,
    block_log_quantizer,
    minifloat_ieee_quantizer,
    minifloat_denorm_quantizer,
    log_quantizer,
)


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
        self.w_quantizer = self._w_quantizer = None
        self.b_quantizer = self._b_quantizer = None

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
                if self.weight_requires_quantisation:
                    self.weight.copy_(self._w_quantizer(self.weight.data))
                    if self.bias is not None:
                        self.bias.copy_(self._b_quantizer(self.bias.data))
                    self.weight_requires_quantisation = False
            return F.linear(x, self.weight, self.bias)
        else:
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
        txt = (
            "{}(in_features={}, out_features={}, bias={}, bypass={}, is_ptq={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.bypass,
                self.is_ptq,
            )
        )
        return txt


@torch.no_grad()
def _copy_weight(linear: _LinearBase, linear_fp32: nn.Linear):
    if linear.is_ptq:
        linear.weight.copy_(linear_fp32.weight)
        if linear.bias is not None:
            linear.bias.copy_(linear_fp32.bias)
    else:
        linear.weight.copy_(linear._w_quantizer(linear_fp32.weight))
        if linear.bias is not None:
            linear.bias.copy_(linear._b_quantizer(linear_fp32.bias))
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
        self._w_quantizer = partial(
            block_fp_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
            block_size=config["weight_block_size"],
            skip_first_dim=False,
        )
        self._b_quantizer = (
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
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x


class LinearBlockLog(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            block_log_quantizer,
            width=config["data_in_width"],
            exponent_bias_width=config["data_in_exponent_bias_width"],
            block_size=config["data_in_block_size"],
            skip_first_dim=True,
        )
        self._w_quantizer = partial(
            block_log_quantizer,
            width=config["weight_width"],
            exponent_bias_width=config["weight_exponent_bias_width"],
            block_size=config["weight_block_size"],
            skip_first_dim=False,
        )
        self._b_quantizer = (
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
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x


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
        self._w_quantizer = partial(
            block_minifloat_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias_width=config["weight_exponent_bias_width"],
            block_size=config["weight_block_size"],
            skip_first_dim=False,
        )
        self._b_quantizer = (
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
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x


class LinearInteger(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            integer_quantizer,
            width=config["data_in_width"],
            frac_width=config["data_in_frac_width"],
            is_signed=True,
        )
        self._w_quantizer = partial(
            integer_quantizer,
            width=config["weight_width"],
            frac_width=config["weight_frac_width"],
            is_signed=True,
        )
        self._b_quantizer = (
            partial(
                integer_quantizer,
                width=config["bias_width"],
                frac_width=config["bias_frac_width"],
                is_signed=True,
            )
            if self.bias is not None
            else None
        )
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x


class LinearLog(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            log_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
        )
        self._w_quantizer = partial(
            log_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
        )
        self._b_quantizer = (
            partial(
                log_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
            )
            if self.bias is not None
            else None
        )
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x


class LinearMinifloatDenorm(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            minifloat_denorm_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
        )
        self._w_quantizer = partial(
            minifloat_denorm_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
        )
        self._b_quantizer = (
            partial(
                minifloat_denorm_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
            )
            if self.bias is not None
            else None
        )
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x


class LinearMinifloatIEEE(_LinearBase):
    def _setup_quantizers(self, config: dict):
        self.x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=config["data_in_width"],
            exponent_width=config["data_in_exponent_width"],
            exponent_bias=config["data_in_exponent_bias"],
        )
        self._w_quantizer = partial(
            minifloat_ieee_quantizer,
            width=config["weight_width"],
            exponent_width=config["weight_exponent_width"],
            exponent_bias=config["weight_exponent_bias"],
        )
        self._b_quantizer = (
            partial(
                minifloat_ieee_quantizer,
                width=config["bias_width"],
                exponent_width=config["bias_exponent_width"],
                exponent_bias=config["bias_exponent_bias"],
            )
            if self.bias is not None
            else None
        )
        if self.is_ptq:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x