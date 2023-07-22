import torch
from functools import partial
import torch.nn.functional as F
import torch.nn as nn

from .quantizers import integer_quantizer, block_fp_quantizer


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
        self.is_qat = config.get("is_qat", True) if not self.bypass else True
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
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            bias = self.b_quantizer(self.bias) if self.bias is not None else None
            return F.linear(x, w, bias)

    @classmethod
    def from_float(cls, linear_fp32: nn.Linear, config: dict):
        raise NotImplementedError


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
        if self.is_qat:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x

    @classmethod
    def from_float(cls, linear_fp32: nn.Linear, config: dict):
        linear = cls(
            linear_fp32.in_features,
            linear_fp32.out_features,
            bias=linear_fp32.bias is not None,
            config=config,
        )
        with torch.no_grad():
            if linear.is_qat:
                linear.weight.copy_(linear_fp32.weight)
                if linear.bias is not None:
                    linear.bias.copy_(linear_fp32.bias)
            else:
                linear.weight.copy_(linear._w_quantizer(linear_fp32.weight))
                if linear.bias is not None:
                    linear.bias.copy_(linear._b_quantizer(linear_fp32.bias))
        return linear

    def __repr__(self):
        txt = "LinearBlockFP(in_features={}, out_features={}, bias={}, bypass={}, is_qat={})".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.bypass,
            self.is_qat,
        )
        return txt


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
        if self.is_qat:
            self.w_quantizer = self._w_quantizer
            self.b_quantizer = self._b_quantizer
        else:
            self.w_quantizer = lambda x: x
            self.b_quantizer = lambda x: x

    @classmethod
    def from_float(cls, linear_fp32: nn.Linear, config: dict):
        linear = cls(
            linear_fp32.in_features,
            linear_fp32.out_features,
            bias=linear_fp32.bias is not None,
            config=config,
        )
        with torch.no_grad():
            if linear.is_qat:
                linear.weight.copy_(linear_fp32.weight)
                if linear.bias is not None:
                    linear.bias.copy_(linear_fp32.bias)
            else:
                linear.weight.copy_(linear._w_quantizer(linear_fp32.weight))
                if linear.bias is not None:
                    linear.bias.copy_(linear._b_quantizer(linear_fp32.bias))
        return linear

    def __repr__(self):
        txt = "LinearInteger(in_features={}, out_features={}, bias={}, bypass={}, is_qat={})".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.bypass,
            self.is_qat,
        )
        return txt


QUANTIZED_MODULE_MAP = {
    "linear": {
        "block_fp": LinearBlockFP,
        "integer": LinearInteger,
    },
}
