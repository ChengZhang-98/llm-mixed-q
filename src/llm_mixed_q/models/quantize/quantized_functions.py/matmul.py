from functools import partial
import torch
from ..quantizers import integer_quantizer, block_fp_quantizer

# PyTorch has torch.matmul and torch.bmm for matrix multiplication
matmul_mapping = {"matmul": torch.matmul, "bmm": torch.bmm}


def generic_matmul_integer(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)
        y_width, y_frac_width = config["weight_width"], config["weight_frac_width"]
        y_quantizer = partial(integer_quantizer, width=y_width, frac_width=y_frac_width)

        x = x_quantizer(x)
        y = y_quantizer(y)
        # y = x_quantizer(y)

        return matmul(x, y)


def generic_matmul_block_fp(x, y, config, style="matmul"):
    bypass = config.get("bypass", False)
    matmul = matmul_mapping[style]
    if bypass:
        return matmul(x, y)
    else:
        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )
        y_width, y_exponent_width, y_exponent_bias, y_block_size = (
            config["weight_width"],
            config["weight_exponent_width"],
            config["weight_exponent_bias"],
            config["weight_block_size"],
        )
        x_more_than_2_dims = x.ndim > 2
        y_more_than_2_dims = y.ndim > 2

        x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )
        y_quantizer = partial(
            block_fp_quantizer,
            width=y_width,
            exponent_width=y_exponent_width,
            exponent_bias=y_exponent_bias,
            block_size=y_block_size,
            skip_first_dim=y_more_than_2_dims,
        )
        # flatten all other dims except for the last two dims for performing matmul
        # this is a hack for allowing block/unblock the last two dims of multiple dim tensors
        x_shape = [i for i in x.shape]
        y_shape = [i for i in y.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, 0, -3)
        if y_more_than_2_dims:
            y = torch.flatten(y, 0, -3)
        x = x_quantizer(x)
        # y = x_quantizer(y)
        y = y_quantizer(y)
        x = torch.reshape(x, x_shape)
        y = torch.reshape(y, y_shape)
        return matmul(x, y)


def bmm_integer(x, y, config):
    return generic_matmul_integer(x, y, config, "bmm")


def matmul_block_fp(x, y, config):
    return generic_matmul_block_fp(x, y, config, "matmul")


def bmm_block_fp(x, y, config):
    return generic_matmul_block_fp(x, y, config, style="bmm")


def matmul_integer(x, y, config):
    return generic_matmul_integer(x, y, config, "matmul")
