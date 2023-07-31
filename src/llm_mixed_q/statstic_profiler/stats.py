import math
import logging
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from numpy import ndarray

logger = logging.getLogger(__name__)

STAT_NAME_TO_CLS = {}


def _add_to_stat_mapping(cls):
    global STAT_NAME_TO_CLS
    assert issubclass(cls, _StatBase)
    STAT_NAME_TO_CLS[cls.name] = cls
    return cls


class _StatBase:
    name: str = None

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def update_a_sample(self, *args, **kwargs) -> None:
        """
        Update the stat with a new sample.

        If the sample is a Tensor, it will be detached and copied to the device specified in the constructor.
        If the sample is a list, tuple, int, or float, it will be converted to a Tensor.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self) -> dict[str, Tensor]:
        """
        Compute/finalize the stat and return a dict of results.

        The results should be a dict of Tensors.
        """
        raise NotImplementedError

    def export(self) -> dict[str, dict[str, list]]:
        """
        Export the stat to a dict of dict of lists.

        This method calls compute() and converts the results to lists, which is more friendly to toml serialization.
        """
        results = self.compute()
        return {
            self.name: {
                k: v.tolist() if isinstance(v, (Tensor)) else v
                for k, v in results.items()
            }
        }

    def __repr__(self) -> str:
        return type(self).__name__.capitalize()


@_add_to_stat_mapping
class VarianceOnline(_StatBase):
    """
    Use Welford's online algorithm to calculate running variance and mean

    This saves memory by not storing all the samples, but the variance is not precise when the count is small.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
    """

    name = "variance_online"

    def __init__(self, device=None, dims: Literal["all"] | None | list = "all") -> None:
        super().__init__()
        self.device = device
        if isinstance(dims, (list, tuple)):
            # assert sorted(dims) == list(
            #     range(min(dims), max(dims) + 1)
            # ), "dims must be consecutive"
            self.dims_to_reduce = sorted(dims)
        else:
            assert dims in ["all", None]
            self.dims_to_reduce = dims

        self.count: int = 0
        self.mean: Tensor = 0
        self.m: Tensor = 0

    @staticmethod
    def _update(
        new_s: Tensor,
        count: int,
        mean: Tensor,
        m: Tensor,
    ):
        count += 1
        delta = new_s - mean
        mean += delta / count
        m += delta * (new_s - mean)

        return count, mean, m

    @staticmethod
    def _reshape_a_sample(new_s: Tensor, dims_to_reduce: list[int]):
        dims_to_keep = [i for i in range(new_s.ndim) if i not in dims_to_reduce]
        transpose_dims = dims_to_keep + dims_to_reduce
        new_s = new_s.permute(*transpose_dims)
        new_s = torch.flatten(new_s, start_dim=len(dims_to_keep), end_dim=-1)
        return new_s

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s)
        assert isinstance(new_s, Tensor)
        new_s = new_s.clone().detach().float()

        if self.device is not None:
            new_s = new_s.to(self.device)

        match self.dims_to_reduce:
            case "all":
                new_s = torch.flatten(new_s)
                n_b = new_s.nelement()
                mean_b = new_s.mean()

                delta = mean_b - self.mean
                self.mean += delta * n_b / (self.count + n_b)
                self.m += new_s.var() * n_b + delta**2 * self.count * n_b / (
                    self.count + n_b
                )
                self.count += n_b
            case None:
                self.count, self.mean, self.m = self._update(
                    new_s=new_s,
                    count=self.count,
                    mean=self.mean,
                    m=self.m,
                )
            case _:
                # self.dims_to_reduce is a list
                new_s = self._reshape_a_sample(
                    new_s, dims_to_reduce=self.dims_to_reduce
                )
                for i in range(new_s.size(-1)):
                    self.count, self.mean, self.m = self._update(
                        new_s=new_s[..., i],
                        count=self.count,
                        mean=self.mean,
                        m=self.m,
                    )

    @torch.no_grad()
    def compute(self) -> dict:
        if self.count < 2:
            logger.warning(
                f"VarianceOnline: count is {self.count}, which is less than 2. "
                "Returning NA for mean and variance."
            )
            return {"mean": "NA", "variance": "NA"}

        var = self.m / self.count
        return {
            "mean": self.mean,
            "variance": var,
            "count": self.count,
        }


@_add_to_stat_mapping
class RangeMinMax(_StatBase):
    """
    Calculate the range of samples based on the min and max values.

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
        abs (bool): if True, take the absolute value of the samples before calculating the min and max.
    """

    name = "range_min_max"

    def __init__(
        self, device=None, dims: Literal["all"] | list | None = "all", abs: bool = False
    ) -> None:
        super().__init__()
        self.device = device
        self.dims = dims
        self.abs = abs
        self.min = None
        self.max = None
        self.count = 0

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s).float()
        new_s = new_s.clone().detach().float()
        if self.device:
            new_s = new_s.to(self.device)

        if self.abs:
            new_s = torch.abs(new_s)

        if self.min is None:
            match self.dims:
                case None:
                    self.min = new_s
                    self.max = new_s
                    self.count += 1
                case "all":
                    self.min = torch.min(new_s)
                    self.max = torch.max(new_s)
                    self.count += new_s.nelement()
                case _:
                    n_elem = 1
                    for dim in self.dims:
                        n_elem *= new_s.size(dim)
                        self.min = torch.min(new_s, dim=dim)
                        self.max = torch.max(new_s, dim=dim)
                    self.count += n_elem
        else:
            match self.dims:
                case None:
                    self.min = torch.min(self.min, new_s)
                    self.max = torch.max(self.max, new_s)
                    self.count += 1
                case "all":
                    self.min = torch.min(self.min, torch.min(new_s))
                    self.max = torch.max(self.max, torch.max(new_s))
                    self.count += new_s.nelement()
                case _:
                    n_elem = 1
                    for dim in self.dims:
                        n_elem *= new_s.size(dim)
                        self.min = torch.min(self.min, torch.min(new_s, dim=dim))
                        self.max = torch.max(self.max, torch.max(new_s, dim=dim))
                    self.count += n_elem

    def compute(self) -> dict:
        if self.count < 2:
            logger.warning(
                f"RangeMinMax: count is {self.count}, which is less than 2. "
                "Returning NA for min and max."
            )
            minimum = "NA"
            maximum = "NA"
            d_range = "NA"
        else:
            minimum = self.min
            maximum = self.max
            d_range = self.max - self.min
        return {"min": minimum, "max": maximum, "range": d_range, "count": self.count}


def create_new_stat(stat_name: str, **stat_kwargs):
    global STAT_NAME_TO_CLS
    assert (
        stat_name in STAT_NAME_TO_CLS
    ), f"Unknown stat name: {stat_name}. Available stat names: {list(STAT_NAME_TO_CLS.keys())}"
    stat_cls = STAT_NAME_TO_CLS[stat_name]
    return stat_cls(**stat_kwargs)
