import torch
from tqdm import tqdm

from .stats import _StatBase, create_new_stat


class ActStatCollection:
    def __init__(self, stats: list[str] | dict[str, dict]):
        self.stats: list[_StatBase] = []
        if isinstance(stats, (list, tuple)):
            for stat_name in stats:
                self.stats.append(create_new_stat(stat_name))
        elif isinstance(stats, dict):
            for stat_name, stat_kwargs in stats.items():
                self.stats.append(create_new_stat(stat_name, **stat_kwargs))
        else:
            raise ValueError(f"Unknown type of stats: {type(stats)}")

    def update(self, batch: torch.Tensor):
        assert isinstance(
            batch, torch.Tensor
        ), f"batch must be a Tensor, got {type(batch)}"
        for stat in self.stats:
            if hasattr(stat, "update_a_batch"):
                stat.update_a_batch(batch)
            else:
                for i in range(batch.size(0)):
                    stat.update_a_sample(batch[[i], ...])

    def compute(self) -> dict:
        results = {}
        for stat in self.stats:
            results.update(stat.export())
        return results

    def __repr__(self) -> str:
        return "ActStatCollection(stats={})".format(
            ", ".join([type(stat).__name__ for stat in self.stats])
        )


class WeightStatCollection:
    def __init__(self, stats: list[str] | dict[str, dict]) -> None:
        self.stats: list[_StatBase] = []
        if isinstance(stats, dict):
            for stat_name, stat_config in stats.items():
                self.stats.append(create_new_stat(stat_name, **stat_config))
        elif isinstance(stats, (list, tuple)):
            for stat_name in stats:
                self.stats.append(create_new_stat(stat_name))
        else:
            raise ValueError(f"Unknown type of stats: {type(stats)}")

    def update(self, weight: torch.Tensor):
        assert isinstance(weight, torch.Tensor)
        for stat in self.stats:
            stat.update_a_sample(weight)

    def compute(self) -> dict[str, dict[str, list]]:
        results = {}
        for stat in self.stats:
            results.update(stat.export())

        return results

    def __repr__(self) -> str:
        return "WeightStatCollection(stats={})".format(
            ", ".join([type(stat).__name__ for stat in self.stats])
        )


class StatManager:
    def __init__(
        self,
        act_stats: tuple[str] | dict[str, dict],
        weight_stats: tuple[str] | dict[str, dict],
    ) -> None:
        self.act_stats = act_stats
        self.weight_stats = weight_stats

        self.registered_stats = {}
        self.weight_collect_updated = {}

    def get_pre_forward_act_hook(self, name: str) -> callable:
        assert (
            name not in self.registered_stats
        ), f"The name `{name}` has been registered for a collection of input activations"
        new_act_clc = ActStatCollection(self.act_stats)
        self.registered_stats[name] = new_act_clc

        def hook(module: torch.nn.Module, input: tuple) -> None:
            new_act_clc.update(input[0])
            return None

        return hook

    def get_post_forward_act_hook(self, name: str) -> callable:
        assert (
            name not in self.registered_stats
        ), f"The name `{name}` has been registered for a collection of output activations"
        new_act_clc = ActStatCollection(self.act_stats)
        self.registered_stats[name] = new_act_clc

        def hook(module: torch.nn.Module, input: tuple, output: tuple) -> None:
            new_act_clc.update(output[0])
            return None

        return hook

    def get_pre_forward_weight_hook(self, name: str, weight_name: str) -> callable:
        assert (
            name not in self.registered_stats
        ), f"The name `{name}` has been registered for a collection of weights"

        new_weight_clc = WeightStatCollection(self.weight_stats)
        self.registered_stats[name] = new_weight_clc
        self.weight_collect_updated[name] = False

        def hook(module: torch.nn.Module, input: tuple) -> None:
            weight = getattr(module, weight_name)
            if self.weight_collect_updated[name]:
                pass
            else:
                new_weight_clc.update(weight)
                self.weight_collect_updated[name] = True
            return None

        return hook

    def finalize(self, show_progress_bar: bool = False) -> dict[str, dict[str, dict]]:
        """
        {
            <name>: {
                <stat_name> : {...}
            }
        }

        <name> is the name of the registered stat collection.
        """
        if show_progress_bar:
            stat_iter = tqdm(
                self.registered_stats.items(),
                desc="Finalizing stats",
                total=len(self.registered_stats),
            )
        else:
            stat_iter = self.registered_stats.items()
        results = {}
        for name, stat in stat_iter:
            delta = {name: stat.compute()}
            results.update(delta)
            if show_progress_bar:
                stat_iter.update(1)
        return results
