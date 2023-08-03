import os
from argparse import ArgumentParser
import toml
import logging
from pathlib import Path
from ..models import get_stat_config_formatter
from ..models.quantize import transform_stat_profile_to_int_quant_config
from ..models import get_config_cls

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def cli_transform_stat_profile_to_int_quant_config():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--statistic_profile", type=str, required=True)
    parser.add_argument("--range_entry", type=str, default="range_min_max")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--integer_width", type=int, default=8)
    parser.add_argument("--integer_width_toml", type=str, default=None)
    parser.add_argument("--frac_width_choices", type=int, nargs="+", default=None)
    parser.add_argument("--frac_width_choices_toml", type=str, default=None)
    parser.add_argument("--is_qat", action="store_true")
    parser.add_argument("--bypass", action="store_true")

    args = parser.parse_args()

    if args.integer_width is not None and args.integer_width_toml is not None:
        raise ValueError(
            "Only one of integer_width and integer_width_toml can be specified"
        )
    elif args.integer_width_toml is not None:
        args.integer_width = toml.load(args.integer_width_toml)

    if args.frac_width_choices is not None and args.frac_width_choices_toml is not None:
        raise ValueError(
            "Only one of frac_width_choices and frac_width_choices_toml can be specified"
        )
    elif args.frac_width_choices_toml is not None:
        args.frac_width_choices = toml.load(args.frac_width_choices_toml)

    config = get_config_cls(args.model_arch).from_pretrained(
        args.model_name, quant_config={"default": {"name": "integer", "bypass": True}}
    )
    config_formatter = get_stat_config_formatter(args.model_arch)

    stat_profile = toml.load(args.statistic_profile)
    quant_config = transform_stat_profile_to_int_quant_config(
        stat_profile,
        range_entry=args.range_entry,
        width=args.integer_width,
        frac_choices=args.frac_width_choices,
        root_name="root",
        is_ptq=not args.is_qat,
        bypass=args.bypass,
    )
    quant_config = config_formatter(
        config=quant_config,
        num_hidden_layers=config.num_hidden_layers,
        default_config={
            "name": "integer",
            "bypass": False,
            "is_ptq": True,
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 4,
            "weight_frac_width": 2,
            "bias_width": 4,
            "bias_frac_width": 2,
        },
        is_ptq=not args.is_qat,
        bypass=args.bypass,
    )

    if args.save_name is not None:
        save_name = Path(args.save_name)
        save_name.parent.mkdir(parents=True, exist_ok=True)
        with open(save_name, "w") as f:
            toml.dump(quant_config, f)
        logger.info(f"Quant config saved to {save_name}")


if __name__ == "__main__":
    cli_transform_stat_profile_to_int_quant_config()
