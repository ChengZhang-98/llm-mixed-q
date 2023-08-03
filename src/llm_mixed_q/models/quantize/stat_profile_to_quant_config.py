import math


def find_int_frac_width(width, max_half_range, frac_choices=None):
    assert max_half_range > 0, f"max_half_range must be positive, got {max_half_range}"
    assert width > 0, f"width must be positive, got {width}"
    upper_limit = 2 ** (width - 1) - 1
    frac_width = math.floor(math.log2(upper_limit / max_half_range))
    if frac_choices is not None:
        frac_width = max(filter(lambda x: x <= frac_width, frac_choices))
    return frac_width


def create_nested_dict(d: dict, key_list: list[str], value):
    if len(key_list) == 1:
        if key_list[0] not in d:
            d[key_list[0]] = value
        elif isinstance(d[key_list[0]], dict):
            d[key_list[0]].update(value)
        else:
            raise ValueError(
                f"Cannot create nested dict at {key_list} with value {value}"
            )
    else:
        if key_list[0] not in d:
            d[key_list[0]] = {}
        create_nested_dict(d[key_list[0]], key_list[1:], value)


def transform_stat_profile_to_int_quant_config(
    stat_profile: dict,
    range_entry: int,
    width: int | dict[str, int],
    frac_choices: list[int] | tuple[int] | dict[str, list[int] | tuple[int]] = None,
    root_name: str = "root",
    is_ptq: bool = True,
    bypass: bool = False,
):
    quant_config = {}
    for name, stat in stat_profile.items():
        tgt_stat = stat[range_entry]
        max_half_range = max(abs(tgt_stat["min"]), abs(tgt_stat["max"]))
        if isinstance(width, dict):
            entry_width = width[f"{name}_width"]
        elif isinstance(width, int):
            entry_width = width
        else:
            raise ValueError(f"Unknown type of width: {type(width)}")

        if isinstance(frac_choices, dict):
            entry_frac_choices = frac_choices[name]
        elif isinstance(frac_choices, (list, tuple)):
            entry_frac_choices = frac_choices
        elif frac_choices is None:
            entry_frac_choices = None
        else:
            raise ValueError(f"Unknown type of frac_choices: {type(frac_choices)}")

        entry_frac_width = find_int_frac_width(
            entry_width, max_half_range, entry_frac_choices
        )

        name = name.removeprefix(f"{root_name}:")
        layer_name_keys, entry_name = name.split(":")[:-1], name.split(":")[-1]

        create_nested_dict(
            d=quant_config,
            key_list=layer_name_keys,
            value={
                f"bypass": bypass,
                f"name": "integer",
                f"is_ptq": is_ptq,
                f"{entry_name}_width": entry_width,
                f"{entry_name}_frac_width": entry_frac_width,
            },
        )

    return quant_config
