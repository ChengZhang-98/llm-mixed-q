def flatten_dict(d: dict, new_d: dict, join: str = ":", name: str = "root") -> dict:
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_d, join, f"{name}{join}{k}")
        else:
            new_d[f"{name}{join}{k}"] = v


def expand_dict(d: dict, new_d: dict, join: str = ":", name: str = "root"):
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

    for k, v in d.items():
        k: str
        key_list = k.removeprefix(f"{name}{join}").split(join)
        create_nested_dict(new_d, key_list, v)
