def flatten_dict(d: dict, new_d: dict, join: str = ":", name: str = "root") -> dict:
    """
    Flatten a nested dict to a flat dict with keys joined by `join`.

    ---
    For example:
    ```python
    d = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3,
                "f": 4,
            },
        },
    }
    new_d = {}
    flatten_dict(d, new_d, join=":", name="root")
    print(new_d)
    ```
    will print
    ```text
    {
        "root:a": 1,
        "root:b:c": 2,
        "root:b:d:e": 3,
        "root:b:d:f": 4,
    }
    ```
    """
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_d, join, f"{name}{join}{k}")
        else:
            new_d[f"{name}{join}{k}"] = v


def expand_dict(d: dict, new_d: dict, join: str = ":", name: str = "root"):
    """
    Expand a flat dict to a nested dict with keys joined by `join`.

    ---
    For example:
    ```python
    d = {
        "root:a": 1,
        "root:b:c": 2,
        "root:b:d:e": 3,
        "root:b:d:f": 4,
    }
    new_d = {}
    expand_dict(d, new_d, join=":", name="root")
    print(new_d)
    ```
    will print
    ```text
    {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3,
                "f": 4,
            },
        },
    }
    ```
    """

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
