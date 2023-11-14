import pathlib
from typing import Any

from ruamel.yaml import YAML


def read_yaml(filename: pathlib.Path, loader: YAML = YAML(typ="safe")) -> Any:
    """Read YAML from file

    Args:
        filename (pathlib.Path): File to read from

    Returns:
        Any: YAML content as python objects (dict, list, etc.)
    """
    if not filename.exists():
        raise FileNotFoundError(f"Unable to read yaml. Filename {filename} does not exists")

    # Read YAML from file
    with open(filename, "r") as f:
        try:
            return loader.load(f)
        except Exception as e:
            raise ValueError(f"Unable to load YAML from {filename}. Exception {e}") from e


def save_yaml(data: Any, filename: pathlib.Path):
    """Saves data to file as YAML format

    Args:
        data (Any): data to save
        filename (pathlib.Path): filename to save to
    """

    def repr_path(representer, data):
        return representer.represent_scalar("tag:yaml.org,2002:str", str(data))

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        with YAML(typ="safe", output=f) as yaml:
            for p in [pathlib.PosixPath, pathlib.WindowsPath]:
                yaml.representer.add_representer(p, repr_path)

            # yaml.register_class(pathlib.PosixPath)
            yaml.explicit_start = True
            yaml.dump(data, f)
