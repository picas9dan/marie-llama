import json
import pathlib
import os


FILE_DIR = pathlib.Path(__file__).parent.resolve()


def _read_json(json_path: str):
    with open(json_path, "r") as f:
        content = json.load(f)
    return content


_TEMPLATE_PATHS = {
    f[:-len(".json")]: os.path.join(dirpath, f) 
    for (dirpath, dirnames, filenames) in os.walk(FILE_DIR) for f in filenames
    if f.endswith(".json")
}

TEMPLATES = {
    template_name: _read_json(filepath) for template_name, filepath in _TEMPLATE_PATHS.items()
}
