import pathlib
import os


FILE_DIR = pathlib.Path(__file__).parent.resolve()


def _read_file(filename: str):
    with open(os.path.join(FILE_DIR, filename), "r") as f:
        content = f.read()
    return content

_template_names = ["alpaca", "marie_no_context", "marie_with_context"]
PROMPT_TEMPLATES = {
    template_name: _read_file(template_name + ".txt") for template_name in _template_names
}
