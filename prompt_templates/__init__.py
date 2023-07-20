import pathlib
import os

FILE_DIR = pathlib.Path(__file__).parent.resolve()

def _read_file(filename: str):
    with open(os.path.join(FILE_DIR, filename), "r") as f:
        content = f.read()
    return content

ALPACA_TEMPLATE = _read_file("alpaca.txt")
MARIE_NO_CONTEXT_TEMPLATE = _read_file("marie_no_context.txt")
MARIE_WITH_CONTEXT_TEMPLATE = _read_file("marie_with_context.txt")
