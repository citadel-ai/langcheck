from __future__ import annotations

import json
from pathlib import Path


def load_json(filepath: str | Path) -> dict | list:
    '''Loads a provided JSON filepath as a Python dictionary or list.'''
    with open(filepath, 'r') as f:
        return json.load(f)
