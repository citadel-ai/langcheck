import json
import tempfile

import pytest

from langcheck.utils import load_json


@pytest.mark.parametrize('json_data', [['a', 'b', 'c', 1, 2, 3], {
    'a': 1,
    'b': 2,
    'c': 3
}, {
    "a": 123,
    "b": 456,
    "c": [1, 2, 3],
    "d": {
        "e": 789
    }
}])
def test_load_json(json_data):
    # Save to JSON file in /tmp
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        json.dump(json_data, f)
        f.flush()  # Ensure the data is written to the file
        loaded_json_data = load_json(f.name)

    assert loaded_json_data == json_data
