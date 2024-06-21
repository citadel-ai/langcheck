import json

import pytest
from langcheck.metrics import (
    contains_all_strings,
    contains_any_strings,
    contains_regex,
    is_float,
    is_int,
    is_json_array,
    is_json_object,
    matches_regex,
    validation_fn,
)

from tests.utils import is_close, lists_are_equal

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,domain,metric_values',
    [
        ('1', None, [1]),
        (['-100', '-1', '0', '1', '100'], None, [1, 1, 1, 1, 1]),
        (['-100', '-1', '0', '1', '100'], range(-5, 6), [0, 1, 1, 1, 0]),
        (['-100', '-1', '0', '1', '100'], {0, 1, 2}, [0, 0, 1, 1, 0]),
        (
            [
                'lorem', 'ipsum', '13.14', '-999.999', 'true', 'True', 'false',
                'False'
            ],
            None,
            [0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ],
)
def test_is_int(generated_outputs, domain, metric_values):
    metric_value = is_int(generated_outputs, domain)
    assert metric_value.metric_name == 'is_int'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,min,max,metric_values',
    [
        ('1', None, None, [1]),
        (['-100.5', '-1', '0', '1', '100.5'], None, None, [1, 1, 1, 1, 1]),
        (['-100.5', '-1', '0', '1', '100.5'], None, 5, [1, 1, 1, 1, 0]),
        (['-100.5', '-1', '0', '1', '100.5'], -5, None, [0, 1, 1, 1, 1]),
        (['-100.5', '-1', '0', '1', '100.5'], -5, 5, [0, 1, 1, 1, 0]),
        (
            [
                'lorem', 'ipsum', '13.14', '-999.999', 'true', 'True', 'false',
                'False'
            ],
            None,
            None,
            [0, 0, 1, 1, 0, 0, 0, 0],
        ),
    ],
)
def test_is_float(generated_outputs, min, max, metric_values):
    metric_value = is_float(generated_outputs, min, max)
    assert metric_value.metric_name == 'is_float'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize('generated_outputs,metric_values', [
    ('''[1, 2, 3]''', [1]),
    (['''[1, 2, 3]'''], [1]),
    (['''["lorem", 1.5, -123.456, {"foo": 9999}, [1, 2, 3]]'''], [1]),
    (['''[1, 2, 3'''], [0]),
    (['''1, 2, 3]'''], [0]),
    (['''1, 2, 3'''], [0]),
    (['''[1, 2, 3]]'''], [0]),
    (['''[[1, 2, 3]'''], [0]),
    (['''"foo"'''], [0]),
    (['''1'''], [0]),
    (['''{"a": 1, "b": 2.5}'''], [0]),
])
def test_is_json_array(generated_outputs, metric_values):
    metric_value = is_json_array(generated_outputs)
    assert metric_value.metric_name == 'is_json_array'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize('generated_outputs,metric_values', [
    ('''{"a": 1, "b": 2.5}''', [1]),
    (['''{"a": 1, "b": 2.5}'''], [1]),
    (['''{"a": "foo", "b": -9999, "c": {"d": "e"}, "f": [1]}'''], [1]),
    (['''"a": 1, "b": 2.5}'''], [0]),
    (['''{"a": 1, "b": 2.5'''], [0]),
    (['''"a": 1, "b": 2.5'''], [0]),
    (['''{"a": 1, "b": 2.5}}'''], [0]),
    (['''{{"a": 1, "b": 2.5}'''], [0]),
    (['''"foo"'''], [0]),
    (['''1'''], [0]),
    (['''[1, 2, 3]'''], [0]),
])
def test_is_json_object(generated_outputs, metric_values):
    metric_value = is_json_object(generated_outputs)
    assert metric_value.metric_name == 'is_json_object'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,regex,metric_values',
    [
        (
            'foo@example.com',
            r'[^@^\s]+@[^@^\s]+\.[^@^\s]+',
            [1],
        ),
        (
            ['foo@example.com', 'his email is foo@example.com'],
            r'[^@^\s]+@[^@^\s]+\.[^@^\s]+',
            [1, 0],
        ),
        (
            ['1234', '123456789', 'my ID is 123456789'],
            r'\d{5,}',
            [0, 1, 0],
        ),
        (
            ['$123', '$123.45', '$123.4567', '¥123', 'the price is $123'],
            r'\$\d+(\.\d\d)?',
            [1, 1, 0, 0, 0],
        ),
    ],
)
def test_matches_regex(generated_outputs, regex, metric_values):
    metric_value = matches_regex(generated_outputs, regex)
    assert metric_value.metric_name == 'matches_regex'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,regex,metric_values',
    [
        (
            'foo@example.com',
            r'[^@^\s]+@[^@^\s]+\.[^@^\s]+',
            [1],
        ),
        (
            ['foo@example.com', 'his email is foo@example.com'],
            r'[^@^\s]+@[^@^\s]+\.[^@^\s]+',
            [1, 1],
        ),
        (['1234', '123456789', 'my ID is 123456789'], r'\d{5,}', [0, 1, 1]),
        (
            ['$123', '$123.45', '$123.4567', '¥123', 'the price is $123'],
            r'\$\d+(\.\d\d)?',
            [1, 1, 1, 0, 1],
        ),
    ],
)
def test_contains_regex(generated_outputs, regex, metric_values):
    metric_value = contains_regex(generated_outputs, regex)
    assert metric_value.metric_name == 'contains_regex'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,strings,case_sensitive,metric_values',
    [
        (
            'As an AI language model, ...',
            ['as an ai language model'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['as an ai language model'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['ai', 'language model'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['ai', 'language model', 'foo'],
            False,
            [0],
        ),
        (
            ['As an AI language model, ...'],
            ['as an ai language model'],
            True,
            [0],
        ),
    ],
)
def test_contains_all_strings(generated_outputs, strings, case_sensitive,
                              metric_values):
    metric_value = contains_all_strings(generated_outputs, strings,
                                        case_sensitive)
    assert metric_value.metric_name == 'contains_all_strings'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize(
    'generated_outputs,strings,case_sensitive,metric_values',
    [
        (
            'As an AI language model, ...',
            ['as an ai language model'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['as an ai language model'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['ai', 'language model'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['ai', 'language model', 'foo'],
            False,
            [1],
        ),
        (
            ['As an AI language model, ...'],
            ['as an ai language model'],
            True,
            [0],
        ),
    ],
)
def test_contains_any_strings(generated_outputs, strings, case_sensitive,
                              metric_values):
    metric_value = contains_any_strings(generated_outputs, strings,
                                        case_sensitive)
    assert metric_value.metric_name == 'contains_any_strings'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)


@pytest.mark.parametrize('generated_outputs,valid_fn,metric_values', [
    ('2', lambda x: int(x) % 2 == 0, [1]),
    (['2', '4', '9', '11'], lambda x: int(x) % 2 == 0, [1, 1, 0, 0]),
    (['''{"myKey": 123}'''], lambda x: 'myKey' in json.loads(x), [1]),
    (['''{"foo": 123}'''], lambda x: 'myKey' in json.loads(x), [0]),
    (['''"myKey": 123'''], lambda x: 'myKey' in json.loads(x), [0]),
    (['lorem ipsum'], lambda x: x / 0, [0]),
])
def test_validation_fn(generated_outputs, valid_fn, metric_values):
    metric_value = validation_fn(generated_outputs, valid_fn)
    assert metric_value.metric_name == 'validation_fn'
    assert metric_value.prompts is None
    assert metric_value.generated_outputs is not None
    assert isinstance(metric_value.generated_outputs, list)
    assert lists_are_equal(generated_outputs, metric_value.generated_outputs)
    assert is_close(metric_value.metric_values, metric_values)
