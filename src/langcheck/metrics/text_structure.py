from __future__ import annotations

import json
import re
from collections.abc import Callable, Container, Iterable

from langcheck.metrics.metric_inputs import (
    get_metric_inputs_with_required_lists,
)
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progress_bar import tqdm_wrapper


def is_int(
    generated_outputs: list[str] | str,
    domain: Iterable[int] | Container[int] | None = None,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs can be parsed as integers, optionally within
    a domain of integers like `range(1, 11)` or `{1, 3, 5}`. This metric takes
    on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        domain: The optional domain of valid integers
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        try:
            output_int = int(output)
            if domain is None or output_int in domain:
                metric_values.append(1)
            else:
                metric_values.append(0)
        except ValueError:
            metric_values.append(0)

    return MetricValue(
        metric_name="is_int",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def is_float(
    generated_outputs: list[str] | str,
    min: float | None = None,
    max: float | None = None,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs can be parsed as floating point numbers,
    optionally within a min/max range. This metric takes on binary 0 or 1
    values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        min: The optional minimum valid float
        max: The optional maximum valid float
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.


    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        try:
            output_float = float(output)
            if min is None and max is None:
                metric_values.append(1)
            elif min is not None and output_float < min:
                metric_values.append(0)
            elif max is not None and output_float > max:
                metric_values.append(0)
            else:
                metric_values.append(1)
        except ValueError:
            metric_values.append(0)

    return MetricValue(
        metric_name="is_float",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def is_json_object(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs can be parsed as JSON objects. This metric
    takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        try:
            json_output = json.loads(output)
            if isinstance(json_output, dict):
                metric_values.append(1)
            else:
                metric_values.append(0)
        except json.JSONDecodeError:
            metric_values.append(0)

    return MetricValue(
        metric_name="is_json_object",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def is_json_array(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs can be parsed as JSON arrays. This metric
    takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        try:
            json_output = json.loads(output)
            if isinstance(json_output, list):
                metric_values.append(1)
            else:
                metric_values.append(0)
        except json.JSONDecodeError:
            metric_values.append(0)

    return MetricValue(
        metric_name="is_json_array",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def matches_regex(
    generated_outputs: list[str] | str,
    regex: str,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs fully match a given regular expression. This
    metric takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        regex: The regular expression to match
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        if re.fullmatch(regex, output) is not None:
            metric_values.append(1)
        else:
            metric_values.append(0)

    return MetricValue(
        metric_name="matches_regex",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def contains_regex(
    generated_outputs: list[str] | str,
    regex: str,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs partially contain a given regular expression.
    This metric takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        regex: The regular expression to match
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        if re.search(regex, output) is not None:
            metric_values.append(1)
        else:
            metric_values.append(0)

    return MetricValue(
        metric_name="contains_regex",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def contains_all_strings(
    generated_outputs: list[str] | str,
    strings: list[str],
    case_sensitive: bool = False,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs contain all strings in of a given list. This
    metric takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        strings: A list of strings to match
        case_sensitive: Whether to match case sensitively or not, default False
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # Convert everything to lowercase if case insensitive
    if not case_sensitive:
        _strings = [string.lower() for string in strings]
        _generated_outputs = [output.lower() for output in generated_outputs]
    else:
        _strings = strings
        _generated_outputs = generated_outputs

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(_generated_outputs):
        if all(string in output for string in _strings):
            metric_values.append(1)
        else:
            metric_values.append(0)

    return MetricValue(
        metric_name="contains_all_strings",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def contains_any_strings(
    generated_outputs: list[str] | str,
    strings: list[str],
    case_sensitive: bool = False,
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs contain any strings in a given list. This
    metric takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        strings: A list of strings to match
        case_sensitive: Whether to match case sensitively or not, default to
            :obj:`False`.
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # Convert everything to lowercase if case insensitive
    if not case_sensitive:
        _strings = [string.lower() for string in strings]
        _generated_outputs = [output.lower() for output in generated_outputs]
    else:
        _strings = strings
        _generated_outputs = generated_outputs

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(_generated_outputs):
        if any(string in output for string in _strings):
            metric_values.append(1)
        else:
            metric_values.append(0)

    return MetricValue(
        metric_name="contains_any_strings",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )


def validation_fn(
    generated_outputs: list[str] | str,
    valid_fn: Callable[[str], bool],
    prompts: list[str] | str | None = None,
) -> MetricValue[int]:
    """Checks if generated outputs are valid according to an arbitrary function.
    This metric takes on binary 0 or 1 values.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        valid_fn: A function that takes a single string and returns a
            bool determining whether the string is valid or not.
            The function can also raise an exception on failure.
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in tqdm_wrapper(generated_outputs):
        try:
            if valid_fn(output):
                metric_values.append(1)
            else:
                metric_values.append(0)
        except Exception:
            metric_values.append(0)

    return MetricValue(
        metric_name="validation_fn",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=metric_values,
        language=None,
    )
