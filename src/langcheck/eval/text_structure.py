from __future__ import annotations

from typing import Container, Iterable, List, Optional

from langcheck.eval.eval_value import EvalValue


def is_int(
        generated_outputs: List[str],
        domain: Iterable[int] | Container[int] | None = None
) -> EvalValue[bool]:
    '''Checks if generated outputs can be parsed as integers, optionally within
    a domain of integers like `range(1, 11)` or `{1, 3, 5}`. This metric takes
    on binary 0 or 1 values.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        domain: The optional domain of valid integers

    Returns:
        An EvalValue object
    '''
    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in generated_outputs:
        try:
            output_int = int(output)
            if domain is None or output_int in domain:
                metric_values.append(1)
            else:
                metric_values.append(0)
        except:
            metric_values.append(0)

    return EvalValue(metric_name='is_int',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     metric_values=metric_values)


def is_float(generated_outputs: List[str],
             min: Optional[float] = None,
             max: Optional[float] = None) -> EvalValue[bool]:
    '''Checks if generated outputs can be parsed as floating point numbers,
    optionally within a min/max range. This metric takes on binary 0 or 1
    values.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        min: The optional minimum valid float
        max: The optional maximum valid float

    Returns:
        An EvalValue object
    '''
    # The values are binary: 1 for success and 0 for failure
    metric_values = []
    for output in generated_outputs:
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
        except:
            metric_values.append(0)

    return EvalValue(metric_name='is_float',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     metric_values=metric_values)
