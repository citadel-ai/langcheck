from typing import List

from langcheck.eval.eval_value import EvalValue


def exact_match(generated_outputs: List[str],
                reference_outputs: List[str]) -> EvalValue[int]:
    '''Checks if the generated outputs exact matches with the reference outputs.
    This metric takes on binary 0 or 1 values.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''
    # The values are binary: 1 if it's an exact match and 0 if not
    metric_values = []
    for gen, ref in zip(generated_outputs, reference_outputs):
        if gen == ref:
            metric_values.append(1)
        else:
            metric_values.append(0)

    return EvalValue(metric_name='exact_match',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     sources=None,
                     metric_values=metric_values,
                     language=None)
