from __future__ import annotations

from typing import List, Optional


def validate_parameters_reference_based(
    generated_outputs: List[str] | str, reference_outputs: List[str] | str,
    prompts: Optional[List[str] | str]
) -> tuple[List[str], List[str], Optional[List[str]]]:
    '''Validates and parses function parameters for reference-based text quality
    metrics in langcheck.metrics.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        A tuple (generated_outputs, reference_outputs, prompts) of the parsed
        parameters. All non-None parameters are converted to lists of strings.
    '''
    _generated_outputs, _prompts, _reference_outputs, _ = _validate_parameters(
        generated_outputs, prompts, reference_outputs, None)

    # For type checking
    assert isinstance(_reference_outputs, list)

    return _generated_outputs, _reference_outputs, _prompts


def validate_parameters_reference_free(
    generated_outputs: List[str] | str, prompts: Optional[List[str] | str]
) -> tuple[List[str], Optional[List[str]]]:
    '''Validates and parses function parameters for reference-free text quality
    metrics in langcheck.metrics.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        A tuple (generated_outputs, prompts) of the parsed parameters. All
        non-None parameters are converted to lists of strings.
    '''
    _generated_outputs, _prompts, _, _ = _validate_parameters(
        generated_outputs, prompts, None, None)

    # For type checking
    assert _prompts is None or isinstance(_prompts, list)

    return _generated_outputs, _prompts


def validate_parameters_text_structure(
    generated_outputs: List[str] | str, prompts: Optional[List[str] | str]
) -> tuple[List[str], Optional[List[str]]]:
    '''Validates and parses function parameters for text structure metrics in
    langcheck.metrics.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        A tuple (generated_outputs, prompts) of the parsed parameters. All
        non-None parameters are converted to lists of strings.
    '''
    _generated_outputs, _prompts, _, _ = _validate_parameters(
        generated_outputs, prompts, None, None)

    # For type checking
    assert _prompts is None or isinstance(_prompts, list)

    return _generated_outputs, _prompts


def validate_parameters_source_based(
    generated_outputs: List[str] | str, sources: List[str] | str,
    prompts: Optional[List[str] | str]
) -> tuple[List[str], List[str], Optional[List[str]]]:
    '''Validates and parses function parameters for source-based text quality
    metrics in langcheck.metrics.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source(s) of the generated output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        A tuple (generated_outputs, sources, prompts) of the parsed parameters.
        All non-None parameters are converted to lists of strings.
    '''
    _generated_outputs, _prompts, _, _sources = _validate_parameters(
        generated_outputs, prompts, None, sources)

    # For type checking
    assert isinstance(_sources, list)

    return _generated_outputs, _sources, _prompts


def validate_parameters_context_relevance(
        prompts: List[str] | str,
        sources: List[str] | str) -> tuple[List[str], List[str]]:
    '''Validates and parses function parameters for the context relevance
    metric.

    Args:
        prompts: The prompt(s)
        sources: The source(s)

    Returns:
        A tuple (prompts, sources) of the parsed parameters, converted to lists
        of strings.
    '''
    # Convert single-string parameters to lists
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(sources, str):
        sources = [sources]

    # Check that prompts and sources are not empty
    if not prompts:
        raise ValueError('Please specify at least one prompt')
    if not sources:
        raise ValueError('Please specify at least one source')

    # Check that the lengths of lists match
    if len(prompts) != len(sources):
        raise ValueError('The number of prompts and sources do not match')

    return prompts, sources


def validate_parameters_answer_relevance(
        generated_outputs: List[str] | str,
        prompts: List[str] | str) -> tuple[List[str], List[str]]:
    '''Validates and parses function parameters for the answer relevance
    metric.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompt(s)

    Returns:
        A tuple (generated_outputs, prompts) of the parsed parameters, converted
        to lists of strings.
    '''
    # Convert single-string parameters to lists
    if isinstance(generated_outputs, str):
        generated_outputs = [generated_outputs]
    if isinstance(prompts, str):
        prompts = [prompts]

    # Check that generated_outputs and prompts are not empty
    if not generated_outputs:
        raise ValueError('Please specify at least one generated output')
    if not prompts:
        raise ValueError('Please specify at least one prompt')

    # Check that the lengths of lists match
    if len(generated_outputs) != len(prompts):
        raise ValueError(
            'The number of generated_outputs and prompts and do not match')

    return generated_outputs, prompts


def _validate_parameters(
    generated_outputs: List[str] | str, prompts: Optional[List[str] | str],
    reference_outputs: Optional[List[str] | str],
    sources: Optional[List[str] | str]
) -> tuple[List[str], Optional[List[str]], Optional[List[str]],
           Optional[List[str]]]:
    '''Validates and parses function parameters for metrics in
    langcheck.metrics.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        reference_outputs: The reference output(s)
        sources: The source(s) of the generated output(s)

    Returns:
        A tuple (generated_outputs, prompts, reference_outputs, sources) of the
        parsed parameters. All non-None parameters are converted to lists of
        strings.
    '''
    # Convert single-string parameters to lists
    if isinstance(generated_outputs, str):
        generated_outputs = [generated_outputs]
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(reference_outputs, str):
        reference_outputs = [reference_outputs]
    if isinstance(sources, str):
        sources = [sources]

    # Check that generated_outputs is not empty
    if not generated_outputs:
        raise ValueError('Please specify at least one generated output')

    # Check that the lengths of lists match
    if prompts is not None and len(generated_outputs) != len(prompts):
        raise ValueError(
            'The number of generated outputs and prompts do not match')
    if reference_outputs is not None and \
            len(generated_outputs) != len(reference_outputs):
        raise ValueError(
            'The number of generated outputs and reference outputs do not match'
        )
    if sources is not None and len(generated_outputs) != len(sources):
        raise ValueError(
            'The number of generated outputs and sources do not match')

    return generated_outputs, prompts, reference_outputs, sources
