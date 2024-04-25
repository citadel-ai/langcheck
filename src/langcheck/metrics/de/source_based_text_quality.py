from __future__ import annotations

from typing import List, Optional, Tuple

from langcheck.metrics._validation import (
    validate_parameters_context_relevance, validate_parameters_source_based)
from langcheck.metrics.de._translation import Translate
from langcheck.metrics.en.source_based_text_quality import \
    factual_consistency as en_factual_consistency
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template

_factual_consistency_translation_model_path = 'Helsinki-NLP/opus-mt-de-en'

LANG = 'de'


def factual_consistency(
        generated_outputs: List[str] | str,
        sources: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        eval_model: str | EvalClient = 'local') -> MetricValue[Optional[float]]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. This metric takes on float values between [0, 1], where 0
    means that the output is not at all consistent with the source text, and 1
    means that the output is fully consistent with the source text. (NOTE: when
    using an EvalClient, the factuality scores are either 0.0, 0.5, or 1.0.
    The score may also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the 'unieval-fact' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.
    This function wraps :func:`~langcheck.metrics.en.factual_consistency`
    using the translation model ``Helsinki-NLP/opus-mt-de-en`` to translate the
    German texts to English before computing the factual consistency
    scores. This is because the UniEval-fact model is trained on English
    text.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'

    Returns:
        An MetricValue object
    '''
    generated_outputs, sources, prompts = validate_parameters_source_based(
        generated_outputs, sources, prompts)

    if eval_model != 'local':  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), 'An EvalClient must be provided for non-local model types.'
        scores, explanations = _factual_consistency_eval_client(
            generated_outputs, sources, eval_model)

        return MetricValue(metric_name='factual_consistency',
                           prompts=prompts,
                           generated_outputs=generated_outputs,
                           reference_outputs=None,
                           sources=sources,
                           explanations=explanations,
                           metric_values=scores,
                           language=LANG)

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    translation = Translate(_factual_consistency_translation_model_path)
    batch_size = 8
    en_source = []
    for i in tqdm_wrapper(range(0, len(sources), batch_size),
                          desc='Translating sources',
                          total=(len(sources) + batch_size - 1) // batch_size):
        batch_sources = sources[i:i + batch_size]
        en_source.extend([translation(src) for src in batch_sources])
    en_generated_outputs = []
    for i in tqdm_wrapper(range(0, len(generated_outputs), batch_size),
                          desc='Translating generated outputs',
                          total=(len(generated_outputs) + batch_size - 1) //
                          batch_size):
        batch_generated_outputs = generated_outputs[i:i + batch_size]
        en_generated_outputs.extend(
            [translation(gen_out) for gen_out in batch_generated_outputs])

    # Compute the factual consistency scores in English.
    metric_value = en_factual_consistency(
        generated_outputs=en_generated_outputs, sources=en_source)
    metric_value.language = LANG
    return metric_value


def _factual_consistency_eval_client(
    generated_outputs: List[str], sources: List[str], eval_client: EvalClient
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    '''Calculates the factual consistency and their associated explanations
    between the generated outputs and the sources using an EvalClient. This
    metric takes on float values that are either 0, 0.5, or 1, where 0 means
    that the output is not at all consistent with the source text, and 1 means
    that the output is fully consistent with the source text. If a score could
    not be computed, `None` is inserted to the score and explanation lists.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
        eval_client: The EvalClient instance used for the evaluation

    Returns:
        score_list: a list of scores
        explanation_list: a list of explanations for the scores
    '''
    factual_consistency_template = get_template(
        'de/metrics/factual_consistency.j2')

    factual_consistency_assessment_to_score = {
        'Fully Consistent': 1.0,
        'Partially Consistent': 0.5,
        'Not Consistent': 0.0
    }
    populated_prompts = [
        factual_consistency_template.render({
            'src': source,
            'gen_output': gen_output
        }) for source, gen_output in zip(sources, generated_outputs)
    ]

    scores, explanations = eval_client.get_score(
        metric_name='factual consistency',
        language='de',
        prompts=populated_prompts,
        score_map=factual_consistency_assessment_to_score,
    )

    return scores, explanations


def context_relevance(sources: List[str] | str, prompts: List[str] | str,
                      eval_model: EvalClient) -> MetricValue[Optional[float]]:
    '''Calculates the relevance of the sources to the prompts. This metric takes
    on float values between [0, 1], where 0 means that the source text is not at
    all relevant to the prompt, and 1 means that the source text is fully
    relevant to the prompt.

    We currently only support the evaluation based on an EvalClient.

    Args:
        sources: The source text(s), one string per prompt
        prompts: The prompt(s)
        eval_model: The EvalClient instance used for the evaluation
    '''
    prompts, sources = validate_parameters_context_relevance(prompts, sources)

    context_relevance_template = get_template('de/metrics/context_relevance.j2')

    context_relevance_assessment_to_score = {
        'Fully Relevant': 1.0,
        'Partially Relevant': 0.5,
        'Not Relevant': 0.0
    }

    populated_prompts = [
        context_relevance_template.render({
            'src': source,
            'user_query': prompt,
        }) for source, prompt in zip(sources, prompts)
    ]

    scores, explanations = eval_model.get_score(
        metric_name='context relevance',
        language='de',
        prompts=populated_prompts,
        score_map=context_relevance_assessment_to_score,
    )

    return MetricValue(metric_name='context_relevance',
                       prompts=prompts,
                       generated_outputs=None,
                       reference_outputs=None,
                       sources=list(sources),
                       explanations=explanations,
                       metric_values=scores,
                       language=LANG)
