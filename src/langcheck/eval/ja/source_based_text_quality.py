from __future__ import annotations

from typing import List

from transformers import pipeline

from langcheck.eval.en.source_based_text_quality import \
    factual_consistency as en_factual_consistency
from langcheck.eval.eval_value import EvalValue

_factual_consistency_translation_model_path = 'staka/fugumt-ja-en'
_factual_consistency_translation_pipeline = None


def factual_consistency(generated_outputs: List[str],
                        sources: List[str]) -> EvalValue[float]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. The factual consistency score for one generated output is
    computed as the average of the per-sentence consistencies of the generated
    output with the source text, where the consistency is computed by querying
    the UniEval-fact model that has been pre-trained to evaluate factual
    consistency. This metric takes on float values between [0, 1], where 0 means
    that the output is not at all consistent with the source text, and 1 means
    that the output is fully consistent with the source text.

    Ref:
        https://github.com/maszhongming/UniEval
        https://huggingface.co/staka/fugumt-ja-en

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        sources: A list of source texts

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object.

    .. note::
        This function wraps :func:`~langcheck.eval.en.en_factual_consistency`
        using the translation model ``staka/fugumt-ja-en`` to translate the
        Japanese texts to English before computing the factual consistency
        scores. This is because the UniEval-fact model is trained on English
        text.
    '''

    # TODO: Unify the validation that we do in all of the evaluation functions
    if len(generated_outputs) != len(sources):
        raise ValueError(
            'The generated outputs and sources lists must be of the same '
            'length')

    # The UniEval-fact model takes quite some time to download, so we early
    # return here to avoid unnecessarily downloading it
    if len(generated_outputs) == 0:
        return EvalValue(metric_name='factual_consistency',
                         prompts=None,
                         generated_outputs=[],
                         reference_outputs=[],
                         sources=[],
                         metric_values=[],
                         language='ja')

    global _factual_consistency_translation_pipeline
    if _factual_consistency_translation_pipeline is None:
        _factual_consistency_translation_pipeline = pipeline(
            'translation', model=_factual_consistency_translation_model_path)

    # Translate the sources and generated outputs to English.
    en_source = [
        d['translation_text']
        for d in _factual_consistency_translation_pipeline(sources)
    ]
    en_generated_outputs = [
        d['translation_text']
        for d in _factual_consistency_translation_pipeline(generated_outputs)
    ]
    # Compute the factual consistency scores in English.
    factual_consistency_scores = en_factual_consistency(
        generated_outputs=en_generated_outputs, sources=en_source).metric_values

    return EvalValue(metric_name='factual_consistency',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=None,
                     sources=sources,
                     metric_values=factual_consistency_scores,
                     language='ja')
