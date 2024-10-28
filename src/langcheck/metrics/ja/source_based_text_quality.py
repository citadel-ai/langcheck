from __future__ import annotations

from typing import cast

from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

from langcheck.metrics.en.source_based_text_quality import (
    factual_consistency as en_factual_consistency,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import (
    get_metric_inputs,
    get_metric_inputs_with_required_lists,
)
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progress_bar import tqdm_wrapper

_factual_consistency_translation_model_path = "Helsinki-NLP/opus-mt-ja-en"
_factual_consistency_translation_pipeline: Pipeline | None = None

LANG = "ja"


def factual_consistency(
    generated_outputs: list[str] | str,
    sources: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
) -> MetricValue[float | None]:
    """Calculates the factual consistency between the generated outputs and
    the sources. This metric takes on float values between [0, 1], where 0
    means that the output is not at all consistent with the source text, and 1
    means that the output is fully consistent with the source text. (NOTE: when
    using an EvalClient, the factuality scores are either 0.0, 0.5, or 1.0.
    The score may also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the 'unieval-fact' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.
    This function wraps :func:`~langcheck.metrics.en.en_factual_consistency`
    using the translation model ``Helsinki-NLP/opus-mt-ja-en`` to translate the
    Japanese texts to English before computing the factual consistency
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
    """
    metric_inputs, [generated_outputs, sources] = (
        get_metric_inputs_with_required_lists(
            generated_outputs=generated_outputs,
            sources=sources,
            prompts=prompts,
            required_params=["generated_outputs", "sources"],
        )
    )

    metric_name = "factual_consistency"
    if eval_model == "local":
        scores = _factual_consistency_local(generated_outputs, sources)
        explanations = None
        return MetricValue(
            metric_name=metric_name,
            metric_inputs=metric_inputs,
            explanations=explanations,
            # type: ignore (pyright doesn't understand that a list of floats is a list of optional floats)
            metric_values=scores,
            language="en",
        )
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), "An EvalClient must be provided for non-local model types."

        factual_consistency_template = eval_model.load_prompt_template(
            language=LANG, metric_name=metric_name
        )
        factual_consistency_assessment_to_score = {
            "Fully Consistent": 1.0,
            "Partially Consistent": 0.5,
            "Not Consistent": 0.0,
        }

        return eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=factual_consistency_template,
            metric_name=metric_name,
            language=LANG,
            score_map=factual_consistency_assessment_to_score,
        )


def _factual_consistency_local(
    generated_outputs: list[str], sources: list[str]
) -> list[float]:
    """Calculates the factual consistency between each generated sentence and
    its corresponding source text. The factual consistency score for one
    generated output is computed as the average of the per-sentence
    consistencies of the generated output with the source text. This function
    wraps :func:`~langcheck.metrics.en.en_factual_consistency` using the
    translation model ``Helsinki-NLP/opus-mt-ja-en`` to translate the Japanese
    texts to English before computing the factual consistency scores. This is
    because the UniEval-fact model is trained on English text.

    Ref:
        https://github.com/maszhongming/UniEval
        https://huggingface.co/Helsinki-NLP/opus-mt-ja-en

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output

    Returns:
        A list of scores
    """

    global _factual_consistency_translation_pipeline
    if _factual_consistency_translation_pipeline is None:
        _factual_consistency_translation_pipeline = pipeline(
            "translation",
            model=_factual_consistency_translation_model_path,
            truncation=True,
        )

    # Translate the sources and generated outputs to English.
    # Currently, the type checks are not working for the pipeline, since
    # too diverse types can be returned.
    batch_size = 8
    en_source = []
    for i in tqdm_wrapper(
        range(0, len(sources), batch_size),
        desc="Translating sources",
        total=(len(sources) + batch_size - 1) // batch_size,
    ):
        batch_sources = sources[i : i + batch_size]
        en_source.extend(
            [
                cast(str, d["translation_text"])  # type: ignore[reportGeneralTypeIssues]
                for d in _factual_consistency_translation_pipeline(
                    batch_sources
                )  # type: ignore[reportGeneralTypeIssues]
            ]
        )
    en_generated_outputs = []
    for i in tqdm_wrapper(
        range(0, len(generated_outputs), batch_size),
        desc="Translating generated outputs",
        total=(len(generated_outputs) + batch_size - 1) // batch_size,
    ):
        batch_generated_outputs = generated_outputs[i : i + batch_size]
        en_generated_outputs.extend(
            [
                cast(str, d["translation_text"])  # type: ignore[reportGeneralTypeIssues]
                for d in _factual_consistency_translation_pipeline(
                    batch_generated_outputs
                )  # type: ignore[reportGeneralTypeIssues]
            ]
        )

    # Compute the factual consistency scores in English.
    factual_consistency_scores = en_factual_consistency(
        generated_outputs=en_generated_outputs, sources=en_source
    ).metric_values

    # Local factual consistency scores are of type list[float]
    return factual_consistency_scores  # type: ignore


def context_relevance(
    sources: list[str] | str, prompts: list[str] | str, eval_model: EvalClient
) -> MetricValue[float | None]:
    """Calculates the relevance of the sources to the prompts. This metric takes
    on float values between [0, 1], where 0 means that the source text is not at
    all relevant to the prompt, and 1 means that the source text is fully
    relevant to the prompt.

    We currently only support the evaluation based on an EvalClient.

    Args:
        sources: The source text(s), one string per prompt
        prompts: The prompt(s)
        eval_model: The EvalClient instance used for the evaluation
    """
    metric_inputs = get_metric_inputs(
        prompts=prompts,
        sources=sources,
        required_params=["prompts", "sources"],
    )

    metric_name = "context_relevance"
    context_relevance_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    context_relevance_assessment_to_score = {
        "Fully Relevant": 1.0,
        "Partially Relevant": 0.5,
        "Not Relevant": 0.0,
    }

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=context_relevance_template,
        metric_name=metric_name,
        language=LANG,
        score_map=context_relevance_assessment_to_score,
    )
