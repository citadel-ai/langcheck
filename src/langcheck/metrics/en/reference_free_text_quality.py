from __future__ import annotations

from langcheck.metrics.en.reference_based_text_quality import (
    semantic_similarity,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import (
    get_metric_inputs,
    get_metric_inputs_with_required_lists,
)
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.scorer.detoxify_models import DetoxifyScorer
from langcheck.metrics.scorer.hf_models import (
    AutoModelForSequenceClassificationScorer,
)
from langcheck.stats import compute_stats
from langcheck.utils.progress_bar import tqdm_wrapper

LANG = "en"


def sentiment(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
    local_overflow_strategy: str = "truncate",
) -> MetricValue[float | None]:
    """Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using an EvalClient, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Twitter-roBERTa-base model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
        local_overflow_strategy: The strategy to handle the inputs that are too
            long for the local model. The supported strategies are 'nullify',
            'truncate', and 'raise'. If 'nullify', the outputs that are too long
            will be assigned a score of None. If 'truncate', the outputs that
            are too long will be truncated. If 'raise', an error will be raised
            when the outputs are too long. The default value is 'nullify'.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    metric_name = "sentiment"
    if eval_model == "local":
        scores = _sentiment_local(generated_outputs, local_overflow_strategy)
        explanations = None
        return MetricValue(
            metric_name="sentiment",
            metric_inputs=metric_inputs,
            explanations=explanations,
            metric_values=scores,
            language=LANG,
        )
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), "An EvalClient must be provided for non-local model types."

        sentiment_template = eval_model.load_prompt_template(
            language=LANG, metric_name=metric_name
        )

        sentiment_assessment_to_score = {
            "Positive": 1.0,
            "Neutral": 0.5,
            "Negative": 0.0,
        }

        return eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=sentiment_template,
            metric_name=metric_name,
            language=LANG,
            score_map=sentiment_assessment_to_score,
        )


def _sentiment_local(
    generated_outputs: list[str], overflow_strategy: str
) -> list[float | None]:
    """Calculates the sentiment scores of generated outputs using the
    Twitter-roBERTa-base model. This metric takes on float values between
    [0, 1], where 0 is negative sentiment and 1 is positive sentiment.

    Ref:
        https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    """
    scorer = AutoModelForSequenceClassificationScorer(
        language="en",
        metric="sentiment",
        # Each class represents a sentiment: 0 for negative, 1 for neutral, and
        # 2 for positive
        class_weights=[0, 0.5, 1],
        overflow_strategy=overflow_strategy,
        max_input_length=512,
    )
    return scorer.score(generated_outputs)


def fluency(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
    local_overflow_strategy: str = "truncate",
) -> MetricValue[float | None]:
    """Calculates the fluency scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low fluency and 1 is high fluency.
    (NOTE: when using an EvalClient, the fluency scores are either 0.0
    (poor), 0.5 (fair), or 1.0 (good). The score may also be `None` if it could
    not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Parrot fluency model is downloaded from
    HuggingFace and run locally. This is the default model type and there is no
    setup needed to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
        local_overflow_strategy: The strategy to handle the inputs that are too
            long for the local model. The supported strategies are 'nullify',
            'truncate', and 'raise'. If 'nullify', the outputs that are too long
            will be assigned a score of None. If 'truncate', the outputs that
            are too long will be truncated. If 'raise', an error will be raised
            when the outputs are too long. The default value is 'nullify'.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    metric_name = "fluency"
    if eval_model == "local":
        scores = _fluency_local(generated_outputs, local_overflow_strategy)
        explanations = None
        return MetricValue(
            metric_name=metric_name,
            metric_inputs=metric_inputs,
            explanations=explanations,
            metric_values=scores,
            language=LANG,
        )
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), "An EvalClient must be provided for non-local model types."

        fluency_template = eval_model.load_prompt_template(
            language=LANG, metric_name=metric_name
        )

        fluency_assessment_to_score = {
            "Poor": 0,
            "Fair": 0.5,
            "Good": 1.0,
        }

        return eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=fluency_template,
            metric_name=metric_name,
            language=LANG,
            score_map=fluency_assessment_to_score,
        )


def _fluency_local(
    generated_outputs: list[str], overflow_strategy: str
) -> list[float | None]:
    """Calculates the fluency scores of generated outputs using the Parrot
    fluency model. This metric takes on float values between [0, 1], where 0 is
    low fluency and 1 is high fluency.

    Ref:
        https://huggingface.co/prithivida/parrot_fluency_model

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        overflow_strategy: The strategy to handle inputs that are longer than
            the maximum input length of the model.

    Returns:
        A list of scores
    """
    scorer = AutoModelForSequenceClassificationScorer(
        language="en",
        metric="fluency",
        # The class 1 is for fluent texts.
        class_weights=[0, 1],
        overflow_strategy=overflow_strategy,
    )
    return scorer.score(generated_outputs)


def toxicity(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
    local_overflow_strategy: str = "truncate",
    eval_prompt_version: str = "v2",
) -> MetricValue[float | None]:
    """Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.
    (NOTE: when using an EvalClient, the toxicity scores are either 0.0
    (nontoxic), or 1.0 (toxic). The score may also be `None` if it could not be
    computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the Detoxify model is downloaded from HuggingFace
    and run locally. This is the default model type and there is no setup needed
    to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
        local_overflow_strategy: The strategy to handle the inputs that are too
            long for the local model. The supported strategies are 'nullify',
            'truncate', and 'raise'. If 'nullify', the outputs that are too long
            will be assigned a score of None. If 'truncate', the outputs that
            are too long will be truncated. If 'raise', an error will be raised
            when the outputs are too long. The default value is 'nullify'.
        eval_prompt_version: The version of the eval prompt to use when the
            EvalClient is used. The default version is 'v2' (latest).

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    metric_name = "toxicity"
    if eval_model == "local":
        scores = _toxicity_local(generated_outputs, local_overflow_strategy)
        explanations = None
        return MetricValue(
            metric_name=metric_name,
            metric_inputs=metric_inputs,
            explanations=explanations,
            metric_values=scores,
            language=LANG,
        )
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), "An EvalClient must be provided for non-local model types."

        toxicity_assessment_to_score = {
            # The v1 prompt returns the toxicity on a scale of 1 to 5
            "v1": {
                "1": 0,
                "2": 0.25,
                "3": 0.5,
                "4": 0.75,
                "5": 1.0,
            },
            # The v2 prompt returns either "Toxic" or "Nontoxic"
            "v2": {
                "Toxic": 1.0,
                "Nontoxic": 0,
            },
        }
        assert (
            eval_prompt_version in toxicity_assessment_to_score
        ), f"Invalid eval_prompt_version: {eval_prompt_version}. The valid versions are {list(toxicity_assessment_to_score.keys())}."

        toxicity_template = eval_model.load_prompt_template(
            language=LANG,
            metric_name=metric_name,
            eval_prompt_version=eval_prompt_version,
        )

        return eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=toxicity_template,
            metric_name=metric_name,
            language=LANG,
            score_map=toxicity_assessment_to_score[eval_prompt_version],
        )


def _toxicity_local(
    generated_outputs: list[str], overflow_strategy: str
) -> list[float | None]:
    """Calculates the toxicity scores of generated outputs using the Detoxify
    model. This metric takes on float values between [0, 1], where 0 is low
    toxicity and 1 is high toxicity.

    Ref:
        https://github.com/unitaryai/detoxify

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        A list of scores
    """
    return DetoxifyScorer(overflow_strategy=overflow_strategy).score(
        generated_outputs
    )


def flesch_reading_ease(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
) -> MetricValue[float]:
    """Calculates the readability of generated outputs using the Flesch Reading
    Ease Score. This metric takes on float values between (-∞, 121.22], but
    typically ranges between 0 and 100, where higher scores mean the text is
    easier to read.

    The score is based on the number of sentences, words, and syllables in the
    text. See "How to Write Plain English" by Rudolf Franz Flesch for more
    details.

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

    output_stats = [
        compute_stats(output)
        for output in tqdm_wrapper(generated_outputs, desc="Computing stats")
    ]
    scores = [
        206.835
        - 1.015 * (stat.num_words / stat.num_sentences)
        - 84.6 * (stat.num_syllables / stat.num_words)
        for stat in output_stats
    ]

    return MetricValue(
        metric_name="flesch_reading_ease",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,
        language="en",
    )


def flesch_kincaid_grade(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
) -> MetricValue[float]:
    """Calculates the readability of generated outputs using the Flesch-Kincaid
    Grade Level metric. This metric takes on float values between [-3.40, ∞),
    but typically ranges between 0 and 12 (corresponding to U.S. grade levels),
    where lower scores mean the text is easier to read.

    Like the Flesch Reading Ease Score, this metric is based on the number of
    sentences, words, and syllables in the text.

    Ref:
        https://apps.dtic.mil/sti/citations/ADA006655

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

    output_stats = [
        compute_stats(output)
        for output in tqdm_wrapper(generated_outputs, desc="Computing stats")
    ]
    scores = [
        0.39 * (stat.num_words / stat.num_sentences)
        + 11.8 * (stat.num_syllables / stat.num_words)
        - 15.59
        for stat in output_stats
    ]
    return MetricValue(
        metric_name="flesch_kincaid_grade",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,
        language="en",
    )


def ai_disclaimer_similarity(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    ai_disclaimer_phrase: str = (
        "I don't have personal opinions, emotions, or consciousness."
    ),
    eval_model: str | EvalClient = "local",
) -> MetricValue[float]:
    """Calculates the degree to which the LLM's output contains a disclaimer
    that it is an AI. This is calculated by computing the semantic similarity
    between the generated outputs and a reference AI disclaimer phrase; by
    default, this phrase is "I don't have personal opinions, emotions, or
    consciousness.", but you can also pass in a custom phrase. Please refer to
    :func:`~langcheck.eval.en.reference_based_text_quality.semantic_similarity`
    for details on the typical output ranges and the supported embedding model
    types.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.
        ai_disclaimer_phrase: Reference AI disclaimer phrase, default "I don't
            have personal opinions, emotions, or consciousness."
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )

    ai_disclaimer_phrase_list = [ai_disclaimer_phrase] * len(generated_outputs)
    semantic_similarity_values = semantic_similarity(
        generated_outputs, ai_disclaimer_phrase_list, prompts, eval_model
    )
    return MetricValue(
        metric_name="ai_disclaimer_similarity",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=semantic_similarity_values.metric_values,
        language="en",
    )


def jailbreak_prompt(
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates whether jailbreak techniques are included in the prompts.
    This metric takes on float values of either 0.0 (Low Risk),
    0.5 (Medium Risk), or 1.0 (High Risk). The score may also be `None`
    if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    metric_inputs = get_metric_inputs(
        prompts=prompts,
        required_params=["prompts"],
    )

    metric_name = "jailbreak_prompt"
    jailbreak_prompt_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=jailbreak_prompt_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )


def prompt_leakage(
    generated_outputs: list[str] | str,
    system_prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the severity of prompt leakage in the generated outputs.
    This metric takes on float values of either 0.0 (Low Risk),
    0.5 (Medium Risk), or 1.0 (High Risk). The score may also be `None`
    if it could not be computed.

    We currently only support the evaluation based on an EvalClient.
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        additional_inputs={
            "system_prompts": system_prompts,
        },
        additional_input_name_to_prompt_var_mapping={
            "system_prompts": "system_prompt",
        },
        required_params=["generated_outputs", "system_prompts"],
    )

    metric_name = "prompt_leakage"

    prompt_leakage_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=prompt_leakage_template,
        metric_name=metric_name,
        language=LANG,
        score_map={
            "Low Risk": 0.0,
            "Medium Risk": 0.5,
            "High Risk": 1.0,
        },
    )
