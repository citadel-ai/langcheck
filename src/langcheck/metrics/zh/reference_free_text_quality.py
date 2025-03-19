from __future__ import annotations

import hanlp
from transformers.pipelines import pipeline

from langcheck.metrics.en.reference_free_text_quality import (
    sentiment as en_sentiment,
)
from langcheck.metrics.en.reference_free_text_quality import (
    toxicity as en_toxicity,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import (
    get_metric_inputs_with_required_lists,
)
from langcheck.metrics.metric_value import MetricValue

LANG = "zh"


def sentiment(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
) -> MetricValue[float | None]:
    """Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using an EvalClient, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where the IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
    model is downloaded from HuggingFace and run locally. This is the default
    model type and there is no setup needed to run this.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Ref:
        https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
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

    if eval_model != "local":  # EvalClient
        assert isinstance(eval_model, EvalClient), (
            "An EvalClient must be provided for non-local model types."
        )

        # This reuses the English prompt.
        # TODO: Update this to use a Chinese prompt.
        metric_value = en_sentiment(generated_outputs, prompts, eval_model)
        metric_value.language = "zh"
        return metric_value

    # {0:"Negative", 1:'Positive'}
    from langcheck.metrics.model_manager import manager

    tokenizer, model = manager.fetch_model(language="zh", metric="sentiment")
    _sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,  # type: ignore[reportGeneralTypeIssues]
        tokenizer=tokenizer,  # type: ignore[reportGeneralTypeIssues]
    )
    _model_id2label = _sentiment_pipeline.model.config.id2label
    _predict_result = _sentiment_pipeline(generated_outputs)  # type: ignore[reportGeneralTypeIssues]
    # if predicted result is 'Positive', use the score directly
    # else, use 1 - score as the sentiment score
    scores = [
        1 - x["score"] if x["label"] == _model_id2label[0] else x["score"]  # type: ignore[reportGeneralTypeIssues]
        for x in _predict_result  # type: ignore[reportGeneralTypeIssues]
    ]
    return MetricValue(
        metric_name="sentiment",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,  # type: ignore[reportGeneralTypeIssues]
        language="zh",
    )


def toxicity(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
    eval_prompt_version: str = "v2",
) -> MetricValue[float | None]:
    """Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.
    (NOTE: when using an EvalClient, the toxicity scores are in steps of
    0.25. The score may also be `None` if it could not be computed.)

    We currently support two evaluation model types:

    1. The 'local' type, where a model file is downloaded from HuggingFace and
    run locally. This is the default model type and there is no setup needed to
    run this.
    The model (alibaba-pai/pai-bert-base-zh-llm-risk-detection) is a
    risky detection model for LLM generated content released by Alibaba group.

    2. The EvalClient type, where you can use an EvalClient typically
    implemented with an LLM. The implementation details are explained in each of
    the concrete EvalClient classes.

    Ref:
        https://huggingface.co/alibaba-pai/pai-bert-base-zh-llm-risk-detection

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'
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

    if eval_model != "local":  # EvalClient
        assert isinstance(eval_model, EvalClient), (
            "An EvalClient must be provided for non-local model types."
        )

        # This reuses the English prompt.
        # TODO: Update this to use a Chinese prompt.
        metric_value = en_toxicity(generated_outputs, prompts, eval_model)
        metric_value.language = "zh"
        return metric_value
    else:
        scores = _toxicity_local(generated_outputs)
        explanations = None

        return MetricValue(
            metric_name="toxicity",
            metric_inputs=metric_inputs,
            explanations=explanations,
            # type: ignore (pyright doesn't understand that a list of floats is a list of optional floats)
            metric_values=scores,
            language="zh",
        )


def _toxicity_local(generated_outputs: list[str]) -> list[float]:
    """Calculates the toxicity scores of generated outputs using a fine-tuned
    model from `alibaba-pai/pai-bert-base-zh-llm-risk-detection`. This metric
    takes on float values between [0, 1], where 0 is low toxicity and 1 is high
    toxicity.

    Ref:
        https://huggingface.co/alibaba-pai/pai-bert-base-zh-llm-risk-detection

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        A list of scores
    """
    # this pipeline output predict probability for each text on each label.
    # the output format is list[list[dict(str)]]
    from langcheck.metrics.model_manager import manager

    tokenizer, model = manager.fetch_model(language="zh", metric="toxicity")
    _toxicity_pipeline = pipeline(
        "text-classification",
        model=model,  # type: ignore[reportOptionalIterable]
        tokenizer=tokenizer,  # type: ignore[reportOptionalIterable]
        top_k=5,
    )
    # {'Normal': 0, 'Pulp': 1, 'Sex': 2, 'Other Risk': 3, 'Adult': 4}
    _model_id2label = _toxicity_pipeline.model.config.id2label
    _predict_results = _toxicity_pipeline(
        generated_outputs  # type: ignore[reportGeneralTypeIssues]
    )
    # labels except Normal are all risky, toxicity_score = 1-score['Normal']
    toxicity_scores = []
    for item_predict_proba in _predict_results:  # type: ignore[reportOptionalIterable]
        for label_proba in item_predict_proba:  # type: ignore[reportGeneralTypeIssues]
            if label_proba["label"] == _model_id2label[0]:  # type: ignore[reportGeneralTypeIssues]
                toxicity_scores.append(1 - label_proba["score"])  # type: ignore[reportGeneralTypeIssues]
    return toxicity_scores  # type: ignore[reportGeneralTypeIssues]


def xuyaochen_report_readability(
    generated_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
) -> MetricValue[float]:
    """Calculates the readability scores of generated outputs introduced in
    "中文年报可读性"(Chinese annual report readability). This metric calculates
    average words per sentence as r1, average of the sum of the numbers of
    adverbs and coordinating conjunction words in a sentence in given generated
    outputs as r2, then, refer to the Fog Index that combine r1 with r2 by
    arithmetic mean as the final outputs. This function uses HanLP Tokenizer and
    POS at the same time, POS in CTB style
    https://hanlp.hankcs.com/docs/annotations/pos/ctb.html.
    The lower the score is, the better the readability. The score is mainly
    influenced by r1, the average number of words in sentences.

    Ref:
        Refer Chinese annual report readability: measurement and test
        Link: https://www.tandfonline.com/doi/full/10.1080/21697213.2019.1701259

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        A list of scores
    """
    # split generated_outputs into sentence
    metric_inputs, [generated_outputs] = get_metric_inputs_with_required_lists(
        generated_outputs=generated_outputs,
        prompts=prompts,
        required_params=["generated_outputs"],
    )
    tokenizer = hanlp.load(
        hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH  # type: ignore[reportGeneralTypeIssues]
    )
    postagger = hanlp.load(
        hanlp.pretrained.pos.CTB9_POS_RADICAL_ELECTRA_SMALL  # type: ignore[reportGeneralTypeIssues]
    )

    pos_pipeline = hanlp.pipeline().append(hanlp.utils.rules.split_sentence)  # type: ignore[reportGeneralTypeIssues]
    pos_pipeline = pos_pipeline.append(tokenizer).append(postagger)

    tokenize_pipeline = hanlp.pipeline().append(
        hanlp.utils.rules.split_sentence  # type: ignore[reportGeneralTypeIssues]
    )
    tokenize_pipeline = tokenize_pipeline.append(tokenizer)
    # OUTPUT: List[List[List[TOKEN]]]
    output_tokens = list(map(tokenize_pipeline, generated_outputs))
    # List[List[List[POS]]]
    output_pos = list(map(pos_pipeline, generated_outputs))

    def count_tokens(sent_tokens: list[str]) -> int:
        count = sum(
            [
                not hanlp.utils.string_util.ispunct(token)  # type: ignore[reportGeneralTypeIssues]
                for token in sent_tokens
            ]
        )
        return count

    def count_postags(sent_poses: list[str]) -> int:
        # AD: adverb, CC: coordinating conjunction,
        # CS: subordinating conjunction
        count = sum([pos in ["AD", "CC", "CS"] for pos in sent_poses])
        return count

    def calc_r1(content: list[list[str]]) -> float:
        token_count_by_sentence = list(map(count_tokens, content))
        if len(token_count_by_sentence) == 0:
            return 0
        else:
            return sum(token_count_by_sentence) / len(token_count_by_sentence)

    def calc_r2(content: list[list[str]]) -> float:
        pos_count_by_sentence = list(map(count_postags, content))
        if len(pos_count_by_sentence) == 0:
            return 0
        else:
            return sum(pos_count_by_sentence) / len(pos_count_by_sentence)

    r1 = list(map(calc_r1, output_tokens))  # type: ignore[reportGeneralTypeIssues]
    r2 = list(map(calc_r2, output_pos))  # type: ignore[reportGeneralTypeIssues]
    r3 = [(r1_score + r2_score) * 0.5 for r1_score, r2_score in zip(r1, r2)]
    return MetricValue(
        metric_name="readability",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=r3,
        language="zh",
    )
