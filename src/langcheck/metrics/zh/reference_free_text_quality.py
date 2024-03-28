from __future__ import annotations

from typing import Dict, List, Optional

import hanlp
from openai import OpenAI
from transformers.pipelines import pipeline

from langcheck.metrics._validation import validate_parameters_reference_free
from langcheck.metrics.en.reference_free_text_quality import _toxicity_openai
from langcheck.metrics.en.reference_free_text_quality import \
    sentiment as en_sentiment
from langcheck.metrics.metric_value import MetricValue


def sentiment(generated_outputs: List[str] | str,
              prompts: Optional[List[str] | str] = None,
              model_type: str = 'local',
              openai_client: Optional[OpenAI] = None,
              openai_args: Optional[Dict[str, str]] = None,
              *,
              use_async: bool = False) -> MetricValue[Optional[float]]:
    '''Calculates the sentiment scores of generated outputs. This metric takes
    on float values between [0, 1], where 0 is negative sentiment and 1 is
    positive sentiment. (NOTE: when using the OpenAI model, the sentiment scores
    are either 0.0 (negative), 0.5 (neutral), or 1.0 (positive). The score may
    also be `None` if it could not be computed.)

    We currently support three model types:

    1. The 'local' type, where the IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
    model is downloaded from HuggingFace and run locally. This is the default
    model type and there is no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this example <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    3. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Ref:
        https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        model_type: The type of model to use ('local', 'openai', or
            'azure_openai'), default 'local'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None but ``model_type`` is 'openai' or 'azure_openai', we will
            attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None
        use_async: Whether to use the asynchronous API of OpenAI, default False

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    if model_type == 'openai' or model_type == 'azure_openai':
        metric_value = en_sentiment(generated_outputs,
                                    prompts,
                                    model_type,
                                    openai_client,
                                    openai_args,
                                    use_async=use_async)
        metric_value.language = 'zh'
        return metric_value

    # {0:"Negative", 1:'Positive'}
    from langcheck.metrics.model_manager import manager
    tokenizer, model = manager.fetch_model(language='zh', metric='sentiment')
    _sentiment_pipeline = pipeline(
        'sentiment-analysis',
        model=model,  # type: ignore[reportGeneralTypeIssues]
        tokenizer=tokenizer  # type: ignore[reportGeneralTypeIssues]
    )
    _model_id2label = _sentiment_pipeline.model.config.id2label
    _predict_result = _sentiment_pipeline(
        generated_outputs
    )  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
    # if predicted result is 'Positive', use the score directly
    # else, use 1 - score as the sentiment score
    # yapf: disable
    scores = [
        1 - x['score'] if x['label'] == _model_id2label[0] else x['score']  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
        for x in _predict_result   # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
    ]
    # yapf: enable
    return MetricValue(
        metric_name='sentiment',
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=None,
        sources=None,
        explanations=None,
        metric_values=scores,  # type: ignore[reportGeneralTypeIssues]
        language='zh')


def toxicity(generated_outputs: List[str] | str,
             prompts: Optional[List[str] | str] = None,
             model_type: str = 'local',
             openai_client: Optional[OpenAI] = None,
             openai_args: Optional[Dict[str, str]] = None,
             *,
             use_async: bool = False) -> MetricValue[Optional[float]]:
    '''Calculates the toxicity scores of generated outputs. This metric takes on
    float values between [0, 1], where 0 is low toxicity and 1 is high toxicity.
    (NOTE: when using the OpenAI model, the toxicity scores are in steps of
    0.25. The score may also be `None` if it could not be computed.)

    We currently support three model types:

    1. The 'local' type, where a model file is downloaded from HuggingFace and
    run locally. This is the default model type and there is no setup needed to
    run this.
    The model (alibaba-pai/pai-bert-base-zh-llm-risk-detection) is a
    risky detection model for LLM generated content released by Alibaba group.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default, in the same way as english counterpart. While the model you use
    is configurable, please make sure to use one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this example <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    3. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Ref:
        https://huggingface.co/alibaba-pai/pai-bert-base-zh-llm-risk-detection

    Args:
        generated_outputs: The model generated output(s) to evaluate
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        model_type: The type of model to use ('local', 'openai', or
            'azure_openai'), default 'local'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None but ``model_type`` is 'openai' or 'azure_openai', we will
            attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None
        use_async: Whether to use the asynchronous API of OpenAI, default False

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    '''
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    if model_type == 'openai' or model_type == 'azure_openai':
        # openai
        scores, explanations = _toxicity_openai(generated_outputs,
                                                model_type,
                                                openai_client,
                                                openai_args,
                                                use_async=use_async)
    else:
        scores = _toxicity_local(generated_outputs)
        explanations = None

    return MetricValue(metric_name='toxicity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=None,
                       explanations=explanations,
                       metric_values=scores,
                       language='zh')


def _toxicity_local(generated_outputs: List[str]) -> List[float]:
    '''Calculates the toxicity scores of generated outputs using a fine-tuned
    model from `alibaba-pai/pai-bert-base-zh-llm-risk-detection`. This metric
    takes on float values between [0, 1], where 0 is low toxicity and 1 is high
    toxicity.

    Ref:
        https://huggingface.co/alibaba-pai/pai-bert-base-zh-llm-risk-detection

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        A list of scores
    '''
    # this pipeline output predict probability for each text on each label.
    # the output format is List[List[Dict(str)]]
    from langcheck.metrics.model_manager import manager
    tokenizer, model = manager.fetch_model(language='zh', metric="toxicity")
    _toxicity_pipeline = pipeline(
        'text-classification',
        model=model,  # type: ignore[reportOptionalIterable]
        tokenizer=tokenizer,  # type: ignore[reportOptionalIterable]
        top_k=5)
    # {'Normal': 0, 'Pulp': 1, 'Sex': 2, 'Other Risk': 3, 'Adult': 4}
    _model_id2label = _toxicity_pipeline.model.config.id2label
    _predict_results = _toxicity_pipeline(
        generated_outputs  # type: ignore[reportGeneralTypeIssues]
    )
    # labels except Normal are all risky, toxicity_score = 1-score['Normal']
    toxicity_scores = []
    for item_predict_proba in _predict_results:  # type: ignore[reportOptionalIterable]  # NOQA: E501
        for label_proba in item_predict_proba:  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
            # yapf: disable
            if label_proba['label'] == _model_id2label[0]:  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
                toxicity_scores.append(1 - label_proba['score'])  # type: ignore[reportGeneralTypeIssues]  # NOQA: E501
            # yapf: enable
    return toxicity_scores  # type: ignore[reportGeneralTypeIssues]


def xuyaochen_report_readability(
        generated_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None) -> MetricValue[float]:
    '''Calculates the readability scores of generated outputs introduced in
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
    '''
    # split generated_outputs into sentence
    generated_outputs, prompts = validate_parameters_reference_free(
        generated_outputs, prompts=prompts)
    # yapf: disable
    tokenizer = hanlp.load(
        hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH  # type: ignore[reportGeneralTypeIssues] # NOQA: E501
    )
    postagger = hanlp.load(
        hanlp.pretrained.pos.CTB9_POS_RADICAL_ELECTRA_SMALL   # type: ignore[reportGeneralTypeIssues] # NOQA: E501
    )

    pos_pipeline = hanlp.pipeline().\
        append(hanlp.utils.rules.split_sentence)  # type: ignore[reportGeneralTypeIssues] # NOQA: E501
    pos_pipeline = pos_pipeline.append(tokenizer).append(postagger)

    tokenize_pipeline = hanlp.pipeline().\
        append(hanlp.utils.rules.split_sentence)  # type: ignore[reportGeneralTypeIssues] # NOQA: E501
    tokenize_pipeline = tokenize_pipeline.append(tokenizer)
    # OUTPUT: List[List[List[TOKEN]]]
    output_tokens = list(map(tokenize_pipeline, generated_outputs))
    # List[List[List[POS]]]
    output_pos = list(map(pos_pipeline, generated_outputs))

    def count_tokens(sent_tokens: List[str]) -> int:
        count = sum([
            not hanlp.utils.string_util.ispunct(token) for token in   # type: ignore[reportGeneralTypeIssues] # NOQA: E501
            sent_tokens
        ])
        return count

    def count_postags(sent_poses: List[str]) -> int:
        # AD: adverb, CC: coordinating conjunction,
        # CS: subordinating conjunction
        count = sum([pos in ['AD', 'CC', 'CS'] for pos in sent_poses])
        return count

    def calc_r1(content: List[List[str]]) -> float:
        token_count_by_sentence = list(map(count_tokens, content))
        if len(token_count_by_sentence) == 0:
            return 0
        else:
            return sum(token_count_by_sentence) / len(token_count_by_sentence)

    def calc_r2(content: List[List[str]]) -> float:
        pos_count_by_sentence = list(map(count_postags, content))
        if len(pos_count_by_sentence) == 0:
            return 0
        else:
            return sum(pos_count_by_sentence) / len(pos_count_by_sentence)

    r1 = list(map(calc_r1, output_tokens))   # type: ignore[reportGeneralTypeIssues] # NOQA: E501
    r2 = list(map(calc_r2, output_pos))   # type: ignore[reportGeneralTypeIssues] # NOQA: E501
    r3 = [(r1_score + r2_score) * 0.5 for r1_score, r2_score in zip(r1, r2)]
    # yapf: enable
    return MetricValue(metric_name='readability',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       sources=None,
                       reference_outputs=None,
                       explanations=None,
                       metric_values=r3,
                       language='zh')
