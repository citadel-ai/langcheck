from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import nltk
import torch
import torch.nn as nn
from openai import OpenAI
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from langcheck.metrics._validation import (
    validate_parameters_context_relevance, validate_parameters_source_based)
from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

_factual_consistency_model_path = 'MingZhong/unieval-fact'
_factual_consistency_config = None
_factual_consistency_tokenizer = None
_factual_consistency_model = None


def factual_consistency(
        generated_outputs: List[str] | str,
        sources: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        model_type: str = 'local',
        openai_client: Optional[OpenAI] = None,
        openai_args: Optional[Dict[str, str]] = None,
        *,
        use_async: bool = False) -> MetricValue[Optional[float]]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. This metric takes on float values between [0, 1], where 0
    means that the output is not at all consistent with the source text, and 1
    means that the output is fully consistent with the source text. (NOTE: when
    using the OpenAI model, the factuality scores are either 0.0, 0.5, or 1.0.
    The score may also be `None` if it could not be computed.)

    We currently support three model types:

    1. The 'local' type, where the 'unieval-fact' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    3. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
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
        An MetricValue object
    '''
    generated_outputs, sources, prompts = validate_parameters_source_based(
        generated_outputs, sources, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')

    if model_type == 'local':
        scores = _factual_consistency_local(generated_outputs, sources)
        explanations = None
    else:  # openai or azure_openai
        scores, explanations = _factual_consistency_openai(generated_outputs,
                                                           sources,
                                                           model_type,
                                                           openai_client,
                                                           openai_args,
                                                           use_async=use_async)

    return MetricValue(metric_name='factual_consistency',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=None,
                       sources=sources,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')


def _factual_consistency_local(generated_outputs: List[str],
                               sources: List[str]) -> List[float]:
    '''Calculates the factual consistency between each generated sentence and
    its corresponding source text. The factual consistency score for one
    generated output is computed as the average of the per-sentence
    consistencies of the generated output with the source text The consistency
    is computed by querying the UniEval-fact model that has been pre-trained to
    evaluate factual consistency.

    Ref:
        https://github.com/maszhongming/UniEval

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output

    Returns:
        A list of scores
    '''
    # Confirm necessary data for nltk.tokenize.sent_tokenize() exists
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Split the generated outputs into individual sentences. This is consistent
    # with how UniEval calculates factual consistency, where the factual
    # consistency of each generated sentence gets averaged.
    # (https://github.com/maszhongming/UniEval/blob/509075cc87bb64f239180ece460025466b260383/metric/evaluator.py#L261)
    srcs_list, gen_sentences_list = [], []
    num_sentences_list = []
    for src, gen in tqdm_wrapper(
            zip(sources, generated_outputs),
            desc='Splitting generated outputs into sentences',
            total=len(generated_outputs)):
        gen_sentences = nltk.tokenize.sent_tokenize(gen)
        num_sentences_list.append(len(gen_sentences))
        gen_sentences_list += gen_sentences
        srcs_list += [src] * len(gen_sentences)

    global _factual_consistency_config, _factual_consistency_tokenizer, \
        _factual_consistency_model
    if _factual_consistency_config is None:
        _factual_consistency_config = AutoConfig.from_pretrained(
            _factual_consistency_model_path)
    if _factual_consistency_tokenizer is None:
        _factual_consistency_tokenizer = AutoTokenizer.from_pretrained(
            _factual_consistency_model_path)
    if _factual_consistency_model is None:
        _factual_consistency_model = AutoModelForSeq2SeqLM.from_pretrained(
            _factual_consistency_model_path, config=_factual_consistency_config)
        _factual_consistency_model.eval()

    pos_id = _factual_consistency_tokenizer('Yes')['input_ids'][0]
    neg_id = _factual_consistency_tokenizer('No')['input_ids'][0]
    softmax = nn.Softmax(dim=1)

    model_input_list = []
    for src, gen in zip(srcs_list, gen_sentences_list):
        model_input = (
            f'question: Is this claim consistent with the document? </s> '
            f'claim: {gen} </s> '
            f'document: {src}')

        model_input_list.append(model_input)

    # Specifying the targets is required to run the model, but has no effect on
    # the score
    target_list = ["No" for _ in range(len(model_input_list))]

    batch_size = 8
    score_list = []
    for i in tqdm_wrapper(range(0, len(model_input_list), batch_size),
                          total=(len(model_input_list) + batch_size - 1) //
                          batch_size):
        inputs = model_input_list[i:i + batch_size]
        targets = target_list[i:i + batch_size]

        with torch.no_grad():
            encoded_inputs = _factual_consistency_tokenizer(inputs,
                                                            truncation=True,
                                                            padding=True,
                                                            return_tensors='pt')
            encoded_targets = _factual_consistency_tokenizer(
                targets, truncation=True, padding=True, return_tensors='pt')
            inputs_tokens = encoded_inputs['input_ids']
            inputs_mask = encoded_inputs['attention_mask']
            targets_tokens = encoded_targets['input_ids'][:, 0].unsqueeze(-1)

            outputs = _factual_consistency_model(input_ids=inputs_tokens,
                                                 attention_mask=inputs_mask,
                                                 labels=targets_tokens)
            logits = outputs.logits.view(
                -1, _factual_consistency_model.config.vocab_size)
            pos_score = softmax(logits)[:, pos_id]
            neg_score = softmax(logits)[:, neg_id]
            score_list += [
                x.item() for x in pos_score / (pos_score + neg_score)
            ]

    # The score for each output is the average of the scores of its sentences
    score_per_output = []
    start_idx = 0
    for num in tqdm_wrapper(num_sentences_list, desc='Calculating scores'):
        scores_for_output = score_list[start_idx:start_idx + num]
        score_per_output.append(sum(scores_for_output) / num)
        start_idx += num
    return score_per_output


def _factual_consistency_openai(
    generated_outputs: List[str],
    sources: List[str],
    client_type: str,
    client: Optional[OpenAI],
    openai_args: Optional[Dict[str, str]],
    *,
    use_async: bool = False
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    '''Calculates the factual consistency and their associated explanations
    between each generated output and its corresponding source text. The
    consistency is computed by calling the OpenAI API, with a prompt similar to
    the one used in OpenAI Evals. We leverage the function calling API to make
    sure that the output is structured such that we can compute a score. If a
    score could not be computed, `None` is inserted to the score and explanation
    lists.

    Ref:
        https://github.com/openai/evals/blob/e49868e550babb7b1c5b4223c9b7a14511bf114d/evals/registry/modelgraded/fact.yaml
        https://platform.openai.com/docs/guides/gpt/function-calling

    Args:
        generated_outputs: The model generated output(s) to evaluate
        sources: The source text(s), one string per generated output
        client_type: The type of OpenAI client ('openai' or 'azure_openai')
        client: (Optional) OpenAI or AzureOpenAI client. If this is None, we
            will attempt to create a default client depending on the
            ``client_type``.
        openai_args: (Optional) Dict of additional args to pass in to the
            ``client.chat.completions.create`` function
        use_async: Whether to use the asynchronous API of OpenAI

    Returns:
        score_list: a list of scores
        explanation_list: a list of explanations for the scores
    '''

    # TODO: The prompt formation, and the scoring system, can do with some
    # improvement. There are some cases where consistent outputs get incorrectly
    # assessed as "Partially Consistent", and there's no differentiation
    # between an output that is unrelated to the source and an output that is
    # straight up contradictory.
    def _prompt(src: str, gen_output: str) -> str:
        return f'''
        You are evaluating the factual consistency of a submitted claim. Here is
        the data:
        [BEGIN DATA]
        ************
        [Source]: {src}
        ************
        [Submission]: {gen_output}
        ************
        [END DATA]

        Determine whether the submitted claim is factually consistent with the
        source. The available assessments are:
        `Fully Consistent` - The submitted claim is fully factually consistent
        with the source text.
        `Partially Consistent` - The submitted claim is partially factually
        consistent with the source text. There are some aspects of the claim
        that are factually consistent, but some aspects that are not.
        `Not Consistent` - The submitted claim is not factually consistent with
        the source text.

        Take a deep breath and work on this problem step-by-step.
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        The following is an assessment on the factual consistency of a claim:
        ************
        [Assessment]: {long_assessment}
        ************

        Save the resulting assessment. The available assessments are:
        `Fully Consistent`
        `Partially Consistent`
        `Not Consistent`
        '''

    factuality_assessment_to_score = {
        'Fully Consistent': 1.0,
        'Partially Consistent': 0.5,
        'Not Consistent': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=factuality_assessment_to_score,
        function_name='save_factual_consistency_assessment',
        function_description=(
            "Saves a submitted claim's factual consistency assessment."),
        argument_name='factuality',
        argument_description='The factual consistency assessment of the claim',
        client_type=client_type,
        client=client,
        openai_args=openai_args,
        use_async=use_async)

    scores, explanations = oai_evaluator.get_score(
        map(_prompt, sources, generated_outputs), _function_call_prompt)

    return scores, explanations


def context_relevance(sources: List[str] | str,
                      prompts: List[str] | str,
                      model_type: str = 'openai',
                      openai_client: Optional[OpenAI] = None,
                      openai_args: Optional[Dict[str, str]] = None,
                      *,
                      use_async: bool = False) -> MetricValue[Optional[float]]:
    '''Calculates the relevance of the sources to the prompts. This metric takes
    on float values between [0, 1], where 0 means that the source text is not at
    all relevant to the prompt, and 1 means that the source text is fully
    relevant to the prompt.

    We currently support two model types:

    1. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    2. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Args:
        sources: The source text(s), one string per prompt
        prompts: The prompt(s)
        model_type: The type of model to use ('openai' or 'azure_openai'),
            default 'openai'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None, we will attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None
        use_async: Whether to use the asynchronous API of OpenAI, default False
    '''
    prompts, sources = validate_parameters_context_relevance(prompts, sources)

    def _prompt(src: str, user_query: str) -> str:
        return f'''
        You are evaluating the relevance of the source to a user's query. Here
        is the data:
        [BEGIN DATA]
        ************
        [Source]: {src}
        ************
        [User Query]: {user_query}
        ************
        [END DATA]

        Determine whether the source contains the relevant and necessary
        information needed to respond to the user's query. The available
        assessments are:
        `Fully Relevant` - The source text contains the information necessary to
        respond to the user's query.
        `Partially Relevant` - The source text is partially relevant to the
        user's query, but does not contain all the information necessary to
        respond to the user's query.
        `Not Relevant` - The source text is not relevant to the user's query.

        Take a deep breath and work on this problem step-by-step.
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        The following is an assessment on the relevance of a source:
        ************
        [Assessment]: {long_assessment}
        ************

        Save the resulting assessment. The available assessments are:
        `Fully Relevant`
        `Partially Relevant`
        `Not Relevant`
        '''

    context_relevance_assessment_to_score = {
        'Fully Relevant': 1.0,
        'Partially Relevant': 0.5,
        'Not Relevant': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=context_relevance_assessment_to_score,
        function_name='save_context_relevance_assessment',
        function_description=("Saves a context relevance assessment."),
        argument_name='context_relevance',
        argument_description='The context relevance assessment',
        client_type=model_type,
        client=openai_client,
        openai_args=openai_args,
        use_async=use_async)

    scores, explanations = oai_evaluator.get_score(
        map(_prompt, sources, prompts), _function_call_prompt)

    return MetricValue(metric_name='context_relevance',
                       prompts=prompts,
                       generated_outputs=None,
                       reference_outputs=None,
                       sources=sources,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')
