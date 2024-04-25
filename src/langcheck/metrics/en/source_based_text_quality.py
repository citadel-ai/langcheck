from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import nltk
import torch
import torch.nn as nn
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from langcheck.metrics._validation import (
    validate_parameters_context_relevance, validate_parameters_source_based)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template

_factual_consistency_model_path = 'MingZhong/unieval-fact'
_factual_consistency_config = None
_factual_consistency_tokenizer = None
_factual_consistency_model = None


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

    if eval_model == 'local':
        scores = _factual_consistency_local(generated_outputs, sources)
        explanations = None
    else:  # EvalClient
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
        'en/metrics/factual_consistency.j2')

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
        language='en',
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

    context_relevance_template = get_template('en/metrics/context_relevance.j2')

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
        language='en',
        prompts=populated_prompts,
        score_map=context_relevance_assessment_to_score,
    )

    return MetricValue(metric_name='context_relevance',
                       prompts=prompts,
                       generated_outputs=None,
                       reference_outputs=None,
                       sources=sources,
                       explanations=explanations,
                       metric_values=scores,
                       language='en')
