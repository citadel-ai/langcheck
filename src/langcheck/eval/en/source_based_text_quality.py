from typing import Dict, List, Optional

import nltk
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from langcheck.eval.en._openai import OpenAIBasedEvaluator
from langcheck.eval.eval_value import EvalValue

_factual_consistency_model_path = 'MingZhong/unieval-fact'
_factual_consistency_config = None
_factual_consistency_tokenizer = None
_factual_consistency_model = None


def factual_consistency(
        generated_outputs: List[str],
        sources: List[str],
        model_type: str = 'local',
        openai_args: Optional[Dict[str, str]] = None) -> EvalValue[float]:
    '''Calculates the factual consistency between the generated outputs and
    the sources. The factual consistency score for one generated output is
    computed as the average of the per-sentence consistencies of the generated
    output with the source text. This metric takes on float values between
    [0, 1], where 0 means that the output is not at all consistent with the
    source text, and 1 means that the output is fully consistent with the source
    text. (NOTE: when uing the OpenAI model, the factuality score for each
    sentence is either 0.0, 0.5, or 1.0.)

    We currently support two model types:

    1. The 'local' type, where the 'unieval-fact' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    https://github.com/citadel-ai/langcheck#evaluate-text for examples on
    setting up the OpenAI API key.

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        sources: A list of source texts
        model_type: The type of model to use ('local' or 'openai'),
            default 'local'
        openai_args: Dict of additional args to pass in to the
            `openai.ChatCompletion.create` function, default None

    Returns:
        An EvalValue object
    '''
    assert model_type in ['local', 'openai'
                         ], ('Unsupported model type. '
                             'The supported ones are ["local", "openai"]')

    # Confirm necessary data for nltk.tokenize.sent_tokenize() exists
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

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
                         language='en')

    # Split the generated outputs into individual sentences. This is consistent
    # with how UniEval calculates factual consistency, where the factual
    # consistency of each generated sentence gets averaged.
    # (https://github.com/maszhongming/UniEval/blob/509075cc87bb64f239180ece460025466b260383/metric/evaluator.py#L261)
    srcs_list, gen_sentences_list = [], []
    num_sentences_list = []
    for src, gen in zip(sources, generated_outputs):
        gen_sentences = nltk.tokenize.sent_tokenize(gen)
        num_sentences_list.append(len(gen_sentences))
        gen_sentences_list += gen_sentences
        srcs_list += [src] * len(gen_sentences)

    if model_type == 'local':
        score_list = _factual_consistency_local(gen_sentences_list, srcs_list)
    else:  # openai
        score_list = _factual_consistency_openai(gen_sentences_list, srcs_list,
                                                 openai_args)

    # The score for each output is the average of the scores of its sentences
    score_per_output = []
    start_idx = 0
    for num in num_sentences_list:
        score_per_output.append(
            sum(score_list[start_idx:start_idx + num]) / num)
        start_idx += num

    return EvalValue(metric_name='factual_consistency',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=None,
                     sources=sources,
                     metric_values=score_per_output,
                     language='en')


def _factual_consistency_local(gen_sentences_list: List[str],
                               srcs_list: List[str]) -> List[float]:
    '''Calculates the factual consistency between each generated sentence and
    its corresponding source text. The consistency is computed by querying the
    UniEval-fact model that has been pre-trained to evaluate factual
    consistency.

    Ref:
        https://github.com/maszhongming/UniEval

    Args:
        gen_sentences_list: A list of model generated sentences to evaluate
        srcs_list: The list of source texts for each generated sentence in
            `gen_sentences_list`

    Returns:
        A list of scores
    '''
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
    for i in range(0, len(model_input_list), batch_size):
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
    return score_list


def _factual_consistency_openai(
        gen_sentences_list: List[str],
        srcs_list: List[str],
        openai_args: Optional[Dict[str, str]] = None) -> List[float]:
    '''Calculates the factual consistency between each generated sentence and
    its corresponding source text. The consistency is computed by calling the
    OpenAI API, with a prompt similar to the one used in OpenAI Evals. We
    leverage the function calling API to make sure that the output is structured
    such that we can compute a score.

    Ref:
        https://github.com/openai/evals/blob/e49868e550babb7b1c5b4223c9b7a14511bf114d/evals/registry/modelgraded/fact.yaml
        https://platform.openai.com/docs/guides/gpt/function-calling

    Args:
        gen_sentences_list: A list of model generated sentences to evaluate
        srcs_list: The list of source texts for each generated sentence in
            `gen_sentences_list`
        openai_args: Dict of additional args to pass in to the
            `openai.ChatCompletion.create` function, default None

    Returns:
        A list of scores
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
        source, and save the resulting assessment. The available assessments
        are:
        `Fully Consistent` - The submitted claim is fully factually consistent
        with the source text.
        `Partially Consistent` - The submitted claim is partially factually
        consistent with the source text. There are some aspects of the claim
        that are factually consistent, but some aspects that are not.
        `Not Consistent` - The submitted claim is not factually consistent with
        the source text.
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
        openai_args=openai_args)

    score_list = []
    for src, gen in zip(srcs_list, gen_sentences_list):
        score = oai_evaluator.get_score(_prompt(src=src, gen_output=gen))
        score_list.append(score)
    return score_list
