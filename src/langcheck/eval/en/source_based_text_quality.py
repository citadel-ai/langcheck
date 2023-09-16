from typing import List

import torch
import torch.nn as nn
from nltk import sent_tokenize
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from langcheck.eval.eval_value import EvalValue

_factual_consistency_model_path = 'MingZhong/unieval-fact'
_factual_consistency_config = None
_factual_consistency_tokenizer = None
_factual_consistency_model = None


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

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        sources: A list of source texts

    Returns:
        An EvalValue object
    '''
    if len(generated_outputs) != len(sources):
        raise ValueError(
            'The generated outputs and sources lists must be of the same '
            'length')
    if len(generated_outputs) == 0:
        return EvalValue(metric_name='factual_consistency',
                         prompts=None,
                         generated_outputs=[],
                         reference_outputs=[],
                         sources=[],
                         metric_values=[],
                         language='en')

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

    # Prepare the inputs to the model. Note that we split the generated outputs
    # into individual sentences, because UniEval calculates the factual
    # consistency score by averaging the factual consistencies of each generated
    # sentence
    # (https://github.com/maszhongming/UniEval/blob/509075cc87bb64f239180ece460025466b260383/metric/evaluator.py#L261)
    model_input_list = []
    num_sentences_list = []
    for src, gen in zip(sources, generated_outputs):
        gen_sentences = sent_tokenize(gen)
        num_sentences_list.append(len(gen_sentences))
        for gen_sent in gen_sentences:
            model_input = (
                f'question: Is this claim consistent with the document? </s> '
                f'claim: {gen_sent} </s> '
                f'document: {src}')

            model_input_list.append(model_input)

    pos_id = _factual_consistency_tokenizer('Yes')['input_ids'][0]
    neg_id = _factual_consistency_tokenizer('No')['input_ids'][0]
    softmax = nn.Softmax(dim=1)

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
