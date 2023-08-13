from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langcheck.eval.eval_value import EvalValue

_sentiment_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_sentiment_tokenizer = None
_sentiment_model = None


def sentiment(generated_outputs: List[str]) -> EvalValue:
    '''Calculates the sentient scores of the generated outputs between
    0 (Negative) and 1 (Positive) based on Twitter-roBERTa-base moel.

    Ref:
    https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

    Args:
        generated_outputs: A list of model generated outputs to evaluate

    Returns:
        An EvalValue object
    '''
    if _sentiment_tokenizer is None or _sentiment_model is None:
        raise RuntimeError(
            'reference_free_text_quality module is not initialized')

    input_tokens = _sentiment_tokenizer(generated_outputs,
                                        return_tensors='pt',
                                        padding=True)

    with torch.no_grad():
        # Probabilities of [negative, neutral, positive]
        probs = torch.nn.functional.softmax(
            _sentiment_model(**input_tokens).logits, dim=1)

    scores = (probs[:, 1] / 2 + probs[:, 2]).tolist()

    return EvalValue(metric_name='sentiment',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     metric_values=scores)


def _init():
    '''Setup pretrained models required for reference free text qualitry analysis.
    '''
    global _sentiment_tokenizer, _sentiment_model
    _sentiment_tokenizer = AutoTokenizer.from_pretrained(_sentiment_model_path)

    # There is a "Some weights are not used warning" but we ignore it because
    # that is intended.
    _sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        _sentiment_model_path)


_init()
