from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langcheck.eval.eval_value import EvalValue

_sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
_sentiment_tokenizer = None
_sentiment_model = None


def sentiment(generated_outputs: List[str],
              prompts: Optional[List[str]] = None) -> EvalValue[float]:
    '''Calculates the sentiment scores of generated outputs in Japanese
    using the Twitter-roBERTa-base model. This metric takes on float values 
    between [0, 1], where 0 is negative sentiment and 1 is positive sentiment.

    Ref:
        https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        prompts: An optional list of prompts used to generate the outputs.
            Prompts are not evaluated and only used as metadata.

    Returns:
        An EvalValue object
    '''
    global _sentiment_tokenizer, _sentiment_model

    if _sentiment_tokenizer is None or _sentiment_model is None:
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(
            _sentiment_model_path)

        # There is a "Some weights are not used warning" but we ignore it because
        # that is intended.
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            _sentiment_model_path)

    input_tokens = _sentiment_tokenizer(generated_outputs,
                                        return_tensors='pt',
                                        padding=True)

    with torch.no_grad():
        # Probabilities of [negative, neutral, positive]
        probs = torch.nn.functional.softmax(
            _sentiment_model(**input_tokens).logits, dim=1)

    scores = (probs[:, 1] / 2 + probs[:, 2]).tolist()

    return EvalValue(metric_name='sentiment',
                     prompts=prompts,
                     generated_outputs=generated_outputs,
                     reference_outputs=None,
                     metric_values=scores,
                     language='ja')