from __future__ import annotations

import torch

from ._base import BaseSingleScorer
from transformers import BatchEncoding


class AutoModelForSequenceClassificationScorer(BaseSingleScorer):
    '''Scorer using Hugging Face's AutoModelForSequenceClassification.
    '''

    def __init__(self, language, metric, validation_mode: str = 'raise'):
        super().__init__(validation_mode)
        from langcheck.metrics.model_manager import manager
        tokenizer, model = manager.fetch_model(language=language, metric=metric)

        self.tokenizer = tokenizer
        self.model = model

    def _tokenize(self, inputs) -> BatchEncoding:
        return self.tokenizer(
            inputs,  # type: ignore
            padding=True,
            truncation=True,
            return_tensors='pt')

    def _validate_tokens(self, tokens: BatchEncoding) -> list[bool]:
        '''Validation based on the maximum input length of the model.
        '''
        input_ids = tokens['input_ids']
        max_valid_input_length = self.tokenizer.model_max_length  # type: ignore

        return [
            len(input_id) <= max_valid_input_length  # type: ignore
            for input_id in input_ids  # type: ignore
        ]

    def _score_tokens(self, tokens: BatchEncoding) -> list[float]:
        '''Return the prediction results as scores.
        '''
        scores = []
        with torch.no_grad():
            logits: torch.Tensor = self.model(**tokens)  # type: ignore
            scores.extend(self._logits_to_scores(logits))
        return scores

    def _logits_to_scores(self, logits: torch.Tensor) -> list[float]:
        '''Turn the logits returned from the models to scores.
        The users can override this method to customize the behavior.
        '''
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs[:, 1].tolist()
