from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import BatchEncoding

from langcheck._handle_logs import _handle_logging_level

from ._base import BaseSimilarityScorer, BaseSingleScorer


class AutoModelForSequenceClassificationScorer(BaseSingleScorer):
    '''Scorer using Hugging Face's AutoModelForSequenceClassification.
    '''

    def __init__(self,
                 language,
                 metric,
                 class_weights,
                 overflow_strategy: str = 'truncate',
                 max_input_length: Optional[int] = None):
        '''
        Initialize the scorer with the provided configs.

        Args:
            language: The language of the model (default 'en')
            metric: The metric of the model (e.g., 'toxicity')
            class_weights: The weights multiplied to the logits to get the
                scores.
            overflow_strategy: The strategy to handle the overflow of the input.
                The value should be either "raise", "truncate" or "nullify".
            max_input_length: The maximum length of the input. If None, the
                maximum length of the model is used.
        '''

        super().__init__()
        self.overflow_strategy = overflow_strategy
        from langcheck.metrics.model_manager import manager
        tokenizer, model = manager.fetch_model(language=language, metric=metric)

        self.tokenizer = tokenizer
        self.model = model
        self.class_weights = class_weights
        if max_input_length is not None:
            self.max_input_length: int = max_input_length
        else:
            self.max_input_length = self.model.config.max_position_embeddings  # type: ignore # NOQA: E501

    def _tokenize(self, inputs: list[str]) -> Tuple[BatchEncoding, list[bool]]:
        '''Tokenize the inputs. It also does the validation on the token length,
        and return the results as a list of boolean values. If the validation
        mode is 'raise', it raises an error when the token length is invalid.
        '''
        truncated_tokens = self.tokenizer(  # type: ignore
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt')

        if self.overflow_strategy == 'truncate':
            return (truncated_tokens, [True] * len(inputs))

        input_validation_results = self._validate_inputs(inputs)

        if self.overflow_strategy == 'raise' and not all(
                input_validation_results):
            raise ValueError('Some of the inputs are too long.')

        assert self.overflow_strategy == 'nullify', 'Overflow strategy is invalid. The value should be either "raise", "truncate" or "nullify".'  # NOQA: E501

        # Return the padded & truncated tokens.
        # The user needs to exclude the invalid tokens from the results.
        return (truncated_tokens, input_validation_results)

    def _validate_inputs(self, inputs: list[str]) -> list[bool]:
        '''Validation based on the maximum input length of the model.
        '''

        validation_results = []
        for input_str in inputs:
            # Tokenize the input and get the length of the input_ids
            # Suppress the warning because we intentionally generate the
            # tokens longer than the maximum length.
            with _handle_logging_level():
                input_ids = self.tokenizer.encode(input_str)  # type: ignore
            validation_results.append(len(input_ids) <= self.max_input_length)

        return validation_results

    def _score_tokens(
            self, tokens: Tuple[BatchEncoding,
                                list[bool]]) -> list[Optional[float]]:
        '''Return the prediction results as scores.
        '''
        input_tokens, validation_results = tokens
        with torch.no_grad():
            logits: torch.Tensor = self.model(
                **input_tokens).logits  # type: ignore
            scores: list[Optional[float]] = self._logits_to_scores(
                logits)  # type: ignore

        for i, validation_result in enumerate(validation_results):
            if not validation_result:
                scores[i] = None

        return scores

    def _slice_tokens(self, tokens: Tuple[BatchEncoding,
                                          list[bool]], start_idx: int,
                      end_idx: int) -> Tuple[BatchEncoding, list[bool]]:

        input_tokens, validation_results = tokens

        return (
            {
                key: value[start_idx:end_idx]
                for key, value in input_tokens.items()
            },  # type: ignore
            validation_results[start_idx:end_idx])

    def _logits_to_scores(self, logits: torch.Tensor) -> list[float]:
        '''Turn the logits returned from the models to scores.
        '''
        probs = torch.nn.functional.softmax(logits, dim=1)
        scores = torch.zeros(probs.shape[0], dtype=torch.float32)

        for i, class_weight in enumerate(self.class_weights):
            scores += probs[:, i] * class_weight

        return scores.tolist()


class SentenceTransformerSimilarityScorer(BaseSimilarityScorer):
    '''Scorer using SentenceTransformer.
    '''

    def __init__(self, language, metric='semantic_similarity'):
        super().__init__()

        from langcheck.metrics.model_manager import manager
        self.model = manager.fetch_model(language=language, metric=metric)

    def _embed(self, inputs: list[str]) -> torch.Tensor:
        return self.model.encode(inputs, convert_to_tensor=True)  # type: ignore
