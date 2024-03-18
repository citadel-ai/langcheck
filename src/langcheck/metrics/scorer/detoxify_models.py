from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from transformers import (BatchEncoding, BertForSequenceClassification,
                          BertTokenizer, XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer)

from langcheck._handle_logs import _handle_logging_level

from ._base import BaseSingleScorer

_checkpoints = {
    "en":
        "https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_original-c1212f89.ckpt",  # NOQA: E501
    "de":
        "https://github.com/unitaryai/detoxify/releases/download/v0.4-alpha/multilingual_debiased-0b549669.ckpt"  # NOQA: E501
}

_model_types = {
    "en": (BertForSequenceClassification, BertTokenizer),
    "de": (XLMRobertaForSequenceClassification, XLMRobertaTokenizer)
}


def load_checkpoint(
    device: str, lang: str
) -> Tuple[Union[BertForSequenceClassification,
                 XLMRobertaForSequenceClassification], Union[
                     BertTokenizer, XLMRobertaTokenizer], List[str]]:
    checkpoint_url = _checkpoints[lang]
    class_model_type, tokenizer_type = _model_types[lang]
    loaded = torch.hub.load_state_dict_from_url(checkpoint_url,
                                                map_location=device)
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model_type = loaded["config"]["arch"]["args"]["model_type"]
    num_classes = loaded["config"]["arch"]["args"]["num_classes"]
    state_dict = loaded["state_dict"]

    model = class_model_type.from_pretrained(
        pretrained_model_name_or_path=model_type,
        num_labels=num_classes,
        state_dict=state_dict)
    tokenizer = tokenizer_type.from_pretrained(model_type)

    # For type check
    assert isinstance(model, class_model_type)
    return model, tokenizer, class_names


class DetoxifyScorer(BaseSingleScorer):
    '''Class for computing scores based on the loaded Detoxify model. The logic
    is partly taken from the Detoxify class in
    https://github.com/unitaryai/detoxify/blob/master/detoxify/detoxify.py.
    '''

    def __init__(self,
                 device: str = 'cpu',
                 lang: str = 'en',
                 overflow_strategy: str = 'truncate',
                 max_input_length: Optional[int] = None):
        '''
        Initialize the scorer with the provided configs.

        Args:
            device: The device on which the model is loaded (default 'cpu')
            lang: The language of the model (default 'en')
            overflow_strategy: The strategy to handle the overflow of the input.
                The value should be either "raise", "truncate" or "nullify".
            max_input_length: The maximum length of the input. If None, the
                maximum length of the model is used.
        '''
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            device, lang)
        self.device = device
        self.model.to(self.device)  # type: ignore
        self.overflow_strategy = overflow_strategy
        self.max_input_length = (max_input_length or
                                 self.tokenizer.model_max_length)

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
            return_tensors='pt').to(self.model.device)

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

    def _score_tokens(
            self, tokens: Tuple[BatchEncoding,
                                list[bool]]) -> list[Optional[float]]:
        input_tokens, validation_results = tokens
        out = self.model(**input_tokens)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = [
                scores[ex_i][i].tolist() for ex_i in range(len(scores))
            ]
        toxicity_scores = results['toxicity']

        for i, validation_result in enumerate(validation_results):
            if not validation_result:
                toxicity_scores[i] = None

        return toxicity_scores
