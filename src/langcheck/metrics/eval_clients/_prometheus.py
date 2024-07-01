from __future__ import annotations

from typing import Iterable

import torch
from transformers import LlamaTokenizer, MistralForCausalLM

from langcheck.utils.progess_bar import tqdm_wrapper

from ._base import EvalClient


class PrometheusEvalClient(EvalClient):
    '''EvalClient defined for the Prometheus 2 model.
    This eval client currently supports only English.
    Presented in `"Prometheus 2: An Open Source Language Model Specialized
    in Evaluating Other Language Models" <https://arxiv.org/abs/2405.01535>`.
    We adapted the prompts in <https://github.com/prometheus-eval/prometheus-
    eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py>.
    '''

    def __init__(self,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda"):
        """
        Initilize the Prometheus evaluation client.

        Args:
            torch_dtype: The torch dtype to use. torch.bfloat16 is recommended.
            device: The device to load the model on.
        """
        self._tokenizer = LlamaTokenizer.from_pretrained(
            "prometheus-eval/prometheus-7b-v2.0")
        self._model: MistralForCausalLM = MistralForCausalLM.from_pretrained(
            "prometheus-eval/prometheus-7b-v2.0",
            torch_dtype=torch_dtype)  # type: ignore
        self._model.to(device)  # type: ignore
        self._device = device

    def get_text_responses(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        """The function that generates resonses to the given prompt texts.

        Args:
            prompts: The prompts you want to get the responses for.
            tqdm_description: The description to be shown in the tqdm bar.
        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        """
        response_texts = []
        tqdm_description = tqdm_description or "Intermediate assessments (1/2)"  # NOQA: E501
        for prompt in tqdm_wrapper(prompts, desc=tqdm_description):
            message = [{"role": "user", "content": prompt}]
            input_ids = self._tokenizer.apply_chat_template(message,
                                                            return_tensors="pt",
                                                            truncation=True,
                                                            max_length=4096)
            input_ids = input_ids.to(self._device)
            generated_ids = self._model.generate(input_ids, max_new_tokens=1000)
            response_text = self._tokenizer.batch_decode(
                generated_ids[:,
                              input_ids.shape[1]:], skip_special_tokens=True)[0]
            response_texts.append(response_text if response_text else None)

        return response_texts

    def get_float_score(
            self,
            metric_name: str,
            language: str,
            unstructured_assessment_result: list[str | None],
            score_map: dict[str, float],
            *,
            tqdm_description: str | None = None) -> list[float | None]:
        '''The function that transforms the unstructured assessments (i.e. long
        texts that describe the evaluation results) into scores. We simple find
        the assessment result which appeared latest in the unstructured text.
        Args:
            metric_name: The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            unstructured_assessment_result: The unstructured assessment results
                for the given assessment prompts.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.
            tqdm_description: The description to be shown in the tqdm bar.

        Returns:
            A list of scores for the given prompts. The scores can be None if
            the evaluation fails.
        '''
        if language != "en":
            raise ValueError(f"Unsupported language: {language}")

        tqdm_description = tqdm_description or "Scores (2/2)"

        options = list(score_map.keys())
        assessments = []
        for unstructured_assessment in tqdm_wrapper(
                unstructured_assessment_result, desc=tqdm_description):
            if unstructured_assessment is None:
                assessments.append(None)
                continue

            # Find the option that appears latest in the assessment
            assessment = max(options, key=unstructured_assessment.rfind)
            if unstructured_assessment.find(assessment) == -1:
                print("No options found in the assessment.")
                assessments.append(None)
            else:
                assessments.append(assessment)

        return [
            score_map[assessment] if assessment else None
            for assessment in assessments
        ]

    def similarity_scorer(self):
        raise NotImplementedError(
            "Embedding-based metrics are not supported in PrometheusEvalClient."
            "Use other EvalClients to get these metrics.")
