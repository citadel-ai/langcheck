from __future__ import annotations

import json
import os
from typing import Any

import requests

from langcheck.utils.progress_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ._base import EvalClient


class OpenRouterEvalClient(EvalClient):
    """EvalClient defined for the OpenRouter API."""

    def __init__(
        self,
        openrouter_args: dict[str, str] | None = None,
        *,
        system_prompt: str | None = None,
    ):
        """
        Initialize the OpenRouter evaluation client.

        Args:
            openrouter_args: (Optional) dict of additional args to pass in to the
            ``client.chat.completions.create`` function.
            system_prompt: (Optional) The system prompt to use. If not provided,
                no system prompt will be used.
        """

        if os.getenv("OPENROUTER_API_KEY") is None:
            raise ValueError("OPENROUTER_API_KEY not set!")

        self._openrouter_args = openrouter_args
        self._system_prompt = system_prompt

    def _call_api(
        self,
        prompts: list[str] | list[str | None],
        config: dict[str, str],
        *,
        tqdm_description: str | None = None,
    ) -> list[Any]:
        def generate_json_dumps(prompt: str):
            system_message = (
                []
                if not self._system_prompt
                else [
                    {
                        "role": "system",
                        "content": self._system_prompt,
                    }
                ]
            )
            msg_dict = {
                "messages": system_message
                + [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
            return msg_dict | config

        responses = []
        for prompt in tqdm_wrapper(prompts, desc=tqdm_description):
            if prompt is not None:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    },
                    data=json.dumps(generate_json_dumps(prompt)),
                )
                responses.append(response.json())

        return responses

    def get_text_responses(
        self,
        prompts: list[str],
        *,
        tqdm_description: str | None = None,
    ) -> list[str | None]:
        """The function that gets responses to the given prompt texts.
        The user's default OpenRouter model is used by default, but you can
        configure it by passing the 'model' parameter in the openrouter_args.

        Args:
            prompts: The prompts you want to get the responses for.

        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        """
        config = self._openrouter_args or {}
        tqdm_description = tqdm_description or "Intermediate assessments (1/2)"
        responses = self._call_api(
            prompts=prompts,
            config=config,
            tqdm_description=tqdm_description,
        )
        response_texts = [
            response["choices"][0]["message"]["content"] if response else None
            for response in responses
        ]

        return response_texts

    def get_float_score(
        self,
        metric_name: str,
        language: str,
        unstructured_assessment_result: list[str | None],
        score_map: dict[str, float],
        *,
        tqdm_description: str | None = None,
    ) -> list[float | None]:
        """The function that transforms the unstructured assessments (i.e. long
        texts that describe the evaluation results) into scores.

        Args:
            metric_name : The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            unstructured_assessment_result: The unstructured assessment results
                for the given assessment prompts.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.
            tqdm_description: The description to be shown in the tqdm bar.
        Returns:
            A list of scores for the given prompts. The scores can be None if
            the evaluation fails.
        """
        if language not in ["en", "ja"]:
            raise ValueError(f"Unsupported language: {language}")

        options = list(score_map.keys())
        get_score_template = get_template(f"{language}/get_score/plain_text.j2")
        get_score_prompts = [
            get_score_template.render(
                {
                    "metric": metric_name,
                    "unstructured_assessment": unstructured_assessment,
                    "options": options,
                }
            )
            if unstructured_assessment
            else None
            for unstructured_assessment in unstructured_assessment_result
        ]

        # If there are any Nones in get_score_prompts,
        # they are excluded from messages to prevent passing those to the model.
        prompts = [prompt for prompt in get_score_prompts if prompt is not None]

        config = {}
        config.update(self._openrouter_args or {})
        tqdm_description = tqdm_description or "Scores (2/2)"
        responses = self._call_api(
            prompts,
            config,
            tqdm_description=tqdm_description,
        )
        raw_response_texts = [
            response["choices"][0]["message"]["content"] if response else None
            for response in responses
        ]

        responses_for_scoring = []
        idx_raw_response_texts = 0
        for idx in range(len(get_score_prompts)):
            if get_score_prompts[idx] is None:
                responses_for_scoring.append(None)
            else:
                responses_for_scoring.append(
                    raw_response_texts[idx_raw_response_texts]
                )
                idx_raw_response_texts += 1

        def _turn_to_score(response: str | None) -> float | None:
            if response is None:
                return None
            option_found = [option for option in options if option in response]
            # if response contains multiple options as substrings, return None
            if len(option_found) != 1:
                return None
            return score_map[option_found[0]]

        return [_turn_to_score(response) for response in responses_for_scoring]

    def get_score(
        self,
        metric_name: str,
        language: str,
        prompts: str | list[str],
        score_map: dict[str, float],
    ) -> tuple[list[float | None], list[str | None]]:
        """Give scores to texts embedded in the given prompts. The function
        itself calls get_text_responses and get_float_score to get the scores.
        The function returns the scores and the unstructured explanation
        strings.

        Args:
            metric_name: The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            prompts: The prompts that contain the original text to be scored,
                the evaluation criteria... etc. Typically it is based on the
                Jinja prompt templates and instantiated withing each metric
                function.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.

        Returns:
            A tuple of two lists. The first list contains the scores for each
            prompt and the second list contains the unstructured assessment
            results for each prompt. Both can be None if the evaluation fails.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        unstructured_assessment_result = self.get_text_responses(prompts)
        scores = self.get_float_score(
            metric_name,
            language,
            unstructured_assessment_result,
            score_map,
        )
        return scores, unstructured_assessment_result

    def similarity_scorer(self):
        raise NotImplementedError(
            "Embedding-based metrics are not supported in OpenRouterEvalClient."
            "Use other EvalClients to get these metrics."
        )
