from __future__ import annotations

import asyncio
from typing import Any, Iterable

from anthropic import Anthropic, AsyncAnthropic

from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ._base import EvalClient


class AnthropicEvalClient(EvalClient):
    '''EvalClient defined for Anthropic API.
    '''

    def __init__(self,
                 anthropic_client: Anthropic | None = None,
                 anthropic_args: dict[str, Any] | None = None,
                 *,
                 use_async: bool = False):
        '''
        Initialize the Anthropic evaluation client. The authentication
        information is automatically read from the environment variables,
        so please make sure ANTHROPIC_API_KEY is set.

        Args:
            anthropic_client: (Optional) The Anthropic client to use.
            anthropic_args: (Optional) dict of additional args to pass in to
                the ``client.messages.create`` function
            use_async: (Optional) If True, the async client will be used.
        '''
        if anthropic_client:
            self._client = anthropic_client
        elif use_async:
            self._client = AsyncAnthropic()
        else:
            self._client = Anthropic()

        self._anthropic_args = anthropic_args or {}
        self._use_async = use_async

    def _call_api(self,
                  prompts: Iterable[str | None],
                  config: dict[str, Any],
                  *,
                  tqdm_description: str | None = None) -> list[Any]:
        # A helper function to call the API with exception filter for alignment
        # of exception handling with the async version.
        def _call_api_with_exception_filter(model_input: dict[str, Any]) -> Any:
            if model_input is None:
                return None
            try:
                return self._client.messages.create(**model_input)
            except Exception as e:
                return e

        model_inputs = [{
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            **config
        } for prompt in prompts]

        if self._use_async:
            # A helper function to call the async API.
            async def _call_async_api() -> list[Any]:
                responses = await asyncio.gather(*map(
                    lambda model_input: self._client.messages.create(
                        **model_input), model_inputs),
                                                 return_exceptions=True)
                return responses

            responses = asyncio.run(_call_async_api())
        else:
            responses = [
                _call_api_with_exception_filter(model_input)
                for model_input in tqdm_wrapper(model_inputs,
                                                desc=tqdm_description)
            ]

        # Filter out exceptions and print them out.
        for i, response in enumerate(responses):
            if not isinstance(response, Exception):
                continue
            print('Anthropic failed to return an assessment corresponding to '
                  f'{i}th prompt: {response}')
            responses[i] = None
        return responses

    def get_text_responses(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        '''The function that gets resonses to the given prompt texts.
        We use Anthropic's 'claude-3-haiku-20240307' model by default, but you
        can configure it by passing the 'model' parameter in the anthropic_args.

        Args:
            prompts: The prompts you want to get the responses for.

        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        '''
        config = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 4096,
            "temperature": 0.0
        }
        config.update(self._anthropic_args or {})
        tqdm_description = tqdm_description or 'Intermediate assessments (1/2)'  # NOQA: E501
        responses = self._call_api(prompts=prompts,
                                   config=config,
                                   tqdm_description=tqdm_description)
        response_texts = [
            response.content[0].text if response else None
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
            tqdm_description: str | None = None) -> list[float | None]:
        '''The function that transforms the unstructured assessments (i.e. long
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
        '''
        if language not in ['en', 'ja', 'de']:
            raise ValueError(f'Unsupported language: {language}')

        options = list(score_map.keys())
        get_score_template = get_template(f'{language}/get_score/plain_text.j2')
        get_score_prompts = [
            get_score_template.render({
                'metric': metric_name,
                'unstructured_assessment': unstructured_assessment,
                'options': options,
            }) if unstructured_assessment else None
            for unstructured_assessment in unstructured_assessment_result
        ]

        config = {"model": "claude-3-haiku-20240307", "max_tokens": 1024}
        config.update(self._anthropic_args or {})
        tqdm_description = tqdm_description or 'Scores (2/2)'
        responses = self._call_api(prompts=get_score_prompts,
                                   config=config,
                                   tqdm_description=tqdm_description)
        raw_response_texts = [
            response.content[0].text if response else None
            for response in responses
        ]

        def _turn_to_score(response: str | None) -> float | None:
            if response is None:
                return None
            option_found = [option for option in options if option in response]
            # if response contains multiple options as substrings, return None
            if len(option_found) != 1:
                return None
            return score_map[option_found[0]]

        return [_turn_to_score(response) for response in raw_response_texts]

    def similarity_scorer(self):
        raise NotImplementedError(
            'Embedding-based metrics are not supported in AnthropicEvalClient.'
            'Use other EvalClients to get these metrics.')
