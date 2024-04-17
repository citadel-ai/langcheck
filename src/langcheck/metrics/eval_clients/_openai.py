from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Iterable

import torch
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer
from ._base import EvalClient


class OpenAIEvalClient(EvalClient):

    def __init__(self,
                 openai_client: OpenAI | None = None,
                 openai_args: dict[str, str] | None = None,
                 *,
                 use_async: bool = False):
        '''
        Intialize the OpenAI evaluation client.

        Args:
            openai_client: (Optional) The OpenAI client to use.
            openai_args: (Optional) dict of additional args to pass in to the
            ``client.chat.completions.create`` function
            use_async: (Optional) If True, the async client will be used.
        '''
        if openai_client:
            self._client = openai_client
        elif use_async:
            self._client = AsyncOpenAI()
        else:
            self._client = OpenAI()

        self._openai_args = openai_args
        self._use_async = use_async

    def _call_api(self,
                  prompts: Iterable[str | None],
                  config: dict[str, str],
                  *,
                  tqdm_description: str | None = None) -> list[Any]:
        # A helper function to call the API with exception filter for alignment
        # of exception handling with the async version.
        def _call_api_with_exception_filter(model_input: dict[str, Any]) -> Any:
            if model_input is None:
                return None
            try:
                return self._client.chat.completions.create(**model_input)
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
                    lambda model_input: self._client.chat.completions.create(
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
            print('OpenAI failed to return an assessment corresponding to '
                  f'{i}th prompt: {response}')
            responses[i] = None
        return responses

    def _unstructured_assessment(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        '''
        TODO
        '''
        config_unstructured_assessments = {
            "model": "gpt-3.5-turbo",
            "seed": 123
        }
        config_unstructured_assessments.update(self._openai_args or {})
        tqdm_description = tqdm_description or 'Intermediate assessments (1/2)'  # NOQA: E501
        responses = self._call_api(prompts=prompts,
                                   config=config_unstructured_assessments,
                                   tqdm_description=tqdm_description)
        unstructured_assessments = [
            response.choices[0].message.content if response else None
            for response in responses
        ]

        return unstructured_assessments

    def _get_float_score(
            self,
            metric_name: str,
            language: str,
            unstructured_assessment_result: list[str | None],
            score_map: dict[str, float],
            *,
            tqdm_description: str | None = None) -> list[float | None]:
        '''
        TODO
        '''
        if language not in ['en', 'ja', 'de', 'zh']:
            raise ValueError(f'Unsupported language: {language}')

        fn_call_template = get_template(f'{language}/get_score/openai.j2')

        options = list(score_map.keys())
        fn_call_messages = [
            fn_call_template.render({
                'metric': metric_name,
                'unstructured_assessment': unstructured_assessment,
                'options': options,
            }) if unstructured_assessment else None
            for unstructured_assessment in unstructured_assessment_result
        ]

        functions = [{
            'name': 'save_assessment',
            'description': f'Save the assessment of {metric_name}.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'assessment': {
                        'type': 'string',
                        'enum': options,
                        'description': f'The assessment of {metric_name}.',
                    },
                },
                'required': ['assessment'],
            },
        }]

        config_structured_assessments = {
            "seed": 123,
            "functions": functions,
            "function_call": {
                "name": 'save_assessment',
            },
            "model": "gpt-3.5-turbo"
        }
        config_structured_assessments.update(self._openai_args or {})

        tqdm_description = tqdm_description or 'Scores (2/2)'
        responses = self._call_api(prompts=fn_call_messages,
                                   config=config_structured_assessments,
                                   tqdm_description=tqdm_description)
        function_args = [
            json.loads(response.choices[0].message.function_call.arguments)
            if response else None for response in responses
        ]
        assessments = [
            function_arg.get('assessment') if function_arg else None
            for function_arg in function_args
        ]

        # Check if any of the assessments are not recognized.
        for assessment in assessments:
            if (assessment is None) or (assessment in options):
                continue
            # By leveraging the function calling API, this should be pretty
            # rare, but we're dealing with LLMs here so nothing is absolute!
            print(f'OpenAI returned an unrecognized assessment: "{assessment}"')

        return [
            score_map[assessment] if assessment else None
            for assessment in assessments
        ]

    def similarity_scorer(self) -> OpenAISimilarityScorer:
        '''
        https://openai.com/blog/new-embedding-models-and-api-updates
        '''
        assert isinstance(
            self._client,
            OpenAI), "Only sync clients are supported for similarity scoring."
        return OpenAISimilarityScorer(openai_client=self._client,
                                      openai_args=self._openai_args)


class AzureOpenAIEvalClient(OpenAIEvalClient):

    def __init__(self,
                 text_model_name: str | None = None,
                 embedding_model_name: str | None = None,
                 openai_args: dict[str, str] | None = None,
                 *,
                 use_async: bool = False):
        '''
        Intialize the Azure OpenAI evaluation client.

        Args:
            model_name: The name of the Azure OpenAI model to use.
            openai_args: (Optional) dict of additional args to pass in to the
            ``client.chat.completions.create`` function
            use_async: (Optional) If True, the async client will be used.
        '''
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#completions
        kargs = {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }
        if use_async:
            self._client = AsyncAzureOpenAI(**kargs)  # type: ignore
        else:
            self._client = AzureOpenAI(**kargs)  # type: ignore

        self._openai_args = openai_args or {}

        self._text_model_name = text_model_name
        self._embedding_model_name = embedding_model_name

        self._use_async = use_async

    def get_score(
        self,
        metric_name: str,
        language: str,
        prompts: str | Iterable[str],
        score_map: dict[str, float],
        *,
        intermediate_tqdm_description: str | None = None,
        score_tqdm_description: str | None = None
    ) -> tuple[list[float | None], list[str | None]]:
        assert self._text_model_name is not None, (
            'You need to specify the text_model_name to get the score for this '
            'metric.')
        self._openai_args['model'] = self._text_model_name
        return super().get_score(
            metric_name,
            language,
            prompts,
            score_map,
            intermediate_tqdm_description=intermediate_tqdm_description,
            score_tqdm_description=score_tqdm_description)

    def similarity_scorer(self) -> OpenAISimilarityScorer:
        assert isinstance(
            self._client, AzureOpenAI
        ), "Only sync clients are supported for similarity scoring."
        assert self._embedding_model_name is not None, (
            'You need to specify the embedding_model_name to get the score for '
            'this metric.')
        openai_args = {**self._openai_args, 'model': self._embedding_model_name}
        return OpenAISimilarityScorer(openai_client=self._client,
                                      openai_args=openai_args)


class OpenAISimilarityScorer(BaseSimilarityScorer):
    '''Similarity scorer that uses the OpenAI API to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    '''

    def __init__(self,
                 openai_client: OpenAI | AzureOpenAI,
                 openai_args: dict[str, Any] | None = None):

        super().__init__()

        self.openai_client = openai_client
        self.openai_args = openai_args

    def _embed(self, inputs: list[str]) -> torch.Tensor:
        '''Embed the inputs using the OpenAI API.
        '''
        # Embed the inputs
        if self.openai_args:
            embed_response = self.openai_client.embeddings.create(
                input=inputs, **self.openai_args)
        else:
            embed_response = self.openai_client.embeddings.create(
                input=inputs, model='text-embedding-3-small')

        return torch.Tensor([item.embedding for item in embed_response.data])
