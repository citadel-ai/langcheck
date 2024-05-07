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
    '''EvalClient defined for OpenAI API.
    '''

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

    def get_text_responses(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        '''The function that gets resonses to the given prompt texts.
        We use OpenAI's 'gpt-turbo-3.5' model by default, but you can configure
        it by passing the 'model' parameter in the openai_args.

        Args:
            prompts: The prompts you want to get the responses for.

        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        '''
        config = {"model": "gpt-3.5-turbo", "seed": 123}
        config.update(self._openai_args or {})
        tqdm_description = tqdm_description or 'Intermediate assessments (1/2)'  # NOQA: E501
        responses = self._call_api(prompts=prompts,
                                   config=config,
                                   tqdm_description=tqdm_description)
        response_texts = [
            response.choices[0].message.content if response else None
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
        texts that describe the evaluation results) into scores. We leverage the
        function calling API to extract the short assessment results from the
        unstructured assessments, so please make sure that the model you use
        supports function calling
        (https://platform.openai.com/docs/guides/gpt/function-calling).

        Ref:
            https://platform.openai.com/docs/guides/gpt/function-calling

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
        if language not in ['en', 'ja', 'de', 'zh']:
            raise ValueError(f'Unsupported language: {language}')

        fn_call_template = get_template(
            f'{language}/get_score/function_calling.j2')

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
                 azure_openai_client: AzureOpenAI | None = None,
                 openai_args: dict[str, str] | None = None,
                 *,
                 use_async: bool = False):
        '''
        Intialize the Azure OpenAI evaluation client.

        Args:
            text_model_name (Optional): The text model name you want to use with
                the Azure OpenAI API. The name is used as
                `{ "model": text_model_name }` parameter when calling the Azure
                OpenAI API for text models.
            embedding_model_name (Optional): The text model name you want to
                use with the Azure OpenAI API. The name is used as
                `{ "model": embedding_model_name }` parameter when calling the
                Azure OpenAI API for embedding models.
            azure_openai_client (Optional): The Azure OpenAI client to use.
            openai_args: (Optional) dict of additional args to pass in to the
            ``client.chat.completions.create`` function
            use_async: (Optional) If True, the async client will be used.
        '''
        assert (text_model_name is not None or
                embedding_model_name is not None), (
                    'You need to specify either the text_model_name or the '
                    'embedding_model_name to use the Azure OpenAI API.')
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#completions
        kargs = {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }
        if azure_openai_client is not None:
            self._client = azure_openai_client
        elif use_async:
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
        '''This method does the sanity check for the text_model_name and then
        calls the parent class's get_score method with the additional "model"
        parameter. See the parent class for the detailed documentation.
        '''
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
        '''This method does the sanity check for the embedding_model_name and
        then calls the parent class's similarity_scorer method with the
        additional "model" parameter. See the parent class for the detailed
        documentation.
        '''
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
