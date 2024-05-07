from __future__ import annotations

import os
from typing import Any, Iterable

import google.ai.generativelanguage as glm
import google.generativeai as genai
import torch

from langcheck.utils.progess_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer
from ._base import EvalClient


class GeminiEvalClient(EvalClient):
    '''EvalClient defined for the Gemini model.
    '''

    def __init__(self,
                 model: genai.GenerativeModel | None = None,
                 model_args: dict[str, Any] | None = None,
                 generate_content_args: dict[str, Any] | None = None,
                 embed_model_name: str | None = None):
        '''
        Initialize the Gemini evaluation client. The authentication
        information is automatically read from the environment variables,
        so please make sure GOOGLE_API_KEY is set.

        TODO: Allow the user to specify the use of async. There currently
        seems to be an issue with `generate_content_async()` that is blocking
        this: https://github.com/google-gemini/generative-ai-python/issues/207

        Ref:
            https://ai.google.dev/api/python/google/generativeai/GenerativeModel

        Args:
            model: (Optional) The Gemini model to use. If not provided, the
                model will be created using the model_args.
            model_args: (Optional) Dict of args to create the Gemini model.
            generate_content_args: (Optional) Dict of args to pass in to the
                ``generate_content`` function.
            embed_model_name: (Optional) The name of the embedding model to use.
                If not provided, the models/embedding-001 model will be used.
        '''
        if model:
            self._model = model
        else:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model_args = model_args or {}
            self._model = genai.GenerativeModel(**model_args)

        self._generate_content_args = generate_content_args or {}
        self._embed_model_name = embed_model_name

    def _call_api(self,
                  prompts: Iterable[str | None],
                  config: dict[str, Any],
                  *,
                  tqdm_description: str | None = None) -> list[Any]:
        # A helper function to call the API with exception filter for alignment
        # of exception handling with the async version.
        def _call_api_with_exception_filter(prompt: str) -> Any:
            try:
                return self._model.generate_content(prompt, **config)
            except Exception as e:
                return e

        responses = [
            _call_api_with_exception_filter(prompt)
            for prompt in tqdm_wrapper(prompts, desc=tqdm_description)
        ]

        # Filter out exceptions and print them out. Also filter out responses
        # that are blocked by safety settings and print out the safety ratings.
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print('Gemini failed to return an assessment corresponding to '
                      f'{i}th prompt: {response}')
                responses[i] = None
            elif response.candidates[0].finish_reason == 3:
                print(
                    f"Gemini's safety settings blocked the {i}th prompt:\n {response.candidates[0].safety_ratings}"  # NOQA: E501
                )
                responses[i] = None
        return responses

    def get_text_responses(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        '''The function that gets resonses to the given prompt texts.

        Args:
            prompts: The prompts you want to get the responses for.
        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        '''
        config = {"generation_config": {"temperature": 0.0}}
        config.update(self._generate_content_args or {})
        tqdm_description = tqdm_description or 'Intermediate assessments (1/2)'  # NOQA: E501
        responses = self._call_api(prompts=prompts,
                                   config=config,
                                   tqdm_description=tqdm_description)
        response_texts = [
            response.text if response else None for response in responses
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
        (https://ai.google.dev/gemini-api/docs/function-calling#supported-models).

        Ref:
            https://ai.google.dev/gemini-api/docs/function-calling

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
        if language not in ['en', 'ja', 'de']:
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

        assessment_schema = glm.Schema(type=glm.Type.STRING, enum=options)

        save_assessment = glm.FunctionDeclaration(
            name='save_assessment',
            description=f'Save the assessment of {metric_name}.',
            parameters=glm.Schema(type=glm.Type.OBJECT,
                                  properties={'assessment': assessment_schema}))
        config_structured_assessments = {
            "tools": [save_assessment],
            "tool_config": {
                'function_calling_config': 'ANY'
            },
            "generation_config": {
                "temperature": 0.0
            }
        }
        config_structured_assessments.update(self._generate_content_args or {})

        tqdm_description = tqdm_description or 'Scores (2/2)'
        responses = self._call_api(prompts=fn_call_messages,
                                   config=config_structured_assessments,
                                   tqdm_description=tqdm_description)
        assessments = []
        for response in responses:
            if response is None:
                assessments.append(None)
                continue
            function_call = response.candidates[0].content.parts[
                0].function_call
            fc_dict = type(function_call).to_dict(function_call)
            assessments.append(fc_dict.get('args', {}).get('assessment'))
        # Check if any of the assessments are not recognized.
        for assessment in assessments:
            if (assessment is None) or (assessment in options):
                continue
            # By leveraging the function calling API, this should be pretty
            # rare, but we're dealing with LLMs here so nothing is absolute!
            print(f'Gemini returned an unrecognized assessment: "{assessment}"')

        return [
            score_map[assessment] if assessment else None
            for assessment in assessments
        ]

    def similarity_scorer(self) -> GeminiSimilarityScorer:
        return GeminiSimilarityScorer(embed_model_name=self._embed_model_name)


class GeminiSimilarityScorer(BaseSimilarityScorer):
    '''Similarity scorer that uses the Gemini API to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    '''

    def __init__(self, embed_model_name: str | None):

        super().__init__()

        self.embed_model_name = embed_model_name or 'models/embedding-001'

    def _embed(self, inputs: list[str]) -> torch.Tensor:
        '''Embed the inputs using the Gemini API.
        '''
        # Embed the inputs
        embed_response = genai.embed_content(model=self.embed_model_name,
                                             content=inputs)

        return torch.Tensor([item for item in embed_response['embedding']])
