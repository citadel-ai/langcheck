import json
import os
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple)

from openai import AzureOpenAI, OpenAI

from langcheck.utils.progess_bar import tqdm_wrapper


class OpenAIBasedEvaluator:
    '''Evaluator class based on OpenAI's API.'''

    def __init__(self, assessment_to_score_mapping: Dict[str, float],
                 function_name: str, function_description: str,
                 argument_name: str, argument_description: str,
                 client_type: str, client: Optional[OpenAI],
                 openai_args: Optional[Dict[str, str]]) -> None:
        '''
        Initialize the OpenAIBasedEvaluator with given parameters.

        Args:
            assessment_to_score_mapping: Mapping from an assessment (str) to a
                numeric score (float)
            function_name: Name of the function that we force the OpenAI API to
                call
            function_description: Description of the function
            argument_name: Name of the argument to the function. This should be
                the name of the metric to evaluate (e.g. sentiment).
            argument_description: Description of the argument
            client_type: The type of OpenAI client ('openai' or 'azure_openai')
            client: (Optional) OpenAI or AzureOpenAI client. If this is None,
                we will attempt to create a default client depending on the
                ``client_type``.
            openai_args: (Optional) Dict of additional args to pass in to the
                ``client.chat.completions.create`` function
        '''
        self._client_type = client_type
        if self._client_type == 'azure_openai' and not openai_args:
            raise AssertionError(
                'The model deployment must be specified in `openai_args` for '
                'the azure_openai type, e.g. '
                '`openai_args={"model": "YOUR_DEPLOYMENT_NAME"}`')

        if client:
            self._client = client
        elif self._client_type == 'openai':
            self._client = OpenAI()
        elif self._client_type == 'azure_openai':
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#completions
            self._client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv(
                    "AZURE_OPENAI_ENDPOINT"))  # type: ignore
        else:
            raise AssertionError(f'Unexpected client type "{client_type}"')

        self._assessment_to_score_mapping = assessment_to_score_mapping
        self._function_name = function_name
        self._function_description = function_description
        self._argument_name = argument_name
        self._argument_description = argument_description
        self._openai_args = openai_args

    def get_score(
        self,
        prompts: Iterator[str] | Sequence[str],
        function_call_prompt_template: Callable,
    ) -> Tuple[List[float | None], List[str | None]]:
        '''
        Retrieves the score and unstructured assessment for a given prompt using
        the OpenAI API. The first API call is a "normal" call, where the API
        will return unstructured text. The second call leverages the function
        calling API, where the API should return a structured response. If the
        API fails to return a response, or the response is not in the expected
        format, `None` is returned for both the score and the unstructured
        assessment.

        Args:
            prompts: :List of prompt that asks the OpenAI API for the
                unstructured assessment response
            function_call_prompt_template: Prompt template used to construct the
                prompt that asks the OpenAI API for the structured assessment
                response

        Returns:
            scores: List of scores associated with the corresponding prompts
                based on the resulting structured assessment. If a score cannot
                be computed for a prompt, it will be represented as `None`.
            unstructured_assessments: List of unstructured assessment text,
                which also serves as the explanation of the score. If a score
                cannot be computed for a prompt, it will be represented as
                `None`.
        '''
        # First, call the API to get an unstructured assessment. The response
        # should include both the assessment itself and the model's reasoning
        # for that assessment.
        # Call the API to get an unstructured assessments.
        config_unstructured_assessments = {
            "model": "gpt-3.5-turbo",
            "seed": 123
        }
        config_unstructured_assessments.update(self._openai_args or {})

        responses = self._call_api(prompts=prompts,
                                   config=config_unstructured_assessments)
        unstructured_assessments = [
            response.choices[0].message.content if response else None
            for response in responses
        ]

        # Next, call the API leveraging function calling to get a structured
        # assessment.
        # Construct the prompt for the function calling API, filled with None
        # for the failed unstructured assessment.
        fn_call_messages = [
            function_call_prompt_template(unstructured_assessment)
            if unstructured_assessment else None
            for unstructured_assessment in unstructured_assessments
        ]
        argument_options = list(self._assessment_to_score_mapping.keys())
        functions = [{
            "name": self._function_name,
            "description": self._function_description,
            "parameters": {
                "type": "object",
                "properties": {
                    self._argument_name: {
                        "type": "string",
                        "enum": argument_options,
                        "description": self._argument_description,
                    },
                },
                "required": [self._argument_name],
            },
        }]
        config_structured_assessments = {
            "seed": 123,
            "functions": functions,
            "function_call": {
                "name": self._function_name
            },
            "model": "gpt-3.5-turbo"
        }
        config_structured_assessments.update(self._openai_args or {})

        responses = self._call_api(
            prompts=fn_call_messages,
            config=config_structured_assessments,
        )
        function_args = [
            json.loads(response.choices[0].message.function_call.arguments)
            if response else None for response in responses
        ]
        assessments = [
            function_arg.get(self._argument_name) if function_arg else None
            for function_arg in function_args
        ]

        # Check if any of the assessments are not recognized.
        for assessment in assessments:
            if (assessment is None) or (assessment
                                        in self._assessment_to_score_mapping):
                continue
            # By leveraging the function calling API, this should be pretty
            # rare, but we're dealing with LLMs here so nothing is
            # absolute!
            print(f'OpenAI returned an unrecognized assessment: "{assessment}"')

        return [
            self._assessment_to_score_mapping[assessment]
            if assessment else None for assessment in assessments
        ], unstructured_assessments

    def _call_api(self, prompts: Iterator[str | None] | Sequence[str | None],
                  config: Dict[str, str]) -> List[Any]:
        # Generates input dict for API call. This procedure is separated as a
        # method because yapf fails when there are too much nests.
        def _generate_model_input(prompt: str | None) -> Dict[str, Any]:
            return {"messages": [{"role": "user", "content": prompt}], **config}

        # A helper function to call the API with exception filter for alignment
        # of exception handling with the async version.
        def _call_api_with_exception_filter(model_input: Dict[str, Any]) -> Any:
            if model_input is None:
                return None
            try:
                return self._client.chat.completions.create(**model_input)
            except Exception as e:
                return e

        model_inputs = map(_generate_model_input, prompts)
        responses = [
            _call_api_with_exception_filter(model_input)
            for model_input in tqdm_wrapper(model_inputs)
        ]

        # Filter out exceptions and print them out.
        for i, response in enumerate(responses):
            if not isinstance(response, Exception):
                continue
            print('OpenAI failed to return an assessment corresponding to '
                  f'{i}th prompt: {response}')
            responses[i] = None
        return responses
