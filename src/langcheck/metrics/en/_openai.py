import json
from typing import Callable, Dict, Optional, Tuple

import openai


class OpenAIBasedEvaluator:
    '''Evaluator class based on OpenAI's API.'''

    def __init__(self,
                 assessment_to_score_mapping: Dict[str, float],
                 function_name: str,
                 function_description: str,
                 argument_name: str,
                 argument_description: str,
                 openai_args: Optional[Dict[str, str]] = None) -> None:
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
            openai_args: Dict of additional args to pass in to the
                `openai.ChatCompletion.create` function, default None
        '''
        self._assessment_to_score_mapping = assessment_to_score_mapping
        self._function_name = function_name
        self._function_description = function_description
        self._argument_name = argument_name
        self._argument_description = argument_description
        self._openai_args = openai_args

    def get_score(
        self, prompt: str, function_call_prompt_template: Callable
    ) -> Tuple[Optional[float], Optional[str]]:
        '''
        Retrieves the score and unstructured assessment for a given prompt using
        the OpenAI API. The first API call is a "normal" call, where the API
        will return unstructured text. The second call leverages the function
        calling API, where the API should return a structured response. If the
        API fails to return a response, or the response is not in the expected
        format, `None` is returned for both the score and the unstructured
        assessment.

        Args:
            prompt: Prompt that asks the OpenAI API for the unstructured
                assessment response
            function_call_prompt_template: Prompt template used to construct the
                prompt that asks the OpenAI API for the structured assessment
                response

        Returns:
            score: score associated with the given prompt based on the resulting
                structured assessment. Returns `None` if the score could not be
                computed.
            unstructured_assessment: unstructured assessment text, which also
                serves as the explanation of the score. Returns `None` if the
                score could not be computed.
        '''
        # First, call the API to get an unstructured assessment. The response
        # should include both the assessment itself and the model's reasoning
        # for that assessment.
        messages = [{"role": "user", "content": prompt}]
        try:
            if self._openai_args is None:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
            else:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    **self._openai_args,
                )
            # This metrics-with-openai-models>`__ is necessary to pass pyright
            # since the openai library is not typed.
            assert isinstance(response, dict)
            unstructured_assessment = response["choices"][0]["message"][
                "content"]
        except Exception as e:
            print(f'OpenAI failed to return an unstructured assessment: {e}')
            print(f'Prompt that triggered the failure is:\n{prompt}')
            return None, None

        # Next, call the API leveraging function calling to get a structured
        # assessment
        fn_call_messages = [{
            "role": "user",
            "content": function_call_prompt_template(unstructured_assessment)
        }]
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
        try:
            if self._openai_args is None:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=fn_call_messages,
                    functions=functions,
                    function_call={"name": self._function_name},
                )
            else:
                response = openai.ChatCompletion.create(
                    messages=fn_call_messages,
                    functions=functions,
                    function_call={"name": self._function_name},
                    **self._openai_args,
                )
            # This sanity check is necessary to pass pyright since the openai
            # library is not typed.
            assert isinstance(response, dict)
            function_args = json.loads(
                response["choices"][0]["message"]["function_call"]["arguments"])
            assessment = function_args.get(self._argument_name)
        except Exception as e:
            print(f'OpenAI failed to return a structured assessment: {e}')
            print('Prompt that triggered the failure is:\n'
                  f'{function_call_prompt_template(unstructured_assessment)}')
            return None, None

        if assessment not in self._assessment_to_score_mapping:
            # By leveraging the function calling API, this should be pretty
            # rare, but we're dealing with LLMs here so nothing is absolute!
            print(f'OpenAI returned an unrecognized assessment: "{assessment}"')
            print(f'Prompt that triggered the failure is:\n{prompt}')
            return None, None

        return self._assessment_to_score_mapping[
            assessment], unstructured_assessment
