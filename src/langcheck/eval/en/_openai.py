import json
from typing import Dict, Optional

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

    def get_score(self, prompt: str) -> float:
        '''
        Retrieves the score for a given prompt using the OpenAI API.

        Args:
            prompt: Prompt that asks the OpenAI API for an assessment

        Returns:
            Score associated with the given prompt based the resulting
                assessment

        Raises:
            AssertionError: If the OpenAI API returns an unrecognized assessment
        '''
        argument_options = list(self._assessment_to_score_mapping.keys())
        messages = [{"role": "user", "content": prompt}]
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
        if self._openai_args is None:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                functions=functions,
                function_call={"name": self._function_name},
            )
        else:
            response = openai.ChatCompletion.create(
                messages=messages,
                functions=functions,
                function_call={"name": self._function_name},
                **self._openai_args,
            )
        response_message = response["choices"][0]["message"]
        function_args = json.loads(
            response_message["function_call"]["arguments"])
        assessment = function_args.get(self._argument_name)

        if assessment not in self._assessment_to_score_mapping:
            # By leveraging the function calling API, this should be pretty
            # rare, but we're dealing with LLMs here so nothing is absolute!
            raise AssertionError(
                'OpenAI returned an unrecognized assessment :(')

        return self._assessment_to_score_mapping[assessment]
