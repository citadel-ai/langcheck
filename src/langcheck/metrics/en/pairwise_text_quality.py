from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI

from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.utils.progess_bar import tqdm_wrapper


def pairwise_comparison(generated_outputs_a: List[str] | str,
                        generated_outputs_b: List[str] | str,
                        prompts: List[str] | str,
                        reference_outputs: Optional[List[str]] = None,
                        model_type: str = 'openai',
                        openai_client: Optional[OpenAI] = None,
                        openai_args: Optional[Dict[str, str]] = None) -> None:
    assert model_type in [
        'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["openai", "azure_openai"]')

    def _prompt(gen_output_a: str, gen_output_b: str, user_query: str) -> str:
        return f'''
        You are comparing the quality of two responses to a user's query. Here
        is the data:
        [BEGIN DATA]
        ************
        [User Query]: {user_query}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        Determine which of the responses is a better response to the user's
        query. Consider factors such as helpfulness, correctness, and relevance
        in your assessment. Do not allow the order in which the responses were
        presented to influence your assessment. Do not allow the length of the
        responses to influence your assessment. The available assessments are:
        `Response A` - Response A is a better response.
        `Response B` - Response B is a better response.
        `Tie` - The two responses are roughly equal in quality.

        Take a deep breath and work on this problem step-by-step.
        '''

    def _prompt_with_reference(gen_output_a: str, gen_output_b: str,
                               user_query: str, ref_output: str) -> str:
        return f'''
        You are comparing the quality of two responses to a user's query. The
        ideal response to the user's query is also provided to you as a
        reference. Here is the data:
        [BEGIN DATA]
        ************
        [Ideal Response]: {ref_output}
        ************
        [User Query]: {user_query}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        Determine which of the responses is a better response to the user's
        query. Consider factors such as helpfulness, correctness, and relevance
        in your assessment, using the provided Ideal Response as a reference. Do
        not allow the order in which the responses were presented to influence
        your assessment. Do not allow the length of the responses to influence
        your assessment. The available assessments are:
        `Response A` - Response A is a better response.
        `Response B` - Response B is a better response.
        `Tie` - The two responses are roughly equal in quality.

        Take a deep breath and work on this problem step-by-step.
        '''

    def _function_call_prompt(long_assessment: str) -> str:
        return f'''
        The following is an assessment on whether Response A or Response B is
        the better response to the user's query:
        ************
        [Assessment]: {long_assessment}
        ************

        Save the resulting assessment. The available assessments are:
        `Response A`
        `Response B`
        `Tie`
        '''

    pairwise_comparison_assessment_to_score = {
        'Response B': 1.0,
        'Tie': 0.5,
        'Response A': 0.0
    }
    oai_evaluator = OpenAIBasedEvaluator(
        assessment_to_score_mapping=pairwise_comparison_assessment_to_score,
        function_name='save_pairwise_comparison_assessment',
        function_description=("Saves a pairwise comparison assessment."),
        argument_name='pairwise_comparison',
        argument_description='The pairwise comparison assessment',
        client_type=model_type,
        client=openai_client,
        openai_args=openai_args)

    score_list = []
    explanation_list = []
    if reference_outputs is not None:
        for gen_a, gen_b, user_query, ref in tqdm_wrapper(
                zip(generated_outputs_a, generated_outputs_b, prompts,
                    reference_outputs),
                desc='Calculating scores',
                total=len(prompts)):
            score, explanation = oai_evaluator.get_score(
                _prompt_with_reference(gen_a, gen_b, user_query, ref),
                _function_call_prompt)
            score_list.append(score)
            explanation_list.append(explanation)
    else:
        for gen_a, gen_b, user_query in tqdm_wrapper(zip(
                generated_outputs_a, generated_outputs_b, prompts),
                                                     desc='Calculating scores',
                                                     total=len(prompts)):
            score, explanation = oai_evaluator.get_score(
                _prompt(gen_a, gen_b, user_query), _function_call_prompt)
            score_list.append(score)
            explanation_list.append(explanation)

    print(score_list)
    print(explanation_list)
