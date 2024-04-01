from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from openai import OpenAI

from langcheck.metrics._validation import \
    validate_parameters_pairwise_comparison
from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper


def pairwise_comparison(
    generated_outputs_a: List[str] | str,
    generated_outputs_b: List[str] | str,
    prompts: List[str] | str,
    sources_a: Optional[List[str] | str] = None,
    sources_b: Optional[List[str] | str] = None,
    reference_outputs: Optional[List[str] | str] = None,
    enforce_consistency: bool = True,
    model_type: str = 'openai',
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str,
                               str]] = None) -> MetricValue[Optional[float]]:
    '''Calculates the pairwise comparison metric. This metric takes on float
    values of either 0.0 (Response A is better), 0.5 (Tie), or 1.0 (Response B
    is better). The score may also be `None` if it could not be computed.

    We currently support two model types:

    1. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default. While the model you use is configurable, please make sure to use
    one that supports function calling
    (https://platform.openai.com/docs/guides/gpt/function-calling). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    2. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Ref:
        Our prompt is similar to the prompt used in
        https://arxiv.org/abs/2306.05685

    Args:
        generated_outputs_a: Model A's generated output(s) to evaluate
        generated_outputs_b: Model B's generated output(s) to evaluate
        prompts: The prompts used to generate the output(s)
        sources_a: The source text(s) for Model A's generated output(s), default
            None
        sources_b: The source text(s) for Model B's generated output(s), default
            None
        reference_outputs: The reference output(s), default None
        enforce_consistency: When this is True, we will only return a score if
            the score is the same when Model A and Model B are swapped. This is
            useful for ensuring that the evaluator's position bias is not
            impacting the scores. Default True.
        model_type: The type of model to use ('openai', or 'azure_openai'),
            default 'openai'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None, we will attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None

    Returns:
        An MetricValue object
    '''
    generated_outputs_a, generated_outputs_b, prompts, sources_a, sources_b, reference_outputs = validate_parameters_pairwise_comparison(  # NOQA: E501
        generated_outputs_a, generated_outputs_b, prompts, sources_a, sources_b,
        reference_outputs)
    assert model_type in [
        'openai', 'azure_openai'
    ], ('Unsupported model type. '
        'The supported ones are ["openai", "azure_openai"]')

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

    prompt_fn, data_iter = _construct_pairwise_comparison_prompt_and_data(
        generated_outputs_a, generated_outputs_b, prompts, sources_a, sources_b,
        reference_outputs)
    all_prompts = [prompt_fn(*data_instance) for data_instance in data_iter]
    scores, explanations = oai_evaluator.get_score(all_prompts,
                                                   _function_call_prompt)

    if enforce_consistency:
        # Swap the generated outputs and calculate the scores again
        prompt_fn, swapped_data_iter = _construct_pairwise_comparison_prompt_and_data(  # NOQA: E501
            generated_outputs_b, generated_outputs_a, prompts, sources_b,
            sources_a, reference_outputs)
        all_swapped_prompts = [
            prompt_fn(*data_instance) for data_instance in swapped_data_iter
        ]
        swapped_scores, swapped_explanations = oai_evaluator.get_score(
            all_swapped_prompts, _function_call_prompt)

        # Iterate through the scores and explanations to check for consistency.
        # If a score is not consistent, set it to None, and merge the two
        # explanations to show the inconsistency.
        for i in range(len(scores)):
            if scores[i] is None or swapped_scores[i] is None:
                # If either score is None, we cannot determine consistency, so
                # we set the score and explanation to None
                scores[i] = None
                explanations[i] = None
                continue
            if scores[i] + swapped_scores[i] != 1.0:  # type: ignore
                scores[i] = None
                explanations[
                    i] = f'Original assessment: {explanations[i]}\nSwapped assessment: {swapped_explanations[i]}'  # NOQA: E501

    return MetricValue(metric_name='pairwise_comparison',
                       prompts=prompts,
                       generated_outputs=(generated_outputs_a,
                                          generated_outputs_b),
                       reference_outputs=reference_outputs,
                       sources=(sources_a, sources_b),
                       explanations=explanations,
                       metric_values=scores,
                       language='en')


def _construct_pairwise_comparison_prompt_and_data(
        generated_outputs_1: List[str], generated_outputs_2: List[str],
        prompts: List[str], sources_1: Optional[List[str]],
        sources_2: Optional[List[str]], reference_outputs: Optional[List[str]]
) -> tuple[Callable, Iterable[Any]]:

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
        [User Query]: {user_query}
        ************
        [Ideal Response]: {ref_output}
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

    def _prompt_with_source(gen_output_a: str, gen_output_b: str,
                            user_query: str, source: str) -> str:
        return f'''
        You are comparing the quality of two responses to a user's query. Source
        text that is supposedly relevant to the user's query is also provided
        to you as a reference (the source text may contain some duplication).
        Here is the data:
        [BEGIN DATA]
        ************
        [User Query]: {user_query}
        ************
        [Source]: {source}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        Determine which of the responses is a better response to the user's
        query. Consider factors such as helpfulness, correctness, and relevance
        in your assessment, using the provided Source as a reference. Do not
        allow the order in which the responses were presented to influence your
        assessment. Do not allow the length of the responses to influence your
        assessment. The available assessments are:
        `Response A` - Response A is a better response.
        `Response B` - Response B is a better response.
        `Tie` - The two responses are roughly equal in quality.

        Take a deep breath and work on this problem step-by-step.
        '''

    def _prompt_with_source_and_reference(gen_output_a: str, gen_output_b: str,
                                          user_query: str, ref_output: str,
                                          source: str) -> str:
        return f'''
        You are comparing the quality of two responses to a user's query. Source
        text that is supposedly relevant to the user's query is also provided
        to you as a reference (the source text may contain some duplication).
        The ideal response to the user's query is also provided to you as a
        reference. Here is the data:
        [BEGIN DATA]
        ************
        [User Query]: {user_query}
        ************
        [Source]: {source}
        ************
        [Ideal Response]: {ref_output}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        Determine which of the responses is a better response to the user's
        query. Consider factors such as helpfulness, correctness, and relevance
        in your assessment, using the provided Source and the Ideal Response as
        references. Do not allow the order in which the responses were presented
        to influence your assessment. Do not allow the length of the responses
        to influence your assessment. The available assessments are:
        `Response A` - Response A is a better response.
        `Response B` - Response B is a better response.
        `Tie` - The two responses are roughly equal in quality.

        Take a deep breath and work on this problem step-by-step.
        '''

    # Combine sources_1 and sources_2 into a single list if both are provided.
    if sources_1 is not None and sources_2 is not None:
        sources = [
            source_a + '\n' + source_b
            for source_a, source_b in zip(sources_1, sources_2)
        ]
    else:
        sources = sources_1 if sources_1 is not None else sources_2

    if sources is not None and reference_outputs is not None:
        prompt_fn = _prompt_with_source_and_reference
        data_iter = zip(generated_outputs_1, generated_outputs_2, prompts,
                        reference_outputs, sources)
    elif sources is not None:
        prompt_fn = _prompt_with_source
        data_iter = zip(generated_outputs_1, generated_outputs_2, prompts,
                        sources)
    elif reference_outputs is not None:
        prompt_fn = _prompt_with_reference
        data_iter = zip(generated_outputs_1, generated_outputs_2, prompts,
                        reference_outputs)
    else:
        prompt_fn = _prompt
        data_iter = zip(generated_outputs_1, generated_outputs_2, prompts)

    return prompt_fn, data_iter
