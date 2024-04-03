from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI

from langcheck.metrics._pairwise_text_quality_utils import (
    PairwiseComparisonPromptGenerator, enforce_pairwise_comparison_consistency)
from langcheck.metrics._validation import \
    validate_parameters_pairwise_comparison
from langcheck.metrics.en._openai import OpenAIBasedEvaluator
from langcheck.metrics.metric_value import MetricValue


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
        openai_args: Optional[Dict[str, str]] = None,
        *,
        use_async: bool = False) -> MetricValue[Optional[float]]:
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
        use_async: Whether to use the asynchronous API of OpenAI, default False

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

    def _prompt(gen_output_a: str, gen_output_b: str, user_query: str) -> str:
        return f'''
        ユーザーの質問に対する2つの回答の品質を比較してください。データは以下の通りです:
        [BEGIN DATA]
        ************
        [ユーザーの質問]: {user_query}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        ユーザーの質問に対してどちらの回答がより良いかを決定してください。有用性、正確さ、
        関連性などの要素を考慮して評価してください。回答が提示された順序に評価が影響されない
        ようにしてください。回答の長さが評価に影響を与えないようにしてください。利用可能な評価
        は以下の通りです:
        `Response A` - Response Aがより良い回答です。
        `Response B` - Response Bがより良い回答です。
        `Tie` - 2つの回答は品質がほぼ同等です。

        深呼吸をして、この問題をステップバイステップで取り組んでください。
        '''

    def _prompt_with_reference(gen_output_a: str, gen_output_b: str,
                               user_query: str, ref_output: str) -> str:
        return f'''
        ユーザーの質問に対する2つの回答の品質を比較してください。ユーザーの質問に対する理想的な
        回答も参考として提供されます。データは以下の通りです:
        [BEGIN DATA]
        ************
        [ユーザーの質問]: {user_query}
        ************
        [理想的な回答]: {ref_output}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        ユーザーの質問に対してどちらの回答がより良いかを決定してください。提供された理想的な
        回答を参考にして、有用性、正確さ、関連性などの要素を考慮して評価してください。回答が
        提示された順序に評価が影響されないようにしてください。回答の長さが評価に影響を与えない
        ようにしてください。利用可能な評価は以下の通りです:
        `Response A` - Response Aがより良い回答です。
        `Response B` - Response Bがより良い回答です。
        `Tie` - 2つの回答は品質がほぼ同等です。

        深呼吸をして、この問題をステップバイステップで取り組んでください。
        '''

    def _prompt_with_source(gen_output_a: str, gen_output_b: str,
                            user_query: str, source: str) -> str:
        return f'''
        ユーザーの質問に対する2つの回答の品質を比較してください。ユーザーの質問に関連があると
        思われるソーステキストも参考として提供されます（ソーステキストには重複が含まれている
        可能性があります）。データは以下の通りです:
        [BEGIN DATA]
        ************
        [ユーザーの質問]: {user_query}
        ************
        [ソース]: {source}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        ユーザーの質問に対してどちらの回答がより良いかを決定してください。提供されたソースを
        参考にして、有用性、正確さ、関連性などの要素を考慮して評価してください。回答が提示された
        順序に評価が影響されないようにしてください。回答の長さが評価に影響を与えないように
        してください。利用可能な評価は以下の通りです:
        `Response A` - Response Aがより良い回答です。
        `Response B` - Response Bがより良い回答です。
        `Tie` - 2つの回答は品質がほぼ同等です。

        深呼吸をして、この問題をステップバイステップで取り組んでください。
        '''

    def _prompt_with_source_and_reference(gen_output_a: str, gen_output_b: str,
                                          user_query: str, ref_output: str,
                                          source: str) -> str:
        return f'''
        ユーザーの質問に対する2つの回答の品質を比較してください。ユーザーの質問に関連があると
        思われるソーステキストも参考として提供されます（ソーステキストには重複が含まれている
        可能性があります）。さらに、ユーザーの質問に対する理想的な回答も参考として提供されます。
        データは以下の通りです:
        [BEGIN DATA]
        ************
        [ユーザーの質問]: {user_query}
        ************
        [ソース]: {source}
        ************
        [理想的な回答]: {ref_output}
        ************
        [Response A]: {gen_output_a}
        ************
        [Response B]: {gen_output_b}
        ************
        [END DATA]

        ユーザーの質問に対してどちらの回答がより良いかを決定してください。提供されたソースと
        理想的な回答を参考に、有用性、正確さ、関連性などの要素を考慮して評価してください。
        回答が提示された順序に評価が影響されないようにしてください。回答の長さが評価に影響を
        与えないようにしてください。利用可能な評価は以下の通りです:
        `Response A` - Response Aがより良い回答です。
        `Response B` - Response Bがより良い回答です。
        `Tie` - 2つの回答は品質がほぼ同等です。

        深呼吸をして、この問題をステップバイステップで取り組んでください。
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
        openai_args=openai_args,
        use_async=use_async)

    prompt_generator = PairwiseComparisonPromptGenerator(
        _prompt, _prompt_with_reference, _prompt_with_source,
        _prompt_with_source_and_reference)
    all_prompts = prompt_generator.generate_prompts(generated_outputs_a,
                                                    generated_outputs_b,
                                                    prompts, sources_a,
                                                    sources_b,
                                                    reference_outputs)
    scores, explanations = oai_evaluator.get_score(all_prompts,
                                                   _function_call_prompt)

    if enforce_consistency:
        # Swap the generated outputs and enforce consistency
        all_swapped_prompts = prompt_generator.generate_prompts(
            generated_outputs_b, generated_outputs_a, prompts, sources_b,
            sources_a, reference_outputs)
        intermediate_tqdm = '[Swapped model outputs order] Intermediate assessments (1/2)'  # NOQA: E501
        score_tqdm = '[Swapped model outputs order] Calculating scores (2/2)'
        swapped_scores, swapped_explanations = oai_evaluator.get_score(
            all_swapped_prompts,
            _function_call_prompt,
            intermediate_tqdm_description=intermediate_tqdm,
            score_tqdm_description=score_tqdm)
        scores, explanations = enforce_pairwise_comparison_consistency(
            scores, explanations, swapped_scores, swapped_explanations)

    return MetricValue(metric_name='pairwise_comparison',
                       prompts=prompts,
                       generated_outputs=(generated_outputs_a,
                                          generated_outputs_b),
                       reference_outputs=reference_outputs,
                       sources=(sources_a, sources_b),
                       explanations=explanations,
                       metric_values=scores,
                       language='ja')
