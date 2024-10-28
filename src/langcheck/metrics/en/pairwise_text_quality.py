from __future__ import annotations

import math
import random
from typing import cast

from langcheck.metrics._pairwise_text_quality_utils import (
    compute_pairwise_comparison_metric_values_with_consistency,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_inputs import get_metric_inputs
from langcheck.metrics.metric_value import MetricValue

from ..eval_clients._base import TextResponseWithLogProbs, TokenLogProb
from ..prompts._utils import get_template, load_few_shot_examples


def simulated_annotators(
    prompt_params: list[dict[str, str | None]],
    eval_model: EvalClient,
    preference_data_path: str = "en/confidence_estimating/preference_data_examples.jsonl",
    k: int = 5,
    n: int = 5,
    seed: int | None = None,
) -> list[float | None]:
    """Compute a confidence score for the pairwise comparison metric based on
    the method Simulated Annotators proposed in the paper "Trust or Escalate:
    LLM Judges with Provable Guarantees for Human Agreement"
    (https://arxiv.org/abs/2407.18370)

    Args:
        prompt_params: The parameters used to populate the prompt template.
        eval_model: The EvalClient instance used for the evaluation.
        preference_data_path: The relative path to preference data labeled by
            human annotators. Users should prepare a pool of preference
            annotations (e.g., 1000 examples) in advance to use this metric.
        k: The number of examples of preference annotations
        n: The numbre of simulated annotators
        seed: The random seed for selecting the few-shot examples
    Returns:
        A confidence score for the pairwise comparison metric
    """
    # Load preprocessed preference data
    preference_data = load_few_shot_examples(preference_data_path)
    assert (
        len(preference_data) >= k
    ), "Not enough examples in the preference data"

    if seed is not None:
        random.seed(seed)

    # Load the prompt template
    prompt_template = get_template(
        "en/confidence_estimating/simulated_annotators.j2"
    )

    confidence_scores = []
    for prompt_param in prompt_params:
        # Simulate n annotators
        prompts = []
        for _ in range(n):
            # Generate few-shot examples
            few_shot_examples = random.sample(preference_data, k)

            # Construct the full prompt using k few-shot examples
            prompt_param["few_shot_examples"] = "\n".join(
                f"[Question]\n{example['prompt']}\n\n"
                "[Assistant A's response]\n{example['model_a']}\n\n"
                "[Assistant B's response]\n{example['model_b']}\n\n"
                "[Verdict]\n{example['winner']}\n"
                for example in few_shot_examples
            )
            prompts.append(prompt_template.render(prompt_param))

        # Get the response and top five logprobs of the first token
        responses: list[TextResponseWithLogProbs | None] = (
            eval_model.get_text_responses_with_log_likelihood(
                prompts, top_logprobs=5
            )
        )
        scores_a, scores_b = [], []
        for i, response in enumerate(responses):
            if response:
                response = cast(TextResponseWithLogProbs, response)
                top_five_first_token_logprobs = cast(
                    list[TokenLogProb], response["response_logprobs"][0]
                )
                # Extract logprobs for tokens 'A' and 'B'
                logprobs_dict = {
                    logprob["token"]: math.exp(float(logprob["logprob"]))
                    for logprob in top_five_first_token_logprobs
                }
                if "A" in logprobs_dict and "B" in logprobs_dict:
                    scores_a.append(logprobs_dict["A"])
                    scores_b.append(logprobs_dict["B"])
                else:
                    print(
                        f"Token 'A' or 'B' was not found for the {i}th simulated annotator"
                    )

        if len(scores_a) != 0 and len(scores_a) == len(scores_b):
            if sum(scores_a) > sum(scores_b):
                confidence_scores.append((sum(scores_a) / len(scores_a)))
            else:
                confidence_scores.append((sum(scores_b) / len(scores_b)))
        else:
            confidence_scores.append(None)

    return confidence_scores


def pairwise_comparison(
    generated_outputs_a: list[str] | str,
    generated_outputs_b: list[str] | str,
    prompts: list[str] | str,
    sources_a: list[str] | str | None = None,
    sources_b: list[str] | str | None = None,
    reference_outputs: list[str] | str | None = None,
    enforce_consistency: bool = True,
    calculated_confidence: bool = False,
    preference_data_path: str = "en/confidence_estimating/preference_data_examples.jsonl",
    k: int = 5,
    n: int = 5,
    seed: int | None = None,
    eval_model: EvalClient | None = None,
) -> MetricValue[float | None]:
    """Calculates the pairwise comparison metric. This metric takes on float
    values of either 0.0 (Response A is better), 0.5 (Tie), or 1.0 (Response B
    is better). The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.

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
        calculated_confidence: When this is True, we will calculate a confidence
            score for the pairwise comparison metric. Default False.
        preference_data_path: The relative path to preference data labeld by
            human annotators. Users should prepare a pool of preference
            annotations (e.g., 1000 examples) in advance to use this metric.
        k: The number of examples of preference annotations
        n: The number of simulated annotators
        seed: The random seed for the simulated annotators
        eval_model: The EvalClient instance used for the evaluation. This is
            marked as Optional so that it can follow the above arguments that
            have default values (for consistency with the other metrics), but
            this is in fact a required argument.

    Returns:
        An MetricValue object
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=(generated_outputs_a, generated_outputs_b),
        prompts=prompts,
        sources=(sources_a, sources_b),
        reference_outputs=reference_outputs,
        required_params=[],
    )

    assert (
        eval_model is not None
    ), "You must pass an EvalClient instance to the pairwise_comparison function."

    pairwise_comparison_assessment_to_score = {
        "Response B": 1.0,
        "Tie": 0.5,
        "Response A": 0.0,
    }

    metric_name = "pairwise_comparison"
    language = "en"
    pairwise_comparison_template = eval_model.load_prompt_template(
        language=language, metric_name=metric_name
    )

    if enforce_consistency:
        metric_value = (
            compute_pairwise_comparison_metric_values_with_consistency(
                eval_client=eval_model,
                metric_inputs=metric_inputs,
                template=pairwise_comparison_template,
                metric_name=metric_name,
                language=language,
                score_map=pairwise_comparison_assessment_to_score,
            )
        )
    else:
        metric_value = eval_model.compute_metric_values_from_template(
            metric_inputs=metric_inputs,
            template=pairwise_comparison_template,
            metric_name=metric_name,
            language=language,
            score_map=pairwise_comparison_assessment_to_score,
        )

    if calculated_confidence:
        print(
            "Warning: The source texts and reference outputs are not used to"
            "calculate the confidence score."
        )
        prompt_template_inputs = metric_inputs.get_inputs_for_prompt_template()
        confidence_scores = simulated_annotators(
            prompt_template_inputs, eval_model, preference_data_path, k, n, seed
        )
        # Append the confidence scores to the explanations
        # TODO: Consider adding the confidence scores to the MetricValue object
        assert metric_value.explanations is not None
        explanations = [
            f"{explanation}\n\nConfidence score: {confidence_score}"
            if explanation and confidence_score
            else explanation
            for explanation, confidence_score in zip(
                metric_value.explanations, confidence_scores
            )
        ]
        metric_value.explanations = explanations

    return metric_value
