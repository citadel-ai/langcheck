from __future__ import annotations

import json
import math
import random
import re
from typing import List, Optional

from langcheck.metrics._pairwise_text_quality_utils import (
    enforce_pairwise_comparison_consistency,
    generate_pairwise_comparison_prompt_params,
)
from langcheck.metrics._validation import (
    validate_parameters_pairwise_comparison,
)
from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.metric_value import MetricValue

from ..prompts._utils import get_template


def simulated_annotators(
    prompt_params: List[dict[str, str | None]],
    eval_model: EvalClient,
    k: int = 5,
    n: int = 5
) -> List[float | None]:
    """Compute a confidence score for the pairwise comparison metric based on
    the method Simulated Annotators proposed in the paper "Trust or Escalate:
    LLM Judges with Provable Guarantees for Human Agreement"
    (https://arxiv.org/abs/2407.18370)

    Args:
        prompt_params: The parameters used to populate the prompt template.
        eval_model: The EvalClient instance used for the evaluation.
        k: the number of examples of preference annotations
        n: the numbre of simulated annotators
    Returns:
        A confidence score for the pairwise comparison metric
    """
    # Load preprocessed chatarena data
    with open("processed_chatarena_examples.jsonl") as f:
        chatarena_data = [json.loads(line) for line in f]
    assert len(
        chatarena_data) >= k, "Not enough examples in the chatarena data"

    # Load the prompt template
    prompt_template = get_template(
        "en/confidence_estimating/simulated_annotators.j2")
    populated_prompts = [
        prompt_template.render(prompt_param)
        for prompt_param in prompt_params
    ]

    confidence_scores = []
    for prompt in populated_prompts:
        # Generate few-shot examples
        few_shot_examples = random.sample(chatarena_data, k)

        # Construct the full prompt using k few-shot examples
        few_shot_prompt = "\n".join(
            f"[Question]\n{example['prompt']}\n\n"
            "[Assistant A's response]\n{example['model_a']}\n\n"
            "[Assistant B's response]\n{example['model_b']}\n\n"
            "[Verdict]\n{example['winner']}\n"
            for example in few_shot_examples
        )
        prompt = re.split(r"\[Few-shot examples\]", prompt)[0] + \
            few_shot_prompt + re.split(r"\[Few-shot examples\]", prompt)[1]

        # Simulate n annotators
        scores = []
        for _ in range(n):
            response = eval_model.get_text_responses_with_log_likelihood(
                [prompt])[0]
            if response and response[1][0][0] in ["A", "B"]:
                scores.append(math.exp(response[1][0][1]))

        if len(scores) != 0:
            confidence_scores.append(sum(scores) / len(scores))
        else:
            confidence_scores.append(None)

    return confidence_scores


def pairwise_comparison(
    generated_outputs_a: List[str] | str,
    generated_outputs_b: List[str] | str,
    prompts: List[str] | str,
    sources_a: Optional[List[str] | str] = None,
    sources_b: Optional[List[str] | str] = None,
    reference_outputs: Optional[List[str] | str] = None,
    enforce_consistency: bool = True,
    calculated_confidence: bool = False,
    eval_model: EvalClient | None = None,
) -> MetricValue[Optional[float]]:
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
        eval_model: The EvalClient instance used for the evaluation. This is
            marked as Optional so that it can follow the above arguments that
            have default values (for consistency with the other metrics), but
            this is in fact a required argument.

    Returns:
        An MetricValue object
    """
    (
        generated_outputs_a,
        generated_outputs_b,
        prompts,
        sources_a,
        sources_b,
        reference_outputs,
    ) = validate_parameters_pairwise_comparison(
        generated_outputs_a,
        generated_outputs_b,
        prompts,
        sources_a,
        sources_b,
        reference_outputs,
    )

    assert (
        eval_model is not None
    ), "You must pass an EvalClient instance to the pairwise_comparison function."

    pairwise_comparison_assessment_to_score = {
        "Response B": 1.0,
        "Tie": 0.5,
        "Response A": 0.0,
    }

    pairwise_comparison_template = eval_model.load_prompt_template(
        language="en", metric_name="pairwise_comparison"
    )
    prompt_params = generate_pairwise_comparison_prompt_params(
        generated_outputs_a,
        generated_outputs_b,
        prompts,
        sources_a,
        sources_b,
        reference_outputs,
    )

    populated_prompts = [
        pairwise_comparison_template.render(prompt_param)
        for prompt_param in prompt_params
    ]

    scores, explanations = eval_model.get_score(
        metric_name="comparison of two responses",
        language="en",
        prompts=populated_prompts,
        score_map=pairwise_comparison_assessment_to_score,
    )

    if enforce_consistency:
        # Swap the generated outputs and enforce consistency
        swapped_prompt_params = generate_pairwise_comparison_prompt_params(
            generated_outputs_b,
            generated_outputs_a,
            prompts,
            sources_b,
            sources_a,
            reference_outputs,
        )

        populated_swapped_prompts = [
            pairwise_comparison_template.render(prompt_param)
            for prompt_param in swapped_prompt_params
        ]

        intermediate_tqdm = (
            "[Swapped model outputs order] Intermediate assessments (1/2)"
        )
        score_tqdm = "[Swapped model outputs order] Calculating scores (2/2)"
        swapped_scores, swapped_explanations = eval_model.get_score(
            metric_name="comparison of two responses",
            language="en",
            prompts=populated_swapped_prompts,
            score_map=pairwise_comparison_assessment_to_score,
            intermediate_tqdm_description=intermediate_tqdm,
            score_tqdm_description=score_tqdm,
        )

        scores, explanations = enforce_pairwise_comparison_consistency(
            scores,
            explanations,
            swapped_scores,
            swapped_explanations,
            pairwise_comparison_assessment_to_score,
        )

    if calculated_confidence:
        print("Warning: The source texts and reference outputs are not used to"
              "calculate the confidence score.")
        confidence_scores = simulated_annotators(
            prompt_params, eval_model
        )
        # Append the confidence scores to the explanations
        explanations = [
            f"{explanation}\n\nConfidence score: {confidence_score}"
            if explanation and confidence_score else explanation
            for explanation, confidence_score in zip(explanations, confidence_scores)
        ]

    return MetricValue(
        metric_name="pairwise_comparison",
        prompts=prompts,
        generated_outputs=(generated_outputs_a, generated_outputs_b),
        reference_outputs=reference_outputs,
        sources=(sources_a, sources_b),
        explanations=explanations,
        metric_values=scores,
        language="en",
    )
