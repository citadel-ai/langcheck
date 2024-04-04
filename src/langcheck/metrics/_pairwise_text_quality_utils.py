from __future__ import annotations

from typing import Callable


class PairwiseComparisonPromptGenerator:

    def __init__(
        self, base_prompt_template: Callable[[str, str, str], str],
        prompt_template_with_reference: Callable[[str, str, str, str], str],
        prompt_template_with_source: Callable[[str, str, str, str], str],
        prompt_template_with_source_and_reference: Callable[
            [str, str, str, str, str], str]
    ) -> None:
        '''Initialize the PairwiseComparisonPromptGenerator.

        Args:
            base_prompt_template: The base prompt template to use when no
                sources or reference outputs are provided. The function's inputs
                should be the generated output from Model A, the generated
                output from Model B, and the input prompt.
            prompt_template_with_reference: The prompt template to use when
                reference outputs are provided. The function's inputs should be
                the generated output from Model A, the generated output from
                Model B, the input prompt, and the reference output.
            prompt_template_with_source: The prompt template to use when
                sources are provided. The function's inputs should be the
                generated output from Model A, the generated output from Model
                B, the input prompt, and the source.
            prompt_template_with_source_and_reference: The prompt template to
                use when sources and reference outputs are provided. The
                function's inputs should be the generated output from Model A,
                the generated output from Model B, the input prompt, the
                reference output, and the source.
        '''
        self.base_prompt_template = base_prompt_template
        self.prompt_template_with_reference = prompt_template_with_reference
        self.prompt_template_with_source = prompt_template_with_source
        self.prompt_template_with_source_and_reference = prompt_template_with_source_and_reference  # NOQA: E501

    def generate_prompts(self, generated_outputs_1: list[str],
                         generated_outputs_2: list[str], prompts: list[str],
                         sources_1: list[str] | None,
                         sources_2: list[str] | None,
                         reference_outputs: list[str] | None) -> list[str]:
        '''Generate prompts for the pairwise comparison metric.

        Args:
            generated_outputs_1: Model 1's generated output(s) to evaluate
            generated_outputs_2: Model 2's generated output(s) to evaluate
            prompts: The prompts used to generate the output(s).
            sources_1: (Optional) the source(s) of Model 1's generated output(s)
            sources_2: (Optional) the source(s) of Model 2's generated output(s)
            reference_outputs: (Optional) the reference output(s)

        Returns:
            A list of prompts
        '''
        # Combine sources_1 and sources_2 into a single list if both are
        # provided.
        if sources_1 is not None and sources_2 is not None:
            sources = [
                source_1 + '\n' + source_2
                for source_1, source_2 in zip(sources_1, sources_2)
            ]
        else:
            sources = sources_1 if sources_1 is not None else sources_2

        if sources is not None and reference_outputs is not None:
            prompt_fn = self.prompt_template_with_source_and_reference
            data_iter = zip(generated_outputs_1, generated_outputs_2, prompts,
                            reference_outputs, sources)
        elif sources is not None:
            prompt_fn = self.prompt_template_with_source
            data_iter = zip(generated_outputs_1, generated_outputs_2, prompts,
                            sources)
        elif reference_outputs is not None:
            prompt_fn = self.prompt_template_with_reference
            data_iter = zip(generated_outputs_1, generated_outputs_2, prompts,
                            reference_outputs)
        else:
            prompt_fn = self.base_prompt_template
            data_iter = zip(generated_outputs_1, generated_outputs_2, prompts)

        return [prompt_fn(*data_instance) for data_instance in data_iter]


def enforce_pairwise_comparison_consistency(
    original_scores: list[float | None],
    original_explanations: list[str | None], swapped_scores: list[float | None],
    swapped_explanations: list[str | None]
) -> tuple[list[float | None], list[str | None]]:
    '''Enforce consistency in pairwise comparison scores.

    Args:
        original_scores: The scores for the original order of the models
        original_explanations: The explanations for the original order of the
            models
        swapped_scores: The scores for the swapped order of the models
        swapped_explanations: The explanations for the swapped order of the
            models
    '''
    # Iterate through the scores and explanations to check for consistency.
    # If a score is not consistent, set it to None, and merge the two
    # explanations to show the inconsistency.
    scores = original_scores.copy()
    explanations = original_explanations.copy()
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
    return scores, explanations
