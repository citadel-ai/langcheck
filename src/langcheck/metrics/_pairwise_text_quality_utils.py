from typing import Callable, Optional


class PairwiseComparisonPromptGenerator:

    def __init__(self, base_prompt_template: Callable,
                 prompt_template_with_reference: Callable,
                 prompt_template_with_source: Callable,
                 prompt_template_with_source_and_reference: Callable) -> None:
        self.base_prompt_template = base_prompt_template
        self.prompt_template_with_reference = prompt_template_with_reference
        self.prompt_template_with_source = prompt_template_with_source
        self.prompt_template_with_source_and_reference = prompt_template_with_source_and_reference  # NOQA: E501

    def generate_prompts(self, generated_outputs_1: list[str],
                         generated_outputs_2: list[str], prompts: list[str],
                         sources_1: Optional[list[str]],
                         sources_2: Optional[list[str]],
                         reference_outputs: Optional[list[str]]) -> list[str]:
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
