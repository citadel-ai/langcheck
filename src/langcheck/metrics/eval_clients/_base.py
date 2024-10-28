from __future__ import annotations

from collections.abc import Iterable
from typing import Union

from jinja2 import Template

from langcheck.metrics.metric_inputs import MetricInputs
from langcheck.metrics.metric_value import MetricValue

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer

TokenLogProb = dict[str, Union[str, float]]
TopKLogProbs = list[list[TokenLogProb]]
TextResponseWithLogProbs = dict[str, Union[str, list[TopKLogProbs]]]


class EvalClient:
    """An abstract class that defines the interface for the evaluation clients.
    Most metrics that use external APIs such as OpenAI API call the methods
    defined in this class to compute the metric values.
    """

    def load_prompt_template(
        self,
        language: str,
        metric_name: str,
        eval_prompt_version: str | None = None,
    ) -> Template:
        """
        Gets a Jinja template from the specified language, eval client, metric
        name, and (optionally) eval prompt version.

        Args:
            language (str): The language of the template.
            metric_name (str): The name of the metric.
            eval_prompt_version (str | None): The version of the eval prompt.
                If None, the default version is used.

        Returns:
            Template: The Jinja template.
        """
        if eval_prompt_version is None:
            return get_template(f"{language}/metrics/{metric_name}.j2")
        return get_template(
            f"{language}/metrics/{metric_name}_{eval_prompt_version}.j2"
        )

    def get_text_responses(
        self,
        prompts: Iterable[str],
        *,
        tqdm_description: str | None = None,
    ) -> list[str | None]:
        """The function that gets responses to the given prompt texts. Each
        concrete subclass needs to define the concrete implementation of this
        function to enable text scoring.

        Args:
            prompts: The prompts you want to get the responses for.

        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        """
        raise NotImplementedError

    def get_text_responses_with_log_likelihood(
        self,
        prompts: Iterable[str],
        top_logprobs: int | None = None,
        *,
        tqdm_description: str | None = None,
    ) -> list[TextResponseWithLogProbs | None]:
        """The function that gets responses with log likelihood to the given
        prompt texts. Each concrete subclass needs to define the concrete
        implementation of this function to enable text scoring.

        Args:
            prompts: The prompts you want to get the responses for.
            top_logprobs: The number of logprobs to return for each token.

        Returns:
            A list of responses to the prompts. Each response is a tuple of the
            output text and the list of tuples of the output tokens and the log
            probabilities. The responses can be None if the evaluation fails.
        """
        raise NotImplementedError

    def get_float_score(
        self,
        metric_name: str,
        language: str,
        unstructured_assessment_result: list[str | None],
        score_map: dict[str, float],
        *,
        tqdm_description: str | None = None,
    ) -> list[float | None]:
        """The function that transforms the unstructured assessments (i.e. long
        texts that describe the evaluation results) into scores. A typical
        workflow can be:

        1. Extract a short assessment result strings from the unstructured
        assessment results.

        2. Map the short assessment result strings to the scores using the
        score_map.

        Each concrete subclass needs to define the concrete implementation of
        this function to enable text scoring.

        Args:
            metric_name: The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            unstructured_assessment_result: The unstructured assessment results
                for the given assessment prompts.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.
            tqdm_description: The description to be shown in the tqdm bar.

        Returns:
            A list of scores for the given prompts. The scores can be None if
            the evaluation fails.
        """
        raise NotImplementedError

    def get_score(
        self,
        metric_name: str,
        language: str,
        prompts: str | Iterable[str],
        score_map: dict[str, float],
        *,
        intermediate_tqdm_description: str | None = None,
        score_tqdm_description: str | None = None,
    ) -> tuple[list[float | None], list[str | None]]:
        """Give scores to texts embedded in the given prompts. The function
        itself calls get_text_responses and get_float_score to get the scores.
        The function returns the scores and the unstructured explanation
        strings.

        Args:
            metric_name: The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            prompts: The prompts that contain the original text to be scored,
                the evaluation criteria... etc. Typically it is based on the
                Jinja prompt templates and instantiated withing each metric
                function.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.
            intermediate_tqdm_description: The description to be
                shown in the tqdm bar for the unstructured assessment.
            score_tqdm_description: The description to be shown in
                the tqdm bar for the score calculation.

        Returns:
            A tuple of two lists. The first list contains the scores for each
            prompt and the second list contains the unstructured assessment
            results for each prompt. Both can be None if the evaluation fails.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        unstructured_assessment_result = self.get_text_responses(
            prompts, tqdm_description=intermediate_tqdm_description
        )
        scores = self.get_float_score(
            metric_name,
            language,
            unstructured_assessment_result,
            score_map,
            tqdm_description=score_tqdm_description,
        )
        return scores, unstructured_assessment_result

    def similarity_scorer(self) -> BaseSimilarityScorer:
        """Get the BaseSimilarityScorer object that corresponds to the
        EvalClient so that the similarity-related metrics can be computed.
        TODO: Intergrate scorer/ with eval_clients/
        """
        raise NotImplementedError

    def compute_metric_values_from_template(
        self,
        metric_inputs: MetricInputs,
        template: Template,
        metric_name: str,
        language: str,
        score_map: dict[str, float],
    ) -> MetricValue[float | None]:
        """Compute the metric values from the given Jinja template with the
        metric inputs. This function assumes that the template parameters are
        already validated and the template is ready to be rendered.

        Args:
            metric_inputs: The metric inputs that contain the prompts,
                generated outputs, reference outputs... etc.
            template: The Jinja template that is ready to be rendered.
            enforce_pairwise_consistency: Whether to enforce pairwise
                consistency when computing the metric values.
            metric_name: The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.

        Returns:
            MetricValue: The metric values computed from the template.
        """
        prompt_template_inputs = metric_inputs.get_inputs_for_prompt_template()
        populated_prompts = [
            template.render(prompt_template_input)
            for prompt_template_input in prompt_template_inputs
        ]

        scores, explanations = self.get_score(
            metric_name=metric_name,
            language=language,
            prompts=populated_prompts,
            score_map=score_map,
        )

        return MetricValue(
            metric_name=metric_name,
            metric_inputs=metric_inputs,
            explanations=explanations,
            metric_values=scores,
            language=language,
        )

    def repeat_requests_from_template(
        self,
        prompt_template_inputs: list[dict[str, str]],
        template: Template,
        num_perturbations: int = 1,
    ) -> list[str | None]:
        """Repeats the request using the given Jinja template for
        `num_perturbations` times. Note that every EvalClient subclass is
        expected to implement `get_text_responses` method to get different
        responses for the same input.

        Args:
            instances: A single string or a list of strings to be augmented.
            template: The Jinja template ready to be rendered.
            num_perturbations: The number of perturbed instances to generate
                for each string in instances.

        Returns:
            A list of responses for each input. If `num_pertuations` is > 1, the
            multiple responses for the same input are included consecutively.
        """

        populated_prompts = [
            template.render(prompt_template_input)
            for prompt_template_input in prompt_template_inputs
            for _ in range(num_perturbations)
        ]

        responses = self.get_text_responses(populated_prompts)

        return responses
