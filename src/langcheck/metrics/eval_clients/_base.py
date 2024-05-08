from __future__ import annotations

from typing import Iterable

from ..scorer._base import BaseSimilarityScorer


class EvalClient:
    '''An abstract class that defines the interface for the evaluation clients.
    Most metrics that use external APIs such as OpenAI API call the methods
    defined in this class to compute the metric values.
    '''

    def get_text_responses(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        '''The function that gets resonses to the given prompt texts. Each
        concrete subclass needs to define the concrete implementation of this
        function to enable text scoring.

        Args:
            prompts: The prompts you want to get the responses for.

        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        '''
        raise NotImplementedError

    def get_float_score(
            self,
            metric_name: str,
            language: str,
            unstructured_assessment_result: list[str | None],
            score_map: dict[str, float],
            *,
            tqdm_description: str | None = None) -> list[float | None]:
        '''The function that transforms the unstructured assessments (i.e. long
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
        '''
        raise NotImplementedError

    def get_score(
        self,
        metric_name: str,
        language: str,
        prompts: str | Iterable[str],
        score_map: dict[str, float],
        *,
        intermediate_tqdm_description: str | None = None,
        score_tqdm_description: str | None = None
    ) -> tuple[list[float | None], list[str | None]]:
        '''Give scores to texts embedded in the given prompts. The function
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
        '''
        if isinstance(prompts, str):
            prompts = [prompts]
        unstructured_assessment_result = self.get_text_responses(
            prompts, tqdm_description=intermediate_tqdm_description)
        scores = self.get_float_score(metric_name,
                                      language,
                                      unstructured_assessment_result,
                                      score_map,
                                      tqdm_description=score_tqdm_description)
        return scores, unstructured_assessment_result

    def similarity_scorer(self) -> BaseSimilarityScorer:
        '''Get the BaseSimilarityScorer object that corresponds to the
        EvalClient so that the similarity-related metrics can be computed.
        TODO: Intergrate scorer/ with eval_clients/
        '''
        raise NotImplementedError
