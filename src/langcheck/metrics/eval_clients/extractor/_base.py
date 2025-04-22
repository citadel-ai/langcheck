from __future__ import annotations


class Extractor:
    """An abstract class that defines the interface for the extractors."""

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
