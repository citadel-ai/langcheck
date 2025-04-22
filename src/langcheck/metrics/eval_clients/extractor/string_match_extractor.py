from __future__ import annotations

from langcheck.metrics.eval_clients.extractor import Extractor
from langcheck.utils.progress_bar import tqdm_wrapper


class StringMatchExtractor(Extractor):
    """Score extractor that uses string matching to find assessment results in
    the text.
    """

    def get_float_score(
        self,
        metric_name: str,
        language: str,
        unstructured_assessment_result: list[str | None],
        score_map: dict[str, float],
        *,
        tqdm_description: str | None = None,
    ) -> list[float | None]:
        """The function that gets the scores from the unstructured assessments
        (i.e. long texts that describe the evaluation results). We simply find
        the assessment result which appeared latest in the unstructured text.
        Args:
            metric_name: The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            unstructured_assessment_result: The unstructured assessment results
                for the given assessment prompts.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.

        Returns:
            A list of scores for the given prompts. The scores can be None if
            the evaluation fails.
        """
        if language != "en":
            raise ValueError(f"Unsupported language: {language}")

        options = list(score_map.keys())
        assessments = []
        for unstructured_assessment in tqdm_wrapper(
            unstructured_assessment_result,
            desc=tqdm_description,
        ):
            if unstructured_assessment is None:
                assessments.append(None)
                continue

            # Find the option that appears latest in the assessment
            assessment = max(options, key=unstructured_assessment.rfind)
            if unstructured_assessment.find(assessment) == -1:
                print("No options found in the assessment.")
                assessments.append(None)
            else:
                assessments.append(assessment)

        return [
            score_map[assessment] if assessment else None
            for assessment in assessments
        ]
