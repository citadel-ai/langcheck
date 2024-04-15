from __future__ import annotations

from typing import Iterable


class EvalClient:

    def _unstructured_assessment(
            self,
            prompts: Iterable[str],
            *,
            tqdm_description: str | None = None) -> list[str | None]:
        '''
        TODO
        '''
        raise NotImplementedError

    def _get_float_score(
            self,
            metric_name: str,
            language: str,
            unstructured_assessment_result: list[str | None],
            score_map: dict[str, float],
            *,
            tqdm_description: str | None = None) -> list[float | None]:
        '''
        TODO
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
        '''
        TODO
        '''
        if isinstance(prompts, str):
            prompts = [prompts]
        unstructured_assessment_result = self._unstructured_assessment(
            prompts, tqdm_description=intermediate_tqdm_description)
        scores = self._get_float_score(metric_name,
                                       language,
                                       unstructured_assessment_result,
                                       score_map,
                                       tqdm_description=score_tqdm_description)
        return scores, unstructured_assessment_result
