from __future__ import annotations

from collections.abc import Iterable

from jinja2 import Template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..prompts._utils import get_template
from ._base import EvalClient


class PrometheusEvalClient(EvalClient):
    """EvalClient defined for the Prometheus 2 model.
    This eval client currently supports only English.
    Presented in `"Prometheus 2: An Open Source Language Model Specialized
    in Evaluating Other Language Models" <https://arxiv.org/abs/2405.01535>`.
    We adapted the prompts in <https://github.com/prometheus-eval/prometheus-
    eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py>.
    """

    def __init__(
        self,
        model_name: str = "prometheus-eval/prometheus-7b-v2.0",
        torch_dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        device: str = "cuda",
    ):
        """
        Initilize the Prometheus evaluation client.

        Args:
            model_name: The name of the model to use.
            torch_dtype: The torch dtype to use. torch.bfloat16 is recommended.
            tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
            device: The device to load the model on.
        """
        self._model = LLM(
            model=model_name,
            max_model_len=8192,
            dtype=torch_dtype,
            tensor_parallel_size=tensor_parallel_size,
            device=device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=1000,
            skip_special_tokens=True,
        )

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
            try:
                return get_template(
                    f"{language}/metrics/prometheus/{metric_name}.j2"
                )
            except FileNotFoundError:
                raise ValueError(
                    f"The {metric_name} metric (language = {language}) is not yet supported by the Prometheus eval client."
                )
        else:
            try:
                return get_template(
                    f"{language}/metrics/prometheus/{metric_name}_{eval_prompt_version}.j2"
                )
            except FileNotFoundError:
                raise ValueError(
                    f"The {metric_name} metric (language = {language}, version = {eval_prompt_version}) is not yet supported by the Prometheus eval client."
                )

    def get_text_responses(self, prompts: Iterable[str]) -> list[str | None]:
        """The function that generates responses to the given prompt texts.

        Args:
            prompts: The prompts you want to get the responses for.
        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        """
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        processed_prompts = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(processed_prompts, str):
            processed_prompts = [processed_prompts]
        else:
            processed_prompts = [str(p) for p in processed_prompts]
        responses = self._model.generate(
            processed_prompts, self._sampling_params
        )
        response_texts = [
            response.outputs[0].text
            if response and response.outputs[0].text != ""
            else None
            for response in responses
        ]

        return response_texts

    def get_float_score(
        self,
        metric_name: str,
        language: str,
        unstructured_assessment_result: list[str | None],
        score_map: dict[str, float],
    ) -> list[float | None]:
        """The function that transforms the unstructured assessments (i.e. long
        texts that describe the evaluation results) into scores. We simple find
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
        for unstructured_assessment in unstructured_assessment_result:
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

    def get_score(
        self,
        metric_name: str,
        language: str,
        prompts: str | Iterable[str],
        score_map: dict[str, float],
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

        Returns:
            A tuple of two lists. The first list contains the scores for each
            prompt and the second list contains the unstructured assessment
            results for each prompt. Both can be None if the evaluation fails.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        unstructured_assessment_result = self.get_text_responses(prompts)
        scores = self.get_float_score(
            metric_name,
            language,
            unstructured_assessment_result,
            score_map,
        )
        return scores, unstructured_assessment_result

    def similarity_scorer(self):
        raise NotImplementedError(
            "Embedding-based metrics are not supported in PrometheusEvalClient."
            "Use other EvalClients to get these metrics."
        )
