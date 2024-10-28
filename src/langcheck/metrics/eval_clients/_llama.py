# Built with Meta Llama 3

from __future__ import annotations

from collections.abc import Iterable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..prompts._utils import get_template
from ._base import EvalClient


class LlamaEvalClient(EvalClient):
    """EvalClient defined for the Llama-based models.
    It currently only supports English and Japanese.
    The default model is set to "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1".
    The following models are also available:
    - `tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1`
    - `elyza/Llama-3-ELYZA-JP-8B`
    - `rinna/llama-3-youko-8b-instruct`
    - `rinna/llama-3-youko-70b-instruct`
    - `meta-llama/Meta-Llama-3.1-8B-Instruct`
    - `meta-llama/Meta-Llama-3.1-70B-Instruct`
    To use the 70B models, set tensor_parallel_size to 8 or more.
    To use the Llama 3.1 models, you need to agree to the terms of service and login with your huggingface account.
    """

    def __init__(
        self,
        model_name: str = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1",
        torch_dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize the Llama evaluation client.

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
            stop="<|eot_id|>",
            skip_special_tokens=True,
        )
        self._system_prompts = {
            "en": "You are a helpful and competent assistant.",
            "ja": "あなたは誠実で優秀な日本人のアシスタントです。以下は、タスクを説明する指示です。要求を適切に満たす応答を日本語で書きなさい。",
        }

    def get_text_responses(
        self,
        prompts: Iterable[str],
        language: str,
    ) -> list[str | None]:
        """The function that generates responses to the given prompt texts.

        Args:
            prompts: The prompts you want to get the responses for.
            language: The language of the prompts. (e.g. "en")
        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        """
        if language not in ["en", "ja"]:
            raise ValueError(f"Unsupported language: {language}")

        messages = [
            [
                {
                    "role": "system",
                    "content": self._system_prompts[language],
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            for prompt in prompts
        ]
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
        texts that describe the evaluation results) into scores.

        Args:
            metric_name : The name of the metric to be used. (e.g. "toxicity")
            language: The language of the prompts. (e.g. "en")
            unstructured_assessment_result: The unstructured assessment results
                for the given assessment prompts.
            score_map: The mapping from the short assessment results
                (e.g. "Good") to the scores.
        Returns:
            A list of scores for the given prompts. The scores can be None if
            the evaluation fails.
        """
        if language not in ["en", "ja"]:
            raise ValueError(f"Unsupported language: {language}")

        options = list(score_map.keys())
        get_score_template = get_template(f"{language}/get_score/plain_text.j2")
        get_score_prompts = [
            get_score_template.render(
                {
                    "metric": metric_name,
                    "unstructured_assessment": unstructured_assessment,
                    "options": options,
                }
            )
            if unstructured_assessment
            else None
            for unstructured_assessment in unstructured_assessment_result
        ]

        # If there are any Nones in get_score_prompts,
        # they are excluded from messages to prevent passing those to the model.
        messages = [
            [
                {
                    "role": "system",
                    "content": self._system_prompts[language],
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            for prompt in get_score_prompts
            if prompt
        ]
        if len(messages):
            prompts = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(prompts, str):
                prompts = [prompts]
            else:
                prompts = [str(p) for p in prompts]
            responses = self._model.generate(prompts, self._sampling_params)
            raw_response_texts = [
                response.outputs[0].text if response else None
                for response in responses
            ]
        else:
            raw_response_texts = []

        responses_for_scoring = []
        idx_raw_response_texts = 0
        for idx in range(len(get_score_prompts)):
            if get_score_prompts[idx] is None:
                responses_for_scoring.append(None)
            else:
                responses_for_scoring.append(
                    raw_response_texts[idx_raw_response_texts]
                )
                idx_raw_response_texts += 1

        def _turn_to_score(response: str | None) -> float | None:
            if response is None:
                return None
            option_found = [option for option in options if option in response]
            # if response contains multiple options as substrings, return None
            if len(option_found) != 1:
                return None
            return score_map[option_found[0]]

        return [_turn_to_score(response) for response in responses_for_scoring]

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
        unstructured_assessment_result = self.get_text_responses(
            prompts, language
        )
        scores = self.get_float_score(
            metric_name,
            language,
            unstructured_assessment_result,
            score_map,
        )
        return scores, unstructured_assessment_result

    def similarity_scorer(self):
        raise NotImplementedError(
            "Embedding-based metrics are not supported in LlamaEvalClient."
            "Use other EvalClients to get these metrics."
        )
