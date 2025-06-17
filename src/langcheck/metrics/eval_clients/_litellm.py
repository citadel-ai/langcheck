from __future__ import annotations

import asyncio
from typing import Any, Literal

import instructor
import torch
from litellm import acompletion, aembedding, completion, embedding
from litellm.types.utils import EmbeddingResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from langcheck.utils.progress_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer
from ._base import EvalClient, TextResponseWithLogProbs
from .extractor import Extractor


class LiteLLMEvalClient(EvalClient):
    """EvalClient defined for OpenAI API."""

    def __init__(
        self,
        model: str,
        embedding_model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        use_async: bool = False,
        system_prompt: str | None = None,
        extractor: Extractor | None = None,
        **kwargs,
    ):
        """
        Initialize the litellm evaluation client.
        """
        self._model = model
        self._embedding_model = embedding_model

        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version

        self._use_async = use_async
        self._system_prompt = system_prompt

        self._kwargs = kwargs

        if extractor is None:
            self._extractor = LiteLLMExtractor(
                model=self._model,
                api_key=self._api_key,
                api_base=self._api_base,
                api_version=self._api_version,
                use_async=self._use_async,
                system_prompt=self._system_prompt,
                **self._kwargs,
            )
        else:
            self._extractor = extractor

    def _call_api(
        self,
        prompts: list[str],
        *,
        top_logprobs: int | None = None,
        tqdm_description: str | None = None,
    ) -> list[Any]:
        # Call API with different seed values for each prompt.
        model_inputs = [
            {
                "messages": [{"role": "user", "content": prompt}]
                + (
                    [{"role": "system", "content": self._system_prompt}]
                    if self._system_prompt
                    else []
                ),
                "seed": i,
            }
            for i, prompt in enumerate(prompts)
        ]

        logprobs = top_logprobs is not None

        if self._use_async:
            # A helper function to call the async API.
            async def _call_async_api() -> list[Any]:
                responses = await asyncio.gather(
                    *[
                        acompletion(
                            model=self._model,
                            messages=model_input["messages"],
                            seed=model_input["seed"],
                            logprobs=logprobs,
                            top_logprobs=top_logprobs,
                            api_key=self._api_key,
                            api_base=self._api_base,
                            api_version=self._api_version,
                            drop_params=True,
                            **self._kwargs,
                        )
                        for model_input in model_inputs
                    ],
                    return_exceptions=True,
                )
                return responses

            responses = asyncio.run(_call_async_api())
        else:
            # A helper function to call the API with exception filter for
            # alignment of exception handling with the async version.
            def _call_api_with_exception_filter(
                model_input: dict[str, Any],
            ) -> Any:
                if model_input is None:
                    return None
                try:
                    return completion(
                        model=self._model,
                        messages=model_input["messages"],
                        seed=model_input["seed"],
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        api_key=self._api_key,
                        api_base=self._api_base,
                        api_version=self._api_version,
                        drop_params=True,
                        **self._kwargs,
                    )
                except Exception as e:
                    return e

            responses = [
                _call_api_with_exception_filter(model_input)
                for model_input in tqdm_wrapper(
                    model_inputs, desc=tqdm_description
                )
            ]

        # Filter out exceptions and print them out.
        for i, response in enumerate(responses):
            if not isinstance(response, Exception):
                continue
            print(
                "OpenAI failed to return an assessment corresponding to "
                f"{i}th prompt: {response}"
            )
            responses[i] = None
        return responses

    def get_text_responses(
        self,
        prompts: list[str],
        *,
        tqdm_description: str | None = None,
    ) -> list[str | None]:
        """The function that gets responses to the given prompt texts.
        We use OpenAI's 'gpt-4o-mini' model by default, but you can configure
        it by passing the 'model' parameter in the openai_args.

        Args:
            prompts: The prompts you want to get the responses for.

        Returns:
            A list of responses to the prompts. The responses can be None if the
            evaluation fails.
        """
        if not isinstance(prompts, list):
            raise ValueError(
                f"prompts must be a list, not a {type(prompts).__name__}"
            )

        tqdm_description = tqdm_description or "Intermediate assessments (1/2)"
        responses = self._call_api(
            prompts=prompts,
            tqdm_description=tqdm_description,
        )
        response_texts = [
            response.choices[0].message.content if response else None
            for response in responses
        ]

        return response_texts

    def get_text_responses_with_log_likelihood(
        self,
        prompts: list[str],
        top_logprobs: int | None = None,
        *,
        tqdm_description: str | None = None,
    ) -> list[TextResponseWithLogProbs | None]:
        """The function that gets responses with log likelihood to the given
        prompt texts. Each concrete subclass needs to define the concrete
        implementation of this function to enable text scoring.

        NOTE: Please make sure that the model you use supports logprobs. In
        Azure OpenAI, the API version 2024-06-01 is the earliest GA version that
        supports logprobs.
        (https://docs.litellm.ai/docs/completion/input#translated-openai-params)

        Args:
            prompts: The prompts you want to get the responses for.
            top_logprobs: The number of logprobs to return for each token.

        Returns:
            A list of responses to the prompts. Each response is a tuple of the
            output text and the list of tuples of the output tokens and the log
            probabilities. The responses can be None if the evaluation fails.
        """
        if not isinstance(prompts, list):
            raise ValueError(
                f"prompts must be a list, not a {type(prompts).__name__}"
            )

        tqdm_description = tqdm_description or "Getting log likelihoods"
        responses = self._call_api(
            prompts=prompts,
            top_logprobs=top_logprobs,
            tqdm_description=tqdm_description,
        )
        response_texts_with_log_likelihood = []
        for response in responses:
            if response is None:
                response_texts_with_log_likelihood.append(None)
            else:
                response_dict = {
                    "response_text": response.choices[0].message.content,
                    "response_logprobs": [],
                }
                for logprob in response.choices[0].logprobs.content:
                    token_top_logprobs = [
                        {
                            "token": token_logprob.token,
                            "logprob": token_logprob.logprob,
                        }
                        for token_logprob in logprob.top_logprobs
                    ]
                    response_dict["response_logprobs"].append(
                        token_top_logprobs
                    )

                response_texts_with_log_likelihood.append(response_dict)

        return response_texts_with_log_likelihood

    def similarity_scorer(self) -> LiteLLMSimilarityScorer:
        """
        https://openai.com/blog/new-embedding-models-and-api-updates
        """
        if self._embedding_model is None:
            raise ValueError("embedding_model is not set")

        return LiteLLMSimilarityScorer(
            model=self._embedding_model,
            api_key=self._api_key,
            use_async=self._use_async,
        )


class LiteLLMExtractor(Extractor):
    """Score extractor defined for OpenAI API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        use_async: bool = False,
        system_prompt: str | None = None,
        **kwargs,
    ):
        """
        Initialize the LLM score extractor.
        """
        self._model = model

        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self._use_async = use_async
        self._system_prompt = system_prompt
        self._kwargs = kwargs

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
        texts that describe the evaluation results) into scores. We leverage the
        structured outputs API to extract the short assessment results from the
        unstructured assessments, so please make sure that the model you use
        supports structured outputs (only available in OpenAI's latest LLMs
        starting with GPT-4o). Also note that structured outputs API is only
        available in OpenAI API version of 2024-08-01-preview or later (See the
        References for more details).

        References:
            https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat

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
        if language not in ["en", "ja", "de", "zh"]:
            raise ValueError(f"Unsupported language: {language}")

        options = list(score_map.keys())

        class Response(BaseModel):
            score: Literal[tuple(options)]  # type: ignore

        structured_output_template = get_template(
            f"{language}/get_score/structured_output.j2"
        )

        model_inputs: list[list[ChatCompletionMessageParam]] = [
            [
                {
                    "role": "user",
                    "content": structured_output_template.render(
                        metric_name=metric_name,
                        unstructured_assessment=unstructured_assessment,
                        options=options,
                    ),
                }
            ]
            for unstructured_assessment in unstructured_assessment_result
        ]

        if self._use_async:
            client = instructor.from_litellm(acompletion)

            # A helper function to call the async API.
            async def _call_async_api() -> list[Any]:
                responses = await asyncio.gather(
                    *[
                        client.chat.completions.create(
                            model=self._model,
                            messages=input,
                            response_model=Response,
                            api_key=self._api_key,
                            api_base=self._api_base,
                            api_version=self._api_version,
                            drop_params=True,
                            **self._kwargs,
                        )
                        for input in model_inputs
                    ],
                    return_exceptions=True,
                )
                return responses

            responses = asyncio.run(_call_async_api())

        else:
            client = instructor.from_litellm(completion)

            # A helper function to call the API with exception filter for alignment
            # of exception handling with the async version.
            def _call_api_with_exception_filter(
                model_input: list[ChatCompletionMessageParam],
            ) -> Any:
                if model_input is None:
                    return None
                try:
                    return client.chat.completions.create(
                        model=self._model,
                        messages=model_input,
                        response_model=Response,
                        api_key=self._api_key,
                        api_base=self._api_base,
                        api_version=self._api_version,
                        drop_params=True,
                        **self._kwargs,
                    )
                except Exception as e:
                    return e

            responses = [
                _call_api_with_exception_filter(model_input)
                for model_input in tqdm_wrapper(
                    model_inputs, desc=tqdm_description
                )
            ]

        # Filter out exceptions and print them out
        for i, response in enumerate(responses):
            if not isinstance(response, Exception):
                continue
            print(
                f"Failed to return an assessment corresponding to {i}th prompt: "
                f"{response}"
            )
            responses[i] = None

        assessments = [
            response.score if response else None for response in responses
        ]

        return [
            score_map[assessment]
            if assessment and assessment in options
            else None
            for assessment in assessments
        ]


class LiteLLMSimilarityScorer(BaseSimilarityScorer):
    """Similarity scorer that uses the OpenAI API to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        use_async: bool = False,
    ):
        super().__init__()

        self._model = model
        self._api_key = api_key
        self._use_async = use_async

    async def _async_embed(self, inputs: list[str]) -> EmbeddingResponse:
        """Embed the inputs using the OpenAI API in async mode."""
        responses = await aembedding(
            input=inputs,
            model=self._model,
            api_key=self._api_key,
        )
        return responses

    def _embed(self, inputs: list[str]) -> torch.Tensor:
        """Embed the inputs using the OpenAI API."""

        if self._use_async:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:  # pragma: py-lt-310
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            embed_response = loop.run_until_complete(self._async_embed(inputs))
            embeddings = [item.embedding for item in embed_response.data]
        else:
            embed_response = embedding(
                input=inputs,
                model=self._model,
                api_key=self._api_key,
            )
            embeddings = [item.embedding for item in embed_response.data]  # type: ignore

        return torch.Tensor(embeddings)
