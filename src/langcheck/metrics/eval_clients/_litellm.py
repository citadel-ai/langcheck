from __future__ import annotations

import asyncio
from typing import Any, Literal

import instructor
import litellm
import torch
from litellm.types.utils import EmbeddingResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from langcheck.utils.progress_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer
from ._base import EvalClient, TextResponseWithLogProbs
from .extractor import Extractor


class LiteLLMEvalClient(EvalClient):
    """EvalClient defined for litellm."""

    def __init__(
        self,
        model: str,
        embedding_model: str | None = None,
        *,
        use_async: bool = False,
        system_prompt: str | None = None,
        extractor: Extractor | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        **kwargs,
    ):
        """
        Initialize the litellm evaluation client.

        References:
            https://docs.litellm.ai/docs/completion/input
            https://docs.litellm.ai/docs/providers

        Args:
            model: The model name for evaluation. The name should be
                <model_provider>/<model_name> (e.g. "openai/gpt-4o-mini").
            embedding_model: The model name for embedding. The name should be
                <model_provider>/<model_name> (e.g. "openai/text-embedding-3-small").
            use_async: Whether to use async mode.
            system_prompt: The system prompt to use for the API.
            extractor: The extractor to use for the API.
            api_key: The API key for the model. This will be checked for all the
                providers.
            api_base: The base URL for the API.
            api_version: The version of the API.
            kwargs: Additional arguments to pass to the API. The credentials for
                cloud providers can be passed here. See the references for the
                supported providers and their credentials.
                Examples:
                - aws_access_key_id, aws_secret_access_key, aws_region_name
                - vertex_location, vertex_credentials
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
                        litellm.acompletion(
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
                    return litellm.completion(
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
                f"Failed to return an assessment corresponding to {i}th prompt: "
                f"{response}"
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

        NOTE: Please make sure that the model you use supports logprobs.
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
        if self._embedding_model is None:
            raise ValueError("embedding_model is not set")

        return LiteLLMSimilarityScorer(
            model=self._embedding_model,
            api_key=self._api_key,
            use_async=self._use_async,
        )


class LiteLLMExtractor(Extractor):
    """Score extractor defined for litellm."""

    def __init__(
        self,
        model: str,
        *,
        use_async: bool = False,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        **kwargs,
    ):
        """
        Initialize the LLM score extractor.

        Args:
            model: The model name for evaluation. The name should be
                <model_provider>/<model_name> (e.g. "openai/gpt-4o-mini").
            use_async: Whether to use async mode.
            api_key: The API key for the model. This will be checked for all the
                providers.
            api_base: The base URL for the API.
            api_version: The version of the API.
            kwargs: Additional arguments to pass to the API. The credentials for
                cloud providers can be passed here.
                Examples:
                - aws_access_key_id, aws_secret_access_key, aws_region_name
                - vertex_location, vertex_credentials
        """
        self._model = model

        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self._use_async = use_async
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
        texts that describe the evaluation results) into scores. `instructor` is
        used to extract the result with robust structured outputs.

        References:
            https://docs.litellm.ai/docs/tutorials/instructor

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
            client = instructor.from_litellm(litellm.acompletion)

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
            client = instructor.from_litellm(litellm.completion)

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
                f"Failed to return an assessment for the {i}th prompt: "
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
    """Similarity scorer to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        *,
        use_async: bool = False,
        **kwargs,
    ):
        """
        Initialize the similarity scorer.

        Args:
            model: The embedding model name. The name should be
                <model_provider>/<model_name> (e.g. "openai/text-embedding-3-small").
            api_key: The API key for the model. This will be checked for all the
                providers.
            api_base: The base URL for the API.
            api_version: The version of the API.
            use_async: Whether to use async mode.
            kwargs: Additional arguments to pass to the API. The credentials for
                cloud providers can be passed here.
                Examples:
                - aws_access_key_id, aws_secret_access_key, aws_region_name
                - vertex_location, vertex_credentials
        """

        super().__init__()

        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self._use_async = use_async

        self._kwargs = kwargs

    async def _async_embed(self, inputs: list[str]) -> EmbeddingResponse:
        """Embed the inputs in async mode."""
        responses = await litellm.aembedding(
            input=inputs,
            model=self._model,
            api_key=self._api_key,
            api_base=self._api_base,
            api_version=self._api_version,
            **self._kwargs,
        )
        return responses

    def _embed(self, inputs: list[str]) -> torch.Tensor:
        """Embed the inputs."""
        if self._use_async:
            # TODO: For Gemini, this outputs some warnings about async client
            # session. https://github.com/BerriAI/litellm/issues/12108
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:  # pragma: py-lt-310
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            embed_response = loop.run_until_complete(self._async_embed(inputs))
        else:
            embed_response = litellm.embedding(
                input=inputs,
                model=self._model,
                api_key=self._api_key,
                api_base=self._api_base,
                api_version=self._api_version,
                **self._kwargs,
            )

        embeddings = [item["embedding"] for item in embed_response.data]  # type: ignore
        return torch.Tensor(embeddings)
