from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any

from anthropic import (
    Anthropic,
    AnthropicVertex,
    AsyncAnthropic,
    AsyncAnthropicVertex,
)

from langcheck.utils.progress_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ._base import EvalClient
from .extractor import Extractor


class AnthropicEvalClient(EvalClient):
    """EvalClient defined for Anthropic API."""

    def __init__(
        self,
        anthropic_client: Anthropic
        | AsyncAnthropic
        | AnthropicVertex
        | AsyncAnthropicVertex
        | None = None,
        anthropic_args: dict[str, Any] | None = None,
        *,
        use_async: bool = False,
        vertexai: bool = False,
        system_prompt: str | None = None,
        extractor: Extractor | None = None,
    ):
        """
        Initialize the Anthropic evaluation client. The authentication
        information is automatically read from the environment variables.
        If you want to use Anthropic API, please set `ANTHROPIC_API_KEY`.
        If you want to use Vertex AI API, set the `vertexai` argument to True,
        and please set the following environment variables:
            - ANTHROPIC_VERTEX_PROJECT_ID=<your-project-id>
            - CLOUD_ML_REGION=<region>  (e.g. europe-west1)
            - GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials-file>

        References:
            - https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude
            - https://cloud.google.com/docs/authentication/application-default-credentials

        Args:
            anthropic_client (Optional): The Anthropic client to use.
            anthropic_args (Optional): dict of additional args to pass in to
                the `client.messages.create` function
            use_async: If True, the async client will be used. Ignored when
                `anthropic_client` is provided. Defaults to False.
            vertexai: If True, the Vertex AI client will be used. Ignored when
                `anthropic_client` is provided. Defaults to False.
            system_prompt (Optional): The system prompt to use. If not provided,
                no system prompt will be used.
            extractor (Optional): The extractor to use. If not provided, the
                default extractor will be used.
        """

        if anthropic_client is None:
            if vertexai:
                # Vertex AI requires these environment variables
                for env_var in [
                    "ANTHROPIC_VERTEX_PROJECT_ID",
                    "CLOUD_ML_REGION",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                ]:
                    if not os.environ.get(env_var):
                        raise ValueError(
                            f"Environment variable '{env_var}' must be set when using Vertex AI."
                        )

                if not os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
                    raise ValueError(
                        "`ANTHROPIC_VERTEX_PROJECT_ID` must be set when using Vertex AI."
                    )

                # Warn that `ANTHROPIC_API_KEY` is not used when using Vertex AI
                if os.environ.get("ANTHROPIC_API_KEY", None):
                    warnings.warn(
                        "`ANTHROPIC_API_KEY` is set when using Vertex AI. "
                        "Vertex AI will take precedence over the API key from "
                        "the environment variable."
                    )

                if use_async:
                    self._client = AsyncAnthropicVertex()
                else:
                    self._client = AnthropicVertex()
            else:
                if os.environ.get("ANTHROPIC_API_KEY", None) is None:
                    raise ValueError(
                        "`ANTHROPIC_API_KEY` is not set when using Anthropic API. "
                        "Please set the `ANTHROPIC_API_KEY` environment variable."
                    )

                if use_async:
                    self._client = AsyncAnthropic()
                else:
                    self._client = Anthropic()

            self._vertexai = vertexai
            self._use_async = use_async
        else:
            self._client = anthropic_client
            self._vertexai = isinstance(
                anthropic_client, (AnthropicVertex, AsyncAnthropicVertex)
            )
            self._use_async = isinstance(
                anthropic_client, (AsyncAnthropic, AsyncAnthropicVertex)
            )

            # Client config will take precedence over the argument, and the
            # argument will be ignored.
            if self._vertexai and not vertexai:
                warnings.warn(
                    "The provided `anthropic_client` is a Vertex AI client, "
                    "so the `vertexai=False` argument will be ignored. The Vertex AI client will be used."
                )
            elif not self._vertexai and vertexai:
                warnings.warn(
                    "The provided `anthropic_client` is an Anthropic client, "
                    "so the `vertexai=True` argument will be ignored. The Anthropic client will be used."
                )

            if self._use_async and not use_async:
                warnings.warn(
                    "The provided `anthropic_client` is an async client, "
                    "so the `use_async=False` argument will be ignored. The async client will be used."
                )
            elif not self._use_async and use_async:
                warnings.warn(
                    "The provided `anthropic_client` is a synchronous client, "
                    "so the `use_async=True` argument will be ignored. The synchronous client will be used."
                )

        self._anthropic_args = anthropic_args or {}
        self._system_prompt = system_prompt

        if system_prompt and "system" in self._anthropic_args:
            warnings.warn(
                '"system" of anthropic_args will be ignored because '
                "system_prompt is provided."
            )

        if extractor is None:
            self._extractor = AnthropicExtractor(
                anthropic_client=self._client,
                use_async=self._use_async,
                vertexai=self._vertexai,
            )
        else:
            self._extractor = extractor

    def get_text_responses(
        self,
        prompts: list[str],
        *,
        tqdm_description: str | None = None,
    ) -> list[str | None]:
        """The function that gets responses to the given prompt texts.
        We use Anthropic's 'claude-3-haiku-20240307' model by default, but you
        can configure it by passing the 'model' parameter in the anthropic_args.

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

        config = {
            # The model names are slightly different for Anthropic API and Vertex AI API
            # Reference: https://docs.anthropic.com/en/docs/about-claude/models/all-models
            "model": "claude-3-haiku@20240307"
            if self._vertexai
            else "claude-3-haiku-20240307",
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        config.update(self._anthropic_args or {})
        tqdm_description = tqdm_description or "Intermediate assessments (1/2)"
        responses = _call_api(
            client=self._client,
            prompts=prompts,
            config=config,
            use_async=self._use_async,
            tqdm_description=tqdm_description,
            system_prompt=self._system_prompt,
        )
        response_texts = [
            response.content[0].text if response else None
            for response in responses
        ]

        return response_texts

    def similarity_scorer(self):
        raise NotImplementedError(
            "Embedding-based metrics are not supported in AnthropicEvalClient."
            "Use other EvalClients to get these metrics."
        )


class AnthropicExtractor(Extractor):
    """Score extractor for Anthropic API."""

    def __init__(
        self,
        anthropic_client: Anthropic
        | AsyncAnthropic
        | AnthropicVertex
        | AsyncAnthropicVertex
        | None = None,
        anthropic_args: dict[str, Any] | None = None,
        *,
        use_async: bool = False,
        vertexai: bool = False,
    ):
        """
        Initialize the Anthropic score extractor. The authentication information
        is automatically read from the environment variables.
        If you want to use Anthropic API, please set `ANTHROPIC_API_KEY`.
        If you want to use Vertex AI API, set the `vertexai` argument to True,
        and please set the following environment variables:
            - ANTHROPIC_VERTEX_PROJECT_ID=<your-project-id>
            - CLOUD_ML_REGION=<region>  (e.g. europe-west1)
            - GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials-file>

        References:
            - https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude
            - https://cloud.google.com/docs/authentication/application-default-credentials

        Args:
            anthropic_client (Optional): The Anthropic client to use.
            anthropic_args (Optional): dict of additional args to pass in to
                the `client.messages.create` function
            use_async: If True, the async client will be used. Ignored when
                `anthropic_client` is provided. Defaults to False.
            vertexai: If True, the Vertex AI client will be used. Ignored when
                `anthropic_client` is provided. Defaults to False.
        """
        if anthropic_client is None:
            if vertexai:
                # Vertex AI requires these environment variables
                for env_var in [
                    "ANTHROPIC_VERTEX_PROJECT_ID",
                    "CLOUD_ML_REGION",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                ]:
                    if not os.environ.get(env_var):
                        raise ValueError(
                            f"Environment variable '{env_var}' must be set when using Vertex AI."
                        )

                if not os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
                    raise ValueError(
                        "`ANTHROPIC_VERTEX_PROJECT_ID` must be set when using Vertex AI."
                    )

                # Warn that `ANTHROPIC_API_KEY` is not used when using Vertex AI
                if os.environ.get("ANTHROPIC_API_KEY", None):
                    warnings.warn(
                        "`ANTHROPIC_API_KEY` is set when using Vertex AI. "
                        "Vertex AI will take precedence over the API key from "
                        "the environment variable."
                    )

                if use_async:
                    self._client = AsyncAnthropicVertex()
                else:
                    self._client = AnthropicVertex()
            else:
                if os.environ.get("ANTHROPIC_API_KEY", None) is None:
                    raise ValueError(
                        "`ANTHROPIC_API_KEY` is not set when using Anthropic API. "
                        "Please set the `ANTHROPIC_API_KEY` environment variable."
                    )

                if use_async:
                    self._client = AsyncAnthropic()
                else:
                    self._client = Anthropic()

            self._use_async = use_async
            self._vertexai = vertexai

        else:
            self._client = anthropic_client
            self._use_async = isinstance(
                anthropic_client, (AsyncAnthropic, AsyncAnthropicVertex)
            )
            self._vertexai = isinstance(
                anthropic_client, (AnthropicVertex, AsyncAnthropicVertex)
            )

            # Client config will take precedence over the argument, and the
            # argument will be ignored.
            if self._vertexai and not vertexai:
                warnings.warn(
                    "The provided `anthropic_client` is a Vertex AI client, "
                    "so the `vertexai=False` argument will be ignored. The Vertex AI client will be used."
                )
            elif not self._vertexai and vertexai:
                warnings.warn(
                    "The provided `anthropic_client` is an Anthropic client, "
                    "so the `vertexai=True` argument will be ignored. The Anthropic client will be used."
                )

            if self._use_async and not use_async:
                warnings.warn(
                    "The provided `anthropic_client` is an async client, "
                    "so the `use_async=False` argument will be ignored. The async client will be used."
                )
            elif not self._use_async and use_async:
                warnings.warn(
                    "The provided `anthropic_client` is a synchronous client, "
                    "so the `use_async=True` argument will be ignored. The synchronous client will be used."
                )

        self._anthropic_args = anthropic_args or {}

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
        texts that describe the evaluation results) into scores.

        Args:
            metric_name : The name of the metric to be used. (e.g. "toxicity")
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
        if language not in ["en", "ja", "de"]:
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

        config = {
            # The model names are slightly different for Anthropic API and Vertex AI API
            # Reference: https://docs.anthropic.com/en/docs/about-claude/models/all-models
            "model": "claude-3-haiku@20240307"
            if self._vertexai
            else "claude-3-haiku-20240307",
            "max_tokens": 1024,
        }
        config.update(self._anthropic_args or {})
        tqdm_description = tqdm_description or "Scores (2/2)"
        responses = _call_api(
            client=self._client,
            prompts=get_score_prompts,
            config=config,
            use_async=self._use_async,
            tqdm_description=tqdm_description,
        )
        raw_response_texts = [
            response.content[0].text if response else None
            for response in responses
        ]

        def _turn_to_score(response: str | None) -> float | None:
            if response is None:
                return None
            option_found = [option for option in options if option in response]
            # if response contains multiple options as substrings, return None
            if len(option_found) != 1:
                return None
            return score_map[option_found[0]]

        return [_turn_to_score(response) for response in raw_response_texts]


def _call_api(
    client: Anthropic | AsyncAnthropic | AnthropicVertex | AsyncAnthropicVertex,
    prompts: list[str] | list[str | None],
    config: dict[str, Any],
    *,
    use_async: bool = False,
    system_prompt: str | None = None,
    tqdm_description: str | None = None,
) -> list[Any]:
    """A helper function to call the Anthropic API."""

    # A helper function to call the API with exception filter for alignment
    # of exception handling with the async version.
    def _call_api_with_exception_filter(model_input: dict[str, Any]) -> Any:
        if model_input is None:
            return None
        try:
            return client.messages.create(**model_input)
        except Exception as e:
            return e

    if system_prompt:
        config["system"] = system_prompt

    model_inputs = [
        {
            "messages": [{"role": "user", "content": prompt}],
            **config,
        }
        for prompt in prompts
    ]

    if use_async:
        # A helper function to call the async API.
        async def _call_async_api() -> list[Any]:
            responses = await asyncio.gather(
                *map(
                    lambda model_input: client.messages.create(**model_input),
                    model_inputs,
                ),
                return_exceptions=True,
            )
            return responses

        responses = asyncio.run(_call_async_api())
    else:
        responses = [
            _call_api_with_exception_filter(model_input)
            for model_input in tqdm_wrapper(model_inputs, desc=tqdm_description)
        ]

    # Filter out exceptions and print them out.
    for i, response in enumerate(responses):
        if not isinstance(response, Exception):
            continue
        print(
            "Anthropic failed to return an assessment corresponding to "
            f"{i}th prompt: {response}"
        )
        responses[i] = None
    return responses
