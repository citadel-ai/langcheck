from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any, Literal

import torch
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from pydantic import BaseModel

from langcheck.utils.progress_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer
from ._base import EvalClient, TextResponseWithLogProbs
from .extractor import Extractor, StringMatchExtractor


class OpenAIEvalClient(EvalClient):
    """EvalClient defined for OpenAI API."""

    def __init__(
        self,
        openai_client: OpenAI | AsyncOpenAI | None = None,
        openai_args: dict[str, str] | None = None,
        *,
        use_async: bool = False,
        system_prompt: str | None = None,
        extractor: Extractor | None = None,
    ):
        """
        Initialize the OpenAI evaluation client. The authentication
        information is automatically read from the environment variables, so
        please make sure `OPENAI_API_KEY` environment variable is set.

        Args:
            openai_client (Optional): The OpenAI client to use.
            openai_args (Optional): dict of additional args to pass in to the
            `client.chat.completions.create` function.
            use_async: If True, the async client will be used. Defaults to
                False.
            system_prompt (Optional): The system prompt to use. If not provided,
                no system prompt will be used.
            extractor (Optional): The extractor to use. If not provided, the
                default extractor will be used.
        """
        if openai_client:
            self._client = openai_client
            self._use_async = isinstance(openai_client, AsyncOpenAI)

            # Client config will take precedence over the argument, and the
            # argument will be ignored.
            if self._use_async and not use_async:
                warnings.warn(
                    "The provided `openai_client` is an async client, "
                    "so the `use_async=False` argument will be ignored. The async client will be used."
                )
            elif not self._use_async and use_async:
                warnings.warn(
                    "The provided `openai_client` is a synchronous client, "
                    "so the `use_async=True` argument will be ignored. The synchronous client will be used."
                )
        else:
            self._client = AsyncOpenAI() if use_async else OpenAI()
            self._use_async = use_async

        self._openai_args = openai_args
        self._system_prompt = system_prompt

        if extractor is None:
            self._extractor = OpenAIExtractor(
                openai_client=self._client,
                openai_args=self._openai_args,
                use_async=self._use_async,
            )
        else:
            self._extractor = extractor

    def _call_api(
        self,
        prompts: list[str],
        config: dict[str, str],
        *,
        tqdm_description: str | None = None,
        system_prompt: str | None = None,
    ) -> list[Any]:
        # A helper function to call the API with exception filter for alignment
        # of exception handling with the async version.
        def _call_api_with_exception_filter(model_input: dict[str, Any]) -> Any:
            if model_input is None:
                return None
            try:
                return self._client.chat.completions.create(**model_input)
            except Exception as e:
                return e

        system_message = []
        if system_prompt:
            system_message.append({"role": "system", "content": system_prompt})

        # Call API with different seed values for each prompt.
        model_inputs = [
            {
                "messages": system_message
                + [{"role": "user", "content": prompt}],
                "seed": i,
                **config,
            }
            for i, prompt in enumerate(prompts)
        ]

        if self._use_async:
            # A helper function to call the async API.
            async def _call_async_api() -> list[Any]:
                responses = await asyncio.gather(
                    *map(
                        lambda model_input: self._client.chat.completions.create(
                            **model_input
                        ),
                        model_inputs,
                    ),
                    return_exceptions=True,
                )
                return responses

            responses = asyncio.run(_call_async_api())
        else:
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
        warnings.warn(
            "The default model is changed to gpt-4o-mini from gpt-3.5-turbo. "
            "If you want to use other models, please set the model "
            "parameter to the desired model name in the `openai_args`."
        )

        if not isinstance(prompts, list):
            raise ValueError(
                f"prompts must be a list, not a {type(prompts).__name__}"
            )

        config = {"model": "gpt-4o-mini"}
        config.update(self._openai_args or {})
        tqdm_description = tqdm_description or "Intermediate assessments (1/2)"
        responses = self._call_api(
            prompts=prompts,
            config=config,
            tqdm_description=tqdm_description,
            system_prompt=self._system_prompt,
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
        supports logprobs (https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#new-ga-api-release).

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

        config = {"model": "gpt-4o-mini", "logprobs": True}
        if top_logprobs:
            config["top_logprobs"] = top_logprobs
        config.update(self._openai_args or {})
        tqdm_description = tqdm_description or "Getting log likelihoods"
        responses = self._call_api(
            prompts=prompts,
            config=config,
            tqdm_description=tqdm_description,
            system_prompt=self._system_prompt,
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

    def similarity_scorer(self) -> OpenAISimilarityScorer:
        """
        https://openai.com/blog/new-embedding-models-and-api-updates
        """
        return OpenAISimilarityScorer(
            openai_client=self._client,
            openai_args=self._openai_args,
        )


class OpenAIExtractor(Extractor):
    """Score extractor defined for OpenAI API."""

    def __init__(
        self,
        openai_client: OpenAI | AsyncOpenAI | None = None,
        openai_args: dict[str, str] | None = None,
        *,
        use_async: bool = False,
    ):
        """
        Initialize the OpenAI score extractor. The authentication information is
        automatically read from the environment variables, so please make sure
        `OPENAI_API_KEY` environment variable is set.

        Args:
            openai_client (Optional): The OpenAI client to use.
            openai_args (Optional): dict of additional args to pass in to the
                `client.chat.completions.create` function.
            use_async: If True, the async client will be used. Defaults to
                False.
        """
        if openai_client:
            self._client = openai_client
            self._use_async = isinstance(openai_client, AsyncOpenAI)

            # Client config will take precedence over the argument, and the
            # argument will be ignored.
            if self._use_async and not use_async:
                warnings.warn(
                    "The provided `openai_client` is an async client, "
                    "so the `use_async=False` argument will be ignored. The async client will be used."
                )
            elif not self._use_async and use_async:
                warnings.warn(
                    "The provided `openai_client` is a synchronous client, "
                    "so the `use_async=True` argument will be ignored. The synchronous client will be used."
                )
        else:
            self._client = AsyncOpenAI() if use_async else OpenAI()
            self._use_async = use_async

        self._openai_args = openai_args

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

        config = {"model": "gpt-4o-mini"}
        config.update(self._openai_args or {})
        model_inputs = [
            {
                **config,
                "messages": [
                    {
                        "role": "user",
                        "content": structured_output_template.render(
                            metric_name=metric_name,
                            unstructured_assessment=unstructured_assessment,
                            options=options,
                        ),
                    }
                ],
                "response_format": Response,
            }
            for unstructured_assessment in unstructured_assessment_result
        ]

        if self._use_async:
            # A helper function to call the async API.
            async def _call_async_api() -> list[Any]:
                responses = await asyncio.gather(
                    *[
                        self._client.beta.chat.completions.parse(**input)
                        for input in model_inputs
                    ],  # type: ignore
                    return_exceptions=True,
                )
                return responses

            responses = asyncio.run(_call_async_api())

        else:
            # A helper function to call the API with exception filter for alignment
            # of exception handling with the async version.
            def _call_api_with_exception_filter(
                model_input: dict[str, Any],
            ) -> Any:
                if model_input is None:
                    return None
                try:
                    return self._client.beta.chat.completions.parse(
                        **model_input
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
                "OpenAI failed to return an assessment corresponding to "
                f"{i}th prompt: {response}"
            )
            responses[i] = None

        assessments = [
            response.choices[0].message.parsed.score if response else None
            for response in responses
        ]

        return [
            score_map[assessment]
            if assessment and assessment in options
            else None
            for assessment in assessments
        ]


class AzureOpenAIEvalClient(OpenAIEvalClient):
    def __init__(
        self,
        text_model_name: str | None = None,
        embedding_model_name: str | None = None,
        azure_openai_client: AzureOpenAI | None = None,
        openai_args: dict[str, str] | None = None,
        *,
        use_async: bool = False,
        system_prompt: str | None = None,
        extractor: Extractor | None = None,
    ):
        """
        Intialize the Azure OpenAI evaluation client.

        Args:
            text_model_name (Optional): The text model name you want to use with
                the Azure OpenAI API. The name is used as
                `{ "model": text_model_name }` parameter when calling the Azure
                OpenAI API for text models.
            embedding_model_name (Optional): The text model name you want to
                use with the Azure OpenAI API. The name is used as
                `{ "model": embedding_model_name }` parameter when calling the
                Azure OpenAI API for embedding models.
            azure_openai_client (Optional): The Azure OpenAI client to use.
            openai_args (Optional): dict of additional args to pass in to the
                `client.chat.completions.create` function.
            use_async (Optional): If True, the async client will be used.
            system_prompt (Optional): The system prompt to use. If not provided,
                no system prompt will be used.
            extractor (Optional): The extractor to use. If not provided, the
                default extractor will be used.
        """
        assert (
            text_model_name is not None or embedding_model_name is not None
        ), (
            "You need to specify either the text_model_name or the "
            "embedding_model_name to use the Azure OpenAI API."
        )
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#completions

        # Check for old environment variable
        if os.getenv("AZURE_OPENAI_KEY") is not None:
            warnings.warn(
                "Environment variable 'AZURE_OPENAI_KEY' is deprecated and will be removed in a future version. "
                "Please use 'AZURE_OPENAI_API_KEY' instead.",
                DeprecationWarning,
            )
            if os.getenv("AZURE_OPENAI_API_KEY") is None:
                warnings.warn(
                    "Environment variable 'AZURE_OPENAI_API_KEY' is not set. "
                    "Falling back to 'AZURE_OPENAI_KEY'.",
                    DeprecationWarning,
                )
                os.environ["AZURE_OPENAI_API_KEY"] = os.environ[
                    "AZURE_OPENAI_KEY"
                ]

        kargs = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }

        if azure_openai_client:
            self._client = azure_openai_client
            self._use_async = isinstance(azure_openai_client, AsyncAzureOpenAI)

            # Client config will take precedence over the argument, and the
            # argument will be ignored.
            if self._use_async and not use_async:
                warnings.warn(
                    "The provided `azure_openai_client` is an async client, "
                    "so the `use_async=False` argument will be ignored. The async client will be used."
                )
            elif not self._use_async and use_async:
                warnings.warn(
                    "The provided `azure_openai_client` is a synchronous client, "
                    "so the `use_async=True` argument will be ignored. The synchronous client will be used."
                )
        else:
            self._client = (
                AsyncAzureOpenAI(**kargs) if use_async else AzureOpenAI(**kargs)  # type: ignore
            )
            self._use_async = use_async

        self._text_model_name = text_model_name
        self._embedding_model_name = embedding_model_name
        self._openai_args = openai_args or {}
        self._system_prompt = system_prompt

        if self._text_model_name is not None:
            self._openai_args["model"] = self._text_model_name

        if extractor is not None:
            self._extractor = extractor
        elif text_model_name is not None:
            self._extractor = AzureOpenAIExtractor(
                text_model_name=text_model_name,
                azure_openai_client=azure_openai_client,
                openai_args=openai_args,
            )
        else:
            self._extractor = StringMatchExtractor()

    def similarity_scorer(self) -> OpenAISimilarityScorer:
        """This method does the sanity check for the embedding_model_name and
        then calls the parent class's similarity_scorer method with the
        additional "model" parameter. See the parent class for the detailed
        documentation.
        """
        assert self._embedding_model_name is not None, (
            "You need to specify the embedding_model_name to get the score for "
            "this metric."
        )
        openai_args = {**self._openai_args, "model": self._embedding_model_name}
        return OpenAISimilarityScorer(
            openai_client=self._client,
            openai_args=openai_args,
        )


class AzureOpenAIExtractor(OpenAIExtractor):
    def __init__(
        self,
        text_model_name: str | None = None,
        azure_openai_client: AzureOpenAI | None = None,
        openai_args: dict[str, str] | None = None,
        *,
        use_async: bool = False,
    ):
        assert text_model_name is not None, (
            "You need to specify the text_model_name to use the Azure OpenAI API."
        )
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#completions
        kargs = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }

        if azure_openai_client:
            self._client = azure_openai_client
            self._use_async = isinstance(azure_openai_client, AsyncAzureOpenAI)

            # Client config will take precedence over the argument, and the
            # argument will be ignored.
            if self._use_async and not use_async:
                warnings.warn(
                    "The provided `azure_openai_client` is an async client, "
                    "so the `use_async=False` argument will be ignored. The async client will be used."
                )
            elif not self._use_async and use_async:
                warnings.warn(
                    "The provided `azure_openai_client` is a synchronous client, "
                    "so the `use_async=True` argument will be ignored. The synchronous client will be used."
                )
        else:
            self._client = (
                AsyncAzureOpenAI(**kargs) if use_async else AzureOpenAI(**kargs)  # type: ignore
            )
            self._use_async = use_async

        self._openai_args = openai_args or {}
        self._openai_args["model"] = text_model_name


class OpenAISimilarityScorer(BaseSimilarityScorer):
    """Similarity scorer that uses the OpenAI API to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    """

    def __init__(
        self,
        openai_client: OpenAI | AzureOpenAI | AsyncOpenAI | AsyncAzureOpenAI,
        openai_args: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.openai_client = openai_client
        self.openai_args = openai_args
        self._use_async = isinstance(openai_client, AsyncOpenAI)

    async def _async_embed(self, inputs: list[str]) -> CreateEmbeddingResponse:
        """Embed the inputs using the OpenAI API in async mode."""
        assert isinstance(self.openai_client, AsyncOpenAI)
        if self.openai_args:
            responses = await self.openai_client.embeddings.create(
                input=inputs, **self.openai_args
            )
        else:
            responses = await self.openai_client.embeddings.create(
                input=inputs, model="text-embedding-3-small"
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
            assert isinstance(self.openai_client, OpenAI)

            if self.openai_args:
                embed_response = self.openai_client.embeddings.create(
                    input=inputs, **self.openai_args
                )
            else:
                embed_response = self.openai_client.embeddings.create(
                    input=inputs, model="text-embedding-3-small"
                )
            embeddings = [item.embedding for item in embed_response.data]

        return torch.Tensor(embeddings)
