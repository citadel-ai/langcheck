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


class OpenAIEvalClient(EvalClient):
    """EvalClient defined for OpenAI API."""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        openai_args: dict[str, str] | None = None,
        *,
        use_async: bool = False,
        system_prompt: str | None = None,
    ):
        """
        Initialize the OpenAI evaluation client.

        Args:
            openai_client: (Optional) The OpenAI client to use.
            openai_args: (Optional) dict of additional args to pass in to the
            ``client.chat.completions.create`` function.
            use_async: If True, the async client will be used. Defaults to
                False.
            system_prompt: (Optional) The system prompt to use. If not provided,
                no system prompt will be used.
        """
        if openai_client:
            self._client = openai_client
        elif use_async:
            self._client = AsyncOpenAI()
        else:
            self._client = OpenAI()

        self._openai_args = openai_args
        self._use_async = use_async
        self._system_prompt = system_prompt

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

    def similarity_scorer(self) -> OpenAISimilarityScorer:
        """
        https://openai.com/blog/new-embedding-models-and-api-updates
        """
        return OpenAISimilarityScorer(
            openai_client=self._client,
            openai_args=self._openai_args,
            use_async=self._use_async,
        )


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
            openai_args: (Optional) dict of additional args to pass in to the
            ``client.chat.completions.create`` function
            use_async: (Optional) If True, the async client will be used.
            system_prompt: (Optional) The system prompt to use. If not provided,
                no system prompt will be used.
        """
        assert (
            text_model_name is not None or embedding_model_name is not None
        ), (
            "You need to specify either the text_model_name or the "
            "embedding_model_name to use the Azure OpenAI API."
        )
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#completions
        kargs = {
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        }
        if azure_openai_client is not None:
            self._client = azure_openai_client
        elif use_async:
            self._client = AsyncAzureOpenAI(**kargs)  # type: ignore
        else:
            self._client = AzureOpenAI(**kargs)  # type: ignore

        self._text_model_name = text_model_name
        self._embedding_model_name = embedding_model_name
        self._openai_args = openai_args or {}
        self._system_prompt = system_prompt

        if self._text_model_name is not None:
            self._openai_args["model"] = self._text_model_name

        self._use_async = use_async

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
            use_async=self._use_async,
        )


class OpenAISimilarityScorer(BaseSimilarityScorer):
    """Similarity scorer that uses the OpenAI API to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    """

    def __init__(
        self,
        openai_client: OpenAI | AzureOpenAI | AsyncOpenAI | AsyncAzureOpenAI,
        openai_args: dict[str, Any] | None = None,
        use_async: bool = False,
    ):
        super().__init__()

        self.openai_client = openai_client
        self.openai_args = openai_args
        self._use_async = use_async

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
