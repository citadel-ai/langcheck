from __future__ import annotations

import asyncio
from typing import Any, Literal

import torch
from google import genai
from google.genai import types
from pydantic import BaseModel

from langcheck.utils.progress_bar import tqdm_wrapper

from ..prompts._utils import get_template
from ..scorer._base import BaseSimilarityScorer
from ._base import EvalClient


class GeminiEvalClient(EvalClient):
    """EvalClient defined for the Gemini model."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        generate_content_args: dict[str, Any] | None = None,
        embed_model_name: str | None = None,
        *,
        use_async: bool = False,
        system_prompt: str | None = None,
    ):
        """
        Initialize the Gemini evaluation client. The authentication
        information is automatically read from the environment variables,
        so please make sure GOOGLE_API_KEY is set.
        If you want to use Vertex AI, please set the following environment
        variables appropriately:
        - GOOGLE_CLOUD_PROJECT=<your-project-id>
        - GOOGLE_CLOUD_LOCATION=<location>  (e.g. europe-west1)
        - GOOGLE_GENAI_USE_VERTEXAI=true
        - GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials-file>

        References:
            - https://ai.google.dev/api/python/google/generativeai/GenerativeModel
            - https://cloud.google.com/docs/authentication/application-default-credentials

        Args:
            model_name: (Optional) The Gemini model to use. Defaults to
                "gemini-1.5-flash".
            generate_content_args: (Optional) Dict of args to pass in to the
                ``generate_content`` function. The keys should be the same as
                the keys in the ``genai.types.GenerateContentConfig`` type.
            embed_model_name: (Optional) The name of the embedding model to use.
                If not provided, the models/text-embedding-004 model will be used.
            use_async: (Optional) If True, the async client will be used.
                Defaults to False.
            system_prompt: (Optional) The system prompt for ``generate_content``
                in ``get_text_responses`` function. If not provided, no system
                prompt will be used.
        """
        self.client = genai.Client()
        self._model_name = model_name
        self._generate_content_args = generate_content_args or {}
        self._embed_model_name = embed_model_name
        self._use_async = use_async
        self._system_instruction = system_prompt

        self._validate_generate_content_config()

    def _validate_generate_content_config(self) -> None:
        try:
            _ = types.GenerateContentConfig(**self._generate_content_args)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid generate_content_args: {self._generate_content_args}"
                f"Error: {e}"
            )

    def _call_api(
        self,
        model: str,
        prompts: list[str],
        config: dict[str, Any],
        *,
        tqdm_description: str | None = None,
    ) -> list[Any]:
        if self._use_async:

            async def _call_async_api() -> list[Any]:
                responses = await asyncio.gather(
                    *map(
                        lambda prompt: self.client.aio.models.generate_content(
                            model=model,
                            contents=types.Part.from_text(text=prompt),
                            config=types.GenerateContentConfig(**config),
                        ),
                        prompts,
                    ),
                    return_exceptions=True,
                )
                return responses

            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(_call_async_api())

        else:
            # A helper function to call the API with exception filter for alignment
            # of exception handling with the async version.
            def _call_api_with_exception_filter(prompt: str) -> Any:
                try:
                    return self.client.models.generate_content(
                        model=model,
                        contents=types.Part.from_text(text=prompt),
                        config=types.GenerateContentConfig(**config),
                    )
                except Exception as e:
                    return e

            responses = [
                _call_api_with_exception_filter(prompt)
                for prompt in tqdm_wrapper(prompts, desc=tqdm_description)
            ]

        # Filter out exceptions and print them out. Also filter out responses
        # that are blocked by safety settings and print out the safety ratings.
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(
                    "Gemini failed to return an assessment corresponding to "
                    f"{i}th prompt: {response}"
                )
                responses[i] = None
            elif response.candidates[0].finish_reason == 3:
                print(
                    f"Gemini's safety settings blocked the {i}th prompt:\n {response.candidates[0].safety_ratings}"
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

        config: dict[str, Any] = {
            "temperature": 0.0,
            "system_instruction": self._system_instruction,
        }
        config.update(self._generate_content_args or {})

        tqdm_description = tqdm_description or "Intermediate assessments (1/2)"
        responses = self._call_api(
            model=self._model_name,
            prompts=prompts,
            config=config,
            tqdm_description=tqdm_description,
        )
        response_texts = [
            response.text if response else None for response in responses
        ]

        return response_texts

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
        structured output API to extract the short assessment results from the
        unstructured assessments, so please make sure that the model you use
        supports structured output (See the References for more details).

        References:
            https://ai.google.dev/gemini-api/docs/structured-output

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
        if language not in ["en", "ja", "de"]:
            raise ValueError(f"Unsupported language: {language}")

        structured_output_template = get_template(
            f"{language}/get_score/structured_output.j2"
        )

        options = list(score_map.keys())

        class Response(BaseModel):
            score: Literal[tuple(options)]  # type: ignore

        config = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": Response,
        }
        config.update(self._generate_content_args or {})

        # Create prompts, filtering out None
        valid_prompts = []
        prompt_indices = []  # Keep track of original indices

        for i, unstructured_assessment in enumerate(
            unstructured_assessment_result
        ):
            if unstructured_assessment is not None:
                valid_prompts.append(
                    structured_output_template.render(
                        {
                            "metric": metric_name,
                            "unstructured_assessment": unstructured_assessment,
                            "options": options,
                        }
                    )
                )
                prompt_indices.append(i)

        tqdm_description = tqdm_description or "Scores (2/2)"

        # Call API for valid prompts
        if valid_prompts:
            api_responses = self._call_api(
                model=self._model_name,
                prompts=valid_prompts,
                config=config,
                tqdm_description=tqdm_description,
            )
        else:
            api_responses = []

        # Reconstruct full responses list with None for invalid prompts
        responses = [None] * len(unstructured_assessment_result)
        for i, response in enumerate(api_responses):
            original_idx = prompt_indices[i]
            responses[original_idx] = response

        assessments = [
            response.parsed.score if response else None
            for response in responses
        ]

        return [
            score_map[assessment]
            if assessment and assessment in options
            else None
            for assessment in assessments
        ]

    def similarity_scorer(self) -> GeminiSimilarityScorer:
        return GeminiSimilarityScorer(
            embed_model_name=self._embed_model_name,
            client=self.client,
            use_async=self._use_async,
        )


class GeminiSimilarityScorer(BaseSimilarityScorer):
    """Similarity scorer that uses the Gemini API to embed the inputs.
    In the current version of langcheck, the class is only instantiated within
    EvalClients.
    """

    def __init__(
        self,
        embed_model_name: str | None,
        client: genai.Client,
        *,
        use_async: bool = False,
    ):
        super().__init__()

        self._embed_model_name = embed_model_name or "text-embedding-004"
        self._client = client
        self._use_async = use_async

    def _embed(self, inputs: list[str]) -> torch.Tensor:
        """Embed the inputs using the Gemini API."""
        if self._use_async:

            async def _call_async_api():
                embed_response = await self._client.aio.models.embed_content(
                    model=self._embed_model_name,
                    contents=[
                        types.Part.from_text(text=prompt) for prompt in inputs
                    ],
                )
                return embed_response

            loop = asyncio.get_event_loop()
            embed_response = loop.run_until_complete(_call_async_api())
        else:
            embed_response = self._client.models.embed_content(
                model=self._embed_model_name,
                contents=[
                    types.Part.from_text(text=prompt) for prompt in inputs
                ],
            )

        assert embed_response.embeddings is not None
        return torch.Tensor(
            [embed.values for embed in embed_response.embeddings]
        )
