from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from openai import AzureOpenAI, OpenAI

from ._base import BaseSimilarityScorer


class OpenAISimilarityScorer(BaseSimilarityScorer):

    def __init__(self,
                 model_type: str = 'openai',
                 openai_client: Optional[OpenAI] = None,
                 openai_args: Optional[Dict[str, Any]] = None):
        assert model_type in [
            'openai', 'azure_openai'
        ], ('Unsupported embedding model type. '
            'The supported ones are ["openai", "azure_openai"]')

        super().__init__()

        if openai_client:
            self.openai_client = openai_client
        else:
            # Initialize the openai object if openai_client is None
            # TODO: Refactor this into OpenAIBasedEvaluator?
            if model_type == 'openai':
                self.openai_client = OpenAI()
            else:  # azure_openai
                # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix#embeddings
                self.openai_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                    api_version=os.getenv("OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv(
                        "AZURE_OPENAI_ENDPOINT"))  # type: ignore

        if model_type == 'azure_openai' and not openai_args:
            raise AssertionError(
                'The embedding model deployment must be specified in '
                '`openai_args` for the azure_openai type, e.g. '
                '`openai_args={"model": "YOUR_DEPLOYMENT_NAME"}`')

        self.openai_args = openai_args

    def _embed(self, inputs: List[str]) -> torch.Tensor:
        '''Embed the inputs using the OpenAI API.
        '''
        # Embed the inputs
        if self.openai_args:
            embed_response = self.openai_client.embeddings.create(
                input=inputs, **self.openai_args)
        else:
            embed_response = self.openai_client.embeddings.create(
                input=inputs, model='text-embedding-3-small')

        return torch.Tensor([item.embedding for item in embed_response.data])
