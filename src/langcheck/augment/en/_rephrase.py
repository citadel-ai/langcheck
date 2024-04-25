from __future__ import annotations

import os
from typing import Optional

from openai import AzureOpenAI, OpenAI


def rephrase(
        instances: list[str] | str,
        *,
        num_perturbations: int = 1,
        model_type: str = 'openai',
        openai_client: Optional[OpenAI] = None,
        openai_args: Optional[dict[str, str]] = None) -> list[Optional[str]]:
    '''Rephrases each string in instances (usually a list of prompts) without
    changing their meaning. We use a modified version of the prompt presented
    in `"Rethinking Benchmark and Contamination for Language Models with
    Rephrased Samples" <https://arxiv.org/abs/2311.04850>`__ to make an LLM
    rephrase the given text.

    We currently support two model types:

    1. The 'openai' type, where we use OpenAI's 'gpt-turbo-3.5' model
    by default.

    2. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify the
    model to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Args:
        instances: A single string or a list of strings to be augmented.
        num_perturbations: The number of perturbed instances to generate for
            each string in instances
        model_type: The type of model to use ('openai' or 'azure_openai'),
            default 'openai'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None, we will attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.chat.completions.create`` function, default None

    Returns:
        A list of rephrased instances.
    '''
    # Initialize the openai object if openai_client is None
    # TODO: Refactor this into OpenAIEvalClient?
    if openai_client is None:
        if model_type == 'openai':
            openai_client = OpenAI()
        elif model_type == 'azure_openai':
            if not openai_args:
                raise AssertionError(
                    'The model deployment must be specified in `openai_args` '
                    'for the azure_openai type, e.g. '
                    '`openai_args={"model": "YOUR_DEPLOYMENT_NAME"}`')
            openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv(
                    "AZURE_OPENAI_ENDPOINT"))  # type: ignore
        else:
            raise AssertionError(f'Unexpected model type "{model_type}"')

    instances = [instances] if isinstance(instances, str) else instances
    rephrased_instances = []
    for instance in instances:
        for i in range(num_perturbations):
            prompt = f'''
            Please rephrase the following prompt without altering its meaning,
            ensuring you adjust the word order appropriately.
            Ensure that no more than five consecutive words are repeated
            and try to use similar words as substitutes where possible.
            [BEGIN DATA]
            ************
            [Prompt]: {instance}
            ************
            [END DATA]
            '''
            messages = [{"role": "user", "content": prompt}]
            chat_completions = openai_client.chat.completions
            try:
                if openai_args is None:
                    response = chat_completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,  # type: ignore
                        seed=i)
                else:
                    response = chat_completions.create(  # type: ignore
                        messages=messages,  # type: ignore
                        seed=i,
                        **openai_args,  # type: ignore
                    )
                rephrased_instance = response.choices[0].message.content
                rephrased_instances.append(rephrased_instance)
            except Exception as e:
                print(f'OpenAI failed to return a rephrased prompt: {e}')
                print(f'Prompt that triggered the failure is:\n{prompt}')
                rephrased_instances.append(None)

    return rephrased_instances
