from __future__ import annotations

from typing import Optional

import openai


def rephrase(
        instances: list[str] | str,
        openai_args: Optional[dict[str, str]] = None) -> list[Optional[str]]:
    '''Rephrases each string in instances (usually a list of prompts) without
    changing their meaning. We use a modified version of the prompt presented
    in `"Rethinking Benchmark and Contamination for Language Models with
    Rephrased Samples" <https://arxiv.org/abs/2311.04850>`__ to make an LLM
    rephrase the given text.
    Currently, this is not available locally and requires an OpenAI API key,
    using the 'gpt-turbo-3.5' model by default. See `this page
    <https://langcheck.readthedocs.io/en/latest/metrics.html#computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    Args:
        instances: A single string or a list of strings to be augmented.
        openai_args: Dict of additional args to pass in to the
            `openai.ChatCompletion.create` function, default None

    Returns:
        A list of rephrased instances.
    '''
    instances = [instances] if isinstance(instances, str) else instances
    rephrased_instances = []
    for instance in instances:
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
        try:
            if openai_args is None:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
            else:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    **openai_args,
                )
            # This metrics-with-openai-models>`__ is necessary to pass pyright
            # since the openai library is not typed.
            assert isinstance(response, dict)
            rephrased_instance = response["choices"][0]["message"]["content"]
            rephrased_instances.append(rephrased_instance)
        except Exception as e:
            print(f'OpenAI failed to return a rephrased prompt: {e}')
            print(f'Prompt that triggered the failure is:\n{prompt}')
            rephrased_instances.append(None)

    return rephrased_instances
