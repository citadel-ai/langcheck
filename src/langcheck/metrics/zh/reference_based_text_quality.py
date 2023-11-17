from __future__ import annotations

from typing import Dict, List, Optional

import torch
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer, util

from langcheck.metrics._validation import validate_parameters_reference_based
from langcheck.metrics.en.reference_based_text_quality import \
    semantic_similarity as en_semantic_similarity
from langcheck.metrics.zh._tokenizers import HanLPTokenizer
from langcheck.metrics.metric_value import MetricValue


def semantic_similarity(
        generated_outputs: List[str] | str,
        reference_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        embedding_model_type: str = 'local',
        openai_args: Optional[Dict[str, str]] = None) -> MetricValue[float]:
    '''
    Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings. This metric
    takes on float values between [-1, 1], but typically ranges between 0 and 1
    where 0 is minimum similarity and 1 is maximum similarity. (NOTE: when using
    OpenAI embeddings, the cosine similarities tend to be skewed quite heavily
    towards higher numbers.)

    We currently support two embedding model types:

    1. The 'local' type, where the 'BAAI/bge-base-zh-v1.5' model
    is downloaded from HuggingFace and run locally. This is the default model
    type and there is no setup needed to run this. this model will return cosine
    similarity around 0.3 when sentences has no semantic similarity. sentences
    with missing punctuation would lower the value to 0.25 ~ 0.3.

    2. The 'openai' type, where we use OpenAI's 'text-embedding-ada-002' model
    by default (this is configurable). See
    `this example <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        https://openai.com/blog/new-and-improved-embedding-model

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        embedding_model_type: The type of embedding model to use ('local' or
            'openai'), default 'local'
        openai_args: Dict of additional args to pass in to the
            `openai.Embedding.create` function, default None

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    
    .. note:: It seems same methods in different language share the same
    pydoc, using sphinx or something to make these pydoc easier to be managed?
    '''
    generated_outputs, reference_outputs, prompts = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)
    assert embedding_model_type in [
        'local', 'openai'
    ], ('Unsupported embedding model type. '
        'The supported ones are ["local", "openai"]')

    if embedding_model_type == 'openai':
        # We can use the same API as english semantic_similarity to compare the
        # similarity
        metric_value = en_semantic_similarity(generated_outputs,
                                              reference_outputs, prompts,
                                              embedding_model_type, openai_args)
        metric_value.language = 'zh'
        return metric_value

    # According to C-MTEB Benchmark
    # https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB
    # 3 different size model provided by BAAI is the Best 3 on embedding task
    # Ref:
    # https://huggingface.co/BAAI/bge-base-zh-v1.5
    # using this model, it is hard to find two sentence cos_sim < 0.25
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    generated_embeddings = model.encode(generated_outputs)
    reference_embeddings = model.encode(reference_outputs)
    cosine_scores = util.pairwise_cos_sim(generated_embeddings,
                                          reference_embeddings)
    # Numerical instability can cause the dot product of almost identical
    # vectors to exceed 1.0 slightly, so we clip the outputs
    cosine_scores = torch.clamp(cosine_scores, -1.0, 1.0)

    return MetricValue(metric_name='semantic_similarity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=reference_outputs,
                       sources=None,
                       explanations=None,
                       metric_values=cosine_scores.tolist(),
                       language='zh')
