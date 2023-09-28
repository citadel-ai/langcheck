from typing import Dict, List, Optional

import openai
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from langcheck.eval.eval_value import EvalValue


def semantic_sim(
        generated_outputs: List[str],
        reference_outputs: List[str],
        embedding_model_type: str = 'local',
        openai_args: Optional[Dict[str, str]] = None) -> EvalValue[float]:
    '''Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings. This metric
    takes on float values between [-1, 1], but typically ranges between 0 and 1
    where 0 is minimum similarity and 1 is maximum similarity. (NOTE: when using
    OpenAI embeddings, the cosine similarities tend to be skewed quite heavily
    towards higher numbers.)

    We currently support two embedding model types:

    1. The 'local' type, where the 'all-mpnet-base-v2' model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'text-embedding-ada-002' model
    by default (this is configurable). See
    https://github.com/citadel-ai/langcheck#evaluate-text for examples on
    setting up the OpenAI API key.

    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        https://openai.com/blog/new-and-improved-embedding-model

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs
        embedding_model_type: The type of embedding model to use ('local' or
            'openai'), default 'local'
        openai_args: Dict of additional args to pass in to the
            `openai.Embedding.create` function, default None

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''
    assert embedding_model_type in [
        'local', 'openai'
    ], ('Unsupported embedding model type. '
        'The supported ones are ["local", "openai"]')

    if len(generated_outputs) != len(reference_outputs):
        raise ValueError(
            'The generated and reference outputs lists must be of the same '
            'length')
    if len(generated_outputs) == 0:
        return EvalValue(metric_name='semantic_sim',
                         prompts=None,
                         generated_outputs=[],
                         reference_outputs=[],
                         sources=None,
                         metric_values=[],
                         language='en')

    if embedding_model_type == 'local':
        # The 'all-mpnet-base-v2' model has the highest average performance out
        # of all the existing sentence-transformer models that have been
        # evaluated.
        # Ref: https://www.sbert.net/docs/pretrained_models.html#model-overview
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        generated_embeddings = model.encode(generated_outputs)
        reference_embeddings = model.encode(reference_outputs)
    else:  # openai
        if openai_args is None:
            gen_embed_response = openai.Embedding.create(
                input=generated_outputs, model='text-embedding-ada-002')
            ref_embed_response = openai.Embedding.create(
                input=reference_outputs, model='text-embedding-ada-002')
        else:
            gen_embed_response = openai.Embedding.create(
                input=generated_outputs, **openai_args)
            ref_embed_response = openai.Embedding.create(
                input=reference_outputs, **openai_args)
        generated_embeddings = [
            item['embedding'] for item in gen_embed_response['data']
        ]
        reference_embeddings = [
            item['embedding'] for item in ref_embed_response['data']
        ]

    cosine_scores = util.pairwise_cos_sim(generated_embeddings,
                                          reference_embeddings)
    # Numerical instability can cause the dot product of almost identical
    # vectors to exceed 1.0 slightly, so we clip the outputs
    cosine_scores = torch.clamp(cosine_scores, -1.0, 1.0)

    return EvalValue(metric_name='semantic_sim',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     sources=None,
                     metric_values=cosine_scores.tolist(),
                     language='en')


def rouge1(generated_outputs: List[str],
           reference_outputs: List[str]) -> EvalValue[float]:
    '''Calculates the F1 metrics of the ROUGE-1 scores between the generated
    outputs and the reference outputs. It evaluates the overlap of unigrams
    (single tokens) between the generated outputs and the reference outputs.
    This metric takes on float values between [0, 1], where 0 is no overlap and
    1 is complete overlap.

    Ref:
        https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''
    scores = _rouge(generated_outputs, reference_outputs, 'rouge1')
    return EvalValue(metric_name='rouge1',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     sources=None,
                     metric_values=scores,
                     language='en')


def rouge2(generated_outputs: List[str],
           reference_outputs: List[str]) -> EvalValue[float]:
    '''Calculates the F1 metrics of the ROUGE-2 scores between the generated
    outputs and the reference outputs. It evaluates the overlap of bigrams
    (two adjacent tokens) between the generated outputs and the reference
    outputs. This metric takes on float values between [0, 1], where 0 is no
    overlap and 1 is complete overlap.

    Ref:
        https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''
    scores = _rouge(generated_outputs, reference_outputs, 'rouge2')
    return EvalValue(metric_name='rouge2',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     sources=None,
                     metric_values=scores,
                     language='en')


def rougeL(generated_outputs: List[str],
           reference_outputs: List[str]) -> EvalValue[float]:
    '''Calculates the F1 metrics of the ROUGE-L scores between the generated
    outputs and the reference outputs. It evaluates the longest common
    subsequence (LCS) between the generated outputs and the reference outputs.
    This metric takes on float values between [0, 1], where 0 means that the LCS
    is empty and 1 means that the reference and generated outputs are the same.

    Ref:
        https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An :class:`~langcheck.eval.eval_value.EvalValue` object
    '''
    # The `rouge_score` package has two flavors of ROUGE-L [1]:
    # - 1) sentence-level, where newline characters are ignored
    # - 2) summary-level, where newline characters are interpreted as sentence
    #      boundaries
    #
    # We use (2) here (i.e. `rougeLsum`) because this is how `pyrouge` computes
    # the ROUGE-L score (https://github.com/bheinzerling/pyrouge), which is a
    # Python wrapper around original perl script implementation.
    #
    # [1] https://github.com/google-research/google-research/tree/master/rouge#two-flavors-of-rouge-l # NOQA E501
    scores = _rouge(generated_outputs, reference_outputs, 'rougeLsum')
    return EvalValue(metric_name='rougeL',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     sources=None,
                     metric_values=scores,
                     language='en')


def _rouge(generated_outputs: List[str], reference_outputs: List[str],
           rouge_type: str) -> List[float]:
    '''Helper function for computing the rouge1, rouge2, and rougeL metrics.
    This uses Google Research's implementation of ROUGE:
    https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs
        rouge_type: rouge1, rouge2, or rougeLsum

    Returns:
        A list of F1 values of the ROUGE scores
    '''
    assert rouge_type in ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = []
    for gen, ref in zip(generated_outputs, reference_outputs):
        score = scorer.score(gen, ref)
        scores.append(score[rouge_type].fmeasure)
    return scores
