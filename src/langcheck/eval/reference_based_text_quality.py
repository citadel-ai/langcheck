from typing import List

import torch
from sentence_transformers import SentenceTransformer, util

from langcheck.eval.eval_value import EvalValue


def semantic_sim(generated_outputs: List[str],
                 reference_outputs: List[str]) -> EvalValue[float]:
    '''Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings. This metric
	takes on float values between [-1, 1], but typically ranges between 0 and 1
    where 0 is minimum similariy and 1 is maximum similarity.

    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An EvalValue object
    '''
    if len(generated_outputs) != len(reference_outputs):
        raise ValueError(
            'The generated and reference outputs lists must be of the same length'
        )
    if len(generated_outputs) == 0:
        return EvalValue(metric_name='semantic_sim',
                         prompts=None,
                         generated_outputs=[],
                         reference_outputs=[],
                         metric_values=[])
    # The 'all-mpnet-base-v2' model has the highest average performance out of
    # all the existing sentence-transformer models that have been evaluated.
    # Ref: https://www.sbert.net/docs/pretrained_models.html#model-overview
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    generated_embeddings = model.encode(generated_outputs)
    reference_embeddings = model.encode(reference_outputs)
    cosine_scores = util.pairwise_cos_sim(generated_embeddings,
                                          reference_embeddings)
    # Numerical instability can cause the dot product of almost identical
    # vectors to exceed 1.0 slightly, so we clip the outputs
    cosine_scores = torch.clamp(cosine_scores, -1.0, 1.0)

    return EvalValue(metric_name='semantic_sim',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     metric_values=cosine_scores.tolist())
