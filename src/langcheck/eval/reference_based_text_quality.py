from typing import List

from sentence_transformers import SentenceTransformer, util

from langcheck.eval.eval_value import EvalValue


def semantic_sim(generated_outputs: List[str],
                 reference_outputs: List[str]) -> EvalValue:
    '''Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings.

    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An EvalValue object
    '''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    generated_embeddings = model.encode(generated_outputs)
    reference_embeddings = model.encode(reference_outputs)
    cosine_scores = util.pairwise_cos_sim(generated_embeddings,
                                          reference_embeddings)

    return EvalValue(metric_name='semantic_sim',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     metric_values=cosine_scores.tolist())
