from sentence_transformers import SentenceTransformer, util

from langcheck.eval.eval_value import EvalValue


def semantic_sim(generated_outputs, reference_outputs):
    '''Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings.

    Ref:
        https://huggingface.co/tasks/sentence-similarity

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs

    Returns:
        An EvalValue object
    '''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    generated_embeddings = [
        model.encode(sentence) for sentence in generated_outputs
    ]
    reference_embeddings = [
        model.encode(sentence) for sentence in reference_outputs
    ]

    similarities = []
    for gen_emb, ref_emb in zip(generated_embeddings, reference_embeddings):
        sim = util.pytorch_cos_sim(gen_emb, ref_emb)
        similarities.append(sim.item())

    return EvalValue(metric_name='semantic_sim',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     metric_values=similarities)
