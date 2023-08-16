from sentence_transformers import SentenceTransformer, util


# Compute semantic similarity
def semantic_sim(generated_outputs, reference_outputs):
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
        similarities.append(sim)

    return similarities


# Example usage
generated_outputs = [
    "The cat sat on the mat.", "Dogs are friendly.",
    "I went to the park today with my dog"
]
reference_outputs = [
    "A cat is on a mat.", "Dogs are usually amiable.",
    "I work at an AI startup in Tokyo"
]
print(semantic_sim(generated_outputs, reference_outputs))
