from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from openai import OpenAI
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from langcheck.metrics._validation import validate_parameters_reference_based
from langcheck.metrics.de._tokenizers import DeTokenizer
from langcheck.metrics.en.reference_based_text_quality import \
    semantic_similarity as en_semantic_similarity
from langcheck.metrics.metric_value import MetricValue
from langcheck.utils.progess_bar import tqdm_wrapper

LANG = "de"


def semantic_similarity(
    generated_outputs: List[str] | str,
    reference_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
    model_type: str = "local",
    openai_client: Optional[OpenAI] = None,
    openai_args: Optional[Dict[str, Any]] = None,
) -> MetricValue[float]:
    """Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings. This metric
    takes on float values between [-1, 1], but typically ranges between 0 and 1
    where 0 is minimum similarity and 1 is maximum similarity. (NOTE: when using
    OpenAI embeddings, the cosine similarities tend to be skewed quite heavily
    towards higher numbers.)

    We currently support three embedding model types:

    1. The 'local' type, where the MODEL_NAME model is downloaded
    from HuggingFace and run locally. This is the default model type and
    there is no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'text-embedding-ada-002' model
    by default (this is configurable). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    on setting up the OpenAI API key.

    3. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        https://openai.com/blog/new-and-improved-embedding-model

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        model_type: The type of embedding model to use ('local', 'openai', or
            'azure_openai'), default 'local'
        openai_client: OpenAI or AzureOpenAI client, default None. If this is
            None but ``model_type`` is 'openai' or 'azure_openai', we will
            attempt to create a default client.
        openai_args: Dict of additional args to pass in to the
            ``client.embeddings.create`` function, default None

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    (
        generated_outputs,
        reference_outputs,
        prompts,
    ) = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)
    assert model_type in [
        "local", "openai", "azure_openai"
    ], ("Unsupported embedding model type. "
        'The supported ones are ["local", "openai", "azure_openai"]')

    if model_type == "openai" or model_type == "azure_openai":
        # We can use the same API as english semantic_similarity to compare the
        # similarity
        metric_value = en_semantic_similarity(
            generated_outputs,
            reference_outputs,
            prompts,
            model_type,
            openai_client,
            openai_args,
        )
        metric_value.language = LANG
        return metric_value

    # we're using 'local' now
    batch_size = 8
    # https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models
    # v1 supports only 15 languages (German included) but is stronger than v2
    # that supports 50+ languages
    # NOTE: it's cased! "Das ist ein Test." != "das ist ein test."
    model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v1")
    generated_embeddings = []
    reference_embeddings = []
    for i in tqdm_wrapper(
            range(0, len(generated_outputs), batch_size),
            total=(len(generated_outputs) + batch_size - 1) // batch_size,
            desc="Getting embeddings",
    ):
        batch_generated_outputs = generated_outputs[i:i + batch_size]
        batch_reference_outputs = reference_outputs[i:i + batch_size]
        batch_generated_embeddings = model.encode(batch_generated_outputs)
        batch_reference_embeddings = model.encode(batch_reference_outputs)
        generated_embeddings.extend(batch_generated_embeddings)
        reference_embeddings.extend(batch_reference_embeddings)

    scores = []
    with torch.no_grad():
        for i in tqdm_wrapper(
                range(0, len(generated_embeddings), batch_size),
                total=(len(generated_embeddings) + batch_size - 1) //
                batch_size,
                desc="Computing semantic similarity",
        ):
            batch_generated_embeddings = generated_embeddings[i:i + batch_size]
            batch_reference_embeddings = reference_embeddings[i:i + batch_size]

            cosine_scores = util.pairwise_cos_sim(
                torch.tensor(batch_generated_embeddings),
                torch.tensor(batch_reference_embeddings),
            )
            # Numerical instability
            # can cause the dot product of almost identical
            # vectors to exceed 1.0 slightly,
            # so we clip the outputs
            cosine_scores = torch.clamp(cosine_scores, -1.0, 1.0)
            scores.extend(cosine_scores.tolist())

    return MetricValue(
        metric_name="semantic_similarity",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=reference_outputs,
        sources=None,
        explanations=None,
        metric_values=scores,
        language=LANG,
    )


def rouge1(
    generated_outputs: List[str] | str,
    reference_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
) -> MetricValue[float]:
    """Calculates the F1 metrics of the ROUGE-1 scores between the generated
    outputs and the reference outputs. It evaluates the overlap of unigrams
    (single tokens) between the generated outputs and the reference outputs.
    This metric takes on float values between [0, 1], where 0 is no overlap and
    1 is complete overlap.

    Ref:
        https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    (
        generated_outputs,
        reference_outputs,
        prompts,
    ) = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)

    scores = _rouge(generated_outputs, reference_outputs, "rouge1")
    return MetricValue(
        metric_name="rouge1",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=reference_outputs,
        sources=None,
        explanations=None,
        metric_values=scores,
        language=LANG,
    )


def rouge2(
    generated_outputs: List[str] | str,
    reference_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
) -> MetricValue[float]:
    """Calculates the F1 metrics of the ROUGE-2 scores between the generated
    outputs and the reference outputs. It evaluates the overlap of bigrams
    (two adjacent tokens) between the generated outputs and the reference
    outputs. This metric takes on float values between [0, 1], where 0 is no
    overlap and 1 is complete overlap.

    Ref:
        https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    (
        generated_outputs,
        reference_outputs,
        prompts,
    ) = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)

    scores = _rouge(generated_outputs, reference_outputs, "rouge2")
    return MetricValue(
        metric_name="rouge2",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=reference_outputs,
        sources=None,
        explanations=None,
        metric_values=scores,
        language=LANG,
    )


def rougeL(
    generated_outputs: List[str] | str,
    reference_outputs: List[str] | str,
    prompts: Optional[List[str] | str] = None,
) -> MetricValue[float]:
    """Calculates the F1 metrics of the ROUGE-L scores between the generated
    outputs and the reference outputs. It evaluates the longest common
    subsequence (LCS) between the generated outputs and the reference outputs.
    This metric takes on float values between [0, 1], where 0 means that the LCS
    is empty and 1 means that the reference and generated outputs are the same.

    Ref:
        https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    (
        generated_outputs,
        reference_outputs,
        prompts,
    ) = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)

    # The `rouge_score` package has two flavors of ROUGE-L [1]:
    # - 1) sentence-level, where newline characters are ignored
    # - 2) summary-level, where newline characters are interpreted as sentence
    #      boundaries
    #
    # We use (2) here (i.e. `rougeLsum`) because this is how `pyrouge` computes
    # the ROUGE-L score (https://github.com/bheinzerling/pyrouge), which is a
    # Python wrapper around original perl script implementation.
    #
    # [1] https://github.com/google-research/google-research/tree/master/rouge#two-flavors-of-rouge-l # NOQA: E501
    scores = _rouge(generated_outputs, reference_outputs, "rougeLsum")
    return MetricValue(
        metric_name="rougeL",
        prompts=prompts,
        generated_outputs=generated_outputs,
        reference_outputs=reference_outputs,
        sources=None,
        explanations=None,
        metric_values=scores,
        language=LANG,
    )


def _rouge(generated_outputs: List[str], reference_outputs: List[str],
           rouge_type: str) -> List[float]:
    """Helper function for computing the rouge1, rouge2, and rougeL metrics.
    This uses Google Research's implementation of ROUGE:
    https://github.com/google-research/google-research/tree/master/rouge

    Args:
        generated_outputs: A list of model generated outputs to evaluate
        reference_outputs: A list of reference outputs
        rouge_type: rouge1, rouge2, or rougeLsum

    Returns:
        A list of F1 values of the ROUGE scores
    """
    assert rouge_type in ["rouge1", "rouge2", "rougeLsum"]

    tokenizer = DeTokenizer()
    scorer = rouge_scorer.RougeScorer([rouge_type],
                                      use_stemmer=True,
                                      tokenizer=tokenizer)
    scores = []
    for gen, ref in tqdm_wrapper(zip(generated_outputs, reference_outputs),
                                 total=len(generated_outputs)):
        score = scorer.score(gen, ref)
        scores.append(score[rouge_type].fmeasure)
    return scores
