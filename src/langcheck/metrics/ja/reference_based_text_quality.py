from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from langcheck.metrics._validation import validate_parameters_reference_based
from langcheck.metrics.ja._tokenizers import JanomeTokenizer
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.scorer.hf_models import \
    SentenceTransformerSimilarityScorer
from langcheck.metrics.scorer.openai_models import OpenAISimilarityScorer
from langcheck.utils.progess_bar import tqdm_wrapper


def semantic_similarity(
        generated_outputs: List[str] | str,
        reference_outputs: List[str] | str,
        prompts: Optional[List[str] | str] = None,
        model_type: str = 'local',
        openai_client: Optional[OpenAI] = None,
        openai_args: Optional[Dict[str, str]] = None) -> MetricValue[float]:
    '''Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings. This metric
    takes on float values between [-1, 1], but typically ranges between 0 and 1
    where 0 is minimum similarity and 1 is maximum similarity.

    We currently support three embedding model types:

    1. The 'local' type, where the 'paraphrase-multilingual-mpnet-base-v2' model
    is downloaded from HuggingFace and run locally. This is the default model
    type and there is no setup needed to run this.

    2. The 'openai' type, where we use OpenAI's 'text-embedding-3-small' model
    by default (this is configurable). See
    `this page <https://langcheck.readthedocs.io/en/latest/metrics.html
    #computing-metrics-with-openai-models>`__
    for examples on setting up the OpenAI API key.

    3. The 'azure_openai' type. Essentially the same as the 'openai' type,
    except that it uses the AzureOpenAI client. Note that you must specify your
    model deployment to use in ``openai_args``, e.g.
    ``openai_args={'model': 'YOUR_DEPLOYMENT_NAME'}``

    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        https://openai.com/blog/new-embedding-models-and-api-updates

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
    '''
    generated_outputs, reference_outputs, prompts = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)
    assert model_type in [
        'local', 'openai', 'azure_openai'
    ], ('Unsupported embedding model type. '
        'The supported ones are ["local", "openai", "azure_openai"]')
    if model_type == 'local':
        scorer = SentenceTransformerSimilarityScorer(language='ja')
    else:  # openai or azure_openai
        scorer = OpenAISimilarityScorer(model_type=model_type,
                                        openai_client=openai_client,
                                        openai_args=openai_args)
    scores = scorer.score(generated_outputs, reference_outputs)

    return MetricValue(metric_name='semantic_similarity',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=reference_outputs,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='ja')


def rouge1(generated_outputs: List[str] | str,
           reference_outputs: List[str] | str,
           prompts: Optional[List[str] | str] = None,
           *,
           tokenizer: Optional[Tokenizer] = None) -> MetricValue[float]:
    '''Calculates the F1 metrics of the ROUGE-1 scores between the generated
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
        An MetricValue object
    '''
    generated_outputs, reference_outputs, prompts = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)

    scores = _rouge(generated_outputs,
                    reference_outputs,
                    'rouge1',
                    tokenizer=tokenizer)
    return MetricValue(metric_name='rouge1',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=reference_outputs,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='ja')


def rouge2(generated_outputs: List[str] | str,
           reference_outputs: List[str] | str,
           prompts: Optional[List[str] | str] = None,
           *,
           tokenizer: Optional[Tokenizer] = None) -> MetricValue[float]:
    '''Calculates the F1 metrics of the ROUGE-2 scores between the generated
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
        An MetricValue object
    '''
    generated_outputs, reference_outputs, prompts = validate_parameters_reference_based(  # NOQA: E501
        generated_outputs, reference_outputs, prompts)

    scores = _rouge(generated_outputs,
                    reference_outputs,
                    'rouge2',
                    tokenizer=tokenizer)
    return MetricValue(metric_name='rouge2',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=reference_outputs,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='ja')


def rougeL(generated_outputs: List[str] | str,
           reference_outputs: List[str] | str,
           prompts: Optional[List[str] | str] = None,
           *,
           tokenizer: Optional[Tokenizer] = None) -> MetricValue[float]:
    '''Calculates the F1 metrics of the ROUGE-L scores between the generated
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
        An MetricValue object
    '''
    generated_outputs, reference_outputs, prompts = validate_parameters_reference_based(  # NOQA: E501
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
    scores = _rouge(generated_outputs,
                    reference_outputs,
                    'rougeLsum',
                    tokenizer=tokenizer)
    return MetricValue(metric_name='rougeL',
                       prompts=prompts,
                       generated_outputs=generated_outputs,
                       reference_outputs=reference_outputs,
                       sources=None,
                       explanations=None,
                       metric_values=scores,
                       language='ja')


def _rouge(generated_outputs: List[str],
           reference_outputs: List[str],
           rouge_type: str,
           *,
           tokenizer: Optional[Tokenizer] = None) -> List[float]:
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
    assert rouge_type in ['rouge1', 'rouge2', 'rougeLsum']

    # The tokenizer is default to JanomeTokenizer
    tokenizer = tokenizer or JanomeTokenizer()
    scorer = rouge_scorer.RougeScorer([rouge_type],
                                      use_stemmer=True,
                                      tokenizer=tokenizer)
    scores = []
    for gen, ref in tqdm_wrapper(zip(generated_outputs, reference_outputs),
                                 total=len(generated_outputs)):
        score = scorer.score(gen, ref)
        scores.append(score[rouge_type].fmeasure)
    return scores
