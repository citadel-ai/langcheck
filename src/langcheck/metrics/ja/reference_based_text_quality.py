from __future__ import annotations

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from langcheck.metrics.eval_clients import EvalClient
from langcheck.metrics.ja._tokenizers import JanomeTokenizer
from langcheck.metrics.metric_inputs import (
    get_metric_inputs,
    get_metric_inputs_with_required_lists,
)
from langcheck.metrics.metric_value import MetricValue
from langcheck.metrics.scorer.hf_models import (
    SentenceTransformerSimilarityScorer,
)
from langcheck.utils.progress_bar import tqdm_wrapper

LANG = "ja"


def answer_correctness(
    generated_outputs: list[str] | str,
    reference_outputs: list[str] | str,
    prompts: list[str] | str,
    eval_model: EvalClient,
) -> MetricValue[float | None]:
    """Calculates the correctness of the generated outputs. This metric takes on
    float values of either 0.0 (Incorrect), 0.5 (Partially Correct), or 1.0
    (Correct). The score may also be `None` if it could not be computed.

    We currently only support the evaluation based on an EvalClient.

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s)
        eval_model: The EvalClient instance used for the evaluation

    Returns:
        A :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        reference_outputs=reference_outputs,
        prompts=prompts,
        required_params=["generated_outputs", "reference_outputs", "prompts"],
    )
    metric_name = "answer_correctness"

    answer_correctness_template = eval_model.load_prompt_template(
        language=LANG, metric_name=metric_name
    )

    return eval_model.compute_metric_values_from_template(
        metric_inputs=metric_inputs,
        template=answer_correctness_template,
        metric_name=metric_name,
        language=LANG,
        score_map={"Correct": 1.0, "Partially Correct": 0.5, "Incorrect": 0.0},
    )


def semantic_similarity(
    generated_outputs: list[str] | str,
    reference_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    eval_model: str | EvalClient = "local",
) -> MetricValue[float]:
    """Calculates the semantic similarities between the generated outputs and
    the reference outputs. The similarities are computed as the cosine
    similarities between the generated and reference embeddings. This metric
    takes on float values between [-1, 1], but typically ranges between 0 and 1
    where 0 is minimum similarity and 1 is maximum similarity.

    We currently support two embedding model types:

    1. The 'local' type, where the 'paraphrase-multilingual-mpnet-base-v2' model
    is downloaded from HuggingFace and run locally. This is the default model
    type and there is no setup needed to run this.

    2. The EvalClient type, where you can use a similarlity scorer returned by
    the given EvalClient. The scorer is typically implemented using the
    embedding APIs of cloud services. The implementation details are explained
    in each of the concrete EvalClient classes.


    Ref:
        https://huggingface.co/tasks/sentence-similarity
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html

    Args:
        generated_outputs: The model generated output(s) to evaluate
        reference_outputs: The reference output(s)
        prompts: The prompts used to generate the output(s). Prompts are
            optional metadata and not used to calculate the metric.
        eval_model: The type of model to use ('local' or the EvalClient instance
            used for the evaluation). default 'local'

    Returns:
        An :class:`~langcheck.metrics.metric_value.MetricValue` object
    """
    metric_inputs, [generated_outputs, reference_outputs] = (
        get_metric_inputs_with_required_lists(
            generated_outputs=generated_outputs,
            reference_outputs=reference_outputs,
            prompts=prompts,
            required_params=["generated_outputs", "reference_outputs"],
        )
    )

    if eval_model == "local":
        scorer = SentenceTransformerSimilarityScorer(language="ja")
    else:  # EvalClient
        assert isinstance(
            eval_model, EvalClient
        ), "An EvalClient must be provided for non-local model types."
        scorer = eval_model.similarity_scorer()

    scores = scorer.score(generated_outputs, reference_outputs)

    return MetricValue(
        metric_name="semantic_similarity",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,
        language="ja",
    )


def rouge1(
    generated_outputs: list[str] | str,
    reference_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    *,
    tokenizer: Tokenizer | None = None,
) -> MetricValue[float]:
    """Calculates the F1 metrics of the ROUGE-1 scores between the generated
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
    """
    metric_inputs, [generated_outputs, reference_outputs] = (
        get_metric_inputs_with_required_lists(
            generated_outputs=generated_outputs,
            reference_outputs=reference_outputs,
            prompts=prompts,
            required_params=["generated_outputs", "reference_outputs"],
        )
    )

    scores = _rouge(
        generated_outputs, reference_outputs, "rouge1", tokenizer=tokenizer
    )

    return MetricValue(
        metric_name="rouge1",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,
        language="ja",
    )


def rouge2(
    generated_outputs: list[str] | str,
    reference_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    *,
    tokenizer: Tokenizer | None = None,
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
        An MetricValue object
    """
    metric_inputs, [generated_outputs, reference_outputs] = (
        get_metric_inputs_with_required_lists(
            generated_outputs=generated_outputs,
            reference_outputs=reference_outputs,
            prompts=prompts,
            required_params=["generated_outputs", "reference_outputs"],
        )
    )

    scores = _rouge(
        generated_outputs, reference_outputs, "rouge2", tokenizer=tokenizer
    )
    return MetricValue(
        metric_name="rouge2",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,
        language="ja",
    )


def rougeL(
    generated_outputs: list[str] | str,
    reference_outputs: list[str] | str,
    prompts: list[str] | str | None = None,
    *,
    tokenizer: Tokenizer | None = None,
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
        An MetricValue object
    """
    metric_inputs, [generated_outputs, reference_outputs] = (
        get_metric_inputs_with_required_lists(
            generated_outputs=generated_outputs,
            reference_outputs=reference_outputs,
            prompts=prompts,
            required_params=["generated_outputs", "reference_outputs"],
        )
    )

    # The `rouge_score` package has two flavors of ROUGE-L [1]:
    # - 1) sentence-level, where newline characters are ignored
    # - 2) summary-level, where newline characters are interpreted as sentence
    #      boundaries
    #
    # We use (2) here (i.e. `rougeLsum`) because this is how `pyrouge` computes
    # the ROUGE-L score (https://github.com/bheinzerling/pyrouge), which is a
    # Python wrapper around original perl script implementation.
    #
    # [1] https://github.com/google-research/google-research/tree/master/rouge#two-flavors-of-rouge-l
    scores = _rouge(
        generated_outputs, reference_outputs, "rougeLsum", tokenizer=tokenizer
    )
    return MetricValue(
        metric_name="rougeL",
        metric_inputs=metric_inputs,
        explanations=None,
        metric_values=scores,
        language="ja",
    )


def _rouge(
    generated_outputs: list[str],
    reference_outputs: list[str],
    rouge_type: str,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[float]:
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

    # The tokenizer is default to JanomeTokenizer
    tokenizer = tokenizer or JanomeTokenizer()
    scorer = rouge_scorer.RougeScorer(
        [rouge_type], use_stemmer=True, tokenizer=tokenizer
    )
    scores = []
    for gen, ref in tqdm_wrapper(
        zip(generated_outputs, reference_outputs), total=len(generated_outputs)
    ):
        score = scorer.score(gen, ref)
        scores.append(score[rouge_type].fmeasure)
    return scores
