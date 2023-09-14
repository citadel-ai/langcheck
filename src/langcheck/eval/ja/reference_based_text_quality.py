from typing import List, Optional

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from langcheck.eval.eval_value import EvalValue
from langcheck.eval.ja._tokenizers import JanomeTokenizer


def rouge1(generated_outputs: List[str],
           reference_outputs: List[str],
           *,
           tokenizer: Optional[Tokenizer] = None) -> EvalValue[float]:
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
        An EvalValue object
    '''
    scores = _rouge(generated_outputs,
                    reference_outputs,
                    'rouge1',
                    tokenizer=tokenizer)
    return EvalValue(metric_name='rouge1',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     metric_values=scores,
                     language='ja')


def rouge2(generated_outputs: List[str],
           reference_outputs: List[str],
           *,
           tokenizer: Optional[Tokenizer] = None) -> EvalValue[float]:
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
        An EvalValue object
    '''
    scores = _rouge(generated_outputs,
                    reference_outputs,
                    'rouge2',
                    tokenizer=tokenizer)
    return EvalValue(metric_name='rouge2',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
                     metric_values=scores,
                     language='ja')


def rougeL(generated_outputs: List[str],
           reference_outputs: List[str],
           *,
           tokenizer: Optional[Tokenizer] = None) -> EvalValue[float]:
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
        An EvalValue object
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
    # [1] https://github.com/google-research/google-research/tree/master/rouge#two-flavors-of-rouge-l
    scores = _rouge(generated_outputs,
                    reference_outputs,
                    'rougeLsum',
                    tokenizer=tokenizer)
    return EvalValue(metric_name='rougeL',
                     prompts=None,
                     generated_outputs=generated_outputs,
                     reference_outputs=reference_outputs,
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
    assert rouge_type in ["rouge1", "rouge2", "rougeLsum"]
    tokenizer = tokenizer or JanomeTokenizer()
    scorer = rouge_scorer.RougeScorer([rouge_type],
                                      use_stemmer=True,
                                      tokenizer=tokenizer)
    scores = []
    for gen, ref in zip(generated_outputs, reference_outputs):
        score = scorer.score(gen, ref)
        scores.append(score[rouge_type].fmeasure)
    return scores
