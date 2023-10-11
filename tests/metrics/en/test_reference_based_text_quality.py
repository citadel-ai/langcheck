from unittest.mock import Mock, patch

import pytest

from langcheck.metrics.en import rouge1, rouge2, rougeL, semantic_sim
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The cat sat on the mat.", "The cat sat on the mat."),
     (["The cat sat on the mat."], ["The cat sat on the mat."])])
def test_semantic_sim_identical(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs,
                              reference_outputs,
                              embedding_model_type='local')
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.99 <= semantic_sim_value <= 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The CAT sat on the MAT.", "The cat sat on the mat."),
     (["The CAT sat on the MAT."], ["The cat sat on the mat."])])
def test_semantic_sim_case_sensitivity(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs,
                              reference_outputs,
                              embedding_model_type='local')
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.9 <= semantic_sim_value <= 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The cat sat on the mat.", "I like to eat ice cream."),
     (["The cat sat on the mat."], ["I like to eat ice cream."])])
def test_semantic_sim_not_similar(generated_outputs, reference_outputs):
    eval_value = semantic_sim(generated_outputs,
                              reference_outputs,
                              embedding_model_type='local')
    semantic_sim_value = eval_value.metric_values[0]
    assert 0.0 <= semantic_sim_value <= 0.1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The cat sat on the mat.", "The cat sat on the mat."),
     (["The cat sat on the mat."], ["The cat sat on the mat."])])
def test_semantic_sim_openai(generated_outputs, reference_outputs):
    mock_embedding_response = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
    # Calling the openai.Embedding.create method requires an OpenAI API key, so
    # we mock the return value instead
    with patch('openai.Embedding.create',
               Mock(return_value=mock_embedding_response)):
        eval_value = semantic_sim(generated_outputs,
                                  reference_outputs,
                                  embedding_model_type='openai')
        semantic_sim_value = eval_value.metric_values[0]
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= semantic_sim_value <= 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The cat sat on the mat.", "The cat sat on the mat."),
     (["The cat sat on the mat."], ["The cat sat on the mat."])])
def test_rouge_identical(generated_outputs, reference_outputs):
    rouge1_eval_value = rouge1(generated_outputs, reference_outputs)
    rouge2_eval_value = rouge2(generated_outputs, reference_outputs)
    rougeL_eval_value = rougeL(generated_outputs, reference_outputs)

    # All ROUGE scores are 1 if the generated and reference outputs are
    # identical
    assert rouge1_eval_value.metric_values[0] == 1
    assert rouge2_eval_value.metric_values[0] == 1
    assert rougeL_eval_value.metric_values[0] == 1


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The cat sat on the mat.", "I like to eat ice cream."),
     (["The cat sat on the mat."], ["I like to eat ice cream."])])
def test_rouge_no_overlap(generated_outputs, reference_outputs):
    rouge1_eval_value = rouge1(generated_outputs, reference_outputs)
    rouge2_eval_value = rouge2(generated_outputs, reference_outputs)
    rougeL_eval_value = rougeL(generated_outputs, reference_outputs)

    # All ROUGE scores are 0 if the generated and reference outputs have no
    # overlapping words
    assert rouge1_eval_value.metric_values[0] == 0
    assert rouge2_eval_value.metric_values[0] == 0
    assert rougeL_eval_value.metric_values[0] == 0


@pytest.mark.parametrize(
    'generated_outputs,reference_outputs',
    [("The cat is sitting on the mat.", "The cat sat on the mat."),
     (["The cat is sitting on the mat."], ["The cat sat on the mat."])])
def test_rouge_some_overlap(generated_outputs, reference_outputs):
    rouge1_eval_value = rouge1(generated_outputs, reference_outputs)
    rouge2_eval_value = rouge2(generated_outputs, reference_outputs)
    rougeL_eval_value = rougeL(generated_outputs, reference_outputs)

    # The ROUGE-2 score is lower than the ROUGE-1 and ROUGE-L scores
    assert is_close(rouge1_eval_value.metric_values, [0.7692307692307692])
    assert is_close(rouge2_eval_value.metric_values, [0.5454545454545454])
    assert is_close(rougeL_eval_value.metric_values, [0.7692307692307692])
