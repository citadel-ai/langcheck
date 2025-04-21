import os
from unittest.mock import Mock, patch

import pytest
from openai.types import CreateEmbeddingResponse

from langcheck.metrics.de import rouge1, rouge2, rougeL, semantic_similarity
from langcheck.metrics.eval_clients import (
    AzureOpenAIEvalClient,
    OpenAIEvalClient,
)
from tests.utils import is_close

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die Katze saß auf der Matte.", "Die Katze saß auf der Matte."),
        (["Die Katze saß auf der Matte."], ["Die Katze saß auf der Matte."]),
    ],
)
def test_semantic_similarity_identical(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(
        generated_outputs, reference_outputs, eval_model="local"
    )
    assert 0.99 <= metric_value <= 1


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die KATZE saß auf der MATTE.", "Die Katze saß auf der Matte."),
        (["Die KATZE saß auf der MATTE."], ["Die Katze saß auf der Matte."]),
    ],
)
def test_semantic_similarity_case_sensitivity(
    generated_outputs, reference_outputs
):
    # nb: the German model is case-sensitive!
    metric_value = semantic_similarity(
        generated_outputs, reference_outputs, eval_model="local"
    )
    assert 0.6 <= metric_value <= 0.7


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die Katze saß auf der Matte.", "Ich esse gerne Eiscreme."),
        (["Die Katze saß auf der Matte."], ["Ich esse gerne Eiscreme."]),
    ],
)
def test_semantic_similarity_not_similar(generated_outputs, reference_outputs):
    metric_value = semantic_similarity(
        generated_outputs, reference_outputs, eval_model="local"
    )
    print(metric_value)
    assert 0.0 <= metric_value <= 0.1


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die Katze saß auf der Matte.", "Die Katze saß auf der Matte."),
        (["Die Katze saß auf der Matte."], ["Die Katze saß auf der Matte."]),
    ],
)
def test_semantic_similarity_openai(generated_outputs, reference_outputs):
    mock_embedding_response = Mock(spec=CreateEmbeddingResponse)
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Calling the openai.resources.Embeddings.create method requires an OpenAI
    # API key, so we mock the return value instead
    with patch(
        "openai.resources.Embeddings.create",
        Mock(return_value=mock_embedding_response),
    ):
        # Set the necessary env vars for the 'openai' embedding model type
        os.environ["OPENAI_API_KEY"] = "dummy_key"
        openai_client = OpenAIEvalClient()
        metric_value = semantic_similarity(
            generated_outputs, reference_outputs, eval_model=openai_client
        )
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= metric_value <= 1

        # Set the necessary env vars for the 'azure_openai' model type
        os.environ["AZURE_OPENAI_API_KEY"] = "dummy_azure_key"
        os.environ["OPENAI_API_VERSION"] = "dummy_version"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "dummy_endpoint"
        azure_openai_client = AzureOpenAIEvalClient(
            text_model_name="foo",
            embedding_model_name="bar",
        )
        metric_value = semantic_similarity(
            generated_outputs, reference_outputs, eval_model=azure_openai_client
        )
        # Since the mock embeddings are the same for the generated and reference
        # outputs, the semantic similarity should be 1.
        assert 0.99 <= metric_value <= 1


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die Katze saß auf der Matte.", "Die Katze saß auf der Matte."),
        (["Die Katze saß auf der Matte."], ["Die Katze saß auf der Matte."]),
    ],
)
def test_rouge_identical(generated_outputs, reference_outputs):
    rouge1_metric_value = rouge1(generated_outputs, reference_outputs)
    rouge2_metric_value = rouge2(generated_outputs, reference_outputs)
    rougeL_metric_value = rougeL(generated_outputs, reference_outputs)

    # All ROUGE scores are 1 if the generated and reference outputs are
    # identical
    assert rouge1_metric_value == 1
    assert rouge2_metric_value == 1
    assert rougeL_metric_value == 1


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die Katze saß auf der Matte", "Ich esse gerne Eiscreme."),
        (["Die Katze saß auf der Matte"], ["Ich esse gerne Eiscreme."]),
    ],
)
def test_rouge_no_overlap(generated_outputs, reference_outputs):
    rouge1_metric_value = rouge1(generated_outputs, reference_outputs)
    rouge2_metric_value = rouge2(generated_outputs, reference_outputs)
    rougeL_metric_value = rougeL(generated_outputs, reference_outputs)

    # rouge1 and rougeL >0 if there is a trailing period in both sentences

    # All ROUGE scores are 0 if the generated and reference outputs have no
    # overlapping words
    assert rouge1_metric_value == 0
    assert rouge2_metric_value == 0
    assert rougeL_metric_value == 0


@pytest.mark.parametrize(
    "generated_outputs,reference_outputs",
    [
        ("Die Katze sitzt auf der Matte.", "Die Katze saß auf der Matte."),
        (["Die Katze sitzt auf der Matte."], ["Die Katze saß auf der Matte."]),
    ],
)
def test_rouge_some_overlap(generated_outputs, reference_outputs):
    rouge1_metric_value = rouge1(generated_outputs, reference_outputs)
    rouge2_metric_value = rouge2(generated_outputs, reference_outputs)
    rougeL_metric_value = rougeL(generated_outputs, reference_outputs)

    # The ROUGE-2 score is lower than the ROUGE-1 and ROUGE-L scores
    assert is_close(rouge1_metric_value.metric_values, [0.8571428571428571])
    assert is_close(rouge2_metric_value.metric_values, [0.6666666666666666])
    assert is_close(rougeL_metric_value.metric_values, [0.8571428571428571])
