from unittest.mock import MagicMock, patch

import pytest
import requests

from langcheck.metrics.model_manager import _model_management
from langcheck.metrics.model_manager._model_management import (
    ModelManager,
    check_model_availability,
)


@pytest.fixture
def temp_config_path(tmp_path) -> str:
    """
    Fixture that creates a temporary configuration file for testing.

    Args:
        tmp_path: A unique temporary directory path provided by pytest.

    Returns:
        The path to the temporary configuration file.
    """
    config = """
    zh:
        toxicity:
            model_name: alibaba-pai/pai-bert-base-zh-llm-risk-detection
            loader_func: load_auto_model_for_text_classification
    ja:
        toxicity:
            model_name: Alnusjaponica/toxicity-score-multi-classification
            model_revision: bc7a465029744889c8252ee858ab04ab9efdb0e7
            tokenizer_name: line-corporation/line-distilbert-base-japanese
            tokenizer_revision: 93bd4811608eecb95ffaaba957646efd9a909cc8
            loader_func: load_auto_model_for_text_classification
    """
    config_path = tmp_path / "metric_config.yaml"
    config_path.write_text(config)
    return str(config_path)


@pytest.fixture
def mock_model_manager(temp_config_path):
    """
    Fixture that creates a mock ModelManager for testing.

    The ModelManager is patched to use the temporary configuration file
    created by the temp_config_path fixture, and to always return True
    when checking model availability.

    Args:
        temp_config_path: The path to the temporary configuration file.

    Returns:
        The mock ModelManager.
    """
    with (
        patch("os.path.join", return_value=temp_config_path),
        patch(
            "langcheck.metrics.model_manager._model_management.check_model_availability",  # NOQA:E501
            return_value=True,
        ),
        patch.object(_model_management, "VALID_LANGUAGE", ["ja", "zh"]),
    ):
        model_manager = ModelManager()
        return model_manager


@pytest.mark.parametrize(
    "model_name,revision, status_code",
    [
        ("bert-base-uncased", "", "200"),
        ("bert-base-uncased", None, "200"),
        ("bert-base-uncased", "main", "200"),
        ("bert-base-uncased", "a265f77", "200"),
        (
            "bert-base-uncased",
            "a265f773a47193eed794233aa2a0f0bb6d3eaa63",
            "200",
        ),
        pytest.param(
            "bert-base-uncased", "a265f78", "404", marks=pytest.mark.xfail
        ),
        pytest.param("", "0e9f4", "404", marks=pytest.mark.xfail),
        pytest.param("terb-base-uncased", "", "404", marks=pytest.mark.xfail),
    ],
)
@patch("requests.get")
def test_check_model_availability(mock_get, model_name, revision, status_code):
    mock_get.return_value.status_code = status_code
    available = check_model_availability(model_name, revision)
    assert available is (status_code == requests.codes.OK)


def test_model_manager_initiation(mock_model_manager):
    mock_config = mock_model_manager.config
    assert "toxicity" in mock_config["zh"]
    assert (
        mock_config["zh"]["toxicity"]["model_name"]
        == "alibaba-pai/pai-bert-base-zh-llm-risk-detection"
    )
    assert (
        mock_config["zh"]["toxicity"]["loader_func"]
        == "load_auto_model_for_text_classification"
    )

    assert "toxicity" in mock_config["ja"]
    assert (
        mock_config["ja"]["toxicity"]["model_name"]
        == "Alnusjaponica/toxicity-score-multi-classification"
    )
    assert (
        mock_config["ja"]["toxicity"]["model_revision"]
        == "bc7a465029744889c8252ee858ab04ab9efdb0e7"
    )
    assert (
        mock_config["ja"]["toxicity"]["tokenizer_name"]
        == "line-corporation/line-distilbert-base-japanese"
    )
    assert (
        mock_config["ja"]["toxicity"]["tokenizer_revision"]
        == "93bd4811608eecb95ffaaba957646efd9a909cc8"
    )
    assert (
        mock_config["ja"]["toxicity"]["loader_func"]
        == "load_auto_model_for_text_classification"
    )


def test_model_manager_fetch_model(mock_model_manager):
    with patch.dict(
        "langcheck.metrics.model_manager._model_management.LOADER_MAP",
        {"load_auto_model_for_text_classification": MagicMock()},
    ):
        model = mock_model_manager.fetch_model(language="zh", metric="toxicity")
        assert model is not None
        model = mock_model_manager.fetch_model(language="ja", metric="toxicity")
        assert model is not None
