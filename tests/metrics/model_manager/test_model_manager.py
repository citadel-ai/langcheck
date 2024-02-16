from unittest.mock import MagicMock, patch

import pytest
import requests
from omegaconf import OmegaConf

from langcheck.metrics.model_manager._model_management import (
    ModelManager, check_model_availability)


@pytest.fixture
def temp_config_path(tmp_path):
    config = """
    zh:
      toxicity:
        model_name: "Alnusjaponica/toxicity-score-multi-classification"
        tokenizer_name: "line-corporation/line-distilbert-base-japanese"
        loader_func: "load_auto_model_for_text_classification"
    """
    config_path = tmp_path / "metric_config.yaml"
    config_path.write_text(config)
    return str(config_path)


@pytest.fixture
def mock_model_manager(temp_config_path):
    with patch("os.path.join", return_value=temp_config_path), \
         patch('langcheck.metrics.model_manager._model_management.check_model_availability',  # NOQA:E501
               return_value=True):
        model_manager = ModelManager()
        return model_manager


@pytest.mark.parametrize(
    "model_name,revision, status_code",
    [("bert-base-uncased", "", "200"), ("bert-base-uncased", None, "200"),
     ("bert-base-uncased", "main", "200"),
     ("bert-base-uncased", "a265f77", "200"),
     ("bert-base-uncased", "a265f773a47193eed794233aa2a0f0bb6d3eaa63", "200"),
     pytest.param(
         "bert-base-uncased", "a265f78", "404", marks=pytest.mark.xfail),
     pytest.param("", "0e9f4", "404", marks=pytest.mark.xfail),
     pytest.param("terb-base-uncased", "", "404", marks=pytest.mark.xfail)],
)
@patch("requests.get")
def test_check_model_availability(mock_get, model_name, revision, status_code):
    mock_get.return_value.status_code = status_code
    available = check_model_availability(model_name, revision)
    assert available is (status_code == requests.codes.OK)


def test_model_manager_initiation(mock_model_manager):
    mock_config = mock_model_manager.config
    assert "toxicity" in mock_config["zh"]
    assert mock_config["zh"]["toxicity"]["model_name"] ==\
        "Alnusjaponica/toxicity-score-multi-classification"
    assert mock_config["zh"]["toxicity"]["tokenizer_name"] == \
        "line-corporation/line-distilbert-base-japanese"
    assert mock_config["zh"]["toxicity"]["loader_func"] == \
        "load_auto_model_for_text_classification"


def test_model_manager_fetch_model(mock_model_manager):
    with patch.dict(
        'langcheck.metrics.model_manager._model_management.LOADER_MAP',
        {'load_auto_model_for_text_classification': MagicMock()}):
        model = mock_model_manager.fetch_model(language='zh', metric='toxicity')
        assert model is not None
