from unittest.mock import mock_open, patch
from langcheck.metrics._model_management import ModelConfig


def test_initialization_with_mock_file():
    try:
        mock_file_content = "[zh]\nsemantic_similarity=test_model\n"
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            config = ModelConfig()
            assert config.model_config['zh']['semantic_similarity'] == 'test_model'  # NOQA:E501
    except AssertionError as err:
        raise err


def test_list_metric_model_with_mock_file(capsys):
    try:
        mock_file_content = "[zh]\nsemantic_similarity=test_model\n"
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            config = ModelConfig()
            config.list_metric_model(language='zh',
                                     metric_type='semantic_similarity')
            captured = capsys.readouterr()  # type: ignore
            assert 'test_model' in captured.out
    except AssertionError as err:
        raise err


def test_set_model_for_metric_with_mock_file():
    try:
        mock_file_content = "[zh]\nsemantic_similarity=test_model\n"
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            config = ModelConfig()
            config.set_model_for_metric(model_name='another_test_model',
                                        language='zh',
                                        metric_type='semantic_similarity')
            assert config.model_config['zh']['semantic_similarity'] == 'another_test_model'  # NOQA:E501
    except AssertionError as err:
        raise err