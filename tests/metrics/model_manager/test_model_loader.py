from unittest.mock import MagicMock, patch

import pytest
from langcheck.metrics.model_manager._model_loader import (
    load_auto_model_for_seq2seq,
    load_auto_model_for_text_classification,
    load_sentence_transformers,
)
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Mock objects for AutoTokenizer and AutoModelForSeq2SeqLM
MockTokenizer = MagicMock(spec=AutoTokenizer)
MockSeq2SeqModel = MagicMock(spec=AutoModelForSeq2SeqLM)
MockSentenceTransModel = MagicMock(spec=SentenceTransformer)
MockSeqClassifcationModel = MagicMock(spec=AutoModelForSequenceClassification)


@pytest.mark.parametrize("model_name,tokenizer_name,revision",
                         [("t5-small", None, "main"),
                          ("t5-small", "t5-base", "main")])
def test_load_auto_model_for_seq2seq(model_name, tokenizer_name, revision):
    with patch('transformers.AutoTokenizer.from_pretrained',
               return_value=MockTokenizer) as mock_tokenizer, \
         patch('transformers.AutoModelForSeq2SeqLM.from_pretrained',
               return_value=MockSeq2SeqModel) as mock_model:
        tokenizer, model = load_auto_model_for_seq2seq(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            model_revision=revision,
            tokenizer_revision=revision)
        if tokenizer_name is None:
            tokenizer_name = model_name

        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        # Assert that the returned objects are instances of the mocked objects
        assert tokenizer == MockTokenizer, \
            "The returned tokenizer is not the expected mock object"
        assert model == MockSeq2SeqModel, \
            "The returned model is not the expected mock object"


@pytest.mark.parametrize("model_name,tokenizer_name,revision",
                         [("bert-base-uncased", None, "main"),
                          ("bert-base-uncased", "bert-large-uncased", "main")])
def test_load_auto_model_for_text_classification(model_name, tokenizer_name,
                                                 revision):
    with patch('transformers.AutoTokenizer.from_pretrained',
               return_value=MockTokenizer) as mock_tokenizer, \
         patch('transformers.AutoModelForSequenceClassification.from_pretrained',  # NOQA:E501
               return_value=MockSeqClassifcationModel) as mock_model:
        tokenizer, model = load_auto_model_for_text_classification(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            model_revision=revision,
            tokenizer_revision=revision)
        if tokenizer_name is None:
            tokenizer_name = model_name

        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        # Assert that the returned objects are instances of the mocked objects
        assert tokenizer == MockTokenizer, \
            "The returned tokenizer is not the expected mock object"
        assert model == MockSeqClassifcationModel, \
            "The returned model is not the expected mock object"


@pytest.mark.parametrize("model_name,tokenizer_name,revision",
                         [("all-MiniLM-L6-v2", None, "main"),
                          ("all-MiniLM-L6-v2", "all-mpnet-base-v2", "main")])
def test_load_sentence_transformers(model_name, tokenizer_name, revision):
    with patch.object(SentenceTransformer, '__init__',
                      return_value=None) as mock_init:
        model = load_sentence_transformers(model_name=model_name,
                                           tokenizer_name=tokenizer_name,
                                           model_revision=revision,
                                           tokenizer_revision=revision)
        # Check if the model was loaded correctly
        mock_init.assert_called_once_with(model_name)

        # Assert that the returned objects are instances of the mocked objects
        assert isinstance(model, SentenceTransformer), \
            "The returned model is not the expected mock object"
