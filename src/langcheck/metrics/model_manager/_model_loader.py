from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import (  # NOQA:E501
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline


def load_sentence_transformers(
        model_name: str,
        tokenizer_name: Optional[str] = None,
        revision: Optional[str] = None) -> SentenceTransformer:
    '''
    Loads a SentenceTransformer model.

    This function currently does not support specifying a tokenizer or a
    revision. If these arguments are provided, a warning message will be
    printed.

    Args:
        model_name: The name of the SentenceTransformer model to load.
        tokenizer_name: The name of the tokenizer to use. Currently not
            supported.
        revision: The model revision to load. Currently not supported.

    Returns:
        model: The loaded SentenceTransformer model.
    '''
    if revision is not None:
        print("Warning: Specifying a revision is not currently supported.")
    if tokenizer_name is not None:
        print("Warning: Customizing the tokenizer is not currently supported.")

    model = SentenceTransformer(model_name)
    return model


def load_auto_model_for_text_classification(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    revision: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    '''
    Loads a sequence classification model and its tokenizer.

    Args:
        model_name: The name of the sequence-classification model to load.
        tokenizer_name: The name of the tokenizer to load. If None, the
            tokenizer associated with the model will be loaded.
        revision: The model revision to load.

    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded sequence classification model.
    '''
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision)
    return tokenizer, model  # type: ignore


def load_auto_model_for_seq2seq(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    revision: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    '''
    Loads a sequence-to-sequence model and its tokenizer.

    Args:
        model_name: The name of the sequence-classification model to load.
        tokenizer_name: The name of the tokenizer to load. If None, the
            tokenizer associated with the model will be loaded.
        revision: The model revision to load.

    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded sequence-to-sequence model.
    '''
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision=revision)  # NOQA: E501
    return tokenizer, model  # type: ignore
