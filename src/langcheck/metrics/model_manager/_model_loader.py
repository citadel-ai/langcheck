from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import (
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from langcheck._handle_logs import _handle_logging_level


def load_sentence_transformers(
        model_name: str,
        model_revision: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_revision: Optional[str] = None) -> SentenceTransformer:
    '''
    Loads a SentenceTransformer model.

    This function currently does not support specifying a tokenizer or a
    revision. If these arguments are provided, a warning message will be
    printed.

    Args:
        model_name: The name of the SentenceTransformer model to load.
        tokenizer_name: The name of the tokenizer to use. Currently not
            supported.
        model_revision: The model revision to load. Currently not supported.
        tokenizerl_revision: The tokenizedr revision to load. Currently not
        supported.

    Returns:
        model: The loaded SentenceTransformer model.
    '''
    if model_revision is not None or tokenizer_revision is not None:
        print("Warning: Specifying a revision is not currently supported.")
    if tokenizer_name is not None:
        print("Warning: Customizing the tokenizer is not currently supported.")

    model = SentenceTransformer(model_name)
    return model


def load_auto_model_for_text_classification(
    model_name: str,
    model_revision: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    tokenizer_revision: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    '''
    Loads a sequence classification model and its tokenizer.

    Args:
        model_name: The name of the sequence-classification model to load.
        tokenizer_name: The name of the tokenizer to load. If None, the
            tokenizer associated with the model will be loaded.
        model_revision: The model revision to load.
        tokenizer_revision: the tokenizer revision to load.

    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded sequence classification model.
    '''
    if tokenizer_name is None:
        tokenizer_name = model_name
    # There are "Some weights are not used warning" for some models, but we
    # ignore it because that is intended.
    with _handle_logging_level():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  trust_remote_code=True,
                                                  revision=tokenizer_revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=model_revision)
    return tokenizer, model  # type: ignore


def load_auto_model_for_seq2seq(
    model_name: str,
    model_revision: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    tokenizer_revision: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    '''
    Loads a sequence-to-sequence model and its tokenizer.

    Args:
        model_name: The name of the sequence-classification model to load.
        tokenizer_name: The name of the tokenizer to load. If None, the
            tokenizer associated with the model will be loaded.
        model_revision: The model revision to load.
        tokenizer_revision: the tokenizer revision to load

    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded sequence-to-sequence model.
    '''
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              revision=tokenizer_revision)
    # There are "Some weights are not used warning" for some models, but we
    # ignore it because that is intended.
    with _handle_logging_level():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                      revision=model_revision)
    return tokenizer, model  # type: ignore
