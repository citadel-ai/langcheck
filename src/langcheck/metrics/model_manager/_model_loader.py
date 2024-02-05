from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import (AutoModelForSeq2SeqLM,
                                                    AutoModelForSequenceClassification)  # NOQA:E501

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline


def load_sentence_transformers(model_name: str,
                               tokenizer_name: Optional[str] = None,
                               revision: Optional[str] = None) -> SentenceTransformer:  # NOQA:E501
    '''
    Return a sequence embeddiing model parsed by sentence-transformer library.

    Args:
        model_name: The name of a sentence-transformer model
    '''
    if revision is not None:
        print("Version Pined not supported in Sentence-Transformers yet.")

    if tokenizer_name is not None:
        print("Tokenizer customize not supported in Sentence-Transformers yet.")

    return SentenceTransformer(model_name)


def load_auto_model_for_text_classification(
    model_name: str, tokenizer_name: Optional[str] = None,
    revision: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    '''
    Return a sequence classification model on huggingface hub.

    Args:
        model_name: The name of a sequence-classification model on Hugging Face
        tokenizer_name: The name of a tokenizer on Hugging Face
        revision: The shortened sha1 string of a model
    '''
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision)  # NOQA: E501
    return tokenizer, model


def load_auto_model_for_seq2seq(
    model_name: str, tokenizer_name: Optional[str] = None,
    revision: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    '''
    Return a sequence to sequence model availble on huggingface hub.

    Args:
        model_name: The name of a sequence-classification model on Hugging Face
        tokenizer_name: The name of a tokenizer on Hugging Face
        revision: The shortened sha1 string of a model
    '''
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision=revision)  # NOQA: E501
    return tokenizer, model
