from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import \
    AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline


def load_sentence_transformers(model_name: str) -> SentenceTransformer:
    '''
    Return a Hugging Face sentence-transformer model.

    Args:
        model_name: The name of a sentence-transformer model
    '''
    return SentenceTransformer(model_name)


def load_auto_model_for_text_classification(
    model_name: str, tokenizer_name: Optional[str], revision: Optional[str]
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    '''
    Return a Hugging Face sequence-classification model.

    Args:
        model_name: The name of a sequence-classification model on Hugging Face
        tokenizer_name: The name of a tokenizer on Hugging Face
        revision: The shortened sha1 string of a model
    '''
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision)  # NOQA: E501
    return tokenizer, model


def load_pipeline_for_text_classification(model_name: str, **kwargs):
    '''
    Return a Hugging Face text-classification pipeline.

    Args:
        model_name: The name of a text-classification pipeline on Hugging Face
    '''
    top_k = kwargs.pop('top_k', None)
    return pipeline('text-classification', model=model_name, top_k=top_k)
