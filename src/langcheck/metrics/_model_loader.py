from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import \
    AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline


def load_sentence_transformers(model_name: str) -> SentenceTransformer:
    """
    return a sentence-transformer model.

    Args:
        model_name: The model name of a sentence-transformers model
    """
    return SentenceTransformer(model_name)


def load_auto_model_for_text_classification(model_name: str,
                                            tokenizer_name: Optional[str],
                                            revision: Optional[str])\
                            -> Tuple[AutoTokenizer,
                                     AutoModelForSequenceClassification]:
    """
    return a Huggingface text-classification pipeline.

    Args:
        model_name: The name of a sequenceclassification model on huggingface hub.  # NOQA:E501
        tokenizer_name: the name of a tokenizer on huggingface hub.
        revisoin: the shorted sha1 string of a model
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)  # NOQA: E501
    return tokenizer, model


def load_pipeline_for_text_classification(model_name: str, **kwargs):
    """
    return a Huggingface text-classification pipeline.

    Args:
        model_name: A huggingface model model for text classification.
    """
    top_k = kwargs.pop('top_k', None)
    return pipeline('text-classification', model=model_name, top_k=top_k)
