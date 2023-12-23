from typing import Tuple, Optional
from transformers.pipelines import pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification  # NOQA:E501
from sentence_transformers import SentenceTransformer


def load_sentence_transformers(model_name: str) -> SentenceTransformer:
    """
    return a sentence-transformer model.

    Args:
        model_name: The model name of a sentence-transformers model
    """
    return SentenceTransformer(model_name)


def load_auto_model_for_text_classification(model_name: str,
                                            tokenizer_name: Optional[str])\
                            -> Tuple[AutoTokenizer,
                                     AutoModelForSequenceClassification]:
    """
    return a Huggingface text-classification pipeline.

    Args:
        model_name: The name of a sequenceclassification model on huggingface hub.  # NOQA:E501
        tokenizer_name: the name of a tokenizer on huggingface hub.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def load_pipeline_for_text_classification(model_name: str, **kwargs):
    """
    return a Huggingface text-classification pipeline.

    Args:
        model_name: A huggingface model model for text classification.
    """
    top_k = kwargs.pop('top_k', None)
    return pipeline('text-classification', model=model_name, top_k=top_k)
