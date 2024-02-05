from ._model_management import ModelManager
from ._model_loader import (load_sentence_transformers,
                            load_auto_model_for_seq2seq,
                            load_auto_model_for_text_classification)

manager = ModelManager()

__all__ = [
    "manager",
    "load_sentence_transformers",
    "load_auto_model_for_seq2seq",
    "load_auto_model_for_text_classification"
]
