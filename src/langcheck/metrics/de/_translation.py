from math import floor

from nltk.tokenize import sent_tokenize
from transformers.pipelines import pipeline


class Translate:
    '''Translation class based on HuggingFace's translation pipeline.'''

    def __init__(self, model_name: str) -> None:
        '''
        Initialize the Translation class with given parameters.

        Args:
            model_name: The name of the model to use for translation
        '''
        self._translation_pipeline = pipeline("translation",
                                              model=model_name,
                                              tokenizer=model_name)
        self._max_length = self._translation_pipeline.model.config.max_length

    def _translate(self, texts: str) -> str:
        '''Translate the texts using the translation pipeline.
        It splits the texts into blocks and translates each block separately,
        avoiding problems with long texts.
        Args:
            texts: The texts to translate
        Returns:
            The translated texts
        '''
        tokenization = self._translation_pipeline.tokenizer(
            texts, return_tensors="pt")  # type: ignore
        if tokenization.input_ids.shape[1] > (self._max_length / 2):
            blocks = floor(
                tokenization.input_ids.shape[1] / self._max_length) + 3
            sentences = sent_tokenize(texts)
            # Split sentences into a number of blocks, e.g., 2 blocks = 2 groups
            len_block = floor(len(sentences) / blocks) + 1
            sentences_list = []
            for i in range(blocks):
                sentences_list.append(sentences[i * len_block:(i + 1) *
                                                len_block])
            texts_ = [" ".join(sent) for sent in sentences_list]
        else:
            texts_ = [texts]
        texts_en = []
        for text in texts_:
            text_en = [
                str(d['translation_text'])  # type: ignore
                for d in self._translation_pipeline(text)  # type: ignore
            ]
            texts_en.append(" ".join(text_en))
        text_en_final = " ".join(texts_en)
        return text_en_final

    def __call__(self, text: str) -> str:
        '''Translate the text using the translation pipeline.
        Args:
            text: The text to translate
        Returns:
            The translated text
        '''
        return self._translate(text)
