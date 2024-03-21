import os
from copy import deepcopy
from functools import lru_cache
from typing import Optional, Tuple, Union

import pandas as pd
import requests
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from transformers.models.auto.modeling_auto import (
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ._model_loader import (load_auto_model_for_seq2seq,
                            load_auto_model_for_text_classification,
                            load_sentence_transformers)

LOADER_MAP = {
    "load_sentence_transformers":
        load_sentence_transformers,
    "load_auto_model_for_text_classification":
        load_auto_model_for_text_classification,
    "load_auto_model_for_seq2seq":
        load_auto_model_for_seq2seq
}
VALID_LOADER_FUNCTION = LOADER_MAP.keys()
VALID_METRICS = [
    'semantic_similarity', 'sentiment', 'toxicity', 'factual_consistency',
    'fluency'
]
VALID_LANGUAGE = ['zh', 'en', 'ja', 'de']


def check_model_availability(model_name: str, revision: Optional[str]) -> bool:
    # TODO: add local cached model availability check for offline environment
    if revision is None or revision == "":
        url = f"https://huggingface.co/api/models/{model_name}"
    else:
        url = f"https://huggingface.co/api/models/{model_name}/revision/{revision}"  # NOQA:E501
    response = requests.get(url, timeout=(1.0, 1.0))
    return response.status_code == 200


class ModelManager:
    '''
    A class to manage different models for multiple languages in LangCheck.
    This class allows setting and retrieving different model names (like
    sentiment_model, semantic_similarity_model, etc.) for each language.
    It also supports loading model configurations from a file.
    '''

    def __init__(self):
        '''
        Initializes the ModelConfig with empty model dictionaries for each
        language.
        '''
        self.config = OmegaConf.create()
        cwd = os.path.dirname(__file__)
        default_config_file_path = os.path.join(cwd, "config",
                                                "metric_config.yaml")
        self.__load_config(default_config_file_path)

    def __load_config(self, path: str) -> None:
        '''
        Loads the model configuration from a file.

        Args:
            path: The path to the configuration file.
        '''
        conf = OmegaConf.load(path)

        for lang, lang_conf in conf.items():
            for metric_name, metric_conf in lang_conf.items():
                # check model availbility, if key not in conf
                # omega conf will return None in default
                assert isinstance(lang, str)
                self.__set_model_for_metric(language=lang,
                                            metric=metric_name,
                                            **metric_conf)
        print('Configuration Load Succeeded!')

    @lru_cache
    def fetch_model(
        self, language: str, metric: str
    ) -> Union[Tuple[AutoTokenizer, AutoModelForSequenceClassification], Tuple[
            AutoTokenizer, AutoModelForSeq2SeqLM], SentenceTransformer]:
        '''
        Return the model (and if applicable, the tokenizer) used for the given
        metric and language.

        Args:
            language: The language for which to get the model
            metric_type: The metric name

        Returns:
            A (tokenizer, modle) tuple, or just the model depending on the
            loader function.
        '''
        if language in self.config:
            if metric in self.config[language]:
                # Deep copy the confguration so that changes to `config` would
                # not affect the original `self.config`.
                config = deepcopy(self.config[language][metric])
                # Get model loader function
                loader_func = config.pop('loader_func')
                loader = LOADER_MAP[loader_func]
                # Call the loader function with the model_name, tokenizer_name
                # (optional), and revision (optional) as arguments
                return loader(**config)
            else:
                raise KeyError(f'Metric {metric} not supported yet.')
        else:
            raise KeyError(f'Language {language} not supported yet')

    @staticmethod
    def validate_config(config,
                        language='all',
                        metric='all',
                        run_check_model_availability=False) -> None:
        '''
        Validate configuration.

        Args:
            config: The configuration dictionary to validate.
            language: The name of the language. Defaults to 'all'.
            metric: The name of the metric. Defaults to 'all'.
            run_check_model_availability: Whether to check the model
                availability on Huggingface Hub. Defaults to False.
        '''
        config = deepcopy(config)
        for lang, lang_setting in config.items():
            if language != 'all' and lang != language:
                continue
            for metric_name, model_setting in lang_setting.items():
                if metric != 'all' and metric_name != metric:
                    continue

                # Check that the model name and loader function are set
                if 'model_name' not in model_setting:
                    raise KeyError(
                        f'{lang} metrics {metric_name} need a model, but found None!'  # NOQA:E501
                    )
                if 'loader_func' not in model_setting:
                    raise KeyError(
                        f'Metrics {metric_name} need a loader, but found None!'  # NOQA:E501
                    )
                loader_func = model_setting.get('loader_func')
                if loader_func not in VALID_LOADER_FUNCTION:
                    raise ValueError(
                        f'loader type should in {VALID_LOADER_FUNCTION}')

                if run_check_model_availability:
                    model_name = model_setting.get('model_name')
                    model_revision = model_setting.get('model_revision')
                    if not check_model_availability(model_name, model_revision):
                        raise ValueError(
                            f'Cannot find {model_name} with {model_revision} at Huggingface Hub'  # NOQA:E501
                        )

                    tokenizer_name = model_setting.get('tokenizer_name')
                    if tokenizer_name is not None and tokenizer_name != model_name:  # NOQA: E501
                        tokenizer_revision = model_setting.get(
                            'tokenizer_revision')
                        if not check_model_availability(tokenizer_name,
                                                        tokenizer_revision):
                            raise ValueError(
                                f'Cannot find {tokenizer_name} with {tokenizer_revision} ay Huggingface Hub'  # NOQA:E501
                            )

    def __set_model_for_metric(self, language: str, metric: str,
                               model_name: str, loader_func: str,
                               **kwargs) -> None:
        '''
        Set model for specified metric in specified language.

        Args:
            language: The name of the language
            metric: The name of the evaluation metric
            model_name: The name of the model
            loader_func: The loader function of the model
            tokenizer_name: (Optional) The name of the tokenizer
            model_revision: (Optional) A version string of the model. If not
                specified, load the latest model by default.
            tokenizer_revision: (Optional) A version string of the tokenizer. If
                not specified, load the latest tokenizer by default.
        '''
        config_copy = deepcopy(self.config)
        try:
            if language not in VALID_LANGUAGE:
                raise KeyError('Language {language} not supported yet')

            if metric not in VALID_METRICS:
                raise KeyError(
                    f'Metric {metric} not supported for language {language} yet'
                )

            # Initialize the configuration for the language and metric if it
            # doesn't exist
            if self.config.get(language) is None:
                self.config[language] = {}
            if self.config.get(language).get(metric) is None:
                self.config[language][metric] = {}

            detail_config = self.config[language][metric]
            # Set the loader function and model name
            detail_config['loader_func'] = loader_func
            detail_config['model_name'] = model_name

            # If tokenizer_name is different from model_name
            tokenizer_name = kwargs.get('tokenizer_name')
            if tokenizer_name:
                detail_config['tokenizer_name'] = tokenizer_name
            # If model's revision is pinned
            model_revision = kwargs.get('model_revision')
            if model_revision:
                detail_config['model_revision'] = model_revision
            # If tokenizer's revision is pinned
            tokenizer_revision = kwargs.get('tokenizer_revision')
            if tokenizer_revision:
                detail_config['tokenizer_revision'] = tokenizer_revision
            # Validate the change
            ModelManager.validate_config(self.config,
                                         language=language,
                                         metric=metric)
            # Clear the LRU cache to make the config change reflected
            # immediately
            self.fetch_model.cache_clear()
        except (ValueError, KeyError) as err:
            # If an error occurred, restore the original configuration
            self.config = config_copy
            raise err

    def list_current_model_in_use(self, language='all', metric='all') -> None:
        '''
        List the models currently in use.

        Args:
            language: The abbrevation name of language
            metric: The evaluation metric name
        '''
        df = pd.DataFrame.from_records(
            [(lang, metric_name, key, value)
             for lang, lang_model_settings in self.config.items()
             for metric_name, model_settings in lang_model_settings.items()
             for key, value in model_settings.items()],
            columns=['language', 'metric_name', 'attribute', 'value'])
        # The code below would generate a dataframe:
        # |index| language | metric_name | loader | model_name | revision |
        # |.....|..........|.............|........|............|..........|
        df_pivot = df.pivot_table(index=['language', 'metric_name'],
                                  columns="attribute",
                                  values="value",
                                  aggfunc='first').reset_index().rename_axis(
                                      None, axis=1)
        df_pivot.columns = [
            'language', 'metric_name', 'loader', 'model_name', 'revision'
        ]

        if language == 'all' and metric == 'all':
            print(
                tabulate(
                    df_pivot,  # type: ignore
                    headers=df_pivot.columns,  # type: ignore
                    tablefmt="github"))
        else:
            if language != "all":
                df_pivot = df_pivot.loc[df_pivot.language == language]
            if metric != 'all':
                df_pivot = df_pivot.loc[df_pivot.metric_name == metric]
            print(
                tabulate(
                    df_pivot,  # type: ignore
                    headers=df_pivot.columns,  # type: ignore
                    tablefmt="github"))
