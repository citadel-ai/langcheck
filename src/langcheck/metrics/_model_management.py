import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple, Union

import pandas as pd
import requests
from configobj import ConfigObj
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import \
    AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ._model_loader import (load_auto_model_for_text_classification,
                            load_sentence_transformers)

# TODO: Use a ENUM class to parse these
VALID_METRIC_NAME = [
    'factual_consistency', 'toxicity', 'sentiment', 'semantic_similarity'
]
VALID_LANGUAGE = ['zh']
VALID_LOADER = ['huggingface', 'sentence-transformers']


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
        self.__init__config()
        self.validate_config()

    def __init__config(self):
        cwd = os.path.dirname(__file__)
        self.config = ConfigObj(
            os.path.join(Path(cwd), 'config', 'metric_config.ini'))

    @lru_cache
    def fetch_model(self, language: str, metric: str)\
        -> Union[Tuple[AutoTokenizer, AutoModelForSequenceClassification],
                 SentenceTransformer]:
        '''
        Return the model used for the given metric and language.

        Args:
            language: The language for which to get the model
            metric_type: The metric name
        '''
        if language in self.config:
            if metric in self.config[language]:
                # Deep copy the confguration so that changes to `config` would
                # not affect the original `self.config`.
                config = deepcopy(self.config[language][metric])
                # Get model name, model loader type
                model_name, loader_type = config['model_name'], config['loader']
                # Check if the model version is fixed
                revision = config.pop("revision", None)
                if loader_type == 'sentence-transformers':
                    if revision is not None:
                        print(
                            'Info: Sentence-Transformers do not support fixed model versions yet'  # NOQA:E501
                        )
                    model = load_sentence_transformers(model_name=model_name)
                    return model
                elif loader_type == 'huggingface':
                    tokenizer_name = config.pop('tokenizer_name', None)
                    tokenizer, model = load_auto_model_for_text_classification(
                        model_name=model_name,
                        tokenizer_name=tokenizer_name,
                        revision=revision)
                    return tokenizer, model
                else:
                    raise KeyError(f'Loader {loader_type} not supported yet.')
            else:
                raise KeyError(f'Metric {metric} not supported yet.')
        else:
            raise KeyError(f'language {language} not supported yet')

    def list_current_model_in_use(self, language='all', metric='all'):
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
                                  aggfunc='first').reset_index().drop(
                                      columns=["attribute"]).reset_index()
        df_pivot.columns = [
            'language', 'metric_name', 'loader', 'model_name', 'revision'
        ]

        if language == 'all' and metric == 'all':
            pprint(df_pivot)
        else:
            if language != "all":
                df_pivot = df_pivot.loc[df_pivot.language == language]
            if metric != 'all':
                df_pivot = df_pivot.loc[df_pivot.metric_name == metric]
            pprint(df_pivot)

    def validate_config(self, language='all', metric='all'):
        '''
        Validate configuration.

        Args:
            language: The name of the language. Defaults to 'all'.
            metric: The name of the metric. Defaults to 'all'.
        '''

        def check_model_availability(model_name, revision):
            if revision is None:
                url = f"https://huggingface.co/api/models/{model_name}"
            else:
                url = f"https://huggingface.co/api/models/{model_name}/revision/{revision}"  # NOQA:E501
            response = requests.get(url)
            return response.status_code == 200

        config = deepcopy(self.config)
        for lang, lang_setting in config.items():
            if language == 'all' or lang == language:
                for metric_name, model_setting in lang_setting.items():
                    if metric == 'all' or metric_name == metric:
                        # If model name not set
                        if 'model_name' not in model_setting:
                            raise KeyError(
                                f'{lang} metrics {metric_name} need a model, but found None!'  # NOQA:E501
                            )
                        if 'loader' not in model_setting:
                            raise KeyError(
                                f'Metrics {metric_name} need a loader, but found None!'  # NOQA:E501
                            )
                        # Check if the model and revision is available on
                        # Hugging Face Hub
                        loader_type = model_setting.pop('loader')
                        if loader_type == 'huggingface':
                            model_name = model_setting.pop('model_name')
                            revision = model_setting.pop('revision', None)
                            if not check_model_availability(
                                    model_name, revision):
                                raise ValueError(
                                    f'Cannot find {model_name} with {revision} and Huggingface Hub'  # NOQA:E501
                                )
                        elif loader_type not in VALID_LOADER:
                            raise ValueError(
                                f'loader type should in {VALID_LOADER}')
                        # TODO: May also need other validations for other loader
                        # not found yet
        print('Configuration Validation Passed')

    def set_model_for_metric(self, language: str, metric: str, model_name: str,
                             loader: Optional[str], **kwargs):
        '''
        Set model for specified metric in specified language.

        Args:
            language: The name of the language
            metric: The name of the evaluation metrics
            model_name: The name of the model
            loader: The loader of the model
            tokenizer_name: (Optional) the name of the tokenizer
            revision: (Optional) a version string of the model
        '''
        config_copy = deepcopy(self.config)
        try:
            if language not in VALID_LANGUAGE:
                raise ValueError('Language {language} not supported yet')

            if metric not in self.config[language]:
                raise ValueError(
                    'Language {language} not supported {metric} yet')

            config = self.config[language][metric]
            config['loader'] = loader
            config['model_name'] = model_name
            # If tokenizer_name is different from model_name
            tokenizer_name = kwargs.pop('tokenizer_name', None)
            if tokenizer_name:
                config['tokenizer_name'] = tokenizer_name
            # If model's revision is pinned
            revision = kwargs.pop('revision', None)
            if revision:
                config['revision'] = revision
            # Validate the change
            if self.validate_config(language=language, metric=metric):
                # Clear the LRU cache to make the config change reflected
                # immediately
                self.fetch_model.cache_clear()
        except (ValueError, KeyError) as err:
            # Trace back the configuration
            self.config = config_copy
            raise err
