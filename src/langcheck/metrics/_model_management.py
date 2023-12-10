import os
import configparser
import collections
from pathlib import Path


class ModelConfig:
    """
    A class to manage different models for multiple languages in the
    langcheck.
    This class allows setting and retrieving different model names.
    (like sentiment_model, semantic_similarity_model, etc.) for each language.
    It also supports loading model configurations from a file.
    """

    def __init__(self):
        """
        Initializes the ModelConfig with empty model dictionaries for each
        language.
        """
        self.__init__config()

    def __init__config(self):
        cwd = os.path.dirname(__file__)
        cfg = configparser.ConfigParser()
        # Initial DEFAULT config from modelconfig.ini
        cfg.read(os.path.join(Path(cwd), 'modelconfig.ini'))
        self.model_config = collections.defaultdict(dict)
        for lang in cfg.sections():
            for metric_type in cfg[lang]:
                self.model_config[lang][metric_type] = cfg.get(section=lang,
                                                               option=metric_type)  # type: ignore[reportGeneralIssue]  # NOQA:E501

    def reset(self):
        ''' reset all model used in langcheck to default'''
        self.__init__config()

    def list_metric_model(self, language: str, metric_type: str):
        """
        return the model used in current metric for a given language.

        Args:
            language: The language for which to get the model.
            metric_type: The metric name.

        Returns:
            str: The name of the specified model.

        Raises:
            KeyError: If the specified language or model type is not found.
        """
        if language in self.model_config:
            if metric_type in self.model_config[language]:
                return self.model_config[language][metric_type]
            else:
                raise KeyError(f"Model type '{metric_type}' not found for language '{language}'.")  # NOQA:E501
        else:
            raise KeyError(f"Language '{language}' not supported.")

    def set_model_for_metric(self, language: str,
                             metric_type: str, model_name: str):
        """
        Sets a specific model used in metric_type for a given language.

        Args:
            language: The language for which to set the model.
            metric_type: The type of the model (e.g., 'sentiment_model').
            model_name: The name of the model.

        Raises:
            KeyError: If the specified language is not supported.
        """
        if language in self.model_config:
            if metric_type in self.model_config[language]:
                self.model_config[language][metric_type] = model_name
            else:
                raise KeyError(f"Metrics '{metric_type}' not used in metric.")
        else:
            raise KeyError(f"Language '{language}' not supported.")

    def load_config_from_file(self, file_path: str):
        """
        Loads model configurations from a specified configuration file.

        The configuration file should have sections for each language with
        key-value pairs for each metrics and model_name.

        Args:
            file_path: The path to the configuration file containing model
            configurations.
        """
        config = configparser.ConfigParser()
        config.read(file_path)

        for lanuage_section in config.sections():
            if lanuage_section in self.model_config:
                for metric_type, model_name in config[lanuage_section].items():
                    if metric_type in self.model_config[lanuage_section]:
                        self.model_config[lanuage_section][metric_type] = model_name  # NOQA:E501

    def save_config_to_disk(self, output_path: str):
        """
        Save Model Configuration to output path.
        Args:
            output_path: The path to save the configuration file
        """
        cfg = configparser.ConfigParser()
        cfg.read_dict(self.model_config)

        with open(output_path, 'w') as f:
            cfg.write(f)
