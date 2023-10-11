<div align="center">

<img src="docs/_static/LangCheck-Logo-square.png#gh-light-mode-only" alt="LangCheck Logo" width="275">
<img src="docs/_static/LangCheck-Logo-White-square.png#gh-dark-mode-only" alt="LangCheck Logo" width="275">

Simple, Pythonic building blocks to evaluate LLM applications.

[Install](#install) •
[Examples](#examples) •
[Docs](https://langcheck.readthedocs.io/en/latest/index.html) •
[Quickstart](https://langcheck.readthedocs.io/en/latest/quickstart.html) •
[日本語](README_ja.md)

</div>

## Install

```
pip install langcheck
```

## Examples

### Evaluate Text

Use LangCheck's suite of metrics to evaluate LLM-generated text.

```python
import langcheck

# Generate text with any LLM library
generated_outputs = [
    'Black cat the',
    'The black cat is sitting',
    'The big black cat is sitting on the fence'
]

# Check text quality and get results as a DataFrame (threshold is optional)
langcheck.metrics.fluency(generated_outputs) > 0.5
```

![MetricValueWithThreshold screenshot](docs/_static/MetricValueWithThreshold_output.png)

It's easy to turn LangCheck metrics into unit tests, just use `assert`:

```python
assert langcheck.metrics.fluency(generated_outputs) > 0.5
```

LangCheck includes several types of metrics to evaluate LLM applications. Some examples:

```python
# 1. Reference-Free Text Quality Metrics
langcheck.metrics.toxicity(generated_outputs)
langcheck.metrics.fluency(generated_outputs)
langcheck.metrics.sentiment(generated_outputs)
langcheck.metrics.flesch_kincaid_grade(generated_outputs)

# 2. Reference-Based Text Quality Metrics
langcheck.metrics.factual_consistency(generated_outputs, reference_outputs)
langcheck.metrics.semantic_sim(generated_outputs, reference_outputs)
langcheck.metrics.rouge2(generated_outputs, reference_outputs)
langcheck.metrics.exact_match(generated_outputs, reference_outputs)

# 3. Text Structure Metrics
langcheck.metrics.is_int(generated_outputs, domain=range(1, 6))
langcheck.metrics.is_float(generated_outputs, min=0, max=None)
langcheck.metrics.is_json_array(generated_outputs)
langcheck.metrics.is_json_object(generated_outputs)
langcheck.metrics.contains_regex(generated_outputs, r"\d{5,}")
langcheck.metrics.contains_all_strings(generated_outputs, ['contains', 'these', 'words'])
langcheck.metrics.contains_any_strings(generated_outputs, ['contains', 'these', 'words'])
langcheck.metrics.validation_fn(generated_outputs, lambda x: 'myKey' in json.loads(x))
```

Some LangCheck metrics support using the OpenAI API. To use the OpenAI option,
make sure to set the API key:

```python
import openai
from langcheck.metrics.en import semantic_sim

# https://platform.openai.com/account/api-keys
openai.api_key = YOUR_OPENAI_API_KEY

generated_outputs = ["The cat is sitting on the mat."]
reference_outputs = ["The cat sat on the mat."]
metric_value = semantic_sim(generated_outputs, reference_outputs, embedding_model_type='openai')
```

Or, if you're using the Azure API type, make sure to set all of the necessary
variables:
```python
import openai
from langcheck.metrics.en import semantic_sim

openai.api_type = 'azure'
openai.api_base = YOUR_AZURE_OPENAI_ENDPOINT
openai.api_version = YOUR_API_VERSION
openai.api_key = YOUR_OPENAI_API_KEY

generated_outputs = ["The cat is sitting on the mat."]
reference_outputs = ["The cat sat on the mat."]

# When using the Azure API type, you need to pass in your model's
# deployment name
metric_value = semantic_sim(generated_outputs,
                          reference_outputs,
                          embedding_model_type='openai',
                          openai_args={'engine': YOUR_EMBEDDING_MODEL_DEPLOYMENT_NAME})
```

### Visualize Metrics

LangCheck comes with built-in, interactive visualizations of metrics.

```python
# Choose some metrics
fluency_values = langcheck.metrics.fluency(generated_outputs)
sentiment_values = langcheck.metrics.sentiment(generated_outputs)

# Interactive scatter plot of one metric
fluency_values.scatter()
```

![Scatter plot for one metric](docs/_static/scatter_one_metric.gif)


```python
# Interactive scatter plot of two metrics
langcheck.plot.scatter(fluency_values, sentiment_values)
```

![Scatter plot for two metrics](docs/_static/scatter_two_metrics.png)


```python
# Interactive histogram of a single metric
fluency_values.histogram()
```

![Histogram for one metric](docs/_static/histogram.png)


### Augment Data (coming soon)

```python
more_prompts = []
more_prompts += langcheck.augment.keyboard_typo(prompts)
more_prompts += langcheck.augment.ocr_typo(prompts)
more_prompts += langcheck.augment.synonym(prompts)
more_prompts += langcheck.augment.gender(prompts, to_gender='male')
more_prompts += langcheck.augment.gpt35_rewrite(prompts)
```

### Building Blocks for Monitoring

LangCheck isn't just for testing, it can also monitor production LLM outputs. Just save the outputs and pass them into LangCheck.

```python
from langcheck.utils import load_json

recorded_outputs = load_json('llm_logs_2023_10_02.json')['outputs']
langcheck.metrics.toxicity(recorded_outputs) < 0.25
langcheck.metrics.is_json_array(recorded_outputs)
```

### Building Blocks for Guardrails

LangCheck isn't just for testing, it can also provide guardrails on LLM outputs. Just filter candidate outputs through LangCheck.

```python
raw_output = my_llm_app(random_user_prompt)
while langcheck.metrics.contains_any_strings(raw_output, blacklist_words).any():
    raw_output = my_llm_app(random_user_prompt)
```
