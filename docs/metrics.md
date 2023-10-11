# Metrics

This page describes LangCheck's metrics for evaluating LLMs.

## Importing Metrics

Inside the LangCheck package, metrics are first categorized by language. For example, {mod}`langcheck.metrics.en` contains all metrics for English text.

:::{tip}
For English text, you can also directly import metrics from {mod}`langcheck.metrics`, which contains all English metrics and language-agnostic metrics:

```python
# Short version
from langcheck.metrics import fluency
from langcheck.metrics import is_json_array

# Long version
from langcheck.metrics.en.reference_free_text_quality import fluency
from langcheck.metrics.text_structure import is_json_array
```
:::

Within each language, metrics are further categorized by metric type. For example, {mod}`langcheck.metrics.ja.reference_free_text_quality` contains all Japanese, reference-free text quality metrics. However, you can also import metrics from {mod}`langcheck.metrics.ja` directly.

So, for Japanese text, you can import Japanese text metrics from {mod}`langcheck.metrics.ja`, and language-agnostic metrics from {mod}`langcheck.metrics`.

```python
from langcheck.metrics.ja import fluency  # Japanese fluency metric
from langcheck.metrics import is_json_array  # Language-agnostic metric
```

## Metric Types

LangCheck metrics are categorized by metric type, which correspond to the kind of ground truth data that's required.

|                                Type of Metric                                 |                                                     Examples                                                     |   Languages   |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------- |
| [Reference-Free Text Quality Metrics](#reference-free-text-quality-metrics)   | `toxicity(generated_outputs)`<br>`sentiment(generated_outputs)`<br>`ai_disclaimer_similarity(generated_outputs)` | EN, JA        |
| [Reference-Based Text Quality Metrics](#reference-based-text-quality-metrics) | `semantic_similarity(generated_outputs, reference_outputs)`<br>`rouge2(generated_outputs, reference_outputs)`    | EN, JA        |
| [Source-Based Text Quality Metrics](#source-based-text-quality-metrics)       | `factual_consistency(generated_outputs, sources)`                                                                | EN, JA        |
| [Text Structure Metrics](#text-structure-metrics)                             | `is_float(generated_outputs, min=0, max=None)`<br>`is_json_object(generated_outputs)`                            | All Languages |

(reference-free-text-quality-metrics)=
### Reference-Free Text Quality Metrics

Reference-free metrics require no ground truth, and directly evaluate the quality of the generated text by itself.

An example metric is {func}`~langcheck.metrics.en.reference_free_text_quality.toxicity`, which directly evaluates the level of toxicity in some text as a score between 0 and 1.

(reference-based-text-quality-metrics)=
### Reference-Based Text Quality Metrics

Reference-based metrics require a ground truth output (a "reference") to compare LLM outputs against. For example, in a Q&A application, you might have human written answers as references.

An example metric is {func}`~langcheck.metrics.en.reference_based_text_quality.semantic_similarity`, which computes the semantic similarity between the LLM-generated text and the reference text as a score between -1 and 1.

(source-based-text-quality-metrics)=
### Source-Based Text Quality Metrics

Source-based metrics require a "source" text. Sources are inputs, but references are outputs. For example, in a Q&A application, the source might be relevant documents that are concatenated to the question and passed into the LLM's context window (this is called Retrieval Augmented Generation or RAG).

An example metric is {func}`~langcheck.metrics.en.source_based_text_quality.factual_consistency`, which compares the factual consistency between the LLM's generated text and the source text as a score between 0 and 1.

(text-structure-metrics)=
### Text Structure Metrics

Text structure metrics validate the format of the text (e.g. is the text valid JSON, an email address, an integer in a specified range). Compared to other metric types which can return floats, these metrics only return 0 or 1.

An example metric is {func}`~langcheck.metrics.text_structure.is_json_object`, which checks if the LLM-generated text is a valid JSON object.

(computing-metrics-with-openai-models)=
### Computing Metrics with OpenAI Models

Several text quality metrics are computed using a model (e.g. `toxicity`, `sentiment`, `semantic_similarity`, `factual_consistency`). By default, LangCheck will download and use a model that can run locally on your machine (often from HuggingFace) so that the metric function works with no additional setup.

However, if you have an OpenAI API key, you can also configure these metrics to use an OpenAI model, which may provide more accurate results for more complex use cases. Here are some examples of how to do this:

```python
import openai
from langcheck.metrics.en import semantic_similarity

# https://platform.openai.com/account/api-keys
openai.api_key = YOUR_OPENAI_API_KEY

generated_outputs = ["The cat is sitting on the mat."]
reference_outputs = ["The cat sat on the mat."]
similarity_value = semantic_similarity(generated_outputs, reference_outputs, embedding_model_type='openai')
```

Or, if you're using the Azure API type, make sure to set all of the necessary variables:

```python
import openai
from langcheck.metrics.en import semantic_similarity

openai.api_type = 'azure'
openai.api_base = YOUR_AZURE_OPENAI_ENDPOINT
openai.api_version = YOUR_API_VERSION
openai.api_key = YOUR_OPENAI_API_KEY

generated_outputs = ["The cat is sitting on the mat."]
reference_outputs = ["The cat sat on the mat."]

# When using the Azure API type, you need to pass in your model's
# deployment name
similarity_value = semantic_similarity(
    generated_outputs,
    reference_outputs,
    embedding_model_type='openai',
    openai_args={'engine': YOUR_EMBEDDING_MODEL_DEPLOYMENT_NAME})
```