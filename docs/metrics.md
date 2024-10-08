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

The same is true for German text:

```python
from langcheck.metrics.de import fluency  # German fluency metric
from langcheck.metrics import is_json_array  # Language-agnostic metric
```


## Metric Types

LangCheck metrics are categorized by metric type, which correspond to the kind of ground truth data that's required.

|                                Type of Metric                                 |                                                     Examples                                                     |   Languages    |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------- |
| [Reference-Free Text Quality Metrics](#reference-free-text-quality-metrics)   | `toxicity(generated_outputs)`<br>`sentiment(generated_outputs)`<br>`ai_disclaimer_similarity(generated_outputs)` | EN, JA, DE, ZH |
| [Reference-Based Text Quality Metrics](#reference-based-text-quality-metrics) | `semantic_similarity(generated_outputs, reference_outputs)`<br>`rouge2(generated_outputs, reference_outputs)`    | EN, JA, DE, ZH |
| [Source-Based Text Quality Metrics](#source-based-text-quality-metrics)       | `factual_consistency(generated_outputs, sources)`                                                                | EN, JA, DE, ZH |
| [Query-Based Text Quality Metrics](#query-based-text-quality-metrics)         | `answer_relevance(generated_outputs, prompts)`                                                                   | EN, JA         |
| [Text Structure Metrics](#text-structure-metrics)                             | `is_float(generated_outputs, min=0, max=None)`<br>`is_json_object(generated_outputs)`                            | All Languages  |
| [Pairwise Text Quality Metrics](#pairwise-text-quality-metrics)               | `pairwise_comparison(generated_outputs_a, generated_outputs_b, prompts)`                                         | EN, JA         |

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

(query-based-text-quality-metrics)=
### Query-Based Text Quality Metrics

Query-based metrics require the input query. For example, in a Q&A application, the query would be the user's question.

An example metric is {func}`~langcheck.metrics.en.query_based_text_quality.answer_relevance`, which computes how relevant the LLM's generated text is with respect to the input query as a score between 0 and 1.

(text-structure-metrics)=
### Text Structure Metrics

Text structure metrics validate the format of the text (e.g. is the text valid JSON, an email address, an integer in a specified range). Compared to other metric types which can return floats, these metrics only return 0 or 1.

An example metric is {func}`~langcheck.metrics.text_structure.is_json_object`, which checks if the LLM-generated text is a valid JSON object.

(pairwise-text-quality-metrics)=
### Pairwise Text Quality Metrics

Pairwise metrics require two generated outputs (A and B) and the corresponding prompt. The reference output and source text are also optional inputs.

An example metric is {func}`~langcheck.metrics.en.pairwise_text_quality.pairwise_comparison`, which compares the quality of two generated outputs (`generated_outputs_a` and `generated_outputs_a`) in response to the given prompt. If a reference output and/or source text are provided, those are also taken into consideration to judge the quality of the outputs. The scores are either 0.0 (Response A is better), 0.5 (Tie), or 1.0 (Response B is better).

(computing-metrics-with-remote-llms)=
### Computing Metrics with Remote LLMs

Several text quality metrics are computed using a machine learning model (e.g. `toxicity`, `semantic_similarity`, `factual_consistency`). By default, LangCheck will download and use a model that can run locally on your machine so that the metric works with no additional setup.

However, you can also configure LangCheck's metrics to use a remotely-hosted LLM, such as OpenAI/Gemini/Claude, which may provide better evaluations for complex text. To do this, you need to provide the appropriate {class}`~langcheck.metrics.eval_clients.EvalClient` instance, such as {class}`~langcheck.metrics.eval_clients.OpenAIEvalClient`, to the metric function.

Here's an example for the OpenAI API:

```python
import os
from langcheck.metrics.en import semantic_similarity
from langcheck.metrics.eval_clients import OpenAIEvalClient

generated_outputs = ["The cat is sitting on the mat."]
reference_outputs = ["The cat sat on the mat."]

# Option 1: Set OPENAI_API_KEY as an environment variable
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'
eval_client = OpenAIEvalClient()
similarity_value = semantic_similarity(generated_outputs,
                                       reference_outputs,
                                       eval_model=eval_client)

# Option 2: Pass in an OpenAI client directly
from openai import OpenAI

openai_client = OpenAI(api_key='YOUR_OPENAI_API_KEY')
eval_client = OpenAIEvalClient(openai_client=openai_client)

similarity_value = semantic_similarity(generated_outputs,
                                       reference_outputs,
                                       eval_model=eval_client)
```

Or, another example for the Azure OpenAI API:

```python
import os
from langcheck.metrics.en import semantic_similarity
from langcheck.metrics.eval_clients import AzureOpenAIEvalClient

generated_outputs = ["The cat is sitting on the mat."]
reference_outputs = ["The cat sat on the mat."]

# Option 1: Set the AZURE_OPENAI_KEY, OPENAI_API_VERSION, and
# AZURE_OPENAI_ENDPOINT environment variables
os.environ["AZURE_OPENAI_KEY"] = 'YOUR_AZURE_OPENAI_KEY'
os.environ["OPENAI_API_VERSION"] = 'YOUR_OPENAI_API_VERSION'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'YOUR_AZURE_OPENAI_ENDPOINT'

# You need to specify embedding_model_name to enable embedding-based evaluations
# for Azure OpenAI
eval_client = AzureOpenAIEvalClient(
    embedding_model_name='YOUR_EMBEDDING_MODEL_DEPLOYMENT_NAME')

similarity_value = semantic_similarity(
    generated_outputs,
    reference_outputs,
    eval_model=eval_client)

# Option 2: Pass in an AzureOpenAI client directly
from openai import AzureOpenAI
from langcheck.metrics.en import fluency

azure_openai_client = AzureOpenAI(api_key='YOUR_AZURE_OPENAI_KEY',
                     api_version='YOUR_OPENAI_API_VERSION',
                     azure_endpoint='YOUR_AZURE_OPENAI_ENDPOINT')
# You need to specify text_model_name to enable text-based evaluations
# for Azure OpenAI
eval_client = AzureOpenAIEvalClient(
    text_model_name='YOUR_TEXT_MODEL_DEPLOYMENT_NAME',
    azure_openai_client=azure_openai_client
)

fluency_value = fluency(
    generated_outputs,
    eval_model=eval_client)
```