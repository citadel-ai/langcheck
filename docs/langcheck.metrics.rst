


langcheck.metrics
=================

.. <UPDATE_AFTER_SPHINX_APIDOC>

``langcheck.metrics`` contains all of LangCheck's evaluation metrics.

Since LangCheck has multi-lingual support, language-specific metrics are organized into sub-packages such as ``langcheck.metrics.en`` or ``langcheck.metrics.ja``.

.. tip::
    As a shortcut, all English and language-agnostic metrics are also directly accessible from ``langcheck.metrics``. For example, you can directly run ``langcheck.metrics.sentiment()`` instead of ``langcheck.metrics.en.reference_free_text_quality.sentiment()``.

    Additionally, ``langcheck.metrics.MetricValue`` is a shortcut for ``langcheck.metrics.metric_value.MetricValue``.

There are several different types of metrics:

+---------------------------------------------+----------------------------------------------------------------------------+----------------+
|               Type of Metric                |                                  Examples                                  |   Languages    |
+=============================================+============================================================================+================+
| :ref:`reference-free-text-quality-metrics`  | ``toxicity(generated_outputs)``                                            | EN, JA, DE, ZH |
|                                             |                                                                            |                |
|                                             | ``sentiment(generated_outputs)``                                           |                |
|                                             |                                                                            |                |
|                                             | ``ai_disclaimer_similarity(generated_outputs)``                            |                |
+---------------------------------------------+----------------------------------------------------------------------------+----------------+
| :ref:`reference-based-text-quality-metrics` | ``semantic_similarity(generated_outputs, reference_outputs)``              | EN, JA, DE, ZH |
|                                             |                                                                            |                |
|                                             | ``rouge2(generated_outputs, reference_outputs)``                           |                |
+---------------------------------------------+----------------------------------------------------------------------------+----------------+
| :ref:`query-based-text-quality-metrics`     | ``answer_relevance(generated_outputs, prompts)``                           | EN, JA         |
|                                             |                                                                            |                |
|                                             | ``answer_safety(generated_outputs, prompts)``                              |                |
+---------------------------------------------+----------------------------------------------------------------------------+----------------+
| :ref:`source-based-text-quality-metrics`    | ``factual_consistency(generated_outputs, sources)``                        | EN, JA, DE, ZH |
+---------------------------------------------+----------------------------------------------------------------------------+----------------+
| :ref:`text-structure-metrics`               | ``is_float(generated_outputs, min=0, max=None)``                           | All Languages  |
|                                             |                                                                            |                |
|                                             | ``is_json_object(generated_outputs)``                                      |                |
+---------------------------------------------+----------------------------------------------------------------------------+----------------+
| :ref:`pairwise-text-quality-metrics`        | ``pairwise_comparison(generated_outputs_a, generated_outputs_b, prompts)`` | EN             |
+---------------------------------------------+----------------------------------------------------------------------------+----------------+


----

.. </UPDATE_AFTER_SPHINX_APIDOC>

.. automodule:: langcheck.metrics
   :members:
   :undoc-members:
   :show-inheritance:


.. toctree::
   :hidden:
   :maxdepth: 4

   langcheck.metrics.de
   langcheck.metrics.en
   langcheck.metrics.eval_clients
   langcheck.metrics.ja
   langcheck.metrics.zh




.. toctree::
   :hidden:
   :maxdepth: 4

   langcheck.metrics.custom_text_quality
   langcheck.metrics.metric_value
   langcheck.metrics.reference_based_text_quality
   langcheck.metrics.text_structure
