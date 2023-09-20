


langcheck.eval
==============

.. <UPDATE_AFTER_SPHINX_APIDOC>

``langcheck.eval`` contains all of LangCheck's evaluation metrics.

Since LangCheck has multi-lingual support, language-specific metrics are organized into sub-packages such as ``langcheck.eval.en`` or ``langcheck.eval.ja``.

.. tip::
    As a shortcut, all English and language-agnostic metrics are also directly accessible from ``langcheck.eval``. For example, you can directly run ``langcheck.eval.sentiment()`` instead of ``langcheck.eval.en.reference_free_text_quality.sentiment()``.

    Additionally, ``langcheck.eval.EvalValue`` is a shortcut for ``langcheck.eval.eval_value.EvalValue``.

There are several different types of metrics:

+-------------------------------------+--------------------------------------------------------+---------------+
| Type of Metric                      | Examples                                               | Languages     |
+=====================================+========================================================+===============+
| Reference-Free Text Quality Metrics | ``toxicity(generated_outputs)``                        |               |
|                                     |                                                        | EN, JA        |
|                                     | ``sentiment(generated_outputs)``                       |               |
+-------------------------------------+--------------------------------------------------------+---------------+
| Reference-Based Text Quality Metrics| ``semantic_sim(generated_outputs, reference_outputs)`` | EN, JA        |
|                                     |                                                        |               |
|                                     | ``rouge2(generated_outputs, reference_outputs)``       |               |
+-------------------------------------+--------------------------------------------------------+---------------+
| Source-Based Text Quality Metrics   | ``factual_consistency(generated_outputs, sources)``    | EN, JA        |
+-------------------------------------+--------------------------------------------------------+---------------+
| Text Structure Metrics              | ``is_float(generated_outputs, min=0, max=None)``       | All Languages |
|                                     |                                                        |               |
|                                     | ``is_json_object(generated_outputs)``                  |               |
+-------------------------------------+--------------------------------------------------------+---------------+


----

.. </UPDATE_AFTER_SPHINX_APIDOC>

.. automodule:: langcheck.eval
   :members:
   :undoc-members:
   :show-inheritance:


.. toctree::
   :hidden:
   :maxdepth: 4

   langcheck.eval.en
   langcheck.eval.ja




.. toctree::
   :hidden:
   :maxdepth: 4

   langcheck.eval.eval_value
   langcheck.eval.reference_based_text_quality
   langcheck.eval.text_structure
