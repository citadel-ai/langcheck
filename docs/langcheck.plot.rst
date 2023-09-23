


langcheck.plot
==============

.. <UPDATE_AFTER_SPHINX_APIDOC>

``langcheck.plot`` contains functions to visualize the output of metric functions (``EvalValue`` objects).

For example, run ``langcheck.plot.scatter()`` in a Jupyter notebook to see this interactive scatter plot:

.. image:: _static/scatter_one_metric.gif

.. tip::
    As a shortcut, you can run plotting functions directly on an :class:`~langcheck.eval.eval_value.EvalValue`. For example, ``toxicity_values.scatter()`` is equivalent to ``langcheck.plot.scatter(toxicity_values)``.

----

.. </UPDATE_AFTER_SPHINX_APIDOC>

.. automodule:: langcheck.plot
   :members:
   :undoc-members:
   :show-inheritance:
