# Installation

To install LangCheck, just run:

```
pip install langcheck
```

LangCheck works with Python 3.8 or higher.

:::{note}
Model files are lazily downloaded the first time you run a metric function. For example, the first time you run the ``langcheck.eval.sentiment()`` function, LangCheck will automatically download the Twitter-roBERTa-base model.
:::
