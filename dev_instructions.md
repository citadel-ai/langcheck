To use the package:

```
# Install the langcheck package in editable mode
> python -m pip install -e .

# Try using langcheck
# (If you edit the package, just restart the Python REPL to reflect your changes)
> python
>>> from langcheck.eval import is_float
>>> is_float(['1', '-2', '3.14', '999', 'asdf'], min=0, max=10)
EvalValue(metric_name='is_float', prompts=None, generated_outputs=['1', '-2', '3.14', '999', 'asdf'], reference_outputs=None, metric_values=[1, 0, 1, 0, 0])
```

To publish the package:

```
> python -m pip install build twine
> python -m build
> twine check dist/*
> twine upload -r testpypi dist/*
```