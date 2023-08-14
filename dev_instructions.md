To use the package:

```
# Install the langcheck package in editable mode with dev dependencies
> python -m pip install -e .[dev]

# Try using langcheck
# (If you edit the package, just restart the Python REPL to reflect your changes)
> python
>>> from langcheck.eval import is_float
>>> is_float(['1', '-2', '3.14', '999', 'asdf'], min=0, max=10)
EvalValue(metric_name='is_float', prompts=None, generated_outputs=['1', '-2', '3.14', '999', 'asdf'], metric_values=[1, 0, 1, 0, 0])
```

To make documentation:

```
# Generate .rst files (use autodoc to generate .rst files at the file-level)
> sphinx-apidoc --no-toc --separate --module-first -o docs src/langcheck/
# Generate .html files from the .rst files (use autodoc to populate .html files at the function-level)
> cd docs && make html
> cd _build/html && python -m http.server
```

To publish the package:

```
> python -m pip install build twine
> python -m build
> twine check dist/*
> twine upload -r testpypi dist/*
```