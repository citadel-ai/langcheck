# Contributing

This page contains instructions for contributing to LangCheck.

## Running LangCheck

To install and run the LangCheck package from your local git repo:

```text
# Install the langcheck package in editable mode with dev dependencies
> python -m pip install -e .[dev]

# Try using langcheck
# (If you edit the package, just restart the Python REPL to reflect your changes)
> python
>>> from langcheck.eval import is_float
>>> is_float(['1', '-2', 'a'])
Metric: is_float
prompt generated_output reference_output  metric_value
0   None                1             None             1
1   None               -2             None             1
2   None                a             None             0

# Run tests
> python -m pytest -s -vv
```

## Documentation

To make documentation:

```text
# Re-generate all docs/langcheck*.rst files. This leaves index.md and other
# .rst files untouched. Since this overwrites all of our custom edits to
# the .rst files, you must check the code diffs for UPDATE_AFTER_SPHINX_APIDOC
# comments and manually re-apply them. (This uses autodoc to generate .rst
# files at the package/module-level)
> sphinx-apidoc -f --no-toc --separate --module-first -t docs/_templates/ -o docs src/langcheck/ src/langcheck/stats.py src/langcheck/plot/css.py

# Re-generate all docs/_build/*.html files from the .rst files. Note that
# you'll see warnings like "duplicate object description of
# langcheck.plot.histogram" and "more than one target found for
# cross-reference 'EvalValue'". Sphinx seems to get confused when we import
# a module's functions/classes into its parent package's __init__.py. This
# seems to be harmless and there doesn't seem to be a way to suppress these.
# https://groups.google.com/g/sphinx-users/c/vuW6OOb96Yo
# https://github.com/sphinx-doc/sphinx/issues/11050
# (This uses autodoc to populate .html files at the function-level)
> make -C docs clean html

# View documentation locally
> python -m http.server -d docs/_build/html
```

## Publishing

To publish the package to PyPi:

```text
> python -m pip install build twine
> python -m build
> twine check dist/*
> twine upload -r testpypi dist/*
```