# Contributing

This page contains instructions for contributing to LangCheck.

## Installing LangCheck from Source

To install and run the LangCheck package from your local git repo:

```bash
# Install the langcheck package in editable mode with dev dependencies
> python -m pip install -e .[all,dev]
# If you are using zsh, make sure to escape the brackets
> python -m pip install -e .\[all,dev\]

# Try using langcheck
# (If you edit the package, just restart the Python REPL to reflect your changes)
> python
>>> from langcheck.metrics import is_float
>>> is_float(['1', '-2', 'a'])
Metric: is_float
prompt generated_output reference_output  metric_value
0   None                1             None             1
1   None               -2             None             1
2   None                a             None             0
```

## Running Formatter
We auto-format the code with [ruff](https://github.com/astral-sh/ruff).
Please format your code with following command before you submit a pull request.

```bash
ruff format src/ tests/
```

## Running Static Checks
We have static checks with [ruff](https://github.com/astral-sh/ruff) and [pyright](https://github.com/microsoft/pyright).
Please make sure that your code passes those two commands as well.

```bash
ruff check src/ tests/
pyright .
```

## Running Tests

To run tests:

```bash
# Run all tests
> python -m pytest -s -vv

# Run non-optional tests only
> python -m pytest -s -vv -m "not optional"

# Run optional tests only (this requires optional Japanese tokenizers like Mecab)
> pip install .[optional]
> python -m pytest -s -vv -m "optional"
```

## Documentation

To make documentation:

1. **Optional:** Re-generate all `docs/langcheck*.rst` files.
   - `sphinx-apidoc -f --no-toc --separate --module-first -t docs/_templates/ -o docs src/langcheck/ src/langcheck/stats.py src/langcheck/metrics/model_manager`
   - **Warning:** This will overwrite all of our custom text in the `.rst` files, so you must check the code diffs for `UPDATE_AFTER_SPHINX_APIDOC` comments and manually re-apply them.
   - This is only necessary when you add or remove entire packages/modules. If you only edit existing packages/modules, you can skip this step.
   - This only modifies the auto-generated `docs/langcheck*.rst` files (the "API Reference" section in the docs). It doesn't touch the `index.md` and other `.md` or `.rst` files.
   - This uses autodoc to generate `.rst` files at the package/module-level.

2. Re-generate all `docs/_build/*.html` files from the raw `.rst` and `.md` files.
    - `make -C docs clean html`
    - This uses autodoc to populate .html files at the function-level.
    - Note: you'll see warnings like "more than one target found for cross-reference 'MetricValue'". Sphinx seems to get confused when we import a module's classes into its parent package's `__init__.py`. This seems to be harmless and there doesn't seem to be a way to suppress it.
        - [https://groups.google.com/g/sphinx-users/c/vuW6OOb96Yo](https://groups.google.com/g/sphinx-users/c/vuW6OOb96Yo)

3. View documentation locally
    - `python -m http.server -d docs/_build/html`

## Publishing

To publish a new version of LangCheck:

1. Increment the version in `pyproject.toml` and `src/langcheck/__init__.py`

2. Cut a new release on GitHub with release notes

3. Build the package

```bash
> python -m pip install build twine
> python -m build
> twine check dist/*
```

4. Publish to PyPi

```bash
# Follow auth token instructions at https://pypi.org/manage/account/token/

# TestPyPi
> twine upload -r testpypi dist/*

# PyPi
> twine upload dist/*
```

5. Log into ReadTheDocs and activate the new version on the "Versions" tab
