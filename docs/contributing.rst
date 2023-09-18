Contributing
============

This page contains instructions for contributing to LangCheck.

Running LangCheck
-----------------

To install and run the LangCheck package from your local git repo:

.. code-block:: text

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

Documentation
-------------

To make documentation:

.. code-block:: text

    # Re-generate all docs/langcheck*.rst files. This leaves index.rst and other
    # .rst files untouched. (This uses autodoc to generate .rst files at the
    # package/module-level)
    > sphinx-apidoc -f --no-toc --separate --module-first -t docs/_templates/ -o docs src/langcheck/ src/langcheck/stats.py src/langcheck/plot/css.py

    # Re-generate all docs/_build/*.html files from the .rst files. (This uses
    # autodoc to populate .html files at the function-level)
    > make -C docs clean html

    # View documentation locally
    > python -m http.server -d docs/_build/html

Publishing
----------

To publish the package to PyPi:

.. code-block:: text

    > python -m pip install build twine
    > python -m build
    > twine check dist/*
    > twine upload -r testpypi dist/*