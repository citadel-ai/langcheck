


langcheck.augment
=================

.. <UPDATE_AFTER_SPHINX_APIDOC>

``langcheck.augment`` contains all of LangCheck's text augmentations.

These augmentations can help you automatically generate test cases to evaluate model robustness to different prompts, typos, gender changes, and more.

Currently, only English text augmentations are available. Japanese text augmentations are in development.

.. Since LangCheck has multi-lingual support, language-specific augmentations are organized into sub-packages such as ``langcheck.augment.en`` or ``langcheck.augment.ja``.

.. tip::
    As a shortcut, all English text augmentations are directly accessible from ``langcheck.augment``. For example, you can directly run ``langcheck.augment.keyboard_typo()`` instead of ``langcheck.augment.en.keyboard_typo()``.

LangCheck's augmentation functions can take either a single string or a list of strings as input. Optionally, you can set the ``num_perturbations`` parameter for most augmentations (except the deterministic ones), which specifies how many perturbed instances to return for each string.

To see more details about each augmentation, refer to the API reference below.

----

.. </UPDATE_AFTER_SPHINX_APIDOC>


.. automodule:: langcheck.augment
   :members:
   :undoc-members:
   :show-inheritance:


.. toctree::
   :hidden:
   :maxdepth: 4

   langcheck.augment.en
