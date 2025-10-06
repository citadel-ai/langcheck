# Installation

To install LangCheck, just run:

```bash
# Install English metrics only
pip install langcheck

# Install English and Japanese metrics
pip install langcheck[ja]

# Install metrics for all languages (requires pip 21.2+)
pip install --upgrade pip
pip install langcheck[all]
```

LangCheck works with Python 3.10 or higher.

:::{note}
Model files are lazily downloaded the first time you run a metric function. For example, the first time you run the ``langcheck.metrics.sentiment()`` function, LangCheck will automatically download the Twitter-roBERTa-base model.
:::

To install LangCheck from source, see [the Contributing page](contributing.md).

## Installation FAQ

Depending on your environment, you might need to install additional system libraries for `pip install langcheck` to work.

If you have any problems installing LangCheck, please check the FAQ below or [open an issue on GitHub](https://github.com/citadel-ai/langcheck/issues).

### 1. Installing LangCheck with Python 3.12

As of February 2024, one of LangCheck's Japanese dependencies, [`fugashi`](https://github.com/polm/fugashi), doesn't provide wheels for Python 3.12, so you'll need to first install MeCab for pip to successfully build `fugashi`.

**Installing MeCab on Linux (Debian/Ubuntu)**

```bash
sudo apt-get update
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
```

**Installing MeCab on macOS**

```bash
brew install mecab
```

**Installing MeCab on Windows**

We haven't tested LangCheck on Windows with Python 3.12, but it may work if you install MeCab with [the official installer](https://taku910.github.io/mecab/#install).

### 2. The error message `command 'gcc' failed`

As of February 2024, one of LangCheck's Japanese dependencies, [`dartsclone`](https://github.com/s-yata/darts-clone), doesn't provide wheels for Python 3.10+. You'll need to have `gcc` installed for pip to successfully build `dartsclone`.

Most systems already have `gcc` installed, with the exception of "slim" Docker images such as [`python:3.11-slim-buster`](https://hub.docker.com/_/python). If you're using this image, you can:

1. Use `python:3.11-buster` instead of `python:3.11-slim-buster`, which includes gcc.
1. Run `apt update && apt install build-essential -y` (e.g. in your Dockerfile) to install gcc.

### 3. The error message `Failed to build sudachipy`

The `sudachipy` package, which is installed with `langcheck[all]` or `langcheck[ja-optional]`, does not have a pre-built wheel for ARM64 Linux devices.
In this case, you need have a Rust compiler in your environment so that the package can be built in the local environment.

Follow the [official documentation](https://www.rust-lang.org/tools/install) and install Rust before installing `langcheck`.

If you do not need the `[ja-optional]` dependencies, you can also install each language individually by

```sh
pip install langcheck[en,ja,de,zh]
```

### 4. The error message `Failed to build vllm`

The `vllm` package, which is installed with `langcheck[all]`, does not have a pre-built wheel for ARM devices, so ARM users would see the errors while building `vllm`.

You should try `pip install langcheck[no-local-llm]`, that will install all the dependencies other than `vllm`.

### 5. The error `AttributeError: module 'langcheck.metrics' has no attribute 'ja'`

If you see this error when calling a LangCheck function, such as `langcheck.metrics.ja.toxicity()`, you probably need to install the Japanese LangCheck package.

Run `pip install langcheck[ja]` (or for your required language) and try again.

### 6. The error `ModuleNotFoundError`

If you see this error when importing a LangCheck package, such as `import langcheck.metrics.ja`, you probably need to install the Japanese LangCheck package.

Run `pip install langcheck[ja]` (or for your required language) and try again.
