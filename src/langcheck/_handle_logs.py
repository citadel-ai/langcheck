import logging
from contextlib import contextmanager


@contextmanager
def _handle_logging_level(maximum_level=logging.WARNING):
    '''Context manager to suppress any logging messages under specified level.

    Args:
        The highest logging level to be disabled. Default to logging.WARNING.
    '''

    current_logging_level = logging.root.getEffectiveLevel()
    logging.disable(maximum_level)

    try:
        yield
    finally:
        logging.disable(current_logging_level)
