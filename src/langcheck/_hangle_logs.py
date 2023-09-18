import logging
from contextlib import contextmanager


@contextmanager
def _handle_logging_level(muximum_level=logging.WARNING):
    '''Context manager to surpress any logging messages under specified level.

    Args:
        The highest logging level to be disabled. Defalut to logging.WARNING.
    '''

    current_logging_level = logging.root.getEffectiveLevel()
    logging.disable(muximum_level)

    try:
        yield
    finally:
        logging.disable(current_logging_level)
