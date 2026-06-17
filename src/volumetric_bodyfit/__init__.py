import logging

from nn_core.console_logging import NNRichHandler


lightning_logger = logging.getLogger("pytorch_lightning")
for handler in lightning_logger.handlers[:]:
    lightning_logger.removeHandler(handler)
lightning_logger.propagate = True

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        NNRichHandler(
            rich_tracebacks=True,
            show_level=True,
            show_path=True,
            show_time=True,
            omit_repeated_times=True,
        )
    ],
)

try:
    from ._version import __version__ as __version__
except ImportError:
    __version__ = "unknown"
