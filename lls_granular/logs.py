import logging
import os

FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"


def setup(level: str | None = None) -> None:
    """Configure logging for scripts. Level: argument, else $LLS_LOG, else INFO."""
    lvl = (level or os.environ.get("LLS_LOG", "INFO")).upper()
    logging.basicConfig(level=lvl, format=FORMAT, datefmt="%H:%M:%S", force=True)
    logging.getLogger(__name__).debug("logging level %s", lvl)
