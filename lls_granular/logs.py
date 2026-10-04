import logging
import os
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"


def setup(level: str | None = None, file: Path | None = None) -> None:
    """Configure logging for scripts. Level: argument, else $LLS_LOG, else INFO. Optional log file."""
    lvl = (level or os.environ.get("LLS_LOG", "INFO")).upper()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file:
        file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file))
    logging.basicConfig(level=lvl, format=FORMAT, datefmt="%H:%M:%S", handlers=handlers, force=True)
    logging.getLogger(__name__).debug("logging level %s, file %s", lvl, file)
