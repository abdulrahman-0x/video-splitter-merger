"""Application entry point for Video Splitter & Merger."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.app import App
from src.utils.logging_setup import setup_logging


def main() -> None:
    """Initialize logging and run the CustomTkinter GUI mainloop."""
    setup_logging()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
