"""
main.py
=======
VisionGuard AI — Smart Face Recognition & Robot Tracking System.

Entry point: initializes configuration/logging, then launches the
CustomTkinter desktop application.

Run with:
    python main.py
"""

from __future__ import annotations

import sys
import traceback

from app.config import get_config
from app.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    try:
        get_config()  # Load config.yaml and ensure directory structure exists.
        logger.info("Starting VisionGuard AI...")

        from app.ui.main_window import MainWindow

        app = MainWindow()
        app.mainloop()
        logger.info("VisionGuard AI closed normally.")
        return 0
    except Exception:  # noqa: BLE001 - top-level guard, never crash silently
        logger.error("Fatal error during startup:\n%s", traceback.format_exc())
        print("A fatal error occurred. See data/logs/visionguard.log for details.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
