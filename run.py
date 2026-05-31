"""民航法律机器人命令行启动脚本。"""

import os
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_DIR / ".venv" / "bin" / "python"


def _in_venv() -> bool:
    """Check we're running inside the project's .venv, not system/conda Python."""
    return sys.prefix.startswith(str(_PROJECT_DIR / ".venv"))


def _relaunch() -> None:
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv[:])


if __name__ == "__main__":
    if _VENV_PYTHON.exists() and not _in_venv():
        _relaunch()

    from legalbot.cli import main

    raise SystemExit(main())
