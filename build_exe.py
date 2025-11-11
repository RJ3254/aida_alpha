"""Create a standalone executable using PyInstaller."""
from __future__ import annotations

import PyInstaller.__main__


if __name__ == "__main__":
    PyInstaller.__main__.run(
        [
            "tts_app/main.py",
            "--name=coqui_tts_studio",
            "--onefile",
            "--noconfirm",
        ]
    )
