"""
Sparkleframe
"""

import os

# Re-exported so users (and sparkleframe.context) can do `from sparkleframe import Engine`.
from .engine import Engine  # noqa: F401

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r", encoding="utf-8") as f:
    __version__ = f.read()
