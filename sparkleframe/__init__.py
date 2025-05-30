"""
Sparkleframe
"""

import os

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r", encoding="utf-8") as f:
    __version__ = f.read()
