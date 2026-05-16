import os
import sys

# Add root to path so absolute imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now import rag directly (server.py must also use absolute import)
from server import app

__all__ = ["app"]