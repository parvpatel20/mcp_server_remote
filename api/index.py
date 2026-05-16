import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now 'src' is a real package, so relative imports inside it work
from server import app

__all__ = ["app"]