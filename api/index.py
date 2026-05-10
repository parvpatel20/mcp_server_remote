import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Vercel unpacks the repo at a path like /var/task — not under a folder named
# `mcp_server_remote`. Register the package at ROOT so `from .rag import ...`
# in server.py resolves without relying on the parent directory name.
_PKG = "mcp_server_remote"
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [ROOT]
    sys.modules[_PKG] = _pkg

from mcp_server_remote.server import app

__all__ = ["app"]
