"""
RescuGuard AI - Test Suite Package Initializer
-----------------------------------------------
Ensures the backend directory is in the Python module search path
so that tests can be executed seamlessly from any working directory.
"""

import sys
from pathlib import Path

# Automatically add backend/ root to sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))
