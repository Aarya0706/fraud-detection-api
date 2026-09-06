import os
import sys

# Ensure the repo root is on sys.path so `models.*` and `api.app` import
# cleanly regardless of where pytest is invoked from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
