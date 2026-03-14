"""
Thin wrapper to import the FastAPI app from the hyphenated filename
so that uvicorn can be pointed at ``src.serving.run:app``.
"""

import importlib

_api = importlib.import_module("src.serving.medllama-api-serve")
app = _api.app
