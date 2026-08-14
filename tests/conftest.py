"""Test-process configuration for the project's PyTorch-only embedding stack."""

import os

# This must be set before test-module collection imports langchain-text-splitters,
# which exposes an optional Sentence Transformers integration from its __init__.
os.environ.setdefault("USE_TF", "0")
