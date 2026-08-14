"""Configure Transformers for this project's PyTorch-only embedding stack."""

import os

os.environ.setdefault("USE_TF", "0")
