"""
Shared text preprocessing for the NLP emotion pipeline.

Keeping this in a standalone, importable module (rather than inside the training
script) matters because the TF-IDF vectoriser stores a reference to this
function when the pipeline is saved with joblib. On load, Python must be able to
import it by its module path - a function defined in ``__main__`` would fail.
"""

import re


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace - a light, explainable preprocessing step."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
