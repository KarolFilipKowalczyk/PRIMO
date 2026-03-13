"""PRIMO: Primacy of Inference over Physics in the Space of Minimal Programs."""

import sys as _sys

# Force UTF-8 stdout/stderr on Windows where the default codepage (e.g. cp1250)
# cannot encode Unicode box-drawing characters or Greek letters used in output.
if hasattr(_sys.stdout, 'reconfigure'):
    try:
        if _sys.stdout.encoding and _sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
            _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if _sys.stderr.encoding and _sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
            _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
