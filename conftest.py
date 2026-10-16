# conftest.py
from __future__ import annotations

import sys
from pathlib import Path

# Asegura que el directorio raíz del proyecto esté siempre en sys.path al invocar pytest
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
