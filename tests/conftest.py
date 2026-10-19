import os
import pytest

@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """
    Aislamiento estricto de entorno para tests:
    1. Base de datos SQLite en memoria (:memory:).
    2. TRADING_MODE en PAPER_L2 por defecto para tests unitarios.
    3. Limpieza de API_AUTH_TOKEN por defecto para tests que no prueban autenticación.
    """
    monkeypatch.setenv("LEDGER_DB_PATH", ":memory:")
    monkeypatch.setenv("TRADING_MODE", "PAPER_L2")
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
