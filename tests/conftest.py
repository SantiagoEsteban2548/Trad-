import os
import pytest

@pytest.fixture(autouse=True)
def isolate_test_ledger_database(monkeypatch):
    """
    Aislamiento estricto de base de datos para tests:
    Fuerza el uso de SQLite en memoria (:memory:) para todas las suites de prueba,
    impidiendo que los tests contaminen o modifiquen la base de datos real en backend/data/trades_ledger.db.
    """
    monkeypatch.setenv("LEDGER_DB_PATH", ":memory:")
