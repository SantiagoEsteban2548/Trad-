"""
Suite de tests de seguridad institucional para Apex Quant HFT (Fase 3):
- Autenticación Bearer token para API REST
- Validación de orígenes CORS
- Confirmación explícita (anti fat-finger) en endpoints destructivos/sensibles
- Bloqueo de inyección de caos fuera de PAPER_L2
- Sanitización de /api/health (cero fuga de secretos)
- Autenticación en WebSockets
"""
import os
import pytest
from fastapi.testclient import TestClient

from backend.telemetry_server import app, engine
from backend.security.auth import verify_bearer_token, get_configured_auth_token


@pytest.fixture
def test_client():
    return TestClient(app)


def test_health_endpoint_public_and_sanitized(test_client):
    """Verifica que /api/health sea público, no requiera auth y no filtre secretos."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_s" in data
    assert "mode" in data
    # Comprobar que no filtra API keys ni secretos
    assert "api_key" not in data
    assert "secret" not in data
    assert "gemini_key" not in data


def test_bearer_auth_rejection_when_configured(test_client, monkeypatch):
    """Verifica que con API_AUTH_TOKEN configurado, requests sin token o con token inválido sean rechazadas con 401."""
    monkeypatch.setenv("API_AUTH_TOKEN", "institutional-secret-token-1234")
    
    # Sin header Authorization
    res_no_auth = test_client.get("/api/telemetry")
    assert res_no_auth.status_code == 401
    assert "WWW-Authenticate" in res_no_auth.headers

    # Con token erróneo
    res_bad_auth = test_client.get(
        "/api/telemetry",
        headers={"Authorization": "Bearer bad-token"}
    )
    assert res_bad_auth.status_code == 401

    # Con token correcto
    res_good_auth = test_client.get(
        "/api/telemetry",
        headers={"Authorization": "Bearer institutional-secret-token-1234"}
    )
    assert res_good_auth.status_code == 200


def test_cors_headers(test_client):
    """Verifica el comportamiento de CORS para orígenes permitidos."""
    # Origin permitido localhost:5173
    res_allowed = test_client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        }
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_dangerous_endpoints_require_confirmation(test_client, monkeypatch):
    """Verifica que las acciones críticas exijan el campo confirm específico para evitar fat-finger."""
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER_L2")
    engine.trading_mode = "PAPER_L2"

    # 1. Reset state
    res_reset_no_confirm = test_client.post("/api/reset-state", json={})
    assert res_reset_no_confirm.status_code == 400
    assert "RESET_STATE" in res_reset_no_confirm.json()["detail"]

    # 2. Reset kill-switch
    res_ks_no_confirm = test_client.post("/api/reset-kill-switch", json={})
    assert res_ks_no_confirm.status_code == 400
    assert "RESET_KILL_SWITCH" in res_ks_no_confirm.json()["detail"]

    # 3. Base capital
    res_cap_no_confirm = test_client.post("/api/base-capital", json={"base_capital_usd": 15000.0})
    assert res_cap_no_confirm.status_code == 400
    assert "UPDATE_CAPITAL" in res_cap_no_confirm.json()["detail"]

    # 4. Trading mode LIVE sin confirm
    res_mode_no_confirm = test_client.post("/api/trading-mode", json={"mode": "LIVE_BINANCE"})
    assert res_mode_no_confirm.status_code == 400
    assert "confirm" in res_mode_no_confirm.json()["detail"]

    # 5. Stress event sin confirm
    res_stress_no_confirm = test_client.post("/api/stress-event", json={"event_type": "FLASH_CRASH_5PCT"})
    assert res_stress_no_confirm.status_code == 400
    assert "INJECT_CHAOS" in res_stress_no_confirm.json()["detail"]

    # 6. Gemini key sin confirm
    res_gemini_no_confirm = test_client.post("/api/gemini-key", json={"api_key": "some-key"})
    assert res_gemini_no_confirm.status_code == 400
    assert "UPDATE_KEY" in res_gemini_no_confirm.json()["detail"]


def test_chaos_injection_blocked_outside_paper_l2(test_client, monkeypatch):
    """Verifica que la inyección de caos esté bloqueada con 403 fuera de PAPER_L2."""
    monkeypatch.setenv("API_AUTH_TOKEN", "valid-auth-token")
    monkeypatch.setenv("TRADING_MODE", "PAPER_L2")
    headers = {"Authorization": "Bearer valid-auth-token"}
    original_mode = engine.trading_mode
    try:
        engine.trading_mode = "LIVE_BINANCE"
        res = test_client.post(
            "/api/stress-event",
            headers=headers,
            json={"event_type": "FLASH_CRASH_5PCT", "confirm": "INJECT_CHAOS"}
        )
        assert res.status_code == 403
        assert "forbidden outside PAPER_L2" in res.json()["detail"]
    finally:
        engine.trading_mode = original_mode


def test_websocket_token_auth(test_client, monkeypatch):
    """Verifica que la conexión WebSocket exija token si API_AUTH_TOKEN está definido."""
    monkeypatch.setenv("API_AUTH_TOKEN", "ws-secret-token")

    # Sin token -> debe cerrarse con WS 1008
    with pytest.raises(Exception):
        with test_client.websocket_connect("/ws/telemetry") as ws:
            ws.receive_text()

    # Con query param ?token=... correcto -> conexión exitosa
    with test_client.websocket_connect("/ws/telemetry?token=ws-secret-token") as ws:
        msg = ws.receive_text()
        assert len(msg) > 0
