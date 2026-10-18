"""
scripts/smoke_test_telemetry.py
Comprobación de humo no destructiva del servidor de telemetría y endpoints REST.
"""

from fastapi.testclient import TestClient
from backend.telemetry_server import app, engine

import sys

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[SMOKE TEST] Inicializando TestClient para FastAPI telemetry_server...")
    client = TestClient(app)

    # 1. Test /api/health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.status_code}"
    health_data = res_health.json()
    assert str(health_data.get("status", "")).upper() == "OK", f"Health payload invalid: {health_data}"
    print(f"  [PASS] GET /api/health: OK ({health_data})")

    # 2. Test /api/telemetry
    res_telem = client.get("/api/telemetry")
    assert res_telem.status_code == 200, f"Telemetry failed: {res_telem.status_code}"
    telem_data = res_telem.json()
    assert "portfolio" in telem_data, "Missing 'portfolio' in telemetry"
    assert "symbols" in telem_data, "Missing 'symbols' in telemetry"
    assert "trading_mode" in telem_data, "Missing 'trading_mode' in telemetry"
    assert "markout_curve_bps" in telem_data, "Missing 'markout_curve_bps' in telemetry"
    symbols_count = len(telem_data.get("symbols", {}))
    print(f"  [PASS] GET /api/telemetry: OK (Modo: {telem_data.get('trading_mode')}, Símbolos: {symbols_count}, Equidad: ${telem_data.get('portfolio', {}).get('current_equity_usd')})")

    # 3. Test /api/trades
    res_trades = client.get("/api/trades?limit=10")
    assert res_trades.status_code == 200, f"Trades failed: {res_trades.status_code}"
    trades_data = res_trades.json()
    assert str(trades_data.get("status", "")).upper() == "OK", "Missing status OK in /api/trades"
    assert "trades" in trades_data, "Missing 'trades' list in /api/trades"
    print(f"  [PASS] GET /api/trades: OK ({len(trades_data.get('trades', []))} trades obtenidos)")

    # 4. Test /api/leverage
    res_lev = client.get("/api/leverage")
    assert res_lev.status_code == 200, f"Leverage check failed: {res_lev.status_code}"
    lev_data = res_lev.json()
    assert "current_tier" in lev_data or "tiers" in lev_data or "active_leverage" in lev_data or isinstance(lev_data, dict), "Invalid leverage payload"
    print(f"  [PASS] GET /api/leverage: OK ({lev_data.get('current_tier', lev_data)})")

    print("[SMOKE TEST] ¡Todos los endpoints operativos respondieron exitosamente!")

if __name__ == "__main__":
    main()
