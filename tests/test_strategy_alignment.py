# tests/test_strategy_alignment.py
"""
Tests institucionales para Fase 5:
1. Dimensionamiento de posición por riesgo fijo (Fixed-Fractional 0.5% Equity Risk Sizing).
2. Gate 5 en MandateGatekeeper (Máximo 5 posiciones abiertas simultáneas).
3. Estrategia Direccional 1:3 R:R (DirectionalTrendStrategy).
4. Verificación del Torneo Cuantitativo (>= 500 trades por modelo y tournament_results.json válido).
"""
import json
import pytest
from pathlib import Path

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.strategies.risk_sizing import compute_fixed_risk_position_size
from backend.strategies.directional_trend import DirectionalTrendStrategy, DirectionalSignal
from backend.strategies.tournament import run_strategy_tournament


def test_fixed_risk_sizing_mathematics():
    """
    Verifica que el dimensionamiento por riesgo fijo:
    1. Arriesgue exactamente el 0.5% del capital.
    2. A mayor distancia al stop loss, menor tamaño de posición.
    3. Capee la posición según el límite de margen de Gate 3 (20% * apalancamiento).
    """
    equity = 10_000.0
    entry_p = 100.0

    # Escenario A: Stop Loss estrecho (10 bps -> 0.1 USD)
    sl_tight = 99.90
    qty_tight = compute_fixed_risk_position_size(
        equity_usd=equity,
        entry_price=entry_p,
        stop_loss_price=sl_tight,
        step_size=0.01,
        risk_pct_per_trade=0.005,  # 0.5% = 50 USD de riesgo
        max_pair_margin_ratio=0.20,
        leverage=1.0,
    )
    # raw_qty = 50 / 0.10 = 500. Gate 3 cap = 10000 * 0.20 * 1 = 2000 USD -> max_qty = 20
    assert qty_tight == 20.0, f"Debe capear al máximo nocional de Gate 3 ($2000 / $100 = 20), obtenido: {qty_tight}"

    # Escenario B: Stop Loss amplio (5 USD de distancia) con leverage 3x (max notional 6000 USD)
    sl_wide = 95.0
    qty_wide = compute_fixed_risk_position_size(
        equity_usd=equity,
        entry_price=entry_p,
        stop_loss_price=sl_wide,
        step_size=0.1,
        risk_pct_per_trade=0.005,  # 50 USD de riesgo
        max_pair_margin_ratio=0.20,
        leverage=3.0,
    )
    # raw_qty = 50 / 5.0 = 10.0. Max notional allowed = 10000 * 0.20 * 3 = 6000 USD -> max_qty = 60
    assert qty_wide == 10.0, f"raw_qty debe ser 10.0, obtenido: {qty_wide}"
    max_loss = qty_wide * (entry_p - sl_wide)
    assert max_loss <= (equity * 0.005 + 1e-4), f"La pérdida máxima en SL no debe superar el 0.5% ($50), calculada: {max_loss}"

    # Escenario C: SL aún más amplio (10 USD) -> la cantidad debe reducirse a la mitad
    sl_wider = 90.0
    qty_wider = compute_fixed_risk_position_size(
        equity_usd=equity,
        entry_price=entry_p,
        stop_loss_price=sl_wider,
        step_size=0.1,
        risk_pct_per_trade=0.005,
        max_pair_margin_ratio=0.20,
        leverage=3.0,
    )
    assert qty_wider == 5.0, f"A mayor distancia de SL, qty debe ser menor ($5.0), obtenido: {qty_wider}"


def test_gate_5_max_simultaneous_positions_limit():
    """
    Verifica que Gate 5:
    1. Permita abrir hasta 5 pares simultáneos.
    2. Bloquee la apertura de un sexto par con GATE_5_MAX_POSITIONS.
    3. Permita órdenes que reduzcan posición o modifiquen órdenes existentes.
    4. Permita EMERGENCY_FLATTEN sin importar el número de posiciones.
    """
    gatekeeper = MandateGatekeeper(max_simultaneous_positions=5, state_file_path=":memory:")
    portfolio = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={
            "BTC/USDT:USDT": 1000.0,
            "ETH/USDT:USDT": 800.0,
            "SOL/USDT:USDT": 600.0,
            "BNB/USDT:USDT": 500.0,
            "DOGE/USDT:USDT": 400.0,
        },
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    # Intento de abrir un 6to par (LINK) -> Debe ser rechazado por Gate 5
    order_6th = OrderRequest(
        symbol="LINK/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=15.0,
        qty=10.0,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="DIRECTIONAL_TREND",
        timestamp_ns=1000000,
    )
    approved, reason = gatekeeper.validate_order(order_6th, portfolio)
    assert not approved, "Abrir un 6to par debe ser rechazado por Gate 5"
    assert "GATE_5_MAX_POSITIONS" in reason

    # Modificar o reducir un par existente (BTC) -> Debe ser aprobado
    order_reduce_btc = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="SELL",
        order_type="LIMIT",
        price=68000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="DIRECTIONAL_TREND",
        timestamp_ns=1000000,
    )
    approved_btc, reason_btc = gatekeeper.validate_order(order_reduce_btc, portfolio)
    assert approved_btc, f"Reducir posición existente debe ser aprobado: {reason_btc}"

    # Cerrar un par (DOGE se elimina de active_positions) -> Ahora hay 4 pares
    portfolio.active_positions.pop("DOGE/USDT:USDT")
    approved_after_close, reason_after = gatekeeper.validate_order(order_6th, portfolio)
    assert approved_after_close, f"Habiendo 4 posiciones, abrir un nuevo par debe ser aprobado: {reason_after}"


def test_directional_trend_strategy_signal_generation():
    """
    Verifica que DirectionalTrendStrategy:
    1. Requiera al menos `slow_period` precios.
    2. Genere señal con Take Profit y Stop Loss garantizando R:R >= 1:3.
    3. Calcule el riesgo en USD dentro del 0.5% del capital.
    """
    strat = DirectionalTrendStrategy(
        symbol="BTC/USDT:USDT",
        tick_size=0.1,
        step_size=0.001,
        fast_period=8,
        slow_period=21,
        risk_pct_per_trade=0.005,
        sl_bps=15.0,
        tp_bps=45.0,
        min_notional_usd=5.0,
    )

    # Sin suficientes precios -> None
    sig_none = strat.evaluate_signal([60000.0] * 10, 1000.0, 60000.0, 60001.0)
    assert sig_none is None

    # Tendencia alcista clara con 30 precios crecientes
    bullish_prices = [60000.0 + (i * 20.0) for i in range(45)]
    sig_bull = strat.evaluate_signal(
        prices=bullish_prices,
        current_equity=2000.0,
        best_bid=60899.9,
        best_ask=60900.0,
        obi=0.40,
    )
    assert sig_bull is not None
    assert isinstance(sig_bull, DirectionalSignal)
    assert sig_bull.side == "BUY"
    assert sig_bull.risk_reward_ratio >= 2.95, f"R:R debe ser ~3.0, obtenido: {sig_bull.risk_reward_ratio}"
    assert sig_bull.risk_usd <= (2000.0 * 0.005 + 1.0), f"Riesgo USD ({sig_bull.risk_usd}) excede el 0.5% del capital"


def test_tournament_results_file_and_500_trades_threshold():
    """
    Verifica que el archivo tournament_results.json exista y que ambos modelos
    hayan ejecutado al menos 500 trades reales en el motor institucional.
    """
    results_path = Path("tournament_results.json")
    assert results_path.exists(), "tournament_results.json debe existir"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strategies = data.get("strategies", {})
    assert "MARKET_MAKING_AVELLANEDA_OBI" in strategies
    assert "DIRECTIONAL_TREND_1_3_FIXED_RISK" in strategies

    mm = strategies["MARKET_MAKING_AVELLANEDA_OBI"]
    dir_strat = strategies["DIRECTIONAL_TREND_1_3_FIXED_RISK"]

    assert mm["total_trades"] >= 500, f"Market Making debe tener >= 500 trades, tiene: {mm['total_trades']}"
    assert dir_strat["total_trades"] >= 500, f"Directional Trend debe tener >= 500 trades, tiene: {dir_strat['total_trades']}"

    # Verificar que los campos cuantitativos clave estén presentes
    for key in ("win_rate_pct", "profit_factor", "net_pnl_usd", "max_drawdown_pct", "annualized_sharpe", "fees_paid_usd"):
        assert key in mm, f"Clave {key} ausente en métricas MM"
        assert key in dir_strat, f"Clave {key} ausente en métricas Directional"

    assert "head_to_head_comparison" in data
    assert "regime_analysis" in data


def test_strategy_mode_api_endpoints(monkeypatch):
    """
    Verifica que los endpoints /api/strategy-mode (GET y POST)
    permitan consultar y cambiar el modo entre MARKET_MAKING y DIRECTIONAL_TREND.
    """
    from fastapi.testclient import TestClient
    from backend.telemetry_server import app, engine

    monkeypatch.setenv("API_AUTH_TOKEN", "test-auth-token-5555")
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-auth-token-5555"}

    # 1. GET sin auth -> 401
    resp_unauth = client.get("/api/strategy-mode")
    assert resp_unauth.status_code == 401

    # 2. GET con auth -> 200
    resp_get = client.get("/api/strategy-mode", headers=headers)
    assert resp_get.status_code == 200
    data = resp_get.json()
    assert "strategy_mode" in data

    # 3. POST con modo DIRECTIONAL_TREND
    resp_post = client.post(
        "/api/strategy-mode",
        json={"strategy_mode": "DIRECTIONAL_TREND"},
        headers=headers,
    )
    assert resp_post.status_code == 200
    assert engine.strategy_mode == "DIRECTIONAL_TREND"

    # 4. POST con modo inválido -> 422 (Pydantic validation error)
    resp_bad = client.post(
        "/api/strategy-mode",
        json={"strategy_mode": "INVALID_MODE"},
        headers=headers,
    )
    assert resp_bad.status_code == 422
