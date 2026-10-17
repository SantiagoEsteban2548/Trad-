# tests/test_institutional_optimizations.py
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.telemetry_server import DryRunExecutionEngine
from backend.mandate_gates.schemas import PortfolioState


def test_compute_microprice_empty_buffer() -> None:
    rb = MarketRingBuffer(capacity=100)
    assert rb.compute_microprice() == 0.0


def test_compute_microprice_balanced_spread() -> None:
    rb = MarketRingBuffer(capacity=100)
    now_ns = time.time_ns()
    # Bid $65,000 con 10 BTC, Ask $65,010 con 10 BTC (volumen idéntico)
    rb.push_order_book(
        bid_prices=[65_000.0, 64_990.0, 64_980.0, 64_970.0, 64_960.0],
        bid_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        ask_prices=[65_010.0, 65_020.0, 65_030.0, 65_040.0, 65_050.0],
        ask_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        timestamp_ns=now_ns,
    )
    # (10 * 65010 + 10 * 65000) / 20 = 65005.0
    micro = rb.compute_microprice()
    assert pytest.approx(micro, rel=1e-5) == 65_005.0


def test_compute_microprice_asymmetric_order_flow() -> None:
    rb = MarketRingBuffer(capacity=100)
    now_ns = time.time_ns()
    # Presión compradora: 30 BTC en Bid_0 vs 10 BTC en Ask_0
    rb.push_order_book(
        bid_prices=[65_000.0, 64_990.0, 64_980.0, 64_970.0, 64_960.0],
        bid_qtys=[30.0, 10.0, 10.0, 10.0, 10.0],
        ask_prices=[65_010.0, 65_020.0, 65_030.0, 65_040.0, 65_050.0],
        ask_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        timestamp_ns=now_ns,
    )
    # (30 * 65010 + 10 * 65000) / 40 = (1950300 + 650000) / 40 = 2600300 / 40 = 65007.5
    micro = rb.compute_microprice()
    assert pytest.approx(micro, rel=1e-5) == 65_007.5
    # El microprice se desplaza hacia el ask por la presión asimétrica de compra
    assert micro > 65_005.0


def test_compute_tfi_empty_and_pure_flow() -> None:
    rb = MarketRingBuffer(capacity=100)
    assert rb.compute_tfi(window_ns=500_000_000) == 0.0

    t0_ns = 1_000_000_000_000  # 1000 s
    # 1. Compra agresiva taker (is_buyer_maker=False) por 5.0 BTC
    rb.push_trade(price=65_000.0, qty=5.0, is_buyer_maker=False, timestamp_ns=t0_ns)
    assert pytest.approx(rb.compute_tfi(window_ns=500_000_000), abs=1e-4) == 1.0

    # 2. Venta agresiva taker (is_buyer_maker=True) por 5.0 BTC en la misma ventana
    rb.push_trade(price=65_000.0, qty=5.0, is_buyer_maker=True, timestamp_ns=t0_ns + 100_000_000)
    assert pytest.approx(rb.compute_tfi(window_ns=500_000_000), abs=1e-4) == 0.0

    # 3. Venta agresiva taker adicional por 10.0 BTC (Total compras=5, ventas=15)
    rb.push_trade(price=64_999.0, qty=10.0, is_buyer_maker=True, timestamp_ns=t0_ns + 200_000_000)
    # TFI = (5 - 15) / (5 + 15) = -10 / 20 = -0.5
    assert pytest.approx(rb.compute_tfi(window_ns=500_000_000), abs=1e-4) == -0.5


def test_compute_tfi_window_decay() -> None:
    rb = MarketRingBuffer(capacity=100)
    t0_ns = 1_000_000_000_000
    # Operación de compra hace 1000 ms (fuera de la ventana de 500 ms)
    rb.push_trade(price=65_000.0, qty=50.0, is_buyer_maker=False, timestamp_ns=t0_ns)
    # Operación de venta hace 100 ms (dentro de la ventana de 500 ms)
    rb.push_trade(price=65_000.0, qty=2.0, is_buyer_maker=True, timestamp_ns=t0_ns + 900_000_000)

    # Ventana de 500 ms: el último trade está en t0 + 900ms. El cutoff es t0 + 400ms.
    # El trade de 50 BTC en t0 está excluido. Sólo queda el trade vendedor de 2 BTC -> TFI = -1.0
    tfi = rb.compute_tfi(window_ns=500_000_000)
    assert pytest.approx(tfi, abs=1e-4) == -1.0


def test_dynamic_normalized_inventory_q() -> None:
    engine = DryRunExecutionEngine()
    engine.portfolio.base_capital_usd = 100.0
    # Límite de concentración Gate 3 = 20% de $100 = 20.0 USD
    engine.portfolio.active_positions.clear()

    # 1. Sin posiciones -> q = 0.0
    assert engine.compute_normalized_inventory_q() == 0.0

    # 2. Posición larga moderada de $10.0 USD -> q = +0.50
    engine.portfolio.active_positions[engine.primary_symbol] = 10.0
    assert pytest.approx(engine.compute_normalized_inventory_q(), abs=1e-4) == 0.50

    # 3. Posición corta moderada de -$15.0 USD -> q = -0.75
    engine.portfolio.active_positions[engine.primary_symbol] = -15.0
    assert pytest.approx(engine.compute_normalized_inventory_q(), abs=1e-4) == -0.75

    # 4. Posición larga saturada al techo de $20.0 USD -> q = +1.0
    engine.portfolio.active_positions[engine.primary_symbol] = 20.0
    assert pytest.approx(engine.compute_normalized_inventory_q(), abs=1e-4) == 1.0

    # 5. Acotamiento en exceso (límite superado a $35 USD) -> clamped a +1.0
    engine.portfolio.active_positions[engine.primary_symbol] = 35.0
    assert engine.compute_normalized_inventory_q() == 1.0

    # 6. Acotamiento en corto excesivo (-$35 USD) -> clamped a -1.0
    engine.portfolio.active_positions[engine.primary_symbol] = -35.0
    assert engine.compute_normalized_inventory_q() == -1.0


def test_dsr_quarantine_persistence_and_restore(tmp_path: Path) -> None:
    test_state_file = tmp_path / "dsr_quarantine_test.json"
    engine = DryRunExecutionEngine()
    engine.dsr_state_path = test_state_file

    # Modificar estado en memoria
    engine.returns_history = [0.0012, 0.0015, -0.0004, 0.0021]
    engine.total_rotations_filled = 48
    engine._last_fill_buy_price = 64_800.0
    engine._last_fill_sell_price = 64_820.0
    engine.portfolio.current_equity_usd = 105.40

    # Persistir
    engine._save_dsr_quarantine_state()
    assert test_state_file.exists()

    with open(test_state_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["total_rotations_filled"] == 48
    assert saved_data["last_fill_buy_price"] == 64_800.0
    assert saved_data["last_fill_sell_price"] == 64_820.0
    assert len(saved_data["returns_history"]) == 4

    # Crear una nueva instancia y restaurar
    engine_restored = DryRunExecutionEngine()
    engine_restored.dsr_state_path = test_state_file
    engine_restored._load_dsr_quarantine_state()

    assert engine_restored.total_rotations_filled == 48
    assert engine_restored._last_fill_buy_price == 64_800.0
    assert engine_restored._last_fill_sell_price == 64_820.0
    assert engine_restored.returns_history == [0.0012, 0.0015, -0.0004, 0.0021]


def test_compute_vpin_calculation() -> None:
    buf = MarketRingBuffer(capacity=100)
    # 1. Sin trades suficientes retorna baseline neutral 0.25
    assert buf.compute_vpin() == 0.25

    # 2. Inyectar flujo altamente tóxico unidireccional (todos seller-initiated agresivos)
    now_ns = time.time_ns()
    for i in range(25):
        # is_buyer_maker=False significa que el comprador fue maker y el vendedor fue taker agresivo
        buf.push_trade(price=65000.0 - i, qty=2.0, is_buyer_maker=True, timestamp_ns=now_ns + (i * 10_000_000))

    vpin = buf.compute_vpin(bucket_volume=5.0, num_buckets=5)
    assert 0.0 <= vpin <= 1.0
    # Con flujo puro en una sola dirección el desequilibrio de volumen por bucket es muy alto (>0.70)
    assert vpin >= 0.70


def test_post_trade_markout_curve_calculation() -> None:
    engine = DryRunExecutionEngine()
    sym = "BTC/USDT:USDT"
    t0 = 1700000000.0

    # 1. Registrar fill de compra a $60,000
    trade = engine.record_trade_fill(
        symbol=sym,
        strategy="AVELLANEDA_OBI",
        side="BUY",
        entry_price=60000.0,
        exit_or_mid_price=60000.0,
        qty=0.01,
        return_pct=0.0,
        pnl_usd=0.0,
        timestamp_ns=int(t0 * 1e9),
    )
    assert len(engine.pending_markouts) >= 1
    pending_item = engine.pending_markouts[-1]
    assert pending_item["symbol"] == sym
    assert pending_item["mid_t0"] == 60000.0

    # 2. Simular que a +1.5s el mid sube a $60,012 (+2.0 bps)
    ctx = engine.symbol_contexts[sym]
    idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
    ctx.ring_buffer.ob_bid_prices[idx, 0] = 60010.0
    ctx.ring_buffer.ob_ask_prices[idx, 0] = 60014.0  # mid = 60012.0

    engine.update_post_trade_markouts(now_s=t0 + 1.5)
    assert "1s" in pending_item["markouts"]
    assert pending_item["markouts"]["1s"] == 2.0

    # 3. Simular que a +5.5s el mid sube a $60,018 (+3.0 bps)
    ctx.ring_buffer.ob_bid_prices[idx, 0] = 60016.0
    ctx.ring_buffer.ob_ask_prices[idx, 0] = 60020.0  # mid = 60018.0
    engine.update_post_trade_markouts(now_s=t0 + 5.5)
    assert "5s" in pending_item["markouts"]
    assert pending_item["markouts"]["5s"] == 3.0

    # 4. Simular que a +16.0s el mid sube a $60,024 (+4.0 bps)
    ctx.ring_buffer.ob_bid_prices[idx, 0] = 60022.0
    ctx.ring_buffer.ob_ask_prices[idx, 0] = 60026.0  # mid = 60024.0
    engine.update_post_trade_markouts(now_s=t0 + 16.0)
    assert "15s" in pending_item["markouts"]
    assert pending_item["markouts"]["15s"] == 4.0

    # 5. Simular que a +31.0s el mid sube a $60,030 (+5.0 bps)
    ctx.ring_buffer.ob_bid_prices[idx, 0] = 60028.0
    ctx.ring_buffer.ob_ask_prices[idx, 0] = 60032.0  # mid = 60030.0
    engine.update_post_trade_markouts(now_s=t0 + 31.0)
    assert "30s" in pending_item["markouts"]
    assert pending_item["markouts"]["30s"] == 5.0
    assert engine.markout_curve_bps["30s"] == 5.0


def test_btc_gold_correlation_and_safe_haven_allocation() -> None:
    import backend.telemetry_server as ts
    ts.GLOBAL_MOCK_TRADFI_STATE = True
    try:
        engine = DryRunExecutionEngine()

        # Inyectar retornos descorrelacionados
        engine.symbol_returns["BTC/USDT:USDT"] = [0.01, -0.015, 0.02, -0.01, 0.005, -0.02, 0.015]
        engine.symbol_returns["XAU/USDT:USDT"] = [0.002, 0.001, -0.001, 0.003, -0.002, 0.004, -0.001]

        corr = engine.compute_rolling_correlation("BTC/USDT:USDT", "XAU/USDT:USDT")
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

        # Activar alerta de caída sistémica en cripto
        engine.contagion_shield.bid_shield_active = True
        weights = engine.compute_multivariate_half_kelly()

        # Si la correlación es baja (<= 0.15), XAU recibe el 20% prioritario de refugio
        if engine.btc_gold_corr <= 0.15:
            assert weights["XAU/USDT:USDT"] == 0.20
            assert engine.contagion_shield.xau_safe_haven_active is True
    finally:
        ts.GLOBAL_MOCK_TRADFI_STATE = None


def test_stress_event_api_endpoint() -> None:
    from fastapi.testclient import TestClient
    from backend.telemetry_server import app, engine

    client = TestClient(app)

    # 1. Inyectar Flash Crash -5%
    res_crash = client.post("/api/stress-event", json={"event_type": "FLASH_CRASH_5PCT", "symbol": "BTC/USDT:USDT"})
    assert res_crash.status_code == 200
    data_crash = res_crash.json()
    assert data_crash["status"] == "STRESS_EVENT_INJECTED"
    assert engine.contagion_shield.bid_shield_active is True
    assert engine.contagion_shield.trigger_reason == "FLASH_CRASH_5PCT_INJECTED"

    # 2. Inyectar Spike de Latencia de 200ms
    res_lat = client.post("/api/stress-event", json={"event_type": "LATENCY_SPIKE_200MS"})
    assert res_lat.status_code == 200
    assert engine.chaos_latency_active is True
    assert engine.symbol_contexts["BTC/USDT:USDT"].l2_sim.latency_ms == 200.0

    # 3. Inyectar Cierre TradFi Weekend Close
    res_tradfi = client.post("/api/stress-event", json={"event_type": "TRADFI_WEEKEND_CLOSE"})
    assert res_tradfi.status_code == 200
    assert engine.mock_tradfi_closed is True

    # 4. Resetear Caos
    res_reset = client.post("/api/stress-event", json={"event_type": "RESET_CHAOS"})
    assert res_reset.status_code == 200
    assert engine.chaos_latency_active is False
    assert engine.mock_tradfi_closed is False
    assert engine.symbol_contexts["BTC/USDT:USDT"].l2_sim.latency_ms == 15.0

