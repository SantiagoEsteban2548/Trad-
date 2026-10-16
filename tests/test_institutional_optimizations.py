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
