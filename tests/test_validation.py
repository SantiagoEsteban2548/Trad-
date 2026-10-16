# tests/test_validation.py
from __future__ import annotations

import math
import numpy as np
import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.schemas import OrderRequest
from backend.validation.l2_simulator_and_dsr import (
    DeflatedSharpeAuditor,
    L2QueueSimulator,
)


@pytest.fixture
def populated_ring_buffer() -> MarketRingBuffer:
    """Buffer con libro L2: Bid0 = 50,000 con 5.0 BTC de volumen inicial en cola."""
    buffer = MarketRingBuffer(capacity=100)
    buffer.push_order_book(
        bid_prices=[50000.0, 49990.0, 49980.0, 49970.0, 49960.0],
        bid_qtys=[5.0, 10.0, 15.0, 20.0, 25.0],
        ask_prices=[50010.0, 50020.0, 50030.0, 50040.0, 50050.0],
        ask_qtys=[4.0, 8.0, 12.0, 16.0, 20.0],
        timestamp_ns=1_000_000_000,
    )
    return buffer


def test_l2_queue_simulator_fifo_queue_ahead(populated_ring_buffer: MarketRingBuffer) -> None:
    """
    Verifica que la orden solo se llene cuando el volumen acumulado de ventas
    a mercado agote completamente `queue_ahead_qty` (5.0 BTC).
    """
    sim = L2QueueSimulator(latency_ms=10.0)
    now_ns = 1_000_000_000

    order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50000.0,
        qty=1.0,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=now_ns,
    )

    sim_order = sim.submit_gtx_order(order, populated_ring_buffer, current_ts_ns=now_ns)
    assert sim_order.status == "OPEN"
    assert sim_order.queue_ahead_qty == 5.0
    assert sim_order.filled_qty == 0.0

    # Trade después del tiempo de llegada (10 ms = 10_000_000 ns)
    trade_ts_ns = now_ns + 15_000_000

    # 1. Venta a mercado de 3.0 BTC: no debe llenar nuestra orden (consume solo parte de queue_ahead)
    fills1 = sim.on_market_trade(
        trade_price=50000.0,
        trade_qty=3.0,
        is_buyer_maker=True,
        trade_ts_ns=trade_ts_ns,
    )
    assert len(fills1) == 0
    assert sim_order.queue_ahead_qty == 2.0
    assert sim_order.status == "OPEN"

    # 2. Venta a mercado de 2.0 BTC: agota exactamente la cola previa, pero no llena nuestra orden
    fills2 = sim.on_market_trade(
        trade_price=50000.0,
        trade_qty=2.0,
        is_buyer_maker=True,
        trade_ts_ns=trade_ts_ns + 1_000_000,
    )
    assert len(fills2) == 0
    assert sim_order.queue_ahead_qty == 0.0
    assert sim_order.filled_qty == 0.0

    # 3. Venta a mercado de 1.0 BTC: finalmente llena nuestra orden de 1.0 BTC
    fills3 = sim.on_market_trade(
        trade_price=50000.0,
        trade_qty=1.0,
        is_buyer_maker=True,
        trade_ts_ns=trade_ts_ns + 2_000_000,
    )
    assert len(fills3) == 1
    assert sim_order.status == "FILLED"
    assert sim_order.filled_qty == 1.0
    assert sim_order.avg_fill_price == 50000.0


def test_l2_queue_simulator_adverse_selection_during_cancel_latency(
    populated_ring_buffer: MarketRingBuffer,
) -> None:
    """
    Simula la latencia de cancelación de 15 ms:
    Si un trade agresor golpea nuestro nivel antes de que la cancelación
    surta efecto, la orden resulta ejecutada (adverse selection in-flight).
    """
    sim = L2QueueSimulator(latency_ms=15.0)  # 15 ms = 15,000,000 ns
    t0_ns = 1_000_000_000

    order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50000.0,
        qty=0.5,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=t0_ns,
    )
    sim_order = sim.submit_gtx_order(order, populated_ring_buffer, current_ts_ns=t0_ns)

    # La orden queda viva a t0 + 15ms = 1_015_000_000 ns
    # A t0 + 20ms, OBI Shield detecta toxicidad y solicita cancelar
    t_cancel_req = t0_ns + 20_000_000
    sim.request_cancel(sim_order.order_id, request_ts_ns=t_cancel_req)
    assert sim_order.status == "CANCEL_PENDING"
    # La cancelación será efectiva a t_cancel_req + 15ms = t0 + 35ms (1_035_000_000 ns)

    # A t0 + 25ms (en pleno vuelo de la cancelación), ocurre un flash crash que barre el nivel
    # Precio cae a 49,990 atravesando nuestro nivel 50,000
    t_crash = t0_ns + 25_000_000
    fills = sim.on_market_trade(
        trade_price=49990.0,
        trade_qty=10.0,
        is_buyer_maker=True,
        trade_ts_ns=t_crash,
    )
    # Debe haberse ejecutado desfavorablemente antes de que la cancelación llegara al exchange
    assert len(fills) == 1
    assert sim_order.status == "FILLED"
    assert sim_order.filled_qty == 0.5


def test_l2_queue_simulator_post_only_crossing_rejected(
    populated_ring_buffer: MarketRingBuffer,
) -> None:
    """Una orden límite GTX que cruzaría el libro inmediatamente debe ser POST_ONLY_REJECTED."""
    sim = L2QueueSimulator(latency_ms=10.0)
    # Best ask está en 50,010. Una orden BUY a 50,015 cruza el spread.
    bad_gtx = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50015.0,
        qty=0.1,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_000_000_000,
    )
    sim_order = sim.submit_gtx_order(bad_gtx, populated_ring_buffer)
    assert sim_order.status == "POST_ONLY_REJECTED"


def test_deflated_sharpe_auditor_clean_strategy_passes() -> None:
    """Una estrategia con alto Sharpe, serie larga y asimetría positiva supera el umbral DSR >= 0.95."""
    result = DeflatedSharpeAuditor.compute_dsr(
        observed_sr=2.8,
        sample_length_t=500,
        skewness_gamma3=0.2,
        kurtosis_gamma4=3.5,
        num_trials_n=10,
    )
    assert result.deflated_sharpe_ratio >= 0.95
    assert result.status == "STATISTICALLY_SIGNIFICANT"
    assert result.is_overfitted is False


def test_deflated_sharpe_auditor_flags_overfitting_warning() -> None:
    """
    Si se probaron 10,000 configuraciones (data-snooping) con una muestra corta (T=50)
    y cola negativa severa (gamma3 = -1.2, gamma4 = 6.0), el DSR debe colapsar < 0.95.
    """
    result = DeflatedSharpeAuditor.compute_dsr(
        observed_sr=0.50,
        sample_length_t=50,
        skewness_gamma3=-1.2,
        kurtosis_gamma4=6.0,
        num_trials_n=10_000,
    )
    assert result.deflated_sharpe_ratio < 0.95
    assert result.status == "OVERFITTING_WARNING"
    assert result.is_overfitted is True


def test_deflated_sharpe_auditor_audit_returns_series() -> None:
    """Audita retornos muestrales directamente con auditor.audit_returns()."""
    rng = np.random.default_rng(seed=42)
    # Serie de 250 retornos diarios normales con media positiva
    returns = rng.normal(loc=0.001, scale=0.01, size=250)

    audit = DeflatedSharpeAuditor.audit_returns(returns, num_trials_n=5)
    assert audit.sample_length_t == 250
    assert audit.observed_sharpe_sr > 0.0
    assert 0.0 <= audit.deflated_sharpe_ratio <= 1.0
    assert isinstance(audit.is_overfitted, bool)
