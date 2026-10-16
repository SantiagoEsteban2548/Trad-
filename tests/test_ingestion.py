# tests/test_ingestion.py
from __future__ import annotations

import time
import numpy as np
import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer


def test_market_ring_buffer_obi_calculation() -> None:
    buffer = MarketRingBuffer(capacity=100)

    # Caso 1: Asimetría compradora pura (Bids llenos, Asks vacíos)
    buffer.push_order_book(
        bid_prices=[50000.0, 49990.0, 49980.0, 49970.0, 49960.0],
        bid_qtys=[2.0, 3.0, 5.0, 1.0, 4.0],  # Suma = 15.0
        ask_prices=[50010.0, 50020.0, 50030.0, 50040.0, 50050.0],
        ask_qtys=[0.0, 0.0, 0.0, 0.0, 0.0],  # Suma = 0.0
        timestamp_ns=1_000_000,
    )
    assert buffer.compute_obi(levels=5) == 1.0

    # Caso 2: Equilibrio perfecto de volumen
    buffer.push_order_book(
        bid_prices=[50000.0, 49990.0, 49980.0, 49970.0, 49960.0],
        bid_qtys=[1.0, 2.0, 3.0, 4.0, 5.0],  # Suma = 15.0
        ask_prices=[50010.0, 50020.0, 50030.0, 50040.0, 50050.0],
        ask_qtys=[1.0, 2.0, 3.0, 4.0, 5.0],  # Suma = 15.0
        timestamp_ns=2_000_000,
    )
    assert buffer.compute_obi(levels=5) == 0.0

    # Caso 3: Asimetría vendedora (Bids = 5, Asks = 15 -> (5-15)/20 = -0.5)
    buffer.push_order_book(
        bid_prices=[50000.0, 49990.0, 49980.0, 49970.0, 49960.0],
        bid_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],  # Suma = 5.0
        ask_prices=[50010.0, 50020.0, 50030.0, 50040.0, 50050.0],
        ask_qtys=[3.0, 3.0, 3.0, 3.0, 3.0],  # Suma = 15.0
        timestamp_ns=3_000_000,
    )
    assert pytest.approx(buffer.compute_obi(levels=5), abs=1e-6) == -0.5


def test_realized_volatility_calculation() -> None:
    buffer = MarketRingBuffer(capacity=500)
    base_ts = 1_000_000_000_000

    prices = [50000.0 + (i * 2.5 if i % 2 == 0 else -i * 1.8) for i in range(100)]
    for i, p in enumerate(prices):
        buffer.push_order_book(
            bid_prices=[p - 0.5] * 5,
            bid_qtys=[1.0] * 5,
            ask_prices=[p + 0.5] * 5,
            ask_qtys=[1.0] * 5,
            timestamp_ns=base_ts + (i * 1_000_000_000),
        )

    vol = buffer.compute_realized_volatility(window_seconds=300)
    assert vol > 0.0
    assert isinstance(vol, float)


def test_liquidation_cluster_and_zscore() -> None:
    buffer = MarketRingBuffer(capacity=1_000)
    now_ns = 1_700_000_000_000_000_000

    for i in range(100):
        p = 50000.0 if i < 50 else 50100.0
        buffer.push_trade(
            price=p,
            qty=1.0,
            is_buyer_maker=False,
            timestamp_ns=now_ns - int((60 - (i * 0.5)) * 1e9),
        )

    for i in range(20):
        buffer.push_liquidation(
            side="SELL",
            price=50000.0,
            qty=0.5,
            timestamp_ns=now_ns - int((100 - i) * 1e9),
        )

    buffer.push_liquidation(
        side="BUY",
        price=50200.0,
        qty=50.0,
        timestamp_ns=now_ns - int(2 * 1e9),
    )

    cluster_info = buffer.detect_liquidation_cluster(percentile=99.0)

    assert cluster_info["is_cluster"] is True
    assert cluster_info["volume_last_10s"] >= 50.0
    assert cluster_info["dominant_side"] == "BUY"
    assert cluster_info["vwap_1m"] == pytest.approx(50050.0, abs=1.0)
    assert cluster_info["z_score"] > 0.0


def test_high_frequency_simulation_latency_under_1ms() -> None:
    buffer = MarketRingBuffer(capacity=10_000)
    total_ticks = 10_000

    rng = np.random.default_rng(seed=42)
    bid_prices = 50000.0 + rng.uniform(-5.0, 5.0, size=(total_ticks, 5))
    bid_qtys = rng.uniform(0.1, 10.0, size=(total_ticks, 5))
    ask_prices = 50001.0 + rng.uniform(-5.0, 5.0, size=(total_ticks, 5))
    ask_qtys = rng.uniform(0.1, 10.0, size=(total_ticks, 5))
    timestamps = np.arange(total_ticks, dtype=np.int64) * 100_000

    latencies_ns = np.zeros(total_ticks, dtype=np.int64)

    # Warmup to eliminate python/numpy cold-start jit or first-touch memory allocation
    for _ in range(50):
        buffer.push_order_book(
            bid_prices=bid_prices[0],
            bid_qtys=bid_qtys[0],
            ask_prices=ask_prices[0],
            ask_qtys=ask_qtys[0],
            timestamp_ns=timestamps[0],
        )
        _ = buffer.compute_obi(levels=5)

    for i in range(total_ticks):
        t0 = time.perf_counter_ns()

        buffer.push_order_book(
            bid_prices=bid_prices[i],
            bid_qtys=bid_qtys[i],
            ask_prices=ask_prices[i],
            ask_qtys=ask_qtys[i],
            timestamp_ns=timestamps[i],
        )
        obi = buffer.compute_obi(levels=5)

        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0
        assert -1.0 <= obi <= 1.0

    avg_latency_ms = float(np.mean(latencies_ns)) / 1_000_000.0
    p999_latency_ms = float(np.percentile(latencies_ns, 99.9)) / 1_000_000.0

    assert avg_latency_ms < 0.05
    assert p999_latency_ms < 1.0
