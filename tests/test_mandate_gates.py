# tests/test_mandate_gates.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import pytest

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.profit_sweeper import (
    daily_profit_sweep,
    should_trigger_profit_sweep,
)
from backend.mandate_gates.schemas import (
    OrderRequest,
    PortfolioState,
    truncate_decimal_step,
)


@pytest.fixture
def healthy_portfolio() -> PortfolioState:
    return PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.04,
        rolling_variance_sigma2=0.01,
    )


@pytest.fixture
def valid_maker_order() -> OrderRequest:
    return OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50_000.0,
        qty=0.03,  # Nocional = $1,500 USD (15% de $10,000 <= 20%)
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_700_000_000_000_000_000,
    )


def test_valid_order_passes_all_gates(
    healthy_portfolio: PortfolioState, valid_maker_order: OrderRequest
) -> None:
    gatekeeper = MandateGatekeeper()
    approved, reason = gatekeeper.validate_order(valid_maker_order, healthy_portfolio)
    assert approved is True
    assert reason == "APPROVED"
    assert gatekeeper.kill_switch_engaged is False


def test_gate_1_rejects_cross_margin(healthy_portfolio: PortfolioState) -> None:
    gatekeeper = MandateGatekeeper()
    cross_order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50_000.0,
        qty=0.02,
        time_in_force="GTX",
        margin_mode="CROSS",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_700_000_000_000_000_000,
    )
    approved, reason = gatekeeper.validate_order(cross_order, healthy_portfolio)
    assert approved is False
    assert "GATE_1_ISOLATED_MARGIN" in reason


@pytest.mark.parametrize(
    "strategy_id,order_type,tif",
    [
        ("AVELLANEDA_OBI", "MARKET", "GTX"),
        ("AVELLANEDA_OBI", "LIMIT", "GTC"),
        ("AVELLANEDA_OBI", "LIMIT", "IOC"),
        ("LIQ_CASCADE", "MARKET", "IOC"),
        ("LIQ_CASCADE", "LIMIT", "GTC"),
    ],
)
def test_gate_2_rejects_non_post_only_maker_strategies(
    healthy_portfolio: PortfolioState,
    strategy_id: str,
    order_type: str,
    tif: str,
) -> None:
    gatekeeper = MandateGatekeeper()
    bad_maker_order = OrderRequest(
        symbol="ETH/USDT:USDT",
        side="BUY",
        order_type=order_type,
        price=2_500.0,
        qty=0.4,
        time_in_force=tif,
        margin_mode="ISOLATED",
        strategy_id=strategy_id,
        timestamp_ns=1_700_000_000_000_000_000,
    )
    approved, reason = gatekeeper.validate_order(bad_maker_order, healthy_portfolio)
    assert approved is False
    assert "GATE_2_MAKER_PROTECTION" in reason


def test_gate_3_rejects_single_pair_over_20_percent_margin(
    healthy_portfolio: PortfolioState,
) -> None:
    gatekeeper = MandateGatekeeper()
    healthy_portfolio.active_positions["BTC/USDT:USDT"] = 1_000.0
    oversized_pair_order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50_000.0,
        qty=0.03,  # $1,500 USD
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_700_000_000_000_000_000,
    )
    approved, reason = gatekeeper.validate_order(oversized_pair_order, healthy_portfolio)
    assert approved is False
    assert "GATE_3_PAIR_CONCENTRATION" in reason


def test_gate_3_rejects_exceeding_dynamic_half_kelly() -> None:
    gatekeeper = MandateGatekeeper()
    low_edge_state = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.002,
        rolling_variance_sigma2=0.01,
    )
    order = OrderRequest(
        symbol="SOL/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=150.0,
        qty=10.0,  # $1,500 USD = 0.15x > 0.10x Half-Kelly
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="LIQ_CASCADE",
        timestamp_ns=1_700_000_000_000_000_000,
    )
    approved, reason = gatekeeper.validate_order(order, low_edge_state)
    assert approved is False
    assert "GATE_3_HALF_KELLY_CAP" in reason


def test_gate_3_enforces_hard_4x_leverage_ceiling() -> None:
    gatekeeper = MandateGatekeeper()
    high_edge_state = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={f"PAIR_{i}/USDT:USDT": 1_950.0 for i in range(20)},
        rolling_edge_mu=0.20,
        rolling_variance_sigma2=0.01,
    )
    new_order = OrderRequest(
        symbol="NEW/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=100.0,
        qty=15.0,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_700_000_000_000_000_000,
    )
    approved, reason = gatekeeper.validate_order(new_order, high_edge_state)
    assert approved is False
    assert "GATE_3_HALF_KELLY_CAP" in reason


def test_gate_4_daily_kill_switch_triggers_emergency_flatten(
    valid_maker_order: OrderRequest,
) -> None:
    flatten_called = False

    def on_flatten() -> None:
        nonlocal flatten_called
        flatten_called = True

    gatekeeper = MandateGatekeeper(emergency_callback=on_flatten)
    drawdown_state = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=9_700.0,  # -3.0% respecto a HWM
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=-300.0,
        active_positions={},
        rolling_edge_mu=0.05,
        rolling_variance_sigma2=0.01,
    )

    approved, reason = gatekeeper.validate_order(valid_maker_order, drawdown_state)
    assert approved is False
    assert "GATE_4_KILL_SWITCH" in reason
    assert gatekeeper.kill_switch_engaged is True
    assert gatekeeper.emergency_flatten_count == 1
    assert flatten_called is True


def test_hot_path_latency_under_0_1_ms(
    healthy_portfolio: PortfolioState, valid_maker_order: OrderRequest
) -> None:
    gatekeeper = MandateGatekeeper()
    iterations = 5_000

    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        gatekeeper.validate_order(valid_maker_order, healthy_portfolio)
    elapsed_ns = time.perf_counter_ns() - start_ns

    avg_latency_ms = (elapsed_ns / iterations) / 1_000_000.0
    assert avg_latency_ms < 0.1, f"Latencia promedio {avg_latency_ms:.5f} ms excede 0.1 ms"


def test_decimal_truncation_and_maker_builder() -> None:
    order = OrderRequest.build_maker_order(
        symbol="BTC/USDT:USDT",
        side="SELL",
        raw_price=50123.456789,
        raw_qty=0.12345678,
        tick_size=0.10,
        step_size=0.001,
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_700_000_000_000_000_000,
    )
    assert order.price == 50123.4
    assert order.qty == 0.123
    assert order.time_in_force == "GTX"
    assert order.order_type == "LIMIT"
    assert truncate_decimal_step(0.1 + 0.2, "0.01") == 0.30


@pytest.mark.asyncio
async def test_daily_profit_sweep_transfers_surplus_and_resets_capital() -> None:
    mock_exchange = AsyncMock()
    state = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_250.50,
        daily_high_watermark_usd=10_250.50,
        realized_pnl_today_usd=250.50,
        active_positions={},
        rolling_edge_mu=0.04,
        rolling_variance_sigma2=0.01,
    )

    assert should_trigger_profit_sweep(state, datetime(2026, 10, 13, 14, 30, tzinfo=timezone.utc)) is True

    swept = await daily_profit_sweep(mock_exchange, state)

    assert swept == 250.50
    mock_exchange.transfer.assert_awaited_once_with(
        code="USDT",
        amount=250.50,
        fromAccount="future",
        toAccount="spot",
    )
    assert state.current_equity_usd == 10_000.0
    assert state.daily_high_watermark_usd == 10_000.0


@pytest.mark.asyncio
async def test_daily_profit_sweep_skips_when_margin_is_committed() -> None:
    mock_exchange = AsyncMock()
    state = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_300.0,
        daily_high_watermark_usd=10_300.0,
        realized_pnl_today_usd=300.0,
        active_positions={"BTC/USDT:USDT": 1_500.0},
        rolling_edge_mu=0.04,
        rolling_variance_sigma2=0.01,
    )

    swept = await daily_profit_sweep(mock_exchange, state)

    assert swept == 0.0
    mock_exchange.transfer.assert_not_awaited()
    assert state.current_equity_usd == 10_300.0
