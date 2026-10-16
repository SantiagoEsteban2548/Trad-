# tests/test_strategies.py
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import PortfolioState
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import FundingArbSnapshot, FundingRateArbitrageEngine
from backend.strategies.liq_cascade import LiquidationCascadeHunter
from backend.strategies.regime_config import RegimeConfig


@pytest.fixture
def portfolio_state() -> PortfolioState:
    return PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.06,
        rolling_variance_sigma2=0.01,
    )


@pytest.fixture
def regime_config() -> RegimeConfig:
    return RegimeConfig(
        gamma=0.15,
        kappa=1.50,
        time_horizon_tau=1.0,
        obi_shield_threshold=0.60,
        obi_volatility_shift_mult=2.5,
        min_annualized_funding_yield=0.18,
        min_basis_spread_bps=-5.0,
    )


@pytest.mark.asyncio
async def test_avellaneda_obi_shield_cancels_bid_and_shifts_down_on_toxic_sell_flow(
    portfolio_state: PortfolioState, regime_config: RegimeConfig
) -> None:
    gatekeeper = MandateGatekeeper()
    engine = AvellanedaStoikovOBIEngine(
        symbol="BTC/USDT:USDT",
        tick_size=0.10,
        step_size=0.001,
        quote_qty=0.02,
        config=regime_config,
        gatekeeper=gatekeeper,
    )
    engine.active_bid_order_id = "BID_OPEN_123"

    ws_gateway = AsyncMock()
    ws_gateway.cancel_order_ws.return_value = True

    ring = MarketRingBuffer(capacity=100)
    ring.push_order_book(
        bid_prices=[50000.0, 49999.0, 49998.0, 49997.0, 49996.0],
        bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0],
        ask_prices=[50001.0, 50002.0, 50003.0, 50004.0, 50005.0],
        ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0],
        timestamp_ns=1_000_000_000,
    )
    neutral_res = await engine.on_l2_update(ring, portfolio_state, ws_gateway, normalized_inventory_q=0.0)
    assert neutral_res.bid_order is not None
    neutral_bid_price = neutral_res.bid_order.price

    engine.active_bid_order_id = "BID_OPEN_123"
    ring.push_order_book(
        bid_prices=[50000.0, 49999.0, 49998.0, 49997.0, 49996.0],
        bid_qtys=[0.5, 0.5, 0.5, 0.5, 0.5],
        ask_prices=[50001.0, 50002.0, 50003.0, 50004.0, 50005.0],
        ask_qtys=[4.5, 4.5, 4.5, 4.5, 4.5],
        timestamp_ns=2_000_000_000,
    )

    toxic_res = await engine.on_l2_update(ring, portfolio_state, ws_gateway, normalized_inventory_q=0.0)

    assert toxic_res.obi == pytest.approx(-0.80, abs=1e-6)
    assert toxic_res.bid_cancelled is True
    ws_gateway.cancel_order_ws.assert_awaited_once_with("BID_OPEN_123", "BTC/USDT:USDT")
    assert engine.active_bid_order_id is None
    assert toxic_res.bid_order is not None
    assert toxic_res.bid_order.price < neutral_bid_price
    assert toxic_res.bid_order.time_in_force == "GTX"


@pytest.mark.asyncio
async def test_avellaneda_obi_shield_cancels_ask_on_bullish_breakout(
    portfolio_state: PortfolioState, regime_config: RegimeConfig
) -> None:
    gatekeeper = MandateGatekeeper()
    engine = AvellanedaStoikovOBIEngine(
        symbol="ETH/USDT:USDT",
        tick_size=0.05,
        step_size=0.01,
        quote_qty=0.40,
        config=regime_config,
        gatekeeper=gatekeeper,
    )
    engine.active_ask_order_id = "ASK_OPEN_999"

    ws_gateway = AsyncMock()
    ring = MarketRingBuffer(capacity=100)

    ring.push_order_book(
        bid_prices=[2500.0, 2499.5, 2499.0, 2498.5, 2498.0],
        bid_qtys=[4.5, 4.5, 4.5, 4.5, 4.5],
        ask_prices=[2500.5, 2501.0, 2501.5, 2502.0, 2502.5],
        ask_qtys=[0.5, 0.5, 0.5, 0.5, 0.5],
        timestamp_ns=1_000_000_000,
    )

    res = await engine.on_l2_update(ring, portfolio_state, ws_gateway)

    assert res.obi == pytest.approx(0.80, abs=1e-6)
    assert res.ask_cancelled is True
    assert res.ask_order is None
    ws_gateway.cancel_order_ws.assert_awaited_once_with("ASK_OPEN_999", "ETH/USDT:USDT")


def test_liquidation_cascade_hunter_triggers_on_3_2_sigma_and_exhaustion(
    portfolio_state: PortfolioState,
) -> None:
    gatekeeper = MandateGatekeeper()
    hunter = LiquidationCascadeHunter(
        symbol="SOL/USDT:USDT",
        tick_size=0.01,
        step_size=0.1,
        order_qty=10.0,
        gatekeeper=gatekeeper,
    )

    ring = MarketRingBuffer(capacity=500)
    now_ns = 1_700_000_000_000_000_000

    for i in range(60):
        price = 150.00 + (0.05 if i % 2 == 0 else -0.05)
        ring.push_trade(price=price, qty=10.0, is_buyer_maker=False, timestamp_ns=now_ns - int((55 - i * 0.5) * 1e9))

    ring.push_trade(price=145.00, qty=2.0, is_buyer_maker=True, timestamp_ns=now_ns - 100_000_000)

    for i in range(30):
        ring.push_liquidation("SELL", 149.50, 1.0, now_ns - int((40 - i) * 1e9))

    ring.push_liquidation("SELL", 145.50, 100.0, now_ns - 750_000_000)
    ring.push_liquidation("SELL", 145.00, 20.0, now_ns - 100_000_000)

    ring.push_order_book(
        bid_prices=[145.00, 144.95, 144.90, 144.85, 144.80],
        bid_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        ask_prices=[145.05, 145.10, 145.15, 145.20, 145.25],
        ask_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        timestamp_ns=now_ns,
    )

    bracket = hunter.evaluate(ring, portfolio_state, now_ns=now_ns)

    assert bracket is not None
    assert bracket.z_score < -3.2
    assert bracket.exhaustion_ratio == pytest.approx(0.20, abs=1e-6)
    assert bracket.entry_order.side == "BUY"
    assert bracket.entry_order.price == 145.00
    assert bracket.entry_order.time_in_force == "GTX"

    expected_tp = 145.00 + 0.40 * (bracket.vwap_1m - 145.00)
    assert bracket.take_profit_order.side == "SELL"
    assert bracket.take_profit_order.price == pytest.approx(expected_tp, abs=0.02)
    assert bracket.reduce_only_tp is True
    assert bracket.stop_loss_price < bracket.entry_order.price


def test_funding_arb_validates_both_legs_in_gatekeeper(
    portfolio_state: PortfolioState, regime_config: RegimeConfig
) -> None:
    gatekeeper = MandateGatekeeper()
    arb_engine = FundingRateArbitrageEngine(
        tick_size=0.10,
        step_size=0.001,
        config=regime_config,
        gatekeeper=gatekeeper,
    )

    snapshot = FundingArbSnapshot(
        symbol="BTC/USDT:USDT",
        spot_price=50_000.0,
        perp_bid_price=50_010.0,
        perp_ask_price=50_015.0,
        projected_funding_rate=0.00025,
        funding_interval_hours=8,
    )

    pkg = arb_engine.evaluate_opportunity(
        snapshot=snapshot,
        target_notional_usd=1_500.0,
        portfolio_state=portfolio_state,
    )

    assert pkg is not None
    assert pkg.annualized_yield == pytest.approx(0.27375, abs=1e-5)
    assert pkg.spot_hedge_leg.side == "BUY"
    assert pkg.perp_futures_leg.side == "SELL"
    assert pkg.spot_hedge_leg.qty == pkg.perp_futures_leg.qty == 0.03
    assert pkg.perp_futures_leg.time_in_force == "GTX"

    rejected_pkg = arb_engine.evaluate_opportunity(
        snapshot=snapshot,
        target_notional_usd=2_500.0,
        portfolio_state=portfolio_state,
    )
    assert rejected_pkg is None
