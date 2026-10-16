# tests/test_live_gateway.py
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.telemetry_server import BinanceLiveWsGateway, app, engine


@pytest.fixture
def mock_exchange() -> MagicMock:
    exchange = MagicMock()
    exchange.set_margin_mode = AsyncMock(return_value={"code": 200, "msg": "success"})
    exchange.set_leverage = AsyncMock(return_value={"leverage": 3, "symbol": "BTC/USDT:USDT"})
    exchange.create_order = AsyncMock(return_value={"id": "binance_123456", "status": "open"})
    exchange.create_order_ws = AsyncMock(return_value={"id": "binance_123456", "status": "open"})
    exchange.cancel_order = AsyncMock(return_value={"id": "binance_123456", "status": "canceled"})
    exchange.cancel_order_ws = AsyncMock(return_value={"id": "binance_123456", "status": "canceled"})
    exchange.cancel_all_orders = AsyncMock(return_value=[{"id": "binance_123456"}])
    exchange.fetch_balance = AsyncMock(
        return_value={
            "total": {"USDT": 12500.50},
            "free": {"USDT": 11000.00},
            "info": {"totalMarginBalance": "12500.50", "totalWalletBalance": "12500.50"},
        }
    )
    exchange.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.25,
                "notional": 16250.0,
                "side": "long",
            }
        ]
    )
    exchange.transfer = AsyncMock(return_value={"tranId": 987654321, "status": "CONFIRMED"})
    exchange.close = AsyncMock()
    return exchange


@pytest.mark.asyncio
async def test_binance_live_ws_gateway_initialization(mock_exchange: MagicMock) -> None:
    gateway = BinanceLiveWsGateway(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
        exchange=mock_exchange,
    )
    assert gateway.api_key == "test_key"
    assert gateway.api_secret == "test_secret"
    assert gateway.testnet is True
    assert gateway.exchange == mock_exchange


@pytest.mark.asyncio
async def test_initialize_symbol_margin(mock_exchange: MagicMock) -> None:
    gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    await gateway.initialize_symbol_margin("BTC/USDT:USDT", max_leverage=3)

    mock_exchange.set_margin_mode.assert_awaited_once_with("ISOLATED", "BTC/USDT:USDT")
    mock_exchange.set_leverage.assert_awaited_once_with(3, "BTC/USDT:USDT")
    assert "BTC/USDT:USDT" in gateway.initialized_symbols


@pytest.mark.asyncio
async def test_place_order_ws_validates_and_submits_gtx(mock_exchange: MagicMock) -> None:
    gatekeeper = MandateGatekeeper()
    portfolio = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )
    gateway = BinanceLiveWsGateway(
        exchange=mock_exchange,
        gatekeeper=gatekeeper,
        portfolio=portfolio,
    )

    valid_order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=65000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_000_000_000,
    )

    order_id = await gateway.place_order_ws(valid_order)
    assert order_id == "binance_123456"
    assert "BTC/USDT:USDT" in gateway.initialized_symbols

    # Test rejection of cross margin or non-GTX order
    invalid_order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=65000.0,
        qty=0.01,
        time_in_force="GTC",
        margin_mode="CROSS",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_000_000_000,
    )
    with pytest.raises(ValueError, match="MandateGatekeeper"):
        await gateway.place_order_ws(invalid_order)


@pytest.mark.asyncio
async def test_cancel_order_ws(mock_exchange: MagicMock) -> None:
    gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    success = await gateway.cancel_order_ws("binance_123456", "BTC/USDT:USDT")
    assert success is True
    # Either cancel_order_ws or cancel_order was called
    assert mock_exchange.cancel_order_ws.await_count + mock_exchange.cancel_order.await_count >= 1


@pytest.mark.asyncio
async def test_emergency_flatten_all(mock_exchange: MagicMock) -> None:
    gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    await gateway.emergency_flatten_all(["BTC/USDT:USDT", "ETH/USDT:USDT"])

    assert mock_exchange.cancel_all_orders.await_count == 2
    # Active position of 0.25 long BTC should trigger market sell with reduceOnly
    mock_exchange.create_order.assert_awaited_with(
        symbol="BTC/USDT:USDT",
        type="market",
        side="sell",
        amount=0.25,
        params={"reduceOnly": True},
    )


@pytest.mark.asyncio
async def test_sync_portfolio_state(mock_exchange: MagicMock) -> None:
    gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    portfolio = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    await gateway.sync_portfolio_state(portfolio)
    assert portfolio.current_equity_usd == 12500.50
    assert portfolio.daily_high_watermark_usd == 12500.50
    assert portfolio.active_positions.get("BTC/USDT:USDT") == 16250.0


@pytest.mark.asyncio
async def test_sync_portfolio_state_resets_closed_order_ids(mock_exchange: MagicMock) -> None:
    mock_exchange.fetch_open_orders = AsyncMock(return_value=[{"id": "open_ask_456"}])
    gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    gateway.live_active_bid_id = "filled_bid_123"
    gateway.live_active_ask_id = "open_ask_456"

    portfolio = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    await gateway.sync_portfolio_state(portfolio)
    assert gateway.live_active_bid_id is None
    assert gateway.live_active_ask_id == "open_ask_456"


def test_api_trading_mode_security_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)

    # 1. Without API keys, attempt to switch to LIVE_BINANCE must return HTTP 400
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    engine.trading_mode = "PAPER_L2"
    engine.gatekeeper.kill_switch_engaged = False

    res = client.post("/api/trading-mode", json={"mode": "LIVE_BINANCE"})
    assert res.status_code == 400
    assert "BINANCE_API_KEY" in res.json()["detail"]
    assert engine.trading_mode == "PAPER_L2"

    # 2. With kill switch engaged, attempt to switch to LIVE_BINANCE must return HTTP 400
    monkeypatch.setenv("BINANCE_API_KEY", "dummy_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "dummy_secret")
    engine.gatekeeper.kill_switch_engaged = True

    res = client.post("/api/trading-mode", json={"mode": "LIVE_BINANCE"})
    assert res.status_code == 400
    assert "Kill-Switch" in res.json()["detail"]
    assert engine.trading_mode == "PAPER_L2"

    # Reset kill switch
    engine.gatekeeper.kill_switch_engaged = False


def test_api_profit_sweep_paper_mode() -> None:
    client = TestClient(app)
    engine.trading_mode = "PAPER_L2"
    engine.portfolio.current_equity_usd = 105.0
    engine.portfolio.base_capital_usd = 100.0
    engine.simulated_vault_usd = 0.0

    res = client.post("/api/profit-sweep")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PROFIT_SWEEP_EXECUTED"
    assert data["surplus_swept_usd"] == 5.0
    assert data["simulated_vault_usd"] == 5.0
    assert engine.portfolio.current_equity_usd == 100.0


@pytest.mark.asyncio
async def test_api_profit_sweep_live_binance(mock_exchange: MagicMock) -> None:
    engine.trading_mode = "LIVE_BINANCE"
    engine.live_gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    engine.portfolio.current_equity_usd = 108.0
    engine.portfolio.base_capital_usd = 100.0

    client = TestClient(app)
    res = client.post("/api/profit-sweep")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PROFIT_SWEEP_EXECUTED"
    assert data["surplus_swept_usd"] == 8.0
    mock_exchange.transfer.assert_awaited_once_with(
        code="USDT",
        amount=8.0,
        fromAccount="future",
        toAccount="spot",
    )
    assert engine.portfolio.current_equity_usd == 100.0

    # Reset back to PAPER_L2
    engine.trading_mode = "PAPER_L2"
    engine.simulated_vault_usd = 0.0
