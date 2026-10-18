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


@pytest.mark.asyncio
async def test_cold_reconciliation_persists_to_ledger_and_db_store(mock_exchange: MagicMock) -> None:
    """
    Verifica que la reconciliación en frío de posiciones preexistentes registre
    correctamente en VirtualStrategyLedger y persista atómicamente en LedgerSqliteStore (SQLite WAL).
    """
    from backend.database.ledger_store import LedgerSqliteStore

    db_store = LedgerSqliteStore(":memory:")
    mock_exchange.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.5,
                "entryPrice": 62000.0,
                "info": {
                    "positionAmt": "0.5",
                    "entryPrice": "62000.0",
                    "breakEvenPrice": "62015.0",
                },
            }
        ]
    )

    gateway = BinanceLiveWsGateway(exchange=mock_exchange, db_store=db_store)
    reconciled = await gateway.reconcile_existing_positions(["BTC/USDT:USDT"])

    assert len(reconciled) == 1
    assert reconciled[0]["symbol"] == "BTC/USDT:USDT"
    assert reconciled[0]["qty"] == 0.5
    assert reconciled[0]["entry_price"] == 62000.0
    assert reconciled[0]["break_even_price"] == 62015.0

    # 1. Verificación en VirtualStrategyLedger
    assert gateway.ledger.get_net_inventory("BTC/USDT:USDT") == 0.5
    lot_key = "BTC/USDT:USDT::AVELLANEDA_OBI::MAIN"
    lot = gateway.ledger.lots.get(lot_key)
    assert lot is not None
    assert lot.inventory_q == 0.5
    assert lot.break_even_price == 62015.0

    # 2. Verificación en LedgerSqliteStore (disco / memoria WAL)
    lots_dict = db_store.load_lots()
    assert lot_key in lots_dict
    saved = lots_dict[lot_key]
    assert saved["symbol"] == "BTC/USDT:USDT"
    assert saved["inventory_q"] == 0.5
    assert saved["entry_price"] == 62000.0
    assert saved["break_even_price"] == 62015.0


@pytest.mark.asyncio
async def test_update_symbol_leverage_blocks_reduction_with_open_position_avoids_4161(mock_exchange: MagicMock) -> None:
    """
    Verifica que las solicitudes de reducción de apalancamiento sean bloqueadas
    proactivamente cuando existen posiciones abiertas, previniendo el error -4161 de Binance.
    """
    mock_exchange.set_leverage.reset_mock()
    gateway = BinanceLiveWsGateway(exchange=mock_exchange)
    gateway.symbol_leverage["BTC/USDT:USDT"] = 5
    gateway.default_leverage = 5

    # Simular posición abierta de 0.5 BTC ($30.000 notional)
    portfolio = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={"BTC/USDT:USDT": 30000.0},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )
    gateway.portfolio = portfolio

    # 1. Intentar reducir de 5x a 3x con posición abierta -> DEBE BLOQUEARSE
    changed = await gateway.update_symbol_leverage_if_needed("BTC/USDT:USDT", optimal_leverage=3)
    assert changed is False
    mock_exchange.set_leverage.assert_not_called()
    assert gateway.symbol_leverage["BTC/USDT:USDT"] == 5

    # 2. initialize_symbol_margin con apalancamiento inferior también debe bloquear set_leverage
    await gateway.initialize_symbol_margin("BTC/USDT:USDT", max_leverage=3)
    mock_exchange.set_leverage.assert_not_called()

    # 3. Limpiar posición abierta -> AHORA SÍ debe permitir reducir apalancamiento
    portfolio.active_positions.clear()
    changed_after = await gateway.update_symbol_leverage_if_needed("BTC/USDT:USDT", optimal_leverage=3)
    assert changed_after is True
    mock_exchange.set_leverage.assert_awaited_once_with(3, "BTC/USDT:USDT")
    assert gateway.symbol_leverage["BTC/USDT:USDT"] == 3


@pytest.mark.asyncio
async def test_place_order_ws_handles_5022_order_immediately_fillable_cleanly(mock_exchange: MagicMock) -> None:
    """
    Verifica que el error -5022 (OrderImmediatelyFillable / post-only cross)
    se capture limpiamente, ajustando el precio 1 tick hacia afuera para reintentar,
    o descartando sin excepción no controlada si el libro sigue cruzando.
    """
    order = OrderRequest(
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

    # Caso 1: Primer envío lanza -5022, el reintento ajustado en tick tiene éxito
    mock_exchange.create_order = AsyncMock(
        side_effect=[
            Exception("binance -5022 OrderImmediatelyFillable: Post-only order would immediately match and take."),
            {"id": "binance_retry_success_888", "status": "open"},
        ]
    )
    mock_exchange.create_order_ws = None

    gateway1 = BinanceLiveWsGateway(exchange=mock_exchange)
    order_id1 = await gateway1.place_order_ws(order)

    assert order_id1 == "binance_retry_success_888"
    assert mock_exchange.create_order.await_count == 2
    # El segundo llamado debe haber ajustado el precio 1 tick hacia abajo para BUY (65000.0 - 0.1 = 64999.9)
    second_call_price = mock_exchange.create_order.await_args_list[1].kwargs["price"]
    assert second_call_price < 65000.0

    # Caso 2: El reintento también falla con -5022 -> se descarta limpiamente retornando ""
    mock_exchange.create_order.reset_mock()
    mock_exchange.create_order = AsyncMock(
        side_effect=Exception("binance -5022 OrderImmediatelyFillable: still crossing")
    )

    gateway2 = BinanceLiveWsGateway(exchange=mock_exchange)
    order_id2 = await gateway2.place_order_ws(order)

    assert order_id2 == ""


@pytest.mark.asyncio
async def test_close_position_market_failure_retains_active_position_and_sl(mock_exchange: MagicMock) -> None:
    """
    Escenario Adversario 1: Si close_position_market falla (ej. timeout / error 502) durante
    un evento de STOP_LOSS en vivo, el motor NO debe borrar la posición de portfolio.active_positions
    ni limpiar ctx.stop_loss_price para que el siguiente ciclo vuelva a intentar cerrarla.
    """
    from backend.telemetry_server import engine
    import time

    sym = "BTC/USDT:USDT"
    ctx = engine.symbol_contexts[sym]
    prev_mode = engine.trading_mode
    prev_gw = engine.live_gateway

    try:
        # Configurar gateway que falla al enviar orden market
        mock_exchange.create_order = AsyncMock(side_effect=Exception("502 Bad Gateway"))
        gw = BinanceLiveWsGateway(exchange=mock_exchange, portfolio=engine.portfolio)
        gw.ledger.reconcile_position(sym, "BUY", 0.01, 65000.0, 65000.0)
        engine.live_gateway = gw
        engine.trading_mode = "LIVE_BINANCE"

        # Simular posición LONG abierta con Stop Loss en $64,800
        engine.portfolio.active_positions[sym] = 650.0  # $650 USD notional
        ctx.avg_entry_price = 65000.0
        ctx.take_profit_price = 65150.0
        ctx.stop_loss_price = 64800.0
        ctx.inventory_open_ts_ns = time.time_ns()

        # Disparar evaluación con mid_px ($64,700) por debajo del Stop Loss ($64,800)
        exit_res = await engine.check_and_execute_tp_sl_closures(
            symbol=sym,
            best_bid=64699.0,
            best_ask=64701.0,
            mid_px=64700.0,
            now_ns=time.time_ns(),
        )

        # Al fallar los 3 intentos de close_position_market, debe retornar None y conservar estado
        assert exit_res is None
        assert sym in engine.portfolio.active_positions
        assert engine.portfolio.active_positions[sym] == 650.0
        assert ctx.stop_loss_price == 64800.0
        assert mock_exchange.create_order.await_count == 3
    finally:
        engine.trading_mode = prev_mode
        engine.live_gateway = prev_gw
        engine.portfolio.active_positions.pop(sym, None)
        ctx.avg_entry_price = None
        ctx.take_profit_price = None
        ctx.stop_loss_price = None
        ctx.inventory_open_ts_ns = None


@pytest.mark.asyncio
async def test_fill_idempotency_prevents_double_accounting_ws_and_rest(mock_exchange: MagicMock) -> None:
    """
    Escenario Adversario 2: Si una orden se llena y el evento llega simultáneamente por
    WebSocket (_handle_order_completion) y por polling REST (sync_portfolio_state / fetch_my_trades),
    el cerrojo de idempotencia (_processed_order_ids) debe evitar duplicar el inventario y el callback on_fill.
    """
    fills_recorded = []

    def on_fill_cb(**kwargs):
        fills_recorded.append(kwargs)

    portfolio = PortfolioState(
        base_capital_usd=1000.0,
        current_equity_usd=1000.0,
        daily_high_watermark_usd=1000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    mock_exchange.fetch_order = AsyncMock(
        return_value={
            "id": "ord_999",
            "status": "closed",
            "filled": 0.01,
            "average": 65000.0,
            "price": 65000.0,
        }
    )

    gw = BinanceLiveWsGateway(exchange=mock_exchange, portfolio=portfolio, on_fill=on_fill_cb)
    gw.active_orders_meta["ord_999"] = {
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",
        "price": 65000.0,
        "qty": 0.01,
        "client_order_id": "AVELLANEDA_OBI_MAIN_1",
    }

    # Invocar _handle_order_completion dos veces concurrentemente para el mismo order_id
    await gw._handle_order_completion("ord_999", "BTC/USDT:USDT", "BUY", portfolio)
    await gw._handle_order_completion("ord_999", "BTC/USDT:USDT", "BUY", portfolio)

    # Debe haberse contabilizado exactamente 1 vez (0.01 BTC = $650.0 USD nocional)
    assert len(fills_recorded) == 1
    assert gw.ledger.get_net_inventory("BTC/USDT:USDT") == pytest.approx(0.01)
    assert portfolio.active_positions["BTC/USDT:USDT"] == pytest.approx(650.0)


@pytest.mark.asyncio
async def test_place_protective_stop_loss_uses_mark_price_and_reduce_only(mock_exchange: MagicMock) -> None:
    """
    Escenario Adversario 3: Verifica que place_protective_stop_loss envíe una orden
    STOP_MARKET nativa al servidor del exchange con workingType='MARK_PRICE' y reduceOnly=True.
    """
    mock_exchange.create_order = AsyncMock(return_value={"id": "sl_native_777", "status": "open"})
    gw = BinanceLiveWsGateway(exchange=mock_exchange)

    sl_id = await gw.place_protective_stop_loss(
        symbol="BTC/USDT:USDT",
        close_side="SELL",
        qty=0.015,
        stop_price=64123.456,
    )

    assert sl_id == "sl_native_777"
    assert gw.protective_sl_by_symbol["BTC/USDT:USDT"] == "sl_native_777"
    mock_exchange.create_order.assert_awaited_once()
    call_kwargs = mock_exchange.create_order.await_args.kwargs
    assert call_kwargs["type"] == "STOP_MARKET"
    assert call_kwargs["side"] == "sell"
    assert call_kwargs["amount"] == 0.015
    assert call_kwargs["params"]["reduceOnly"] is True
    assert call_kwargs["params"]["workingType"] == "MARK_PRICE"
    assert call_kwargs["params"]["stopPrice"] == 64123.4


def test_gatekeeper_loss_streak_circuit_breaker_and_429_cooldown() -> None:
    """
    Escenario Adversario 4: Verifica que MandateGatekeeper active el Kill-Switch tras alcanzar
    max_consecutive_losses (5 pérdidas seguidas) y que bloquee temporalmente el envío de órdenes
    durante un enfriamiento por HTTP 429 / -1003 (Rate Limit).
    """
    valid_portfolio = PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )
    gk = MandateGatekeeper(max_consecutive_losses=5)
    order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=60000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_000_000_000,
    )

    # 1. Simular error 429 Rate Limit -> debe activar cooldown temporal
    gk.record_api_error("429 Too Many Requests (-1003 Way too much request weight)")
    allowed_rl, reason_rl = gk.validate_order(order, valid_portfolio)
    assert allowed_rl is False
    assert "RATE_LIMIT_COOLDOWN" in reason_rl

    # Limpiar cooldown para probar racha de pérdidas
    gk.rate_limit_cooldown_until_ns = 0

    # 2. Registrar 4 pérdidas seguidas y 1 ganancia -> resetea contador a 0
    for _ in range(4):
        gk.record_trade_outcome(-1.50)
    assert gk.consecutive_losses == 4
    gk.record_trade_outcome(2.00)
    assert gk.consecutive_losses == 0

    # 3. Registrar 5 pérdidas consecutivas -> dispara Circuit Breaker y Kill-Switch
    for _ in range(5):
        gk.record_trade_outcome(-1.25)
    assert gk.consecutive_losses == 5
    assert gk.kill_switch_engaged is True

    allowed_ls, reason_ls = gk.validate_order(order, valid_portfolio)
    assert allowed_ls is False
    assert "KILL_SWITCH" in reason_ls or "LOSS_STREAK" in reason_ls

