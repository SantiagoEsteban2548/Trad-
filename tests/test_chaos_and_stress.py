# tests/test_chaos_and_stress.py
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.ingestion.ws_multiplexer import WebSocketMultiplexer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.telemetry_server import BinanceLiveWsGateway


@pytest.fixture
def default_portfolio() -> PortfolioState:
    return PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )


@pytest.mark.asyncio
async def test_ws_multiplexer_silent_drop_triggers_timeout_and_reconnection() -> None:
    """
    Chaos Test 1: Simula un cuelgue silencioso (silent drop TCP) en watch_order_book
    que supera los 15s de timeout. Verifica que _order_book_stream capture el TimeoutError,
    emita la advertencia de reconexión y continúe el bucle sin tirar abajo el multiplexer.
    """
    mock_ex = MagicMock()
    mock_ex.price_to_precision = MagicMock(side_effect=lambda s, p: p)
    mock_ex.amount_to_precision = MagicMock(side_effect=lambda s, a: a)

    mock_ex.watch_order_book = AsyncMock(
        side_effect=[
            asyncio.TimeoutError("Simulated 15s TCP silent drop"),
            {
                "bids": [[65000.0, 1.0], [64990.0, 1.0], [64980.0, 1.0], [64970.0, 1.0], [64960.0, 1.0]],
                "asks": [[65010.0, 1.0], [65020.0, 1.0], [65030.0, 1.0], [65040.0, 1.0], [65050.0, 1.0]],
            },
        ]
    )

    ring = MarketRingBuffer(capacity=100)
    mux = WebSocketMultiplexer(symbols=["BTC/USDT:USDT"])
    mux.exchange = mock_ex
    mux.buffers["BTC/USDT:USDT"] = ring
    mux.running = True

    orig_sleep = asyncio.sleep
    sleep_called = []

    async def fast_sleep(sec):
        sleep_called.append(sec)
        await orig_sleep(0)

    with patch("backend.ingestion.ws_multiplexer.asyncio.sleep", side_effect=fast_sleep):
        task = asyncio.create_task(mux._order_book_stream("BTC/USDT:USDT"))
        for _ in range(5):
            await asyncio.sleep(0.01)
        mux.running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert mock_ex.watch_order_book.await_count >= 2
    assert len(sleep_called) >= 1
    assert ring.ob_count >= 1


@pytest.mark.asyncio
async def test_rate_limit_avalanche_engages_gatekeeper_cooldown(default_portfolio: PortfolioState) -> None:
    """
    Chaos Test 2: Simula una avalancha de respuestas HTTP 429 Too Many Requests (-1003).
    Verifica que MandateGatekeeper active inmediatamente el cooldown de 60s en Gate 4,
    bloqueando el hot path de despacho para evitar que la IP sea baneada por el exchange.
    """
    gk = MandateGatekeeper()
    order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=65000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=time.time_ns(),
    )

    # Antes del error 429, la orden pasa los filtros
    is_valid, _ = gk.validate_order(order, default_portfolio)
    assert is_valid is True

    # Inyección de avalancha de errores 429
    for i in range(5):
        gk.record_api_error(f"HTTP 429 Too Many Requests: (-1003) Way too much request weight used (burst {i})")

    now_ns = time.time_ns()
    assert gk.rate_limit_cooldown_until_ns > now_ns
    assert gk.consecutive_api_errors >= 5

    # Con cooldown activo, las órdenes subsiguientes deben ser bloqueadas inmediatamente
    blocked, reason = gk.validate_order(order, default_portfolio)
    assert blocked is False
    assert "RATE_LIMIT_COOLDOWN" in reason

    # Avanzar el reloj simulado más allá del cooldown -> el sistema debe reanudar la operativa
    gk.rate_limit_cooldown_until_ns = now_ns - 1_000_000_000
    resumed, _ = gk.validate_order(order, default_portfolio)
    assert resumed is True


@pytest.mark.asyncio
async def test_loss_streak_auto_flatten_trigger(default_portfolio: PortfolioState) -> None:
    """
    Chaos Test 3: Simula una racha destructiva de 5 pérdidas consecutivas.
    Verifica que el Circuit Breaker de pérdidas consecutivas active el Kill-Switch
    y que todas las órdenes posteriores queden inmediatamente rechazadas.
    """
    gk = MandateGatekeeper(max_consecutive_losses=5)
    order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=65000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=time.time_ns(),
    )

    # 4 pérdidas consecutivas -> todavía activo
    for _ in range(4):
        gk.record_trade_outcome(-5.0)
    assert gk.kill_switch_engaged is False
    v, _ = gk.validate_order(order, default_portfolio)
    assert v is True

    # 5ta pérdida consecutiva -> dispara el Kill-Switch institucional
    gk.record_trade_outcome(-10.0)
    assert gk.consecutive_losses == 5
    assert gk.kill_switch_engaged is True

    # La siguiente orden es rechazada por el Kill-Switch
    blocked, reason = gk.validate_order(order, default_portfolio)
    assert blocked is False
    assert "KILL_SWITCH" in reason or "LOSS_STREAK" in reason


@pytest.mark.asyncio
async def test_post_only_cross_storm_backoff() -> None:
    """
    Chaos Test 4: Simula volatilidad hostil extrema en el libro donde la orden enviada
    y el reintento de ajuste por tick vuelven a cruzar el libro (error -5022 continuado).
    Verifica que el Gateway descarte limpiamente la orden sin excepciones fatales ni fugas de memoria.
    """
    mock_exchange = MagicMock()
    mock_exchange.id = "binanceusdm"
    mock_exchange.create_order_ws = None
    mock_exchange.create_order = AsyncMock(
        side_effect=[
            Exception("binance -5022 OrderImmediatelyFillable: Post-only order would cross (attempt 1)"),
            Exception("binance -5022 OrderImmediatelyFillable: still crossing (attempt 2)"),
        ]
    )

    gw = BinanceLiveWsGateway(exchange=mock_exchange)
    gw.initialized_symbols.add("BTC/USDT:USDT")

    order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=65000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=time.time_ns(),
    )

    # Debe retornar "" orden descartada limpiamente
    oid = await gw.place_order_ws(order)
    assert oid == ""
    assert mock_exchange.create_order.await_count == 2
    # El diccionario de órdenes activas no debe contener residuos
    assert len(gw.active_orders_meta) == 0
