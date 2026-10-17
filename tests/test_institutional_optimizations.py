# tests/test_institutional_optimizations.py
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from backend.database.ledger_store import LedgerSqliteStore
from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import (
    CrossExchangeFundingSnapshot,
    FundingRateArbitrageEngine,
)
from backend.strategies.regime_config import RegimeConfig
from backend.strategies.strategy_ledger import VirtualStrategyLedger
from backend.telemetry_server import BinanceLiveWsGateway, BingXLiveWsGateway, CROSS_HEDGE_PAIRS


# =============================================================================
# 1. Test Persistencia Transaccional SQLite WAL (Módulo 7)
# =============================================================================

def test_ledger_store_sqlite_persistence_and_restore(tmp_path):
    db_file = str(tmp_path / "test_trades.db")
    store = LedgerSqliteStore(db_path=db_file)

    # 1. Guardar trades
    trade_1 = {
        "id": "TRD_001",
        "timestamp_ms": 1700000000000,
        "time_str": "12:00:00",
        "symbol": "BTC/USDT:USDT",
        "strategy": "AVELLANEDA_OBI_L1",
        "side": "BUY",
        "entry_price": 65000.0,
        "exit_or_mid_price": 65100.0,
        "qty": 0.05,
        "return_pct": 0.15,
        "pnl_usd": 5.0,
        "fee_cost": 0.35,
        "exchange": "BINANCE",
    }
    trade_2 = {
        "id": "TRD_002",
        "timestamp_ms": 1700000005000,
        "time_str": "12:00:05",
        "symbol": "ETH/USDT:USDT",
        "strategy": "TAKE_PROFIT",
        "side": "SELL",
        "entry_price": 3500.0,
        "exit_or_mid_price": 3520.0,
        "qty": 1.0,
        "return_pct": 0.57,
        "pnl_usd": 20.0,
        "fee_cost": 0.80,
        "exchange": "BINGX",
    }
    store.save_trade(trade_1, exchange="BINANCE")
    store.save_trade(trade_2, exchange="BINGX")

    # 2. Guardar métricas y lotes
    store.save_metrics(wins_count=12, losses_count=3, gross_profit_usd=150.0, gross_loss_usd=25.0, current_equity_usd=1125.0)
    store.save_lot(
        lot_key="BTC/USDT:USDT::AVELLANEDA_OBI::L1",
        symbol="BTC/USDT:USDT",
        strategy_id="AVELLANEDA_OBI",
        sub_lot_id="L1",
        inventory_q=0.05,
        entry_price=65000.0,
        break_even_price=65026.0,
        entry_ts_ns=1700000000000000000,
        realized_pnl_usd=5.0,
    )
    store.save_markout({
        "trade_id": "TRD_001",
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",
        "fill_price": 65000.0,
        "m1s_bps": 1.5,
        "m5s_bps": 2.2,
        "m15s_bps": 2.8,
        "m30s_bps": 3.4,
    })

    # 3. Reabrir store desde otro objeto y verificar recuperación exacta
    store2 = LedgerSqliteStore(db_path=db_file)
    recent = store2.load_recent_trades(limit=10)
    assert len(recent) == 2
    assert recent[0]["id"] == "TRD_002"  # Más reciente primero
    assert recent[1]["id"] == "TRD_001"

    metrics = store2.load_metrics()
    assert metrics is not None
    assert metrics["wins_count"] == 12
    assert metrics["losses_count"] == 3
    assert metrics["gross_profit_usd"] == 150.0
    assert metrics["current_equity_usd"] == 1125.0

    lots = store2.load_lots()
    assert "BTC/USDT:USDT::AVELLANEDA_OBI::L1" in lots
    assert lots["BTC/USDT:USDT::AVELLANEDA_OBI::L1"]["inventory_q"] == 0.05

    markouts = store2.load_recent_markouts()
    assert len(markouts) == 1
    assert markouts[0]["trade_id"] == "TRD_001"
    assert markouts[0]["m30s_bps"] == 3.4


# =============================================================================
# 2. Test Order Flow Imbalance (OFI) Diferencial Cont-Kukanov-Stoikov (Módulo 3)
# =============================================================================

def test_order_flow_imbalance_calculation_and_shield():
    buf = MarketRingBuffer(capacity=100)
    now = time.time_ns()

    # Iniciar con un snapshot estable
    buf.push_order_book(
        bid_prices=[100.0, 99.9, 99.8, 99.7, 99.6],
        bid_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        ask_prices=[100.1, 100.2, 100.3, 100.4, 100.5],
        ask_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        timestamp_ns=now,
    )

    # 1. Empuje comprador institucional agresivo: bid sube de precio y ask sube
    buf.push_order_book(
        bid_prices=[100.1, 100.0, 99.9, 99.8, 99.7],
        bid_qtys=[15.0, 10.0, 10.0, 10.0, 10.0],
        ask_prices=[100.2, 100.3, 100.4, 100.5, 100.6],
        ask_qtys=[8.0, 10.0, 10.0, 10.0, 10.0],
        timestamp_ns=now + 10_000_000,
    )
    ofi_buy = buf.compute_ofi(window_ticks=10)
    assert ofi_buy > 0.50, f"OFI debería ser fuertemente positivo ante presión compradora, obtenido: {ofi_buy}"

    # 2. Retiro masivo institucional en Bids (spoofing cancel / micro-crash)
    buf.push_order_book(
        bid_prices=[99.5, 99.4, 99.3, 99.2, 99.1],  # Bids colapsaron
        bid_qtys=[2.0, 2.0, 2.0, 2.0, 2.0],
        ask_prices=[100.0, 100.1, 100.2, 100.3, 100.4],  # Asks entraron más abajo
        ask_qtys=[25.0, 20.0, 20.0, 20.0, 20.0],
        timestamp_ns=now + 20_000_000,
    )
    ofi_sell = buf.compute_ofi(window_ticks=10)
    assert ofi_sell < -0.40, f"OFI debería ser fuertemente negativo ante retiro masivo en bids, obtenido: {ofi_sell}"


# =============================================================================
# 3. Test Calibración Online de Kappa Dinámico por Activo (Módulo 4)
# =============================================================================

def test_dynamic_kappa_calculation_by_symbol():
    # 1. Simulación Activo Líquido (ej. BTC): alto flujo de trades, spreads estrechos (1 bps)
    buf_btc = MarketRingBuffer(capacity=500)
    now = time.time_ns()
    # Llenar trades frecuentes (50 trades en 10 segundos = 5 trades/sec)
    for i in range(50):
        t_ns = now - int((50 - i) * 0.2 * 1e9)
        buf_btc.push_trade(price=65000.0, qty=0.1, is_buyer_maker=False, timestamp_ns=t_ns)
        buf_btc.push_order_book(
            bid_prices=[65000.0, 64999.0, 64998.0, 64997.0, 64996.0],
            bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0],
            ask_prices=[65000.5, 65001.5, 65002.5, 65003.5, 65004.5],
            ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0],
            timestamp_ns=t_ns,
        )

    kappa_btc = buf_btc.compute_dynamic_kappa(window_seconds=10.0, default_kappa=1.5)
    assert kappa_btc >= 1.5, f"BTC líquido debería tener kappa elevado (spreads estrechos), obtenido: {kappa_btc}"
    assert kappa_btc <= 5.0

    # 2. Simulación Altcoin Ilíquida: pocos trades (2 en 60s = 0.03 trades/s), spreads abiertos (30 bps)
    buf_alt = MarketRingBuffer(capacity=500)
    buf_alt.push_trade(price=1.0, qty=10.0, is_buyer_maker=True, timestamp_ns=now - int(50 * 1e9))
    buf_alt.push_trade(price=1.002, qty=5.0, is_buyer_maker=False, timestamp_ns=now - int(10 * 1e9))
    for i in range(20):
        buf_alt.push_order_book(
            bid_prices=[0.995, 0.990, 0.985, 0.980, 0.975],
            bid_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
            ask_prices=[1.005, 1.010, 1.015, 1.020, 1.025],
            ask_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
            timestamp_ns=now - int((20 - i) * 1e9),
        )

    kappa_alt = buf_alt.compute_dynamic_kappa(window_seconds=60.0, default_kappa=1.5)
    assert kappa_alt <= 1.2, f"Altcoin ilíquida debería tener kappa bajo (spreads defensivos), obtenido: {kappa_alt}"
    assert kappa_alt >= 0.20


# =============================================================================
# 4. Test Batch Orders en Gateways (Módulo 2)
# =============================================================================

@pytest.mark.asyncio
async def test_batch_orders_placement_and_cancellation():
    mock_exchange = MagicMock()
    mock_exchange.id = "binanceusdm"
    mock_exchange.create_orders_ws = None
    mock_exchange.cancel_orders_ws = None
    mock_exchange.create_orders = AsyncMock(return_value=[
        {"id": "BIN_001", "clientOrderId": "AV_L1_123"},
        {"id": "BIN_002", "clientOrderId": "AV_L2_123"},
        {"id": "BIN_003", "clientOrderId": "AV_L3_123"},
    ])
    mock_exchange.cancel_orders = AsyncMock(return_value={"status": "ok"})

    gw = BinanceLiveWsGateway(exchange=mock_exchange, api_key="dummy", api_secret="dummy")
    gw.initialized_symbols.add("BTC/USDT:USDT")

    orders = [
        OrderRequest.build_maker_order(
            symbol="BTC/USDT:USDT", side="BUY", raw_price=65000.0, raw_qty=0.01,
            tick_size=0.1, step_size=0.001, strategy_id="AVELLANEDA_OBI", timestamp_ns=time.time_ns(),
            client_order_id="AV_L1_123",
        ),
        OrderRequest.build_maker_order(
            symbol="BTC/USDT:USDT", side="BUY", raw_price=64950.0, raw_qty=0.02,
            tick_size=0.1, step_size=0.001, strategy_id="AVELLANEDA_OBI", timestamp_ns=time.time_ns(),
            client_order_id="AV_L2_123",
        ),
        OrderRequest.build_maker_order(
            symbol="BTC/USDT:USDT", side="BUY", raw_price=64900.0, raw_qty=0.03,
            tick_size=0.1, step_size=0.001, strategy_id="AVELLANEDA_OBI", timestamp_ns=time.time_ns(),
            client_order_id="AV_L3_123",
        ),
    ]

    placed = await gw.place_orders_batch(orders)
    assert placed == ["BIN_001", "BIN_002", "BIN_003"]
    assert mock_exchange.create_orders.call_count == 1

    cancelled = await gw.cancel_orders_batch(placed, "BTC/USDT:USDT")
    assert cancelled == ["BIN_001", "BIN_002", "BIN_003"]
    assert mock_exchange.cancel_orders.call_count == 1


# =============================================================================
# 5. Test User Stream WS y Colocación Inmediata de TP Maker (Módulo 1)
# =============================================================================

@pytest.mark.asyncio
async def test_ws_user_stream_instant_tp_maker_dispatch():
    mock_exchange = MagicMock()
    mock_exchange.id = "binanceusdm"
    mock_exchange.create_order = AsyncMock(return_value={"id": "TP_MAKER_01"})

    on_fill_called = []
    def mock_on_fill(*args, **kwargs):
        on_fill_called.append((args, kwargs))

    gw = BinanceLiveWsGateway(exchange=mock_exchange, api_key="dummy", api_secret="dummy", on_fill=mock_on_fill)
    gw.initialized_symbols.add("BTC/USDT:USDT")

    # Registrar orden activa en meta
    oid = "ORD_ENTRY_001"
    gw.active_orders_meta[oid] = {
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",
        "price": 65000.0,
        "qty": 0.05,
        "strategy_id": "AVELLANEDA_OBI",
        "sub_lot_id": "L1",
        "client_order_id": "AV_L1_12345",
        "timestamp_ns": time.time_ns(),
    }
    mock_exchange.fetch_order = AsyncMock(return_value={
        "id": oid,
        "status": "closed",
        "filled": 0.05,
        "average": 65000.0,
    })

    portfolio = PortfolioState(
        base_capital_usd=100.0, current_equity_usd=100.0, daily_high_watermark_usd=100.0,
        realized_pnl_today_usd=0.0, active_positions={}, rolling_edge_mu=0.045, rolling_variance_sigma2=0.009,
    )

    # Simular evento de fill completado
    await gw._handle_order_completion(oid, "BTC/USDT:USDT", "BUY", portfolio)
    await asyncio.sleep(0.05)  # Breve pausa para la tarea async del TP Maker

    assert len(on_fill_called) == 1
    # Verificar que el Take-Profit Maker fue lanzado
    assert mock_exchange.create_order.called
    call_kwargs = mock_exchange.create_order.call_args[1]
    assert call_kwargs["side"] == "sell"
    assert call_kwargs["params"]["timeInForce"] == "GTX"
    assert call_kwargs["price"] > 65000.0  # +15 bps TP


# =============================================================================
# 6. Test Escáner de Arbitraje de Funding Rate en Vivo (Módulo 6)
# =============================================================================

def test_funding_arb_evaluates_high_yield_diff():
    regime = RegimeConfig(min_annualized_funding_yield=0.20)
    gate = MandateGatekeeper()
    engine = FundingRateArbitrageEngine(tick_size=0.1, step_size=0.001, config=regime, gatekeeper=gate)

    portfolio = PortfolioState(
        base_capital_usd=1000.0, current_equity_usd=1000.0, daily_high_watermark_usd=1000.0,
        realized_pnl_today_usd=0.0, active_positions={}, rolling_edge_mu=0.045, rolling_variance_sigma2=0.009,
    )

    # Simular Binance = +0.06% (0.0006) y BingX = -0.01% (-0.0001) -> Δ = 0.0007 por 8h
    # 0.0007 * 3 * 365 = 76.65% APY (> 25% umbral de ejecución)
    snap = CrossExchangeFundingSnapshot(
        symbol="ETH/USDT:USDT",
        binance_funding_rate=0.0006,
        bingx_funding_rate=-0.0001,
        binance_bid_price=3500.0,
        binance_ask_price=3500.5,
        bingx_bid_price=3500.0,
        bingx_ask_price=3500.5,
    )

    pkg = engine.evaluate_cross_exchange_opportunity(
        snapshot=snap,
        target_notional_usd=100.0,
        portfolio_state=portfolio,
    )

    assert pkg is not None
    assert pkg.annualized_yield_diff >= 0.25, f"Rendimiento esperado >= 25% APY, obtenido: {pkg.annualized_yield_diff}"
    assert pkg.short_exchange == "BINANCE"  # Cobra fondeo -> short en Binance
    assert pkg.long_exchange == "BINGX"     # Paga fondeo -> long en BingX
    assert pkg.long_leg.side == "BUY"
    assert pkg.short_leg.side == "SELL"


# =============================================================================
# 7. Test Cobertura Cruzada Instantánea (Módulo 5)
# =============================================================================

def test_cross_hedge_pairs_definition():
    assert "BTC/USDT:USDT" in CROSS_HEDGE_PAIRS
    assert "ETH/USDT:USDT" in CROSS_HEDGE_PAIRS
    assert "SOL/USDT:USDT" in CROSS_HEDGE_PAIRS
    assert len(CROSS_HEDGE_PAIRS) >= 5
