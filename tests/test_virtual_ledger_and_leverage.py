# tests/test_virtual_ledger_and_leverage.py
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.leverage_controller import AutonomousLeverageController
from backend.strategies.regime_config import RegimeConfig
from backend.strategies.strategy_ledger import VirtualPositionLot, VirtualStrategyLedger
from backend.telemetry_server import BinanceLiveWsGateway, DryRunWsGateway
from backend.validation.l2_simulator_and_dsr import L2QueueSimulator


def test_virtual_strategy_ledger_prefix_and_cids():
    ledger = VirtualStrategyLedger()

    # Generación para diferentes motores y niveles
    cid_l1 = ledger.generate_client_order_id("AVELLANEDA_OBI", "L1")
    cid_l2 = ledger.generate_client_order_id("AVELLANEDA_OBI", "L2")
    cid_l3 = ledger.generate_client_order_id("AVELLANEDA_OBI", "L3")
    cid_liq = ledger.generate_client_order_id("LIQ_CASCADE", "MAIN")
    cid_farb = ledger.generate_client_order_id("FUNDING_ARB", "MAIN")

    # Longitud máxima Binance USDⓈ-M (máx 36 caracteres)
    for cid in [cid_l1, cid_l2, cid_l3, cid_liq, cid_farb]:
        assert len(cid) <= 36
        assert len(cid) >= 10

    # Parseo de prefijos
    strat_1, sub_1 = ledger.parse_client_order_id(cid_l1)
    assert strat_1 == "AVELLANEDA_OBI"
    assert sub_1 == "L1"

    strat_2, sub_2 = ledger.parse_client_order_id(cid_l2)
    assert strat_2 == "AVELLANEDA_OBI"
    assert sub_2 == "L2"

    strat_liq, sub_liq = ledger.parse_client_order_id(cid_liq)
    assert strat_liq == "LIQ_CASCADE"
    assert sub_liq == "MAIN"

    strat_farb, sub_farb = ledger.parse_client_order_id(cid_farb)
    assert strat_farb == "FUNDING_ARB"
    assert sub_farb == "MAIN"


def test_virtual_strategy_ledger_independent_lots():
    ledger = VirtualStrategyLedger()
    symbol = "BTC/USDT:USDT"

    # Registrar órdenes para L1 y L2
    cid_l1 = ledger.generate_client_order_id("AVELLANEDA_OBI", "L1")
    cid_l2 = ledger.generate_client_order_id("AVELLANEDA_OBI", "L2")

    ledger.register_order(cid_l1, symbol, "AVELLANEDA_OBI", "L1", exchange_order_id="BIN_1001")
    ledger.register_order(cid_l2, symbol, "AVELLANEDA_OBI", "L2", exchange_order_id="BIN_1002")

    # Fill de compra para L1 a $80,000 (0.010 BTC)
    res1 = ledger.record_fill("BIN_1001", symbol, "BUY", 80000.0, 0.010)
    assert res1["lot_key"] == f"{symbol}::AVELLANEDA_OBI::L1"
    assert res1["inventory_q"] == 0.010
    assert res1["entry_price"] == 80000.0
    assert not res1["is_closed"]

    # Fill de compra para L2 a $79,000 (0.010 BTC)
    res2 = ledger.record_fill("BIN_1002", symbol, "BUY", 79000.0, 0.010)
    assert res2["lot_key"] == f"{symbol}::AVELLANEDA_OBI::L2"
    assert res2["inventory_q"] == 0.010
    assert res2["entry_price"] == 79000.0

    # Comprobación de independencia: L1 y L2 no se mezclaron
    lot_l1 = ledger.get_or_create_lot(symbol, "AVELLANEDA_OBI", "L1")
    lot_l2 = ledger.get_or_create_lot(symbol, "AVELLANEDA_OBI", "L2")
    assert lot_l1.entry_price == 80000.0
    assert lot_l2.entry_price == 79000.0

    # Inventario neto consolidado en el símbolo (coincide con Binance positionAmt)
    assert round(ledger.get_net_inventory(symbol), 4) == 0.020

    # Fill de venta de cierre para L1 a $80,500 (+500 USD de ganancia sobre 0.01 BTC = +5.0 USD)
    res_close_l1 = ledger.record_fill(cid_l1, symbol, "SELL", 80500.0, 0.010)
    assert res_close_l1["is_closed"]
    assert res_close_l1["inventory_q"] == 0.0
    assert round(res_close_l1["realized_pnl_usd"], 2) == 5.0
    assert not lot_l1.is_open

    # L2 sigue completamente abierta e intacta
    assert lot_l2.is_open
    assert lot_l2.inventory_q == 0.010
    assert lot_l2.entry_price == 79000.0
    assert round(ledger.get_net_inventory(symbol), 4) == 0.010


def test_autonomous_leverage_controller():
    ring = MarketRingBuffer(capacity=100)
    portfolio = PortfolioState(
        base_capital_usd=100.0,
        current_equity_usd=100.0,
        daily_high_watermark_usd=100.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    # 1. Sin datos en buffer -> apalancamiento por defecto 3x
    assert AutonomousLeverageController.compute_optimal_leverage("BTC/USDT:USDT", ring, portfolio) == 3

    # Sembrar L2 en calma: volatilidad realizada muy baja (< 0.15%)
    now_ns = time.time_ns()
    for i in range(10):
        # Precios estables alrededor de 80,000
        p = 80000.0 + (i % 2) * 2.0
        ring.push_order_book(
            bid_prices=[p - 1.0, p - 2.0, p - 3.0, p - 4.0, p - 5.0],
            bid_qtys=[2.0, 2.0, 2.0, 2.0, 2.0],
            ask_prices=[p + 1.0, p + 2.0, p + 3.0, p + 4.0, p + 5.0],
            ask_qtys=[2.0, 2.0, 2.0, 2.0, 2.0],
            timestamp_ns=now_ns + i * 1_000_000_000,
        )

    calm_lev = AutonomousLeverageController.compute_optimal_leverage("BTC/USDT:USDT", ring, portfolio)
    # En régimen de calma con baja volatilidad debe elegir apalancamiento alto (entre 6x y 10x)
    assert calm_lev >= 6
    assert calm_lev <= 10

    # 2. Régimen de volatilidad explosiva (> 0.60%)
    ring_shock = MarketRingBuffer(capacity=100)
    for i in range(10):
        # Desviaciones salvajes de 1,000 USD entre ticks
        p = 80000.0 + (1000.0 if i % 2 == 0 else -1000.0)
        ring_shock.push_order_book(
            bid_prices=[p - 5.0, p - 10.0, p - 15.0, p - 20.0, p - 25.0],
            bid_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
            ask_prices=[p + 5.0, p + 10.0, p + 15.0, p + 20.0, p + 25.0],
            ask_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
            timestamp_ns=now_ns + i * 1_000_000_000,
        )

    shock_lev = AutonomousLeverageController.compute_optimal_leverage("BTC/USDT:USDT", ring_shock, portfolio)
    # Volatilidad extrema comprime automáticamente a 1x o 2x
    assert shock_lev in (1, 2)


def test_min_profitable_spread_bps_floor():
    config = RegimeConfig(
        gamma=0.01,  # Aversión muy baja para producir spread teórico estrecho
        kappa=5.0,
        time_horizon_tau=0.005,
        obi_shield_threshold=0.60,
        obi_volatility_shift_mult=0.5,
    )
    gatekeeper = MandateGatekeeper()
    engine = AvellanedaStoikovOBIEngine(
        symbol="BTC/USDT:USDT",
        tick_size=0.10,
        step_size=0.001,
        quote_qty=0.001,
        config=config,
        gatekeeper=gatekeeper,
    )

    mid_price = 80000.0
    _, optimal_spread = engine.compute_reservation_and_spread(
        mid_price=mid_price,
        normalized_inventory_q=0.0,
        sigma=0.0001,  # Muy baja volatilidad
    )

    # El spread óptimo debe respetar estrictamente el piso mínimo de 10.0 bps
    spread_bps = (optimal_spread / mid_price) * 10_000.0
    assert spread_bps >= 10.0
    assert optimal_spread >= mid_price * (10.0 / 10_000.0)


@pytest.mark.asyncio
async def test_break_even_and_momentum_trailing_exit():
    config = RegimeConfig(
        gamma=0.05,
        kappa=1.5,
        time_horizon_tau=0.05,
        obi_shield_threshold=0.60,
        obi_volatility_shift_mult=0.5,
    )
    gatekeeper = MandateGatekeeper()
    engine = AvellanedaStoikovOBIEngine(
        symbol="BTC/USDT:USDT",
        tick_size=0.10,
        step_size=0.001,
        quote_qty=0.001,
        config=config,
        gatekeeper=gatekeeper,
    )

    ring = MarketRingBuffer(capacity=50)
    now_ns = time.time_ns()
    # Mercado alrededor de 80,000 con OBI alcista (+0.70)
    ring.push_order_book(
        bid_prices=[80000.0, 79990.0, 79980.0, 79970.0, 79960.0],
        bid_qtys=[10.0, 8.0, 6.0, 5.0, 4.0],  # Libro de compra fuerte
        ask_prices=[80005.0, 80010.0, 80020.0, 80030.0, 80040.0],
        ask_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
        timestamp_ns=now_ns,
    )

    portfolio = PortfolioState(
        base_capital_usd=100.0,
        current_equity_usd=100.0,
        daily_high_watermark_usd=100.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )
    mock_gw = AsyncMock()

    # Caso 1: Posición LONG (q = +0.50), precio de compra entrada = $79,900
    res_long = await engine.on_l2_update(
        ring,
        portfolio,
        mock_gw,
        normalized_inventory_q=0.50,
        entry_price=79900.0,
    )

    # Las órdenes ASK son de cierre (reduceOnly virtual):
    # Deben exigir precio >= entry * (1 + 8 bps) = 79900 * 1.0008 = 79963.92
    assert res_long.ask_ladder is not None
    for ask_ord in res_long.ask_ladder:
        assert ask_ord.price >= 79900.0 * (1.0 + 8.0 / 10_000.0)


@pytest.mark.asyncio
async def test_prune_stale_orders():
    # Test del podador en DryRunWsGateway
    sim = L2QueueSimulator(latency_ms=10.0)
    ring = MarketRingBuffer(capacity=50)
    now_ns = time.time_ns()
    ring.push_order_book(
        bid_prices=[80000.0, 79990.0, 79980.0, 79970.0, 79960.0],
        bid_qtys=[2.0, 2.0, 2.0, 2.0, 2.0],
        ask_prices=[80010.0, 80020.0, 80030.0, 80040.0, 80050.0],
        ask_qtys=[2.0, 2.0, 2.0, 2.0, 2.0],
        timestamp_ns=now_ns,
    )

    gw = DryRunWsGateway(sim, ring)

    # Enviar 5 órdenes de compra y 5 órdenes de venta
    for i in range(5):
        sim.submit_gtx_order(
            OrderRequest(
                symbol="BTC/USDT:USDT",
                side="BUY",
                price=79900.0 - i * 10,
                qty=0.001,
                order_type="LIMIT",
                margin_mode="ISOLATED",
                time_in_force="GTX",
                strategy_id="AVELLANEDA_OBI",
                timestamp_ns=now_ns + i,
            ),
            ring,
        )
        sim.submit_gtx_order(
            OrderRequest(
                symbol="BTC/USDT:USDT",
                side="SELL",
                price=80100.0 + i * 10,
                qty=0.001,
                order_type="LIMIT",
                margin_mode="ISOLATED",
                time_in_force="GTX",
                strategy_id="AVELLANEDA_OBI",
                timestamp_ns=now_ns + i,
            ),
            ring,
        )

    # Verificar que inicialmente hay 10 órdenes abiertas
    open_orders_initial = [o for o in sim.orders.values() if o.status == "OPEN"]
    assert len(open_orders_initial) == 10

    # Ejecutar poda con tope de máx 3 por lado (máx 6 en total)
    pruned = await gw.prune_stale_orders("BTC/USDT:USDT", max_orders_per_side=3)
    assert len(pruned) == 4  # 2 bids + 2 asks podadas

    remaining_bids = [o for o in sim.orders.values() if o.status == "OPEN" and o.request.side == "BUY"]
    remaining_asks = [o for o in sim.orders.values() if o.status == "OPEN" and o.request.side == "SELL"]
    assert len(remaining_bids) == 3
    assert len(remaining_asks) == 3
