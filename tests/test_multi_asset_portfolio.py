# tests/test_multi_asset_portfolio.py
from __future__ import annotations

import time
import numpy as np
import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.profit_sweeper import step_up_ratchet
from backend.mandate_gates.schemas import PortfolioState
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.regime_config import RegimeConfig
from backend.telemetry_server import (
    DryRunExecutionEngine,
    SYMBOL_SPECS,
    SymbolMarketSpec,
    ContagionShieldState,
)


def test_symbol_specs_and_context_initialization() -> None:
    """Verifica que SYMBOL_SPECS contenga los 12 pares con sus ticks y steps exactos."""
    assert len(SYMBOL_SPECS) == 12
    for sym in [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT",
        "XRP/USDT:USDT", "SUI/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "PEPE/USDT:USDT",
        "NEAR/USDT:USDT", "ARB/USDT:USDT",
    ]:
        assert sym in SYMBOL_SPECS

    assert SYMBOL_SPECS["BTC/USDT:USDT"].tick_size == 0.10
    assert SYMBOL_SPECS["BTC/USDT:USDT"].step_size == 0.001

    assert SYMBOL_SPECS["ETH/USDT:USDT"].tick_size == 0.01
    assert SYMBOL_SPECS["ETH/USDT:USDT"].step_size == 0.001

    assert SYMBOL_SPECS["SOL/USDT:USDT"].tick_size == 0.01
    assert SYMBOL_SPECS["SOL/USDT:USDT"].step_size == 0.1

    assert SYMBOL_SPECS["BNB/USDT:USDT"].tick_size == 0.01
    assert SYMBOL_SPECS["BNB/USDT:USDT"].step_size == 0.01

    assert SYMBOL_SPECS["DOGE/USDT:USDT"].tick_size == 0.00001
    assert SYMBOL_SPECS["DOGE/USDT:USDT"].step_size == 1.0


def test_dry_run_engine_multi_symbol_contexts() -> None:
    """Verifica que el motor inicialice contenedores independientes para los 12 símbolos y seleccione el Top 5 activo."""
    engine = DryRunExecutionEngine()
    assert len(engine.symbol_contexts) == 12
    assert len(engine.top_active_symbols) == 5
    for sym in SYMBOL_SPECS:
        assert sym in engine.symbol_contexts
        ctx = engine.symbol_contexts[sym]
        assert ctx.symbol == sym
        assert ctx.spec.tick_size == SYMBOL_SPECS[sym].tick_size
        assert ctx.spec.step_size == SYMBOL_SPECS[sym].step_size
        assert ctx.ring_buffer is not None
        assert ctx.avellaneda is not None
        assert ctx.liq_hunter is not None
        assert ctx.l2_sim is not None


def test_multivariate_half_kelly_weights_and_bounds() -> None:
    """Verifica la asignacion matricial Half-Kelly F* = 0.5 * C^-1 * M acotada a Gate 3."""
    engine = DryRunExecutionEngine()

    # Caso 1: Sin historial suficiente -> distribucion uniforme acotada (<= 20%)
    weights_init = engine.compute_multivariate_half_kelly()
    assert len(weights_init) == 12
    for sym, w in weights_init.items():
        assert 0.02 <= w <= 0.20

    # Caso 2: Con retornos simulados
    np.random.seed(42)
    for sym in engine.symbols:
        engine.symbol_returns[sym] = list(np.random.normal(loc=0.002, scale=0.01, size=40))

    weights_active = engine.compute_multivariate_half_kelly()
    for sym, w in weights_active.items():
        assert 0.02 <= w <= 0.20  # Gate 3: max 20% por par
    assert sum(weights_active.values()) <= 3.0  # Gate 3: max 3.0x apalancamiento global


def test_cross_asset_contagion_shield_btc_dump() -> None:
    """
    Verifica que el Escudo de Contagio se active cuando BTC OBI < -0.70,
    suprimiendo los BIDs en todas las altcoins.
    """
    engine = DryRunExecutionEngine()
    now_ns = time.time_ns()

    # Inyectar presion de venta extrema en BTC (OBI < -0.70)
    btc_ctx = engine.symbol_contexts["BTC/USDT:USDT"]
    btc_ctx.ring_buffer.push_order_book(
        bid_prices=[65000.0, 64990.0, 64980.0, 64970.0, 64960.0],
        bid_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
        ask_prices=[65010.0, 65020.0, 65030.0, 65040.0, 65050.0],
        ask_qtys=[50.0, 50.0, 50.0, 50.0, 50.0],
        timestamp_ns=now_ns,
    )
    btc_obi = btc_ctx.ring_buffer.compute_obi(levels=5)
    assert btc_obi < -0.70

    # Simular evaluacion del escudo
    obis = {s: ctx.ring_buffer.compute_obi(levels=5) for s, ctx in engine.symbol_contexts.items()}
    bearish_alts = sum(1 for s in engine.symbols if s != "BTC/USDT:USDT" and obis[s] < -0.60)
    bid_shield = (obis["BTC/USDT:USDT"] < -0.70) or (bearish_alts >= 2)

    assert bid_shield is True


def test_cross_asset_contagion_shield_multi_alt_cascade() -> None:
    """
    Verifica que el Escudo de Contagio se active cuando 2 o mas altcoins
    presenten OBI < -0.60 aun si BTC esta neutral.
    """
    engine = DryRunExecutionEngine()
    now_ns = time.time_ns()

    # BTC neutral
    btc_ctx = engine.symbol_contexts["BTC/USDT:USDT"]
    btc_ctx.ring_buffer.push_order_book(
        bid_prices=[65000.0, 64990.0, 64980.0, 64970.0, 64960.0],
        bid_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        ask_prices=[65010.0, 65020.0, 65030.0, 65040.0, 65050.0],
        ask_qtys=[10.0, 10.0, 10.0, 10.0, 10.0],
        timestamp_ns=now_ns,
    )

    # Inyectar venta fuerte en ETH y SOL (OBI < -0.60)
    for alt in ["ETH/USDT:USDT", "SOL/USDT:USDT"]:
        ctx = engine.symbol_contexts[alt]
        p_bid = 2500.0 if "ETH" in alt else 150.0
        p_ask = 2501.0 if "ETH" in alt else 150.1
        ctx.ring_buffer.push_order_book(
            bid_prices=[p_bid - i * 0.1 for i in range(5)],
            bid_qtys=[1.0, 1.0, 1.0, 1.0, 1.0],
            ask_prices=[p_ask + i * 0.1 for i in range(5)],
            ask_qtys=[40.0, 40.0, 40.0, 40.0, 40.0],
            timestamp_ns=now_ns,
        )

    obis = {s: ctx.ring_buffer.compute_obi(levels=5) for s, ctx in engine.symbol_contexts.items()}
    bearish_alts = sum(1 for s in engine.symbols if s != "BTC/USDT:USDT" and obis[s] < -0.60)
    bid_shield = (obis["BTC/USDT:USDT"] < -0.70) or (bearish_alts >= 2)

    assert bearish_alts >= 2
    assert bid_shield is True


def test_telemetry_snapshot_multi_asset_schema() -> None:
    """Verifica que get_telemetry_snapshot incluya las estructuras multimoneda, top 5 y el estado del escudo."""
    engine = DryRunExecutionEngine()
    snapshot = engine.get_telemetry_snapshot()

    assert "symbols" in snapshot
    assert "contagion_shield" in snapshot
    assert "top_active_symbols" in snapshot
    assert len(snapshot["top_active_symbols"]) == 5

    # Schema del escudo
    cs = snapshot["contagion_shield"]
    assert "bid_shield_active" in cs
    assert "ask_shield_active" in cs
    assert "trigger_reason" in cs
    assert "btc_obi" in cs

    # Schema de terminal DEX Pro cuantitativo
    assert "order_book_depth" in snapshot
    assert "bids" in snapshot["order_book_depth"]
    assert "asks" in snapshot["order_book_depth"]
    assert len(snapshot["order_book_depth"]["bids"]) == 5
    assert len(snapshot["order_book_depth"]["asks"]) == 5

    assert "candles_history" in snapshot
    assert len(snapshot["candles_history"]) == 40
    c0 = snapshot["candles_history"][0]
    for key in ("open", "high", "low", "close", "volume", "is_bullish", "reservation_price"):
        assert key in c0

    assert "equity_comparison" in snapshot
    assert len(snapshot["equity_comparison"]) == 50
    eq0 = snapshot["equity_comparison"][0]
    for key in ("timestamp", "bot_equity_usd", "benchmark_equity_usd", "trade_pnl_bar"):
        assert key in eq0

    assert "whale_and_liq_tape" in snapshot
    assert len(snapshot["whale_and_liq_tape"]) >= 10
    w0 = snapshot["whale_and_liq_tape"][0]
    for key in ("symbol", "side", "notional_usd", "intensity_pct", "age_str"):
        assert key in w0

    assert "portfolio_allocation" in snapshot
    assert len(snapshot["portfolio_allocation"]) == 5
    a0 = snapshot["portfolio_allocation"][0]
    for key in ("symbol", "base_asset", "weight_pct", "notional_usd", "color"):
        assert key in a0

    # Schema de los 12 pares
    syms = snapshot["symbols"]
    assert len(syms) == 12
    for expected in SYMBOL_SPECS:
        assert expected in syms
        data = syms[expected]
        assert "symbol" in data
        assert "mid_price" in data
        assert "obi" in data
        assert "inventory_q" in data
        assert "position_usd" in data
        assert "rotations_today" in data
        assert "pnl_today_usd" in data
        assert "half_kelly_weight" in data


def test_step_up_ratchet_85_15_math() -> None:
    """
    Verifica que el trinquete 85/15:
    1. No dispare si la ganancia es menor al +2.0%.
    2. Cuando equity supera +2.0%, transfiera 15% a bóveda y reinvierta 85% a base_capital_usd.
    3. Permita escalar el límite de concentración de Gate 3 (20% del nuevo base_capital).
    """
    state = PortfolioState(
        base_capital_usd=100.0,
        current_equity_usd=101.5,  # +1.5% -> no alcanza el umbral de 2.0%
        daily_high_watermark_usd=101.5,
        realized_pnl_today_usd=1.5,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )
    swept = step_up_ratchet(state, threshold_pct=0.02, vault_ratio=0.15, reinvest_ratio=0.85)
    assert swept == 0.0
    assert state.base_capital_usd == 100.0

    # Ahora equity sube a 104.0 USD (+4.0 USD de ganancia sobre 100 base = +4.0%)
    state.current_equity_usd = 104.0
    swept = step_up_ratchet(state, threshold_pct=0.02, vault_ratio=0.15, reinvest_ratio=0.85)
    # Excedente = 4.0 USD.
    # Boveda 15% = 0.60 USD
    # Reinversion 85% = 3.40 USD -> nuevo base_capital = 103.40 USD
    assert pytest.approx(swept, abs=1e-4) == 0.60
    assert pytest.approx(state.base_capital_usd, abs=1e-4) == 103.40

    # El límite de concentración por par ahora es 20% de 103.40 = 20.68 USD (escalado dinámico)
    gk = MandateGatekeeper()
    assert gk.MAX_SINGLE_PAIR_MARGIN_RATIO == 0.20
    max_notional_before = 100.0 * 0.20
    max_notional_after = state.base_capital_usd * 0.20
    assert max_notional_after > max_notional_before
    assert pytest.approx(max_notional_after, abs=1e-4) == 20.68


def test_step_up_ratchet_hard_cap_25k() -> None:
    """
    Verifica que al alcanzar o superar el tope físico de $25,000 USD (MAX_ACTIVE_CAPITAL_CAP_USD):
    1. El capital base no exceda nunca $25,000 USD.
    2. El 100% de las ganancias posteriores se transfieran a la Bóveda Spot.
    """
    # Caso 1: Cerca del tope ($24,800 USD) y gana $1,000 USD (+4.0%)
    state = PortfolioState(
        base_capital_usd=24_800.0,
        current_equity_usd=25_800.0,
        daily_high_watermark_usd=25_800.0,
        realized_pnl_today_usd=1000.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )
    # Excedente = 1000 USD. Reinvest nominal 85% = 850 USD.
    # Pero solo faltan 200 USD para el tope de $25,000.
    # Reinvest real = 200 USD. Boveda = 800 USD.
    swept = step_up_ratchet(state, threshold_pct=0.02, vault_ratio=0.15, reinvest_ratio=0.85, max_capital_cap_usd=25_000.0)
    assert pytest.approx(state.base_capital_usd, abs=1e-4) == 25_000.0
    assert pytest.approx(swept, abs=1e-4) == 800.0

    # Caso 2: Ya en el tope ($25,000 USD) y gana $500 USD (+2.0%)
    state.current_equity_usd = 25_500.0
    swept2 = step_up_ratchet(state, threshold_pct=0.02, vault_ratio=0.15, reinvest_ratio=0.85, max_capital_cap_usd=25_000.0)
    # 100% del excedente ($500) transferido a boveda, capital base congelado en $25,000
    assert pytest.approx(state.base_capital_usd, abs=1e-4) == 25_000.0
    assert pytest.approx(swept2, abs=1e-4) == 500.0
    assert pytest.approx(state.current_equity_usd, abs=1e-4) == 25_000.0


def test_top_5_active_microstructure_scanner() -> None:
    """Verifica que select_top_active_symbols ordene y devuelva 5 monedas activas garantizando ancla en BTC."""
    engine = DryRunExecutionEngine()
    # Sembrar volatilidad y spreads en un par para que sea el top 1
    sui_ctx = engine.symbol_contexts["SUI/USDT:USDT"]
    now_ns = time.time_ns()
    sui_ctx.ring_buffer.push_order_book(
        bid_prices=[2.0000, 1.9990, 1.9980, 1.9970, 1.9960],
        bid_qtys=[500.0, 500.0, 500.0, 500.0, 500.0],
        ask_prices=[2.0100, 2.0110, 2.0120, 2.0130, 2.0140],
        ask_qtys=[500.0, 500.0, 500.0, 500.0, 500.0],
        timestamp_ns=now_ns,
    )
    sui_ctx.ring_buffer.push_liquidation(price=2.000, qty=1000.0, side="SELL", timestamp_ns=now_ns)
    sui_score = engine.compute_activity_score("SUI/USDT:USDT")
    assert sui_score > 0.0

    top_5 = engine.select_top_active_symbols(k=5)
    assert len(top_5) == 5
    assert "BTC/USDT:USDT" in top_5


def test_3_level_avellaneda_ladder_structure() -> None:
    """Verifica que on_l2_update construya una escalera de 3 niveles Post-Only (45%, 35%, 20%)."""
    cfg = RegimeConfig(
        market_regime="LOW_VOLATILITY",
        gamma=0.10,
        kappa=1.5,
        time_horizon_tau=0.5,
        obi_shield_threshold=0.60,
        obi_volatility_shift_mult=0.5,
        maker_fee_rebate_bps=-0.5,
    )
    gk = MandateGatekeeper()
    engine = AvellanedaStoikovOBIEngine(
        symbol="BTC/USDT:USDT",
        tick_size=0.10,
        step_size=0.001,
        quote_qty=0.010,
        config=cfg,
        gatekeeper=gk,
    )
    rb = MarketRingBuffer(capacity=100)
    now_ns = time.time_ns()
    rb.push_order_book(
        bid_prices=[65000.0, 64990.0, 64980.0, 64970.0, 64960.0],
        bid_qtys=[5.0, 5.0, 5.0, 5.0, 5.0],
        ask_prices=[65010.0, 65020.0, 65030.0, 65040.0, 65050.0],
        ask_qtys=[5.0, 5.0, 5.0, 5.0, 5.0],
        timestamp_ns=now_ns,
    )

    p_state = PortfolioState(
        base_capital_usd=2000.0,
        current_equity_usd=2000.0,
        daily_high_watermark_usd=2000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    class DummyGateway:
        async def cancel_order_ws(self, oid: str, sym: str) -> bool:
            return True

    import asyncio
    quote_res = asyncio.run(
        engine.on_l2_update(rb, p_state, DummyGateway(), normalized_inventory_q=0.0)
    )

    assert quote_res.bid_ladder is not None
    assert quote_res.ask_ladder is not None
    assert len(quote_res.bid_ladder) == 3
    assert len(quote_res.ask_ladder) == 3

    # Nivel 1 más cercano que Nivel 2, y Nivel 2 más cercano que Nivel 3
    bids = quote_res.bid_ladder
    assert bids[0].price > bids[1].price > bids[2].price
    # Proporciones de tamaño: Nivel 1 (45%), Nivel 2 (35%), Nivel 3 (20%)
    assert bids[0].qty >= bids[1].qty >= bids[2].qty


def test_inventory_holding_time_decay() -> None:
    """Verifica que holding_time_decay incremente gamma en 25% tras 15 minutos (900s)."""
    cfg = RegimeConfig(
        market_regime="LOW_VOLATILITY",
        gamma=0.10,
        kappa=1.5,
        time_horizon_tau=0.5,
        obi_shield_threshold=0.60,
        obi_volatility_shift_mult=0.5,
        maker_fee_rebate_bps=-0.5,
    )
    gk = MandateGatekeeper()
    engine = AvellanedaStoikovOBIEngine(
        symbol="BTC/USDT:USDT",
        tick_size=0.10,
        step_size=0.001,
        quote_qty=0.010,
        config=cfg,
        gatekeeper=gk,
    )

    mid = 65000.0
    q = 0.50  # Inventario largo abierto
    sigma = 0.005

    # Caso A: Reciente (holding_time = 60s)
    res_a, spread_a = engine.compute_reservation_and_spread(
        mid_price=mid,
        normalized_inventory_q=q,
        sigma=sigma,
        inventory_holding_time_s=60.0,
    )

    # Caso B: Antiguo (> 900s) -> gamma * 1.25 incrementa la penalización de inventario
    res_b, spread_b = engine.compute_reservation_and_spread(
        mid_price=mid,
        normalized_inventory_q=q,
        sigma=sigma,
        inventory_holding_time_s=950.0,
    )

    # Al ser posición larga (q > 0), un mayor gamma hace que reservation_price baje más
    # para acelerar la venta agresiva del inventario hacia el mercado
    assert res_b < res_a


def test_execution_stats_and_trade_blotter_telemetry() -> None:
    """
    Verifica que el servidor registre operaciones en self.recent_trades,
    actualice Win Rate, Profit Factor, curva de equidad y emita las claves
    'execution_stats', 'open_orders' y 'recent_trades' en el snapshot de telemetría.
    """
    engine = DryRunExecutionEngine()

    # 1. Verificar estado inicial
    initial_snap = engine.get_telemetry_snapshot()
    assert "execution_stats" in initial_snap
    assert "open_orders" in initial_snap
    assert "recent_trades" in initial_snap
    assert initial_snap["execution_stats"]["total_trades"] == 0
    assert initial_snap["execution_stats"]["win_rate_pct"] == 0.0

    # 2. Registrar trade ganador
    trade_win = engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_OBI",
        side="BUY",
        entry_price=65000.0,
        exit_or_mid_price=65050.0,
        qty=0.01,
        return_pct=0.077,
        pnl_usd=0.50,
    )
    assert trade_win["pnl_usd"] == 0.50
    assert engine.wins_count == 1
    assert engine.gross_profit_usd == 0.50

    # 3. Registrar trade perdedor
    trade_loss = engine.record_trade_fill(
        symbol="ETH/USDT:USDT",
        strategy="LIQ_CASCADE",
        side="SELL",
        entry_price=2500.0,
        exit_or_mid_price=2505.0,
        qty=0.1,
        return_pct=-0.20,
        pnl_usd=-0.50,
    )
    assert trade_loss["pnl_usd"] == -0.50
    assert engine.losses_count == 1
    assert engine.gross_loss_usd == 0.50

    # 4. Verificar snapshot actualizado
    snap = engine.get_telemetry_snapshot()
    stats = snap["execution_stats"]
    assert stats["total_trades"] == 2
    assert stats["wins_count"] == 1
    assert stats["losses_count"] == 1
    assert stats["win_rate_pct"] == 50.0
    assert stats["profit_factor"] == 1.0
    assert stats["avg_trade_pnl_usd"] == 0.0
    assert len(snap["recent_trades"]) == 2
    assert snap["recent_trades"][0]["symbol"] == "ETH/USDT:USDT"  # Más reciente primero
    assert snap["recent_trades"][1]["symbol"] == "BTC/USDT:USDT"
    assert "age_str" in snap["recent_trades"][0]
    assert "price" in snap["recent_trades"][0]
    assert snap["recent_trades"][0]["price"] == 2505.0
    assert len(stats["equity_curve"]) >= 10

