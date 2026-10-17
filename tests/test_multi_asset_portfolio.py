# tests/test_multi_asset_portfolio.py
from __future__ import annotations

from datetime import datetime, timezone
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.profit_sweeper import step_up_ratchet
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.regime_config import RegimeConfig
from backend.telemetry_server import (
    BingXLiveWsGateway,
    ContagionShieldState,
    DryRunExecutionEngine,
    SYMBOL_SPECS,
    SymbolMarketSpec,
    is_tradfi_market_open,
)


def test_symbol_specs_and_context_initialization() -> None:
    """Verifica que SYMBOL_SPECS contenga los 14 pares (12 cripto + 2 TradFi commodities BingX) con sus ticks y steps exactos."""
    assert len(SYMBOL_SPECS) == 14
    for sym in [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT",
        "XRP/USDT:USDT", "SUI/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "PEPE/USDT:USDT",
        "NEAR/USDT:USDT", "ARB/USDT:USDT", "XAU/USDT:USDT", "XAG/USDT:USDT",
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

    # BingX TradFi Commodities
    assert SYMBOL_SPECS["XAU/USDT:USDT"].tick_size == 0.10
    assert SYMBOL_SPECS["XAU/USDT:USDT"].step_size == 0.01
    assert SYMBOL_SPECS["XAU/USDT:USDT"].min_spread_bps == 6.0
    assert SYMBOL_SPECS["XAU/USDT:USDT"].asset_class == "TRADFI_COMMODITY"
    assert SYMBOL_SPECS["XAU/USDT:USDT"].exchange == "bingx"

    assert SYMBOL_SPECS["XAG/USDT:USDT"].tick_size == 0.01
    assert SYMBOL_SPECS["XAG/USDT:USDT"].step_size == 0.1
    assert SYMBOL_SPECS["XAG/USDT:USDT"].min_spread_bps == 6.5
    assert SYMBOL_SPECS["XAG/USDT:USDT"].asset_class == "TRADFI_COMMODITY"
    assert SYMBOL_SPECS["XAG/USDT:USDT"].exchange == "bingx"


def test_dry_run_engine_multi_symbol_contexts() -> None:
    """Verifica que el motor inicialice contenedores independientes para los 14 símbolos y seleccione el Top 5 activo."""
    engine = DryRunExecutionEngine()
    assert len(engine.symbol_contexts) == 14
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
    assert len(weights_init) == 14
    for sym, w in weights_init.items():
        assert 0.0 <= w <= 0.20

    # Caso 2: Con retornos simulados
    np.random.seed(42)
    for sym in engine.symbols:
        engine.symbol_returns[sym] = list(np.random.normal(loc=0.002, scale=0.01, size=40))

    weights_active = engine.compute_multivariate_half_kelly()
    for sym, w in weights_active.items():
        assert 0.0 <= w <= 0.20  # Gate 3: max 20% por par
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

    # Schema de los 14 pares
    syms = snapshot["symbols"]
    assert len(syms) == 14
    assert "tradfi_market_open" in snapshot
    assert "xau_safe_haven_active" in snapshot["contagion_shield"]
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


def test_is_tradfi_market_open_sessions() -> None:
    """Verifica con precisión los horarios de apertura y cierre de mercados TradFi (XAU/XAG)."""
    # 1. Viernes pre-cierre y cierre (20:30 UTC)
    fri_open = datetime(2026, 10, 16, 20, 29, 0, tzinfo=timezone.utc)
    fri_closed = datetime(2026, 10, 16, 20, 30, 0, tzinfo=timezone.utc)
    assert is_tradfi_market_open(fri_open) is True
    assert is_tradfi_market_open(fri_closed) is False

    # 2. Sábado completo cerrado
    sat_midday = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)
    assert is_tradfi_market_open(sat_midday) is False

    # 3. Domingo antes y después de reapertura (22:05 UTC)
    sun_pre = datetime(2026, 10, 18, 22, 4, 0, tzinfo=timezone.utc)
    sun_reopen = datetime(2026, 10, 18, 22, 5, 0, tzinfo=timezone.utc)
    assert is_tradfi_market_open(sun_pre) is False
    assert is_tradfi_market_open(sun_reopen) is True

    # 4. Mantenimiento diario (Lunes a Jueves 20:45 UTC a 22:05 UTC)
    tue_regular = datetime(2026, 10, 20, 20, 44, 0, tzinfo=timezone.utc)
    tue_maint = datetime(2026, 10, 20, 20, 45, 0, tzinfo=timezone.utc)
    tue_maint_2 = datetime(2026, 10, 20, 21, 30, 0, tzinfo=timezone.utc)
    tue_maint_3 = datetime(2026, 10, 20, 22, 4, 0, tzinfo=timezone.utc)
    tue_post_maint = datetime(2026, 10, 20, 22, 5, 0, tzinfo=timezone.utc)
    assert is_tradfi_market_open(tue_regular) is True
    assert is_tradfi_market_open(tue_maint) is False
    assert is_tradfi_market_open(tue_maint_2) is False
    assert is_tradfi_market_open(tue_maint_3) is False
    assert is_tradfi_market_open(tue_post_maint) is True


def test_tradfi_reallocation_when_market_closed() -> None:
    """Verifica que cuando el mercado TradFi está cerrado, el cupo de XAU y XAG se reasigne a los pares cripto 24/7."""
    engine = DryRunExecutionEngine()

    with patch("backend.telemetry_server.is_tradfi_market_open", return_value=False):
        weights = engine.compute_multivariate_half_kelly()
        assert weights["XAU/USDT:USDT"] == 0.0
        assert weights["XAG/USDT:USDT"] == 0.0

        # Los 12 pares cripto 24/7 deben tener ponderaciones positivas y <= 0.20 (Gate 3)
        for sym in engine.symbols:
            if sym not in ("XAU/USDT:USDT", "XAG/USDT:USDT"):
                assert weights[sym] > 0.0
                assert weights[sym] <= 0.20

        assert sum(weights.values()) <= 3.0


def test_cross_asset_contagion_shield_gold_safe_haven_exemption() -> None:
    """
    Verifica que en una caída de BTC (OBI < -0.70), si Oro (XAU) tiene OBI > -0.40
    y correlación baja con BTC (< 0.25), Oro queda exento del bloqueo y actúa como activo refugio.
    """
    engine = DryRunExecutionEngine()
    now_ns = time.time_ns()

    # Inyectar caída severa en BTC (OBI < -0.70)
    btc_ctx = engine.symbol_contexts["BTC/USDT:USDT"]
    btc_ctx.ring_buffer.push_order_book(
        bid_prices=[65000.0 - i * 0.1 for i in range(5)],
        bid_qtys=[1.0] * 5,
        ask_prices=[65010.0 + i * 0.1 for i in range(5)],
        ask_qtys=[60.0] * 5,
        timestamp_ns=now_ns,
    )
    assert btc_ctx.ring_buffer.compute_obi(levels=5) < -0.70

    # Oro saludable (OBI > -0.40)
    xau_ctx = engine.symbol_contexts["XAU/USDT:USDT"]
    xau_ctx.ring_buffer.push_order_book(
        bid_prices=[2650.0 - i * 0.1 for i in range(5)],
        bid_qtys=[20.0] * 5,
        ask_prices=[2650.2 + i * 0.1 for i in range(5)],
        ask_qtys=[5.0] * 5,
        timestamp_ns=now_ns,
    )
    xau_obi = xau_ctx.ring_buffer.compute_obi(levels=5)
    assert xau_obi > -0.40

    # Simular retornos descorrelacionados
    np.random.seed(99)
    engine.symbol_returns["BTC/USDT:USDT"] = list(np.random.normal(-0.01, 0.02, 30))
    engine.symbol_returns["XAU/USDT:USDT"] = list(np.random.normal(0.005, 0.005, 30))

    corr = engine.compute_rolling_correlation("XAU/USDT:USDT", "BTC/USDT:USDT")
    assert corr < 0.25

    # Evaluar condición de refugio
    is_gold_haven = (btc_ctx.ring_buffer.compute_obi(levels=5) < -0.70) and (xau_obi > -0.40) and (corr < 0.25)
    assert is_gold_haven is True


@pytest.mark.asyncio
async def test_bingx_live_ws_gateway_post_only_and_margin() -> None:
    """Verifica que BingXLiveWsGateway configure margen aislado, posición unidireccional y traduzca GTX a PO."""
    mock_exchange = MagicMock()
    mock_exchange.set_position_mode = AsyncMock(return_value={"code": 0, "msg": "success"})
    mock_exchange.set_margin_mode = AsyncMock(return_value={"code": 0, "msg": "success"})
    mock_exchange.set_leverage = AsyncMock(return_value={"code": 0, "leverage": 3})
    mock_exchange.create_order = AsyncMock(return_value={"id": "bingx_order_999", "status": "open"})
    mock_exchange.create_order_ws = AsyncMock(return_value={"id": "bingx_order_999", "status": "open"})
    mock_exchange.cancel_order = AsyncMock(return_value={"id": "bingx_order_999", "status": "canceled"})
    mock_exchange.cancel_order_ws = AsyncMock(return_value={"id": "bingx_order_999", "status": "canceled"})
    mock_exchange.cancel_all_orders = AsyncMock(return_value=[{"id": "bingx_order_999"}])
    mock_exchange.fetch_positions = AsyncMock(return_value=[])

    gatekeeper = MandateGatekeeper()
    portfolio = PortfolioState(
        base_capital_usd=5000.0,
        current_equity_usd=5000.0,
        daily_high_watermark_usd=5000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    gateway = BingXLiveWsGateway(
        api_key="mock_bingx_key",
        api_secret="mock_bingx_secret",
        exchange=mock_exchange,
        gatekeeper=gatekeeper,
        portfolio=portfolio,
    )

    # 1. Configuración de margen y apalancamiento
    await gateway.initialize_symbol_margin("XAU/USDT:USDT", max_leverage=3)
    mock_exchange.set_position_mode.assert_awaited_once_with(False, "XAU/USDT:USDT")
    mock_exchange.set_margin_mode.assert_awaited_once_with("ISOLATED", "XAU/USDT:USDT")
    mock_exchange.set_leverage.assert_awaited_once_with(3, "XAU/USDT:USDT")
    assert "XAU/USDT:USDT" in gateway.initialized_symbols

    # 2. Despacho de orden con traducción obligatoria de GTX a Post-Only PO
    order = OrderRequest(
        symbol="XAU/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=2650.0,
        qty=0.02,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=1_000_000_000,
    )
    order_id = await gateway.place_order_ws(order)
    assert order_id == "bingx_order_999"
    assert gateway.live_active_bid_id == "bingx_order_999"

    # Verificar que los parámetros enviados a CCXT tengan PO y postOnly=True
    create_call_kwargs = mock_exchange.create_order_ws.call_args.kwargs
    assert create_call_kwargs["params"]["timeInForce"] == "PO"
    assert create_call_kwargs["params"]["postOnly"] is True

    # 3. Cancelación de orden
    cancelled = await gateway.cancel_order_ws("bingx_order_999", "XAU/USDT:USDT")
    assert cancelled is True
    assert gateway.live_active_bid_id is None

    # 4. Emergency Flatten
    await gateway.emergency_flatten_all(["XAU/USDT:USDT"])
    mock_exchange.cancel_all_orders.assert_awaited_once_with("XAU/USDT:USDT")


@pytest.mark.asyncio
async def test_short_position_sign_in_sync_and_emergency_flatten() -> None:
    """
    Verifica que cuando CCXT devuelve una posición SHORT (contracts > 0 pero side == 'short'),
    sync_portfolio_state registre active_positions con signo negativo (-notional) y
    emergency_flatten_all envíe una orden BUY con reduceOnly=True (evitando el error -2022).
    """
    from backend.telemetry_server import BinanceLiveWsGateway

    mock_exchange = MagicMock()
    mock_exchange.cancel_all_orders = AsyncMock(return_value=[])
    mock_exchange.create_order = AsyncMock(return_value={"id": "close_short_1", "status": "closed"})
    mock_exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 5000.0}})
    mock_exchange.fetch_open_orders = AsyncMock(return_value=[])
    mock_exchange.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.016,
                "notional": 1305.31,
                "side": "short",
                "info": {"positionAmt": "-0.016"},
            }
        ]
    )

    gw = BinanceLiveWsGateway(exchange=mock_exchange)
    portfolio = PortfolioState(
        base_capital_usd=5000.0,
        current_equity_usd=5000.0,
        daily_high_watermark_usd=5000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.045,
        rolling_variance_sigma2=0.0090,
    )

    await gw.sync_portfolio_state(portfolio)
    assert portfolio.active_positions["BTC/USDT:USDT"] == -1305.31

    await gw.emergency_flatten_all(["BTC/USDT:USDT"])
    mock_exchange.create_order.assert_awaited_once_with(
        symbol="BTC/USDT:USDT",
        type="market",
        side="buy",
        amount=0.016,
        params={"reduceOnly": True},
    )


@pytest.mark.asyncio
async def test_position_brackets_and_tp_sl_closures() -> None:
    """
    Verifica que update_position_brackets calcule TP y SL exactos tanto para LONG como para SHORT,
    y que check_and_execute_tp_sl_closures cierre determinísticamente la posición en TAKE_PROFIT
    y en STOP_LOSS actualizando el equity y limpiando active_positions.
    """
    eng = DryRunExecutionEngine()
    sym = "BTC/USDT:USDT"
    ctx = eng.symbol_contexts[sym]
    now_ns = time.time_ns()

    # 1. Caso LONG: entrada a 65000.0 con +1000 USDT nocional
    eng.portfolio.base_capital_usd = 5000.0
    eng.portfolio.current_equity_usd = 5000.0
    eng.portfolio.active_positions[sym] = 1000.0
    ctx.inventory_open_ts_ns = now_ns
    eng.update_position_brackets(sym, entry_price=65000.0, signed_notional=1000.0)

    assert ctx.avg_entry_price == 65000.0
    assert ctx.take_profit_price is not None and ctx.take_profit_price > 65000.0
    assert ctx.stop_loss_price is not None and ctx.stop_loss_price < 65000.0

    # Simular que el bid alcanza el Take Profit
    tp_target = ctx.take_profit_price
    reason_tp = await eng.check_and_execute_tp_sl_closures(
        symbol=sym,
        best_bid=tp_target + 1.0,
        best_ask=tp_target + 2.0,
        mid_px=tp_target + 1.5,
        now_ns=now_ns + 1_000_000_000,
    )
    assert reason_tp == "TAKE_PROFIT"
    assert sym not in eng.portfolio.active_positions
    assert eng.tp_exits_count == 1
    assert eng.portfolio.current_equity_usd > 5000.0

    # 2. Caso SHORT: entrada a 65000.0 con -1000 USDT nocional
    eq_before_sl = eng.portfolio.current_equity_usd
    eng.portfolio.active_positions[sym] = -1000.0
    ctx.inventory_open_ts_ns = now_ns
    eng.update_position_brackets(sym, entry_price=65000.0, signed_notional=-1000.0)

    assert ctx.take_profit_price is not None and ctx.take_profit_price < 65000.0
    assert ctx.stop_loss_price is not None and ctx.stop_loss_price > 65000.0

    # Simular que el precio sube y toca el Stop Loss del SHORT
    sl_target = ctx.stop_loss_price
    reason_sl = await eng.check_and_execute_tp_sl_closures(
        symbol=sym,
        best_bid=sl_target,
        best_ask=sl_target + 1.0,
        mid_px=sl_target + 0.5,
        now_ns=now_ns + 2_000_000_000,
    )
    assert reason_sl == "STOP_LOSS"
    assert sym not in eng.portfolio.active_positions
    assert eng.sl_exits_count == 1
    assert eng.portfolio.current_equity_usd < eq_before_sl


@pytest.mark.asyncio
async def test_multi_tier_leverage_evaluation_and_switching() -> None:
    """
    Verifica que evaluate_leverage_tiers evalúe x3, x5, x7 y x10 con sus métricas de riesgo/retorno,
    y que set_leverage_tier actualice MandateGatekeeper y los brackets TP/SL.
    """
    eng = DryRunExecutionEngine()
    eng.portfolio.base_capital_usd = 5000.0

    analysis = eng.evaluate_leverage_tiers()
    assert analysis["selected_leverage"] == 3
    assert analysis["recommended_leverage"] in (3, 5, 7, 10)
    tiers = {t["leverage"]: t for t in analysis["tiers"]}
    assert set(tiers.keys()) == {3, 5, 7, 10}
    assert tiers[3]["max_portfolio_notional_usd"] == 15000.0
    assert tiers[5]["max_portfolio_notional_usd"] == 25000.0
    assert tiers[7]["max_portfolio_notional_usd"] == 35000.0
    assert tiers[10]["max_portfolio_notional_usd"] == 50000.0
    # A mayor apalancamiento, el SL en bps es más ajustado para proteger el margen aislado
    assert tiers[3]["sl_bps"] > tiers[5]["sl_bps"] > tiers[7]["sl_bps"] > tiers[10]["sl_bps"]

    # Cambiar a x7
    updated = await eng.set_leverage_tier(7)
    assert updated["selected_leverage"] == 7
    assert eng.gatekeeper.max_leverage_cap == 7.0
    assert eng.symbol_contexts["BTC/USDT:USDT"].tp_bps == 14.0
    assert eng.symbol_contexts["BTC/USDT:USDT"].sl_bps == 18.0


