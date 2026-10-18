import pytest
import numpy as np
import time
from backend.telemetry_server import DryRunExecutionEngine
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, DSRAuditResult


def test_entry_fill_has_zero_pnl_and_no_win_loss_increment() -> None:
    """
    Verifica que las órdenes de entrada (apertura de inventario) registren estrictamente
    pnl_usd = 0.0, return_pct = 0.0 y NO incrementen contadores de victorias ni derrotas.
    """
    engine = DryRunExecutionEngine()
    engine.portfolio.base_capital_usd = 1000.0
    engine.portfolio.current_equity_usd = 1000.0

    assert engine.wins_count == 0
    assert engine.losses_count == 0
    assert engine.gross_profit_usd == 0.0
    assert engine.gross_loss_usd == 0.0

    # Simular fill de apertura
    entry_fill = engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_OBI",
        side="BUY",
        entry_price=80000.0,
        exit_or_mid_price=80000.0,
        qty=0.01,
        return_pct=0.0,
        pnl_usd=0.0,
        fee_cost=0.16,
        is_close=False,
    )

    assert entry_fill["pnl_usd"] == 0.0
    assert entry_fill["return_pct"] == 0.0
    assert entry_fill["is_close"] is False
    assert engine.wins_count == 0
    assert engine.losses_count == 0
    assert engine.gross_profit_usd == 0.0
    assert engine.gross_loss_usd == 0.0


def test_win_rate_strict_positive_net_pnl_only() -> None:
    """
    Verifica que un trade solo se considere victoria si PnL_neto > 0.
    Si el PnL bruto es positivo pero las comisiones lo vuelven negativo, debe ser derrota.
    """
    engine = DryRunExecutionEngine()

    # Caso 1: Gross PnL +$0.50, comisiones $0.60 -> Net PnL -$0.10 (Derrota)
    loss_trade = engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_CLOSE",
        side="SELL",
        entry_price=80000.0,
        exit_or_mid_price=80050.0,
        qty=0.01,
        return_pct=0.05,
        pnl_usd=0.50,
        fee_cost=0.60,
        is_close=True,
        fee_already_deducted=False,
    )
    assert engine.wins_count == 0
    assert engine.losses_count == 1
    assert engine.gross_loss_usd == 0.10

    # Caso 2: Gross PnL +$1.00, comisiones $0.20 -> Net PnL +$0.80 (Victoria)
    win_trade = engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_TAKE_PROFIT",
        side="SELL",
        entry_price=80000.0,
        exit_or_mid_price=80100.0,
        qty=0.01,
        return_pct=0.10,
        pnl_usd=1.00,
        fee_cost=0.20,
        is_close=True,
        fee_already_deducted=False,
    )
    assert engine.wins_count == 1
    assert engine.losses_count == 1
    assert engine.gross_profit_usd == 0.80

    # Caso 3: Breakeven neto ($0.00) no altera ni victorias ni derrotas
    engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_CLOSE",
        side="SELL",
        entry_price=80000.0,
        exit_or_mid_price=80000.0,
        qty=0.01,
        return_pct=0.0,
        pnl_usd=0.20,
        fee_cost=0.20,
        is_close=True,
        fee_already_deducted=False,
    )
    assert engine.wins_count == 1
    assert engine.losses_count == 1


@pytest.mark.asyncio
async def test_paper_trading_tp_maker_sl_taker_slippage_and_funding() -> None:
    """
    Verifica que check_and_execute_tp_sl_closures descuente:
    - Maker fee (2 bps) en TP
    - Taker fee (5 bps) + Slippage adverso (2 bps) en SL
    - Costo de financiación acumulado por tiempo de retención
    """
    engine = DryRunExecutionEngine()
    engine.trading_mode = "PAPER_L2"
    engine.portfolio.base_capital_usd = 1000.0
    engine.portfolio.current_equity_usd = 1000.0
    engine.latest_funding_rate = 0.0001  # 1 bp por 8h

    ctx = engine.symbol_contexts["BTC/USDT:USDT"]
    ctx.avg_entry_price = 80000.0
    ctx.inventory_open_ts_ns = time.time_ns() - int(8 * 3600 * 1e9)  # 8 horas de retención
    engine.portfolio.active_positions["BTC/USDT:USDT"] = 800.0  # 0.01 BTC largo

    # 1. Configurar TP en 80160 (+20 bps) y simular mejor bid/ask alcanzando el TP
    ctx.take_profit_price = 80160.0
    ctx.stop_loss_price = 79800.0

    # Llenar ring buffer con bid/ask que toque el TP
    ctx.ring_buffer.push_order_book(
        bid_prices=np.array([80160.0, 80155.0, 80150.0, 80145.0, 80140.0]),
        bid_qtys=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ask_prices=np.array([80165.0, 80170.0, 80175.0, 80180.0, 80185.0]),
        ask_qtys=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        timestamp_ns=time.time_ns(),
    )

    exit_reason = await engine.check_and_execute_tp_sl_closures(
        symbol="BTC/USDT:USDT",
        best_bid=80160.0,
        best_ask=80165.0,
        mid_px=80162.5,
        now_ns=time.time_ns(),
    )
    assert exit_reason == "TAKE_PROFIT"

    # Verificar que se registraron los fees de maker (2 bps entrada + 2 bps salida = 4 bps)
    # y funding de 8h (1 bp). Total 5 bps sobre 800 USD = 0.40 USD.
    # Bruto: (80160 - 80000) * 0.01 = $1.60.
    # Neto esperado ~ $1.20 USD.
    assert len(engine.recent_trades) == 1
    last_trade = engine.recent_trades[-1]
    assert last_trade["strategy"] == "TAKE_PROFIT"
    assert last_trade["pnl_usd"] < 1.60
    assert last_trade["fee_cost"] > 0.0
    assert engine.wins_count == 1

    # 2. Probar SL con Slippage adverso y Taker fee
    ctx.avg_entry_price = 80000.0
    ctx.inventory_open_ts_ns = time.time_ns() - int(4 * 3600 * 1e9)
    engine.portfolio.active_positions["BTC/USDT:USDT"] = 800.0
    ctx.take_profit_price = 80500.0
    ctx.stop_loss_price = 79600.0  # -50 bps

    # Simular caída que traspasa el SL
    ctx.ring_buffer.push_order_book(
        bid_prices=np.array([79550.0, 79540.0, 79530.0, 79520.0, 79510.0]),
        bid_qtys=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ask_prices=np.array([79560.0, 79570.0, 79580.0, 79590.0, 79600.0]),
        ask_qtys=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        timestamp_ns=time.time_ns(),
    )

    exit_reason_sl = await engine.check_and_execute_tp_sl_closures(
        symbol="BTC/USDT:USDT",
        best_bid=79550.0,
        best_ask=79560.0,
        mid_px=79555.0,
        now_ns=time.time_ns(),
    )
    assert exit_reason_sl == "STOP_LOSS"
    assert engine.losses_count == 1
    sl_trade = engine.recent_trades[-1]
    assert sl_trade["strategy"] == "STOP_LOSS"
    # El precio efectivo de salida debe ser inferior a 79550 por slippage adverso
    assert sl_trade["exit_or_mid_price"] < 79550.0


def test_dsr_telemetry_insufficient_data_status_and_null_metrics() -> None:
    """
    Verifica que con < 30 trades la telemetría reporte status INSUFICIENTE
    y retorne null en deflated_sharpe_ratio y observed_sharpe_sr sin inventar valores.
    """
    engine = DryRunExecutionEngine()
    engine.returns_history = [0.001, -0.0005, 0.002]  # Solo 3 retornos

    snap = engine.get_telemetry_snapshot()
    dsr_audit = snap["dsr_audit"]

    assert "INSUFICIENTE (3 trades)" in dsr_audit["status"]
    assert dsr_audit["deflated_sharpe_ratio"] is None
    assert dsr_audit["observed_sharpe_sr"] is None
    assert dsr_audit["benchmark_null_sr0"] is None
    assert dsr_audit["is_overfitted"] is False

    # Ahora agregar 35 retornos reales para superar el umbral de 30
    rng = np.random.default_rng(seed=123)
    engine.returns_history = list(rng.normal(0.0005, 0.002, 35))

    snap_sufficient = engine.get_telemetry_snapshot()
    dsr_sufficient = snap_sufficient["dsr_audit"]

    assert dsr_sufficient["deflated_sharpe_ratio"] is not None
    assert dsr_sufficient["observed_sharpe_sr"] is not None
    assert dsr_sufficient["status"] in ("STATISTICALLY_SIGNIFICANT", "OVERFITTING_WARNING")


def test_dynamic_half_kelly_and_leverage_tier_clamp() -> None:
    """
    Verifica que:
    1. LEVERAGE_TIERS_CONFIG ya no contenga la constante fija est_daily_roi_pct.
    2. Con N < 100 trades, el multiplicador Half-Kelly esté acotado a <= 1.0x.
    3. Con N >= 100 trades, el multiplicador Half-Kelly se calcule dinámicamente de mu y sigma2.
    """
    engine = DryRunExecutionEngine()

    for lev, cfg in engine.LEVERAGE_TIERS_CONFIG.items():
        assert "est_daily_roi_pct" not in cfg

    # Caso 1: N < 100 con alta rentabilidad simulada
    engine.returns_history = [0.05] * 20  # Rendimiento extremo pero solo 20 trades
    analysis = engine.evaluate_leverage_tiers()
    assert analysis["raw_half_kelly_x"] <= 1.0  # Clamped por seguridad institucional

    # Caso 2: N >= 100 con distribución calibrada
    rng = np.random.default_rng(seed=42)
    # Media 0.1%, Desviación estándar 0.5% -> Varianza 0.000025
    # Half-Kelly teórico = 0.5 * (0.001 / 0.000025) = 20.0 (desbloqueado de 1.0x)
    engine.returns_history = list(rng.normal(0.001, 0.005, 120))
    analysis_100 = engine.evaluate_leverage_tiers()
    assert analysis_100["raw_half_kelly_x"] > 1.0
