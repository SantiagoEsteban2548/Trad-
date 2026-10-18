# backend/backtest/engine.py
"""
Motor Cuantitativo Unificado de Backtest Institucional (Zero Look-Ahead & Real-Friction).
Implementa:
1. Zero look-ahead: cotizaciones fijadas estrictamente con t-1 (o apertura t).
2. Fills maker verificados con penetración estricta de precio (low < bid_px).
3. Motor B (mecha de liquidación): señal detectada al cierre de N y orden colocada en N+1.
4. Media móvil de volumen de 20 periodos reales.
5. Prelación conservadora: en colisión simultánea de TP y SL en la misma vela, el STOP LOSS gana.
6. Estructura de comisiones institucionales (2 bps maker, 5 bps taker + 2 bps slippage en stops).
7. Marcado a mercado (MTM) continuo barra a barra para cálculo verídico de Max Drawdown.
8. Control de ruina: detención inmediata si la equidad MTM <= 0.
9. Auditoría DSR sobre retornos diarios agregados y cálculo de Sharpe Anualizado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

import numpy as np

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.profit_sweeper import step_up_ratchet
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.strategies.regime_config import RegimeConfig
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, TrialRegistry


@dataclass
class BacktestTrade:
    entry_bar_idx: int
    exit_bar_idx: int
    entry_ts_ms: int
    exit_ts_ms: int
    symbol: str
    side: str
    strategy: str
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl_usd: float
    fee_cost_usd: float
    funding_cost_usd: float
    net_pnl_usd: float
    return_pct: float
    exit_reason: str  # "TAKE_PROFIT" | "STOP_LOSS" | "TIME_STOP" | "RUIN"
    bars_held: int


@dataclass
class BacktestRunResult:
    status: Literal["COMPLETED", "RUIN"]
    symbol: str
    strategy_mode: str
    bars_count: int
    start_date: str
    end_date: str
    initial_capital_usd: float
    final_equity_usd: float
    spot_vault_swept_usd: float
    total_wealth_usd: float
    compounded_roi_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_per_trade_usd: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    daily_returns: list[float]
    observed_daily_sharpe: float
    annualized_sharpe: float
    deflated_sharpe_ratio: float
    dsr_status: str
    is_overfitted: bool
    num_trials_n: int
    kill_switch_triggers: int
    trades: list[BacktestTrade] = field(default_factory=list)


class InstitutionalBacktestEngine:
    """
    Motor de simulación cuantitativa de derivados cripto con física de ejecución L2 rigurosa.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        initial_capital_usd: float = 1000.0,
        maker_fee_bps: float = 2.0,      # 0.0002 (SymbolMarketSpec)
        taker_fee_bps: float = 5.0,      # 0.0005 (SymbolMarketSpec)
        slippage_bps: float = 2.0,       # 0.0002 en salidas a mercado/stop
        target_spread_bps: float = 18.0, # 18 bps de spread base
        tick_size: float = 0.10,
        step_size: float = 0.0001,
        min_notional_usd: float = 5.0,
        stop_loss_mult: float = 2.5,     # 2.5x el spread
        take_profit_mult: float = 1.0,   # 1.0x el spread
        compound_growth: bool = False,
        leverage: float = 1.0,
        max_single_pair_margin_ratio: float = 0.20,
        require_tick_penetration: bool = True,  # low_p < bid_px (no toque rasante)
        fixed_risk_pct: float | None = None,    # ej. 0.005 (0.5% del capital) para Fase 5
        risk_reward_ratio: float | None = None, # ej. 3.0 para R:R 1:3
        max_holding_bars: int = 12,      # 12 velas (3h en timeframe 15m)
        regime_config: RegimeConfig | None = None,
    ) -> None:
        self.symbol = symbol
        self.initial_capital_usd = float(initial_capital_usd)
        self.maker_fee = maker_fee_bps / 10_000.0
        self.taker_fee = taker_fee_bps / 10_000.0
        self.slippage = slippage_bps / 10_000.0
        self.target_spread_bps = target_spread_bps
        self.tick_size = tick_size
        self.step_size = step_size
        self.min_notional_usd = min_notional_usd
        self.stop_loss_mult = stop_loss_mult
        self.take_profit_mult = take_profit_mult
        self.compound_growth = compound_growth
        self.leverage = float(leverage)
        self.require_tick_penetration = require_tick_penetration
        self.fixed_risk_pct = fixed_risk_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.max_holding_bars = max_holding_bars
        self.max_single_pair_margin_ratio = float(max_single_pair_margin_ratio)
        self.regime_config = regime_config or RegimeConfig.default_normal()

    def run(
        self,
        candles: Sequence[tuple[int, float, float, float, float, float]],
        strategy_mode: Literal["MARKET_MAKING", "DIRECTIONAL_TREND"] = "MARKET_MAKING",
        num_trials_n: int | None = None,
    ) -> BacktestRunResult:
        """
        Ejecuta la simulación de backtest barra a barra sin look-ahead.
        Cada vela es: (timestamp_ms, open, high, low, close, volume).
        """
        n_trials = num_trials_n if num_trials_n is not None else TrialRegistry.get_trials_count()

        if len(candles) < 25:
            raise ValueError(f"Se requieren al menos 25 velas para el backtest. Recibidas: {len(candles)}")

        gatekeeper = MandateGatekeeper(max_single_pair_margin_ratio=self.max_single_pair_margin_ratio)
        if self.leverage > 1.0:
            gatekeeper.set_leverage_tier(self.leverage)
            if self.max_single_pair_margin_ratio != 0.20:
                gatekeeper.max_single_pair_margin_ratio = self.max_single_pair_margin_ratio
        portfolio = PortfolioState(
            base_capital_usd=self.initial_capital_usd,
            current_equity_usd=self.initial_capital_usd,
            daily_high_watermark_usd=self.initial_capital_usd,
            realized_pnl_today_usd=0.0,
            active_positions={},
            rolling_edge_mu=0.045,
            rolling_variance_sigma2=0.0090,
        )

        swept_spot_vault_usd = 0.0
        active_inventory_qty = 0.0
        avg_entry_price = 0.0
        entry_bar_idx = 0
        entry_ts_ms = 0
        is_cascade_position = False
        current_strategy = strategy_mode
        active_sl_price = 0.0
        active_tp_price = 0.0

        trades: list[BacktestTrade] = []
        rolling_returns: list[float] = []
        ema_obi = 0.0
        current_day_str = ""
        base_funding_rate_8h = 0.00008  # ~0.008% cada 8h en Binance
        kill_switch_triggers = 0

        peak_mtm_equity = self.initial_capital_usd
        max_drawdown_pct = 0.0
        max_drawdown_usd = 0.0

        daily_equity_snapshots: list[tuple[int, float]] = []
        daily_equity_snapshots.append((candles[0][0], self.initial_capital_usd))

        # Señales pendientes calculadas al cierre de i-1 para ejecución en i
        pending_order: dict[str, Any] | None = None

        for i in range(1, len(candles)):
            ts_ms, open_p, high_p, low_p, close_p, volume = candles[i]
            prev_ts, prev_open, prev_high, prev_low, prev_close, prev_vol = candles[i - 1]
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            now_ns = int(ts_ms * 1_000_000)

            # Volatilidad realizada rolling (30 barras previas, cerradas hasta i-1)
            bar_ret = (prev_close - candles[max(0, i - 2)][4]) / max(candles[max(0, i - 2)][4], 1e-8)
            rolling_returns.append(bar_ret)
            if len(rolling_returns) > 30:
                rolling_returns.pop(0)
            sigma = max(float(np.std(rolling_returns)), 0.0003)

            # OBI cerrado hasta i-1
            prev_range = max(prev_high - prev_low, self.tick_size)
            prev_body = prev_close - prev_open
            raw_obi = float(np.clip(prev_body / prev_range, -0.95, 0.95))
            ema_obi = 0.85 * ema_obi + 0.15 * raw_obi

            # Media móvil de volumen de las 20 velas anteriores (i-20 a i-1)
            vol_window = [c[5] for c in candles[max(0, i - 20):i]]
            vol_ma = float(np.mean(vol_window)) if vol_window else prev_vol

            # -----------------------------------------------------------------
            # 1. Mantenimiento Diario y Bóveda (a las 00:00 UTC)
            # -----------------------------------------------------------------
            if current_day_str != date_str:
                current_day_str = date_str
                # Registrar snapshot diario para DSR
                daily_equity_snapshots.append((ts_ms, portfolio.current_equity_usd + swept_spot_vault_usd))
                portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            if dt.hour == 0 and dt.minute == 0 and active_inventory_qty == 0.0:
                if not self.compound_growth:
                    if portfolio.current_equity_usd > self.initial_capital_usd:
                        surplus = round(portfolio.current_equity_usd - self.initial_capital_usd, 2)
                        swept_spot_vault_usd += surplus
                        portfolio.current_equity_usd = self.initial_capital_usd
                        portfolio.daily_high_watermark_usd = self.initial_capital_usd
                else:
                    portfolio.base_capital_usd = portfolio.current_equity_usd
                    portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            # -----------------------------------------------------------------
            # 2. Financiamiento de Derivados (Funding Rate cada 8h: 00:00, 08:00, 16:00 UTC)
            # -----------------------------------------------------------------
            if dt.minute == 0 and dt.hour in (0, 8, 16):
                if abs(active_inventory_qty) > 0:
                    funding_cost = (active_inventory_qty * prev_close) * base_funding_rate_8h
                    portfolio.current_equity_usd -= funding_cost

            # -----------------------------------------------------------------
            # 3. Marcado a Mercado (MTM) continuo y Chequeo de Ruina
            # -----------------------------------------------------------------
            if active_inventory_qty > 0.0:
                # Peor excursión adversa dentro de la vela i (el low)
                mtm_unrealized = (low_p - avg_entry_price) * active_inventory_qty
                mtm_equity = portfolio.current_equity_usd + swept_spot_vault_usd + mtm_unrealized
            else:
                mtm_equity = portfolio.current_equity_usd + swept_spot_vault_usd

            if mtm_equity <= 0.0:
                # Quiebra del portafolio: registrar ruina y detener simulación
                final_tot = max(0.0, mtm_equity)
                roi_pct = -100.0
                return BacktestRunResult(
                    status="RUIN",
                    symbol=self.symbol,
                    strategy_mode=strategy_mode,
                    bars_count=i + 1,
                    start_date=datetime.fromtimestamp(candles[0][0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d"),
                    end_date=date_str,
                    initial_capital_usd=self.initial_capital_usd,
                    final_equity_usd=0.0,
                    spot_vault_swept_usd=swept_spot_vault_usd,
                    total_wealth_usd=0.0,
                    compounded_roi_pct=roi_pct,
                    total_trades=len(trades),
                    winning_trades=sum(1 for t in trades if t.net_pnl_usd > 0),
                    losing_trades=sum(1 for t in trades if t.net_pnl_usd <= 0),
                    win_rate_pct=round((sum(1 for t in trades if t.net_pnl_usd > 0) / max(len(trades), 1)) * 100.0, 2),
                    profit_factor=0.0,
                    expectancy_per_trade_usd=-self.initial_capital_usd / max(len(trades), 1),
                    max_drawdown_pct=100.0,
                    max_drawdown_usd=peak_mtm_equity,
                    daily_returns=[-1.0],
                    observed_daily_sharpe=0.0,
                    annualized_sharpe=0.0,
                    deflated_sharpe_ratio=0.0,
                    dsr_status="OVERFITTING_WARNING",
                    is_overfitted=True,
                    num_trials_n=n_trials,
                    kill_switch_triggers=kill_switch_triggers,
                    trades=trades,
                )

            if mtm_equity > peak_mtm_equity:
                peak_mtm_equity = mtm_equity

            cur_dd_usd = peak_mtm_equity - mtm_equity
            cur_dd_pct = (cur_dd_usd / peak_mtm_equity) * 100.0 if peak_mtm_equity > 0 else 0.0
            if cur_dd_pct > max_drawdown_pct:
                max_drawdown_pct = cur_dd_pct
            if cur_dd_usd > max_drawdown_usd:
                max_drawdown_usd = cur_dd_usd

            # -----------------------------------------------------------------
            # 4. Gestión de Posición Abierta: TP, Stop Loss y Prelación Conservadora
            # -----------------------------------------------------------------
            if active_inventory_qty > 0.0:
                bars_held = i - entry_bar_idx
                is_sl_hit = low_p <= active_sl_price
                is_tp_hit = high_p >= active_tp_price
                is_time_stop = (bars_held >= self.max_holding_bars) and (close_p < avg_entry_price)

                # Regla Conservadora 4: Si ambos se tocan en la misma vela, GANA EL STOP LOSS
                if is_sl_hit and is_tp_hit:
                    chosen_exit = "STOP_LOSS"
                elif is_sl_hit:
                    chosen_exit = "STOP_LOSS"
                elif is_tp_hit:
                    chosen_exit = "TAKE_PROFIT"
                elif is_time_stop:
                    chosen_exit = "TIME_STOP"
                else:
                    chosen_exit = None

                if chosen_exit is not None:
                    if chosen_exit == "STOP_LOSS":
                        # Taker fill con slippage adverso. Si abre con gap bajista por debajo del stop, fill en open_p
                        actual_sl_px = min(active_sl_price, open_p)
                        exit_px = actual_sl_px * (1.0 - self.slippage)
                        exit_fee = (exit_px * active_inventory_qty) * self.taker_fee
                    elif chosen_exit == "TAKE_PROFIT":
                        # Maker fill sin slippage
                        exit_px = active_tp_price
                        exit_fee = (exit_px * active_inventory_qty) * self.maker_fee
                    else:  # TIME_STOP
                        exit_px = close_p * (1.0 - self.slippage)
                        exit_fee = (exit_px * active_inventory_qty) * self.taker_fee

                    gross_pnl = (exit_px - avg_entry_price) * active_inventory_qty
                    entry_fee = (avg_entry_price * active_inventory_qty) * self.maker_fee
                    total_fee = entry_fee + exit_fee
                    net_pnl = gross_pnl - total_fee

                    portfolio.current_equity_usd += net_pnl
                    if (portfolio.current_equity_usd + swept_spot_vault_usd) <= 0.0:
                        return BacktestRunResult(
                            status="RUIN",
                            symbol=self.symbol,
                            strategy_mode=strategy_mode,
                            bars_count=i + 1,
                            start_date=datetime.fromtimestamp(candles[0][0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d"),
                            end_date=date_str,
                            initial_capital_usd=self.initial_capital_usd,
                            final_equity_usd=0.0,
                            spot_vault_swept_usd=swept_spot_vault_usd,
                            total_wealth_usd=0.0,
                            compounded_roi_pct=-100.0,
                            total_trades=len(trades) + 1,
                            winning_trades=sum(1 for t in trades if t.net_pnl_usd > 0),
                            losing_trades=sum(1 for t in trades if t.net_pnl_usd <= 0) + 1,
                            win_rate_pct=round((sum(1 for t in trades if t.net_pnl_usd > 0) / max(len(trades) + 1, 1)) * 100.0, 2),
                            profit_factor=0.0,
                            expectancy_per_trade_usd=-self.initial_capital_usd / max(len(trades) + 1, 1),
                            max_drawdown_pct=100.0,
                            max_drawdown_usd=peak_mtm_equity,
                            daily_returns=[-1.0],
                            observed_daily_sharpe=0.0,
                            annualized_sharpe=0.0,
                            deflated_sharpe_ratio=0.0,
                            dsr_status="OVERFITTING_WARNING",
                            is_overfitted=True,
                            num_trials_n=n_trials,
                            kill_switch_triggers=kill_switch_triggers,
                            trades=trades,
                        )

                    if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                        portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

                    ret_pct = round((net_pnl / max(avg_entry_price * active_inventory_qty, 1e-8)) * 100.0, 3)
                    trades.append(
                        BacktestTrade(
                            entry_bar_idx=entry_bar_idx,
                            exit_bar_idx=i,
                            entry_ts_ms=entry_ts_ms,
                            exit_ts_ms=ts_ms,
                            symbol=self.symbol,
                            side="BUY",
                            strategy=current_strategy,
                            entry_price=round(avg_entry_price, 2),
                            exit_price=round(exit_px, 2),
                            qty=round(active_inventory_qty, 4),
                            gross_pnl_usd=round(gross_pnl, 2),
                            fee_cost_usd=round(total_fee, 4),
                            funding_cost_usd=0.0,
                            net_pnl_usd=round(net_pnl, 2),
                            return_pct=ret_pct,
                            exit_reason=chosen_exit,
                            bars_held=bars_held,
                        )
                    )

                    active_inventory_qty = 0.0
                    avg_entry_price = 0.0
                    portfolio.active_positions.clear()
                    pending_order = None

            # -----------------------------------------------------------------
            # 5. Ejecución de Orden Pendiente (Generada con datos de i-1)
            # -----------------------------------------------------------------
            if active_inventory_qty == 0.0 and pending_order is not None:
                p_order = pending_order
                p_px = p_order["price"]
                p_qty = p_order["qty"]
                p_strat = p_order["strategy"]

                # Verificación Maker: penetración estricta de precio
                is_filled = (low_p < p_px) if self.require_tick_penetration else (low_p <= p_px)
                if is_filled:
                    # Validar Gatekeeper antes de admitir
                    req_bid = OrderRequest(
                        symbol=self.symbol,
                        side="BUY",
                        order_type="LIMIT",
                        price=p_px,
                        qty=p_qty,
                        time_in_force="GTX",
                        margin_mode="ISOLATED",
                        strategy_id=p_strat,
                        timestamp_ns=now_ns,
                    )
                    is_valid, _ = gatekeeper.validate_order(req_bid, portfolio)
                    if is_valid:
                        active_inventory_qty = p_qty
                        avg_entry_price = p_px
                        entry_bar_idx = i
                        entry_ts_ms = ts_ms
                        current_strategy = p_strat
                        portfolio.active_positions[self.symbol] = p_qty * p_px

                        # Fijar TP y SL inmediatamente según estrategia
                        if p_order.get("tp_price") and p_order.get("sl_price"):
                            active_tp_price = p_order["tp_price"]
                            active_sl_price = p_order["sl_price"]
                        else:
                            spread_usd = p_order["spread_usd"]
                            active_tp_price = p_px + (spread_usd * self.take_profit_mult)
                            active_sl_price = p_px - (spread_usd * self.stop_loss_mult)
                    else:
                        if gatekeeper.kill_switch_engaged:
                            kill_switch_triggers += 1

                pending_order = None

            # -----------------------------------------------------------------
            # 6. Generación de Señal para la Siguiente Barra (Estrictamente con datos hasta i)
            #    El precio resultante se enviará para evaluación en la barra i+1.
            # -----------------------------------------------------------------
            if active_inventory_qty == 0.0:
                current_cap = portfolio.current_equity_usd if self.compound_growth else self.initial_capital_usd
                spread_usd = close_p * (self.target_spread_bps / 10_000.0)

                # Cálculo de Sizing
                if self.fixed_risk_pct is not None and self.fixed_risk_pct > 0.0:
                    # Fase 5: Sizing por Riesgo Fijo (ej. 0.5% del capital)
                    # Qty = (Equity * risk_pct) / Distancia_al_Stop
                    sl_dist = spread_usd * self.stop_loss_mult
                    risk_budget = current_cap * self.fixed_risk_pct
                    calc_qty = risk_budget / max(sl_dist, 1e-4)
                    order_qty = round(math.floor(calc_qty / self.step_size) * self.step_size, 4)
                else:
                    max_notional_gate3 = current_cap * self.max_single_pair_margin_ratio
                    order_qty = round((max_notional_gate3 * 0.90) / close_p, 4)

                order_qty = max(self.step_size, order_qty)

                # Validar mínimo nocional institucional (ej. 100 USDT para BTC)
                if (order_qty * close_p) >= self.min_notional_usd:
                    if strategy_mode == "MARKET_MAKING":
                        # Detección Motor B (mecha de liquidación) al cierre de la vela actual i
                        curr_lower_wick = min(open_p, close_p) - low_p
                        is_liq_cascade = (curr_lower_wick > (spread_usd * 1.8)) and (volume > (vol_ma * 1.5))

                        if is_liq_cascade:
                            # Motor B: la orden se sitúa al límite de spread de la apertura siguiente
                            bid_px = round(close_p - spread_usd * 0.5, 1)
                            tp_px = bid_px + (0.40 * curr_lower_wick)
                            sl_px = bid_px - (spread_usd * self.stop_loss_mult)
                            pending_order = {
                                "price": bid_px,
                                "qty": order_qty,
                                "strategy": "LIQ_CASCADE",
                                "spread_usd": spread_usd,
                                "tp_price": tp_px,
                                "sl_price": sl_px,
                            }
                        elif ema_obi > -0.60:
                            # Motor A: Avellaneda-Stoikov + OBI Shield
                            bid_px = round(close_p - spread_usd / 2.0, 1)
                            pending_order = {
                                "price": bid_px,
                                "qty": order_qty,
                                "strategy": "AVELLANEDA_OBI",
                                "spread_usd": spread_usd,
                                "tp_price": bid_px + spread_usd,
                                "sl_price": bid_px - (spread_usd * self.stop_loss_mult),
                            }

                    elif strategy_mode == "DIRECTIONAL_TREND":
                        # Fase 5: Estrategia Direccional 1:3 R:R (Ruptura con filtro de régimen OBI)
                        # Compra en ruptura alcista sobre high de las últimas 10 barras
                        recent_high = max(c[2] for c in candles[max(0, i - 10):i + 1])
                        recent_low = min(c[3] for c in candles[max(0, i - 10):i + 1])
                        breakout_range = recent_high - recent_low

                        if close_p >= (recent_high - self.tick_size) and ema_obi > 0.20:
                            entry_target = close_p
                            sl_dist = max(breakout_range * 0.5, spread_usd * 1.5)
                            rr_mult = self.risk_reward_ratio if self.risk_reward_ratio else 3.0
                            tp_dist = sl_dist * rr_mult
                            pending_order = {
                                "price": entry_target,
                                "qty": order_qty,
                                "strategy": "DIRECTIONAL_TREND",
                                "spread_usd": spread_usd,
                                "tp_price": entry_target + tp_dist,
                                "sl_price": entry_target - sl_dist,
                            }

        # ---------------------------------------------------------------------
        # 7. Métricas Finales y Auditoría DSR
        # ---------------------------------------------------------------------
        total_wealth = portfolio.current_equity_usd + swept_spot_vault_usd
        compounded_roi = round(((total_wealth - self.initial_capital_usd) / self.initial_capital_usd) * 100.0, 2)

        winning_trades = sum(1 for t in trades if t.net_pnl_usd > 0)
        losing_trades = sum(1 for t in trades if t.net_pnl_usd <= 0)
        total_trades = len(trades)
        win_rate = round((winning_trades / max(total_trades, 1)) * 100.0, 2)

        gross_profit = sum(t.net_pnl_usd for t in trades if t.net_pnl_usd > 0)
        gross_loss = sum(abs(t.net_pnl_usd) for t in trades if t.net_pnl_usd < 0)
        profit_factor = round(gross_profit / max(gross_loss, 1e-4), 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        expectancy = round(sum(t.net_pnl_usd for t in trades) / max(total_trades, 1), 2)

        # Retornos diarios para Sharpe y DSR
        daily_equity_snapshots.append((candles[-1][0], total_wealth))
        daily_returns = DeflatedSharpeAuditor.aggregate_daily_returns(daily_equity_snapshots)

        if len(daily_returns) >= 3:
            dsr_audit = DeflatedSharpeAuditor.audit_returns(
                daily_returns,
                num_trials_n=n_trials,
                annualization_factor=365.0,
            )
            obs_sharpe = round(dsr_audit.observed_sharpe_sr, 4)
            ann_sharpe = round(dsr_audit.annualized_sharpe, 4)
            dsr_score = round(dsr_audit.deflated_sharpe_ratio, 4)
            dsr_status = dsr_audit.status
            is_overfit = dsr_audit.is_overfitted
        else:
            obs_sharpe = 0.0
            ann_sharpe = 0.0
            dsr_score = 0.0
            dsr_status = "INSUFFICIENT_DATA"
            is_overfit = True

        return BacktestRunResult(
            status="COMPLETED",
            symbol=self.symbol,
            strategy_mode=strategy_mode,
            bars_count=len(candles),
            start_date=datetime.fromtimestamp(candles[0][0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d"),
            end_date=datetime.fromtimestamp(candles[-1][0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d"),
            initial_capital_usd=self.initial_capital_usd,
            final_equity_usd=round(portfolio.current_equity_usd, 2),
            spot_vault_swept_usd=round(swept_spot_vault_usd, 2),
            total_wealth_usd=round(total_wealth, 2),
            compounded_roi_pct=compounded_roi,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            expectancy_per_trade_usd=expectancy,
            max_drawdown_pct=round(max_drawdown_pct, 2),
            max_drawdown_usd=round(max_drawdown_usd, 2),
            daily_returns=daily_returns,
            observed_daily_sharpe=obs_sharpe,
            annualized_sharpe=ann_sharpe,
            deflated_sharpe_ratio=dsr_score,
            dsr_status=dsr_status,
            is_overfitted=is_overfit,
            num_trials_n=n_trials,
            kill_switch_triggers=kill_switch_triggers,
            trades=trades,
        )
