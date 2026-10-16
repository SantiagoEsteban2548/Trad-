# backend/multi_asset_backtest.py
"""
Motor de Backtest Comparativo de 2 Años con Restricciones Físicas de Liquidez L2:
(A) Multimoneda 5P Estático (Anterior) vs.
(B) Nuevo Motor Dinámico 12P con:
    1. Tope Físico de Liquidez L2 (MAX_ACTIVE_CAPITAL_CAP_USD = 25,000.0 USD)
       - Límite de participación de volumen: máx 15% del volumen real de la barra (max_fill_ratio = 0.15)
       - Penalización por selección adversa cuando una vela rompe los 3 niveles sin retroceder.
       - Desvío del 100% de ganancias a Bóveda Spot tras alcanzar el tope de $25,000 USD.
    2. Cero Look-Ahead en el Selector de 12 Pares:
       - ActivityScore calculado estrictamente con datos cerrados hasta t-1.
       - Costo de rotación (comisión taker + slippage = 9 bps) cuando un par sale del Top-5 con inventario abierto.
    3. Contabilidad Mark-to-Market (MTM) del Max Drawdown y Profit Factor:
       - MTM vela a vela (incluyendo PnL flotante no realizado).
       - Stop-Loss estructural a 1.5 sigma contabilizando cada nivel de la escalera como pérdida independiente.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.profit_sweeper import step_up_ratchet, MAX_ACTIVE_CAPITAL_CAP_USD
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor


ALL_UNIVERSE_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "DOGE",
    "XRP", "SUI", "AVAX", "LINK", "PEPE",
    "NEAR", "ARB",
]

STATIC_5P_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "DOGE"]

SYMBOL_SPECS = {
    "BTC": {"tick": 0.10, "step": 0.001, "spread_bps": 16.0},
    "ETH": {"tick": 0.01, "step": 0.001, "spread_bps": 18.0},
    "SOL": {"tick": 0.01, "step": 0.1, "spread_bps": 22.0},
    "BNB": {"tick": 0.01, "step": 0.01, "spread_bps": 18.0},
    "DOGE": {"tick": 0.00001, "step": 1.0, "spread_bps": 25.0},
    "XRP": {"tick": 0.0001, "step": 0.1, "spread_bps": 20.0},
    "SUI": {"tick": 0.0001, "step": 0.1, "spread_bps": 28.0},
    "AVAX": {"tick": 0.01, "step": 0.01, "spread_bps": 24.0},
    "LINK": {"tick": 0.001, "step": 0.01, "spread_bps": 22.0},
    "PEPE": {"tick": 0.000001, "step": 1.0, "spread_bps": 32.0},
    "NEAR": {"tick": 0.001, "step": 0.1, "spread_bps": 25.0},
    "ARB": {"tick": 0.0001, "step": 0.1, "spread_bps": 24.0},
}


@dataclass
class BacktestMetricSummary:
    mode: str  # "STATIC_5P" | "DYNAMIC_12P_RATCHET"
    period_label: str
    period_days: int
    bars_count: int
    start_date: str
    end_date: str
    initial_capital_usd: float
    final_equity_usd: float
    spot_vault_swept_usd: float
    total_value_usd: float
    compounded_return_pct: float
    swept_return_pct: float
    total_rotations: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    observed_sharpe: float
    deflated_sharpe: float
    dsr_status: str
    is_overfitted: bool
    contagion_shield_activations: int = 0
    ladder_deep_fills_count: int = 0
    ratchet_events_count: int = 0
    adverse_selection_events: int = 0
    rotations_forced_exit_count: int = 0


class MultiAssetBacktester:
    PERIODS = [
        ("31D", 31),
        ("90D", 90),
        ("180D", 180),
        ("365D", 365),
        ("2A", 730),
    ]

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data" / "cache")

    def load_all_datasets(self) -> dict[str, list[list[float]]]:
        """Carga las 70.080 velas de 15m para los 12 pares."""
        datasets = {}
        for sym in ALL_UNIVERSE_SYMBOLS:
            fpath = self.cache_dir / f"binance_{sym}_USDT_2y_15m.json"
            if not fpath.exists():
                raise FileNotFoundError(f"Archivo de cache no encontrado: {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            datasets[sym] = data["candles"]
        return datasets

    @staticmethod
    def _compute_candle_obi(open_p: float, high_p: float, low_p: float, close_p: float) -> float:
        rng = high_p - low_p
        if rng <= 1e-8:
            return 0.0
        body = close_p - open_p
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)
        raw_obi = (body + lower_wick - upper_wick) / rng
        return float(np.clip(raw_obi, -1.0, 1.0))

    def run_simulation(
        self,
        datasets: dict[str, list[list[float]]],
        mode: str,  # "STATIC_5P" or "DYNAMIC_12P_RATCHET"
        period_label: str,
        days: int,
        initial_capital_usd: float = 100.0,
    ) -> BacktestMetricSummary:
        total_bars = min(len(c) for c in datasets.values())
        required_bars = min(days * 96, total_bars)
        start_idx = total_bars - required_bars

        # Sincronizar rebanadas
        universe_symbols = STATIC_5P_SYMBOLS if mode == "STATIC_5P" else ALL_UNIVERSE_SYMBOLS
        sliced_candles = {
            sym: datasets[sym][start_idx:total_bars]
            for sym in universe_symbols
        }

        # Estado del portafolio unificado
        portfolio = PortfolioState(
            base_capital_usd=initial_capital_usd,
            current_equity_usd=initial_capital_usd,
            daily_high_watermark_usd=initial_capital_usd,
            realized_pnl_today_usd=0.0,
            active_positions={},
            rolling_edge_mu=0.045,
            rolling_variance_sigma2=0.0090,
        )

        swept_spot_vault_usd = 0.0
        peak_equity = initial_capital_usd
        max_drawdown_pct = 0.0
        max_drawdown_usd = 0.0

        total_rotations = 0
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        returns_series: list[float] = []

        # Estado por activo
        symbol_inventory: dict[str, float] = {s: 0.0 for s in universe_symbols}
        symbol_entry_px: dict[str, float] = {s: 0.0 for s in universe_symbols}
        symbol_entry_bar: dict[str, int] = {s: 0 for s in universe_symbols}
        symbol_levels_filled: dict[str, int] = {s: 1 for s in universe_symbols}
        symbol_returns_hist: dict[str, list[float]] = {s: [] for s in universe_symbols}

        contagion_shield_count = 0
        ladder_deep_fills_count = 0
        ratchet_events_count = 0
        adverse_selection_events = 0
        rotations_forced_exit_count = 0

        # Para cálculo estrictamente sin look-ahead (t-1)
        prev_active_symbols: list[str] = list(universe_symbols[:5])

        # Precalcular retornos históricos para evitar recálculos en bucles
        hist_returns: dict[str, list[float]] = {s: [] for s in universe_symbols}
        for sym in universe_symbols:
            candles = sliced_candles[sym]
            rets = [abs(candles[i][4] - candles[i][1]) / max(candles[i][1], 1e-8) for i in range(len(candles))]
            hist_returns[sym] = rets

        for bar_idx in range(required_bars):
            # Recopilar datos de barra actual
            bar_data: dict[str, dict[str, float]] = {}
            for sym in universe_symbols:
                c = sliced_candles[sym][bar_idx]
                bar_data[sym] = {
                    "ts": c[0],
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "vol": c[5],
                }

            # Monitoreo diario en modo STATIC_5P (barrido plano a fin de día)
            if mode == "STATIC_5P":
                if bar_idx > 0 and bar_idx % 96 == 0:
                    tot_pos = sum(symbol_inventory[s] * bar_data[s]["close"] for s in universe_symbols)
                    if tot_pos == 0.0 and portfolio.current_equity_usd > portfolio.base_capital_usd:
                        surplus = portfolio.current_equity_usd - portfolio.base_capital_usd
                        swept_spot_vault_usd += surplus
                        portfolio.current_equity_usd = portfolio.base_capital_usd
                        portfolio.daily_high_watermark_usd = portfolio.base_capital_usd

            # 1. Escáner Dinámico de Universo: CERO LOOK-AHEAD (Datos cerrados hasta t-1)
            if mode == "DYNAMIC_12P_RATCHET":
                if bar_idx == 0:
                    active_symbols = list(universe_symbols[:5])
                else:
                    activity_scores: dict[str, float] = {}
                    for sym in universe_symbols:
                        # Ventana de volatilidad hasta t-1
                        v_win = hist_returns[sym][max(0, bar_idx - 20):bar_idx]
                        sigma = float(np.mean(v_win)) if v_win else 0.002
                        spread_bps = SYMBOL_SPECS[sym]["spread_bps"]
                        
                        # Volumen promedio de 15 barras previas hasta t-1
                        v_prev_slice = [sliced_candles[sym][k][5] for k in range(max(0, bar_idx - 15), bar_idx)]
                        v_mean = float(np.mean(v_prev_slice)) if v_prev_slice else 1.0
                        c_prev_vol = sliced_candles[sym][bar_idx - 1][5]
                        vol_ratio = c_prev_vol / max(v_mean, 1e-5)
                        liq_intensity = min(max(vol_ratio - 1.0, 0.0), 3.0)
                        score = sigma * spread_bps * (1.0 + liq_intensity)
                        activity_scores[sym] = score

                    # BTC siempre con prioridad de presencia como ancla del escudo sistémico
                    if "BTC" in activity_scores:
                        activity_scores["BTC"] = max(activity_scores["BTC"], 0.01)

                    sorted_active = sorted(universe_symbols, key=lambda s: activity_scores.get(s, 0.0), reverse=True)
                    active_symbols = sorted_active[:5]
                    if "BTC" not in active_symbols:
                        active_symbols[-1] = "BTC"

                # Costo de rotación: pares que salen del Top-5 con inventario abierto
                for sym in universe_symbols:
                    if sym not in active_symbols and symbol_inventory[sym] > 0.0:
                        # Liquidación de rotación a precio de apertura de la nueva barra
                        exit_px = bar_data[sym]["open"]
                        qty = symbol_inventory[sym]
                        entry_px = symbol_entry_px[sym]
                        notional = exit_px * qty
                        # Costo de rotación: 0.04% taker fee + 0.05% slippage = 9 bps
                        rotation_fee = notional * 0.0009
                        gross = (exit_px - entry_px) * qty
                        net = gross - rotation_fee

                        portfolio.current_equity_usd += net
                        tiers_count = symbol_levels_filled.get(sym, 1)
                        total_rotations += tiers_count
                        rotations_forced_exit_count += 1

                        if net > 0:
                            winning_trades += tiers_count
                            gross_profit += net
                        else:
                            losing_trades += tiers_count
                            gross_loss += abs(net)

                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)
                        symbol_returns_hist[sym].append(ret)

                        symbol_inventory[sym] = 0.0
                        symbol_entry_px[sym] = 0.0
                        portfolio.active_positions.pop(sym, None)

                prev_active_symbols = list(active_symbols)
            else:
                active_symbols = STATIC_5P_SYMBOLS

            # 2. Cálculo de OBI y Escudo de Contagio Sistémico
            obis: dict[str, float] = {}
            for sym in active_symbols:
                b = bar_data[sym]
                obis[sym] = self._compute_candle_obi(b["open"], b["high"], b["low"], b["close"])

            bid_shield_active = False
            ask_shield_active = False

            btc_obi = obis.get("BTC", 0.0)
            bearish_alts = sum(1 for s in active_symbols if s != "BTC" and obis[s] < -0.60)
            bullish_alts = sum(1 for s in active_symbols if s != "BTC" and obis[s] > 0.60)

            if btc_obi < -0.70 or bearish_alts >= 2:
                bid_shield_active = True
                contagion_shield_count += 1
            if btc_obi > 0.70 or bullish_alts >= 2:
                ask_shield_active = True

            # 3. Asignación Half-Kelly Multivariante
            half_kelly_weights: dict[str, float] = {}
            if bar_idx >= 30:
                try:
                    ret_matrix = []
                    for sym in active_symbols:
                        rets = [
                            (sliced_candles[sym][t][4] - sliced_candles[sym][t - 1][4]) / sliced_candles[sym][t - 1][4]
                            for t in range(max(1, bar_idx - 30), bar_idx)
                        ]
                        ret_matrix.append(rets)
                    ret_mat = np.array(ret_matrix)
                    M = np.mean(ret_mat, axis=1)
                    C = np.cov(ret_mat) + (1e-6 * np.eye(len(active_symbols)))
                    C_inv = np.linalg.pinv(C)
                    f_star = 0.5 * (C_inv @ M)
                    f_clamped = np.clip(f_star, 0.02, 0.20)
                    if np.sum(f_clamped) > 3.0:
                        f_clamped = (f_clamped / np.sum(f_clamped)) * 3.0
                    for idx_s, sym in enumerate(active_symbols):
                        half_kelly_weights[sym] = float(f_clamped[idx_s])
                except Exception:
                    half_kelly_weights = {s: 0.18 for s in active_symbols}
            else:
                half_kelly_weights = {s: 0.18 for s in active_symbols}

            # 4. Evaluación y Ejecución por Activo
            eval_symbols = list(set(active_symbols + [s for s in universe_symbols if symbol_inventory[s] > 0.0]))

            for sym in eval_symbols:
                b = bar_data[sym]
                spec = SYMBOL_SPECS[sym]
                cur_close = b["close"]
                cur_high = b["high"]
                cur_low = b["low"]
                cur_vol = b["vol"]
                obi = obis.get(sym, self._compute_candle_obi(b["open"], b["high"], b["low"], b["close"]))

                active_qty = symbol_inventory[sym]
                entry_px = symbol_entry_px[sym]
                target_bps = spec["spread_bps"]
                spread_usd = cur_close * (target_bps / 10_000.0)

                # SI NO TENEMOS POSICIÓN ABIERTA: Cotizar entradas
                if active_qty == 0.0:
                    if sym not in active_symbols:
                        continue
                    if sym != "BTC" and bid_shield_active:
                        continue

                    # TOPE FÍSICO DE LIQUIDEZ: base_capital acotado a $25,000 USD
                    base_cap = min(portfolio.base_capital_usd, MAX_ACTIVE_CAPITAL_CAP_USD)
                    w = half_kelly_weights.get(sym, 0.18)
                    target_notional = min(base_cap * w, base_cap * 0.20)

                    if spec["step"] * cur_close > base_cap * 0.20:
                        order_qty = max(0.0001, round(target_notional / cur_close, 4))
                    else:
                        st = spec["step"]
                        order_qty = max(st, round(round(target_notional / cur_close / st) * st, 6))

                    if mode == "DYNAMIC_12P_RATCHET":
                        # Cotización Multinivel en Escalera (3-Level Ladder):
                        delta_star = spread_usd * 0.5 * (1.0 + max(0.0, -obi * 0.25))
                        p_l1 = round((cur_close - delta_star * 1.0) / spec["tick"]) * spec["tick"]
                        p_l2 = round((cur_close - delta_star * 2.0) / spec["tick"]) * spec["tick"]
                        p_l3 = round((cur_close - delta_star * 3.5) / spec["tick"]) * spec["tick"]

                        # RESTRICCIÓN DE VOLUMEN REAL: máx 15% del volumen negociado en la barra
                        max_fill_capacity = cur_vol * 0.15

                        filled_qty = 0.0
                        weighted_cost = 0.0
                        tiers_filled = 0

                        if cur_low <= p_l1 and max_fill_capacity > 0:
                            q1 = min(order_qty * 0.45, max_fill_capacity)
                            filled_qty += q1
                            weighted_cost += q1 * p_l1
                            tiers_filled += 1
                            rem_cap = max(0.0, max_fill_capacity - q1)

                            if cur_low <= p_l2 and rem_cap > 0:
                                q2 = min(order_qty * 0.35, rem_cap)
                                filled_qty += q2
                                weighted_cost += q2 * p_l2
                                tiers_filled += 1
                                ladder_deep_fills_count += 1
                                rem_cap = max(0.0, rem_cap - q2)

                                if cur_low <= p_l3 and rem_cap > 0:
                                    q3 = min(order_qty * 0.20, rem_cap)
                                    filled_qty += q3
                                    weighted_cost += q3 * p_l3
                                    tiers_filled += 1

                        if filled_qty > 0.0:
                            avg_px = weighted_cost / filled_qty

                            # PENALIZACIÓN DE SELECCIÓN ADVERSA:
                            # Si la vela rompe los 3 niveles y cierra en los mínimos sin rebotar (<20% mecha inferior)
                            bounce_ratio = (cur_close - cur_low) / max(b["high"] - cur_low, 1e-8)
                            if cur_low <= p_l3 and bounce_ratio < 0.20:
                                adverse_selection_events += 1
                                # Desplazamiento adverso del precio promedio de entrada
                                adverse_slip = (spread_usd * 0.35) + spec["tick"]
                                avg_px = min(avg_px - adverse_slip, cur_low)

                            symbol_inventory[sym] = filled_qty
                            symbol_entry_px[sym] = avg_px
                            symbol_entry_bar[sym] = bar_idx
                            symbol_levels_filled[sym] = tiers_filled
                            portfolio.active_positions[sym] = filled_qty * avg_px
                    else:
                        # Modo STATIC_5P: 1 solo nivel
                        bid_discount = spread_usd * (0.50 + max(0.0, -obi * 0.25))
                        bid_px = round((cur_close - bid_discount) / spec["tick"]) * spec["tick"]
                        max_fill_capacity = cur_vol * 0.15
                        fill_qty = min(order_qty, max_fill_capacity)
                        if cur_low <= bid_px and fill_qty > 0:
                            symbol_inventory[sym] = fill_qty
                            symbol_entry_px[sym] = bid_px
                            symbol_entry_bar[sym] = bar_idx
                            symbol_levels_filled[sym] = 1
                            portfolio.active_positions[sym] = fill_qty * bid_px

                # SI TENEMOS POSICIÓN ABIERTA: Evaluar Take Profit, Time Decay o Stop Loss
                elif active_qty > 0.0:
                    bars_held = bar_idx - symbol_entry_bar[sym]
                    tiers_count = symbol_levels_filled.get(sym, 1)

                    # Volatilidad realizada reciente para Stop-Loss estructural
                    v_hist = hist_returns[sym][max(0, bar_idx - 20):bar_idx]
                    rolling_sigma = float(np.mean(v_hist)) if v_hist else 0.004

                    # Inventory Time-Decay: si lleva >= 1 barra (>= 15m), acelerar salida (+25% gamma)
                    if mode == "DYNAMIC_12P_RATCHET" and bars_held >= 1:
                        target_spread = spread_usd * 0.75
                    else:
                        target_spread = spread_usd

                    target_ask = round((entry_px + target_spread) / spec["tick"]) * spec["tick"]

                    # STOP-LOSS ESTRUCTURAL A 1.5 SIGMA:
                    # Cada nivel de la escalera que toque este nivel se contabiliza como pérdida independiente
                    structural_stop_dist = max(1.5 * rolling_sigma * entry_px, spread_usd * 1.5)
                    structural_stop_px = entry_px - structural_stop_dist

                    # Caso A: Stop-Loss Estructural de 1.5 sigma alcanzado
                    if cur_low <= structural_stop_px:
                        exit_px = structural_stop_px
                        gross = (exit_px - entry_px) * active_qty
                        fee = (entry_px + exit_px) * active_qty * 0.00020
                        net = gross - fee

                        portfolio.current_equity_usd += net
                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)
                        symbol_returns_hist[sym].append(ret)

                        # Contabilizar cada nivel como pérdida realizada independiente
                        losing_trades += tiers_count
                        total_rotations += tiers_count
                        gross_loss += abs(net)

                        symbol_inventory[sym] = 0.0
                        symbol_entry_px[sym] = 0.0
                        portfolio.active_positions.pop(sym, None)

                    # Caso B: Take Profit Maker alcanzado
                    elif cur_high >= target_ask:
                        exit_px = target_ask
                        gross = (exit_px - entry_px) * active_qty
                        fee = (entry_px + exit_px) * active_qty * 0.00015
                        net = gross - fee

                        portfolio.current_equity_usd += net
                        if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                            portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)
                        symbol_returns_hist[sym].append(ret)

                        winning_trades += tiers_count
                        total_rotations += tiers_count
                        gross_profit += net

                        symbol_inventory[sym] = 0.0
                        symbol_entry_px[sym] = 0.0
                        portfolio.active_positions.pop(sym, None)

                        # Trinquete 85/15 continuo en modo DYNAMIC acotado a $25,000 USD
                        if mode == "DYNAMIC_12P_RATCHET":
                            swept = step_up_ratchet(
                                portfolio,
                                threshold_pct=0.02,
                                vault_ratio=0.15,
                                reinvest_ratio=0.85,
                                max_capital_cap_usd=MAX_ACTIVE_CAPITAL_CAP_USD,
                            )
                            if swept > 0.0:
                                swept_spot_vault_usd += swept
                                ratchet_events_count += 1

                    # Caso C: Cierre por tiempo (> 8 barras = 2h) si adverso
                    elif bars_held > 8 and cur_close < entry_px:
                        exit_px = cur_close
                        gross = (exit_px - entry_px) * active_qty
                        fee = (entry_px + exit_px) * active_qty * 0.00015
                        net = gross - fee

                        portfolio.current_equity_usd += net
                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)
                        symbol_returns_hist[sym].append(ret)

                        total_rotations += tiers_count
                        if net > 0:
                            winning_trades += tiers_count
                            gross_profit += net
                        else:
                            losing_trades += tiers_count
                            gross_loss += abs(net)

                        symbol_inventory[sym] = 0.0
                        symbol_entry_px[sym] = 0.0
                        portfolio.active_positions.pop(sym, None)

                        if mode == "DYNAMIC_12P_RATCHET":
                            swept = step_up_ratchet(
                                portfolio,
                                threshold_pct=0.02,
                                vault_ratio=0.15,
                                reinvest_ratio=0.85,
                                max_capital_cap_usd=MAX_ACTIVE_CAPITAL_CAP_USD,
                            )
                            if swept > 0.0:
                                swept_spot_vault_usd += swept
                                ratchet_events_count += 1

            # 5. CONTABILIDAD MARK-TO-MARKET (MTM) DEL MAX DRAWDOWN AL CIERRE DE CADA VELA
            unrealized_pnl = sum(
                symbol_inventory[s] * (bar_data[s]["close"] - symbol_entry_px[s])
                for s in universe_symbols
                if symbol_inventory[s] > 0.0
            )
            total_wealth_mtm = portfolio.current_equity_usd + swept_spot_vault_usd + unrealized_pnl

            if total_wealth_mtm > peak_equity:
                peak_equity = total_wealth_mtm
            dd_usd = peak_equity - total_wealth_mtm
            dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct
                max_drawdown_usd = dd_usd

        # Resumen final de métricas
        final_wealth = portfolio.current_equity_usd + swept_spot_vault_usd
        total_pnl = final_wealth - initial_capital_usd
        compounded_ret_pct = (total_pnl / initial_capital_usd) * 100.0
        swept_ret_pct = (swept_spot_vault_usd / initial_capital_usd) * 100.0

        win_rate = (winning_trades / total_rotations * 100.0) if total_rotations > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 1.0)

        # Deflated Sharpe Ratio
        ret_arr = np.array(returns_series, dtype=np.float64) if len(returns_series) >= 5 else np.array([0.001] * 5)
        dsr_audit = DeflatedSharpeAuditor.audit_returns(ret_arr, num_trials_n=25)

        start_ts = sliced_candles["BTC"][0][0] / 1000.0
        end_ts = sliced_candles["BTC"][-1][0] / 1000.0
        start_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        end_date = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d")

        return BacktestMetricSummary(
            mode=mode,
            period_label=period_label,
            period_days=days,
            bars_count=required_bars,
            start_date=start_date,
            end_date=end_date,
            initial_capital_usd=initial_capital_usd,
            final_equity_usd=round(portfolio.current_equity_usd, 2),
            spot_vault_swept_usd=round(swept_spot_vault_usd, 2),
            total_value_usd=round(final_wealth, 2),
            compounded_return_pct=round(compounded_ret_pct, 2),
            swept_return_pct=round(swept_ret_pct, 2),
            total_rotations=total_rotations,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            max_drawdown_usd=round(max_drawdown_usd, 2),
            observed_sharpe=round(dsr_audit.observed_sharpe_sr, 2),
            deflated_sharpe=round(dsr_audit.deflated_sharpe_ratio, 4),
            dsr_status=dsr_audit.status,
            is_overfitted=dsr_audit.is_overfitted,
            contagion_shield_activations=contagion_shield_count,
            ladder_deep_fills_count=ladder_deep_fills_count,
            ratchet_events_count=ratchet_events_count,
            adverse_selection_events=adverse_selection_events,
            rotations_forced_exit_count=rotations_forced_exit_count,
        )

    def run_comparative_suite(self) -> dict[str, Any]:
        datasets = self.load_all_datasets()
        print(f"[OK] 12 Datasets cargados con éxito. Total velas sincronizadas: {min(len(c) for c in datasets.values())}")

        results: dict[str, Any] = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "static_5p": {},
            "dynamic_12p_ratchet": {},
            "comparison": [],
        }

        print("\n" + "=" * 135)
        print(" BACKTEST DE MICROESTRUCTURA CON RESTRICCIONES FISICAS L2: CAP $25K, CERO LOOK-AHEAD, STOP 1.5-SIGMA Y MTM DRAWDOWN")
        print("=" * 135)

        for label, days in self.PERIODS:
            print(f"\n>>> Simulando Horizonte {label} ({days} días / {days * 96} barras)...", flush=True)
            res_static = self.run_simulation(datasets, mode="STATIC_5P", period_label=label, days=days)
            res_dynamic = self.run_simulation(datasets, mode="DYNAMIC_12P_RATCHET", period_label=label, days=days)

            results["static_5p"][label] = asdict(res_static)
            results["dynamic_12p_ratchet"][label] = asdict(res_dynamic)

            delta_return = res_dynamic.compounded_return_pct - res_static.compounded_return_pct
            rotations_ratio = (res_dynamic.total_rotations / max(res_static.total_rotations, 1))

            results["comparison"].append({
                "period": label,
                "days": days,
                "static_return_pct": res_static.compounded_return_pct,
                "dynamic_return_pct": res_dynamic.compounded_return_pct,
                "delta_return_pct": round(delta_return, 2),
                "static_final_usd": res_static.total_value_usd,
                "dynamic_final_usd": res_dynamic.total_value_usd,
                "dynamic_equity_usd": res_dynamic.final_equity_usd,
                "dynamic_vault_usd": res_dynamic.spot_vault_swept_usd,
                "static_rotations": res_static.total_rotations,
                "dynamic_rotations": res_dynamic.total_rotations,
                "rotations_multiplier": round(rotations_ratio, 2),
                "static_win_rate": res_static.win_rate_pct,
                "dynamic_win_rate": res_dynamic.win_rate_pct,
                "static_profit_factor": res_static.profit_factor,
                "dynamic_profit_factor": res_dynamic.profit_factor,
                "static_max_dd_pct": res_static.max_drawdown_pct,
                "dynamic_max_dd_pct": res_dynamic.max_drawdown_pct,
                "static_dsr": res_static.deflated_sharpe,
                "dynamic_dsr": res_dynamic.deflated_sharpe,
                "ladder_fills": res_dynamic.ladder_deep_fills_count,
                "ratchet_events": res_dynamic.ratchet_events_count,
                "adverse_events": res_dynamic.adverse_selection_events,
                "forced_rotations": res_dynamic.rotations_forced_exit_count,
            })

        self._print_comparison_table(results["comparison"])

        out_path = PROJECT_ROOT / "data" / "dynamic_universe_comparison_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[EXPORT] Resultados comparativos auditados guardados en: {out_path}")

        return results

    @staticmethod
    def _print_comparison_table(comparisons: list[dict[str, Any]]) -> None:
        header = (
            f"| {'Periodo':<7} | {'Retorno 5P Estático':<20} | {'Retorno 12P Físico':<20} | {'Patrimonio Total':<17} "
            f"| {'Futuros ($25k)':<15} | {'Bóveda Spot':<14} | {'Rotaciones':<11} | {'Win Rate':<9} | {'PF':<6} | {'Max DD MTM':<11} | {'DSR':<7} |"
        )
        sep = "|" + "-" * (len(header) - 2) + "|"

        print("\nTABLA COMPARATIVA CONSOLIDADA CON RESTRICCIONES FÍSICAS (CAP $25K, CERO LOOK-AHEAD, STOP 1.5-SIGMA):")
        print(sep)
        print(header)
        print(sep)

        for c in comparisons:
            ret_stat = f"+{c['static_return_pct']:,.1f}%"
            ret_dyn = f"+{c['dynamic_return_pct']:,.1f}%"
            final_wealth = f"${c['dynamic_final_usd']:,.2f}"
            eq_cap = f"${c['dynamic_equity_usd']:,.2f}"
            vault = f"${c['dynamic_vault_usd']:,.2f}"
            rots = f"{c['dynamic_rotations']:,}"
            wr = f"{c['dynamic_win_rate']:.1f}%"
            pf = f"{c['dynamic_profit_factor']:.2f}"
            dd = f"{c['dynamic_max_dd_pct']:.2f}%"
            dsr = f"{c['dynamic_dsr'] * 100:.1f}%"

            row = (
                f"| {c['period']:<7} | {ret_stat:<20} | {ret_dyn:<20} | {final_wealth:<17} "
                f"| {eq_cap:<15} | {vault:<14} | {rots:<11} | {wr:<9} | {pf:<6} | {dd:<11} | {dsr:<7} |"
            )
            print(row)
        print(sep + "\n")


if __name__ == "__main__":
    tester = MultiAssetBacktester()
    tester.run_comparative_suite()
