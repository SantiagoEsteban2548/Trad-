# backend/backtest_2021_2022.py
"""
Motor de Backtest Paralelo del Ciclo Historico 2021-2022 (730 Dias / 70,080 Barras de 15m)
Simulacion Cuantitativa Multi-Activo bajo Restricciones Fisicas de Liquidez L2 y los 4 Mandate Gates:
- Periodo Completo: 2021-01-01 00:00 UTC a 2022-12-31 23:59 UTC (730D)
- Sub-Periodo 1: 2021 Bull Run (2021-01-01 a 2021-12-31, 365D)
- Sub-Periodo 2: 2022 Bear Market (2022-01-01 a 2022-12-31, 365D)
- Stress-Test Luna/Terra: 2022-04-01 a 2022-06-30 (90D)
- Stress-Test FTX Crash: 2022-10-01 a 2022-12-31 (92D)
"""
from __future__ import annotations

import json
import math
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

from backend.mandate_gates.profit_sweeper import step_up_ratchet, MAX_ACTIVE_CAPITAL_CAP_USD
from backend.mandate_gates.schemas import PortfolioState
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor

CORE_5P_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "DOGE"]

SYMBOL_SPECS = {
    "BTC": {"tick": 0.10, "step": 0.001, "spread_bps": 16.0},
    "ETH": {"tick": 0.01, "step": 0.001, "spread_bps": 18.0},
    "SOL": {"tick": 0.01, "step": 0.1, "spread_bps": 22.0},
    "BNB": {"tick": 0.01, "step": 0.01, "spread_bps": 18.0},
    "DOGE": {"tick": 0.00001, "step": 1.0, "spread_bps": 25.0},
    "XRP": {"tick": 0.0001, "step": 0.1, "spread_bps": 20.0},
    "AVAX": {"tick": 0.01, "step": 0.01, "spread_bps": 24.0},
    "LINK": {"tick": 0.001, "step": 0.01, "spread_bps": 22.0},
    "NEAR": {"tick": 0.001, "step": 0.1, "spread_bps": 25.0},
    "ADA": {"tick": 0.0001, "step": 0.1, "spread_bps": 20.0},
    "DOT": {"tick": 0.001, "step": 0.01, "spread_bps": 22.0},
    "LTC": {"tick": 0.01, "step": 0.01, "spread_bps": 20.0},
}


@dataclass
class PeriodEvaluationResult:
    period_id: str
    period_name: str
    start_date: str
    end_date: str
    bars_count: int
    initial_capital_usd: float
    # Modo A: Static 5P
    static_final_futures_usd: float
    static_swept_vault_usd: float
    static_total_wealth_usd: float
    static_roi_pct: float
    static_max_drawdown_pct: float
    static_win_rate_pct: float
    static_profit_factor: float
    static_total_rotations: int
    static_observed_sharpe: float
    static_dsr_score: float
    static_is_overfitted: bool
    static_kill_switch_triggers: int
    # Modo B: Dynamic Multi-Asset Ratchet
    dynamic_final_futures_usd: float
    dynamic_swept_vault_usd: float
    dynamic_total_wealth_usd: float
    dynamic_roi_pct: float
    dynamic_max_drawdown_pct: float
    dynamic_win_rate_pct: float
    dynamic_profit_factor: float
    dynamic_total_rotations: int
    dynamic_observed_sharpe: float
    dynamic_dsr_score: float
    dynamic_is_overfitted: bool
    dynamic_ratchet_sweeps_count: int
    dynamic_ladder_deep_fills_count: int
    dynamic_adverse_selection_events: int
    dynamic_contagion_shield_activations: int
    dynamic_kill_switch_triggers: int
    # Benchmarks HODL
    btc_hodl_roi_pct: float
    eth_hodl_roi_pct: float
    sol_hodl_roi_pct: float
    bnb_hodl_roi_pct: float
    doge_hodl_roi_pct: float
    equal_weight_hodl_roi_pct: float
    alpha_over_equal_weight_pct: float


class Backtest20212022Runner:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data" / "cache")
        self.datasets: dict[str, list[list[float]]] = {}

    def load_data(self, symbols: list[str]) -> None:
        print(f"[LOADER] Cargando datasets 2021-2022 para {symbols}...")
        for sym in symbols:
            fp = self.cache_dir / f"binance_{sym}_USDT_2021_2022_15m.json"
            if not fp.exists():
                raise FileNotFoundError(f"Falta archivo de cache: {fp}")
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.datasets[sym] = d["candles"]
            print(f"  -> {sym}: {len(d['candles']):,} velas cargadas")

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

    def simulate_slice(
        self,
        sliced_candles: dict[str, list[list[float]]],
        symbols: list[str],
        mode: str,  # "STATIC_5P" | "DYNAMIC_MULTI_RATCHET"
        initial_capital_usd: float = 100.0,
    ) -> dict[str, Any]:
        bars_count = min(len(c) for c in sliced_candles.values())
        portfolio = PortfolioState(
            base_capital_usd=initial_capital_usd,
            current_equity_usd=initial_capital_usd,
            daily_high_watermark_usd=initial_capital_usd,
            realized_pnl_today_usd=0.0,
            active_positions={},
            rolling_edge_mu=0.045,
            rolling_variance_sigma2=0.0090,
        )

        swept_vault_usd = 0.0
        peak_wealth = initial_capital_usd
        max_dd_pct = 0.0
        total_rotations = 0
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        returns_series: list[float] = []

        symbol_inventory: dict[str, float] = {s: 0.0 for s in symbols}
        symbol_entry_px: dict[str, float] = {s: 0.0 for s in symbols}
        symbol_entry_bar: dict[str, int] = {s: 0 for s in symbols}
        symbol_levels_filled: dict[str, int] = {s: 1 for s in symbols}

        contagion_shield_count = 0
        ladder_deep_fills_count = 0
        ratchet_sweeps_count = 0
        adverse_selection_events = 0
        kill_switch_triggers = 0
        kill_switch_active = False

        # Precalcular retornos de barra
        hist_returns: dict[str, list[float]] = {s: [] for s in symbols}
        for s in symbols:
            c = sliced_candles[s]
            rets = [abs(c[i][4] - c[i][1]) / max(c[i][1], 1e-8) for i in range(len(c))]
            hist_returns[s] = rets

        for bar_idx in range(bars_count):
            bar_data: dict[str, dict[str, float]] = {}
            for s in symbols:
                c = sliced_candles[s][bar_idx]
                bar_data[s] = {
                    "ts": c[0],
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "vol": c[5],
                }

            # Reinicio diario UTC cada 96 barras (24 horas)
            if bar_idx > 0 and bar_idx % 96 == 0:
                kill_switch_active = False
                portfolio.daily_high_watermark_usd = max(portfolio.current_equity_usd, 0.01)

                if mode == "STATIC_5P":
                    tot_pos = sum(symbol_inventory[s] * bar_data[s]["close"] for s in symbols)
                    if tot_pos == 0.0 and portfolio.current_equity_usd > portfolio.base_capital_usd:
                        surplus = portfolio.current_equity_usd - portfolio.base_capital_usd
                        swept_vault_usd += surplus
                        portfolio.current_equity_usd = portfolio.base_capital_usd
                        portfolio.daily_high_watermark_usd = portfolio.base_capital_usd

            # Verificacion Gate 4: Kill Switch Diario (-3% HWM)
            if not kill_switch_active:
                hwm = portfolio.daily_high_watermark_usd
                if hwm > 0:
                    dd_daily = (portfolio.current_equity_usd - hwm) / hwm
                    if dd_daily <= -0.030:
                        kill_switch_active = True
                        kill_switch_triggers += 1

            # OBI y Escudo de Contagio
            obis: dict[str, float] = {}
            for s in symbols:
                b = bar_data[s]
                obis[s] = self._compute_candle_obi(b["open"], b["high"], b["low"], b["close"])

            bid_shield_active = False
            btc_obi = obis.get("BTC", 0.0)
            bearish_alts = sum(1 for s in symbols if s != "BTC" and obis[s] < -0.60)
            if btc_obi < -0.70 or bearish_alts >= 2:
                bid_shield_active = True
                contagion_shield_count += 1

            # Asignacion Half-Kelly
            half_kelly_weights: dict[str, float] = {}
            if bar_idx >= 30:
                try:
                    ret_matrix = []
                    for s in symbols:
                        rets = [
                            (sliced_candles[s][t][4] - sliced_candles[s][t - 1][4]) / sliced_candles[s][t - 1][4]
                            for t in range(max(1, bar_idx - 30), bar_idx)
                        ]
                        ret_matrix.append(rets)
                    ret_mat = np.array(ret_matrix)
                    M = np.mean(ret_mat, axis=1)
                    C = np.cov(ret_mat) + (1e-6 * np.eye(len(symbols)))
                    C_inv = np.linalg.pinv(C)
                    f_star = 0.5 * (C_inv @ M)
                    f_clamped = np.clip(f_star, 0.02, 0.20)
                    if np.sum(f_clamped) > 3.0:
                        f_clamped = (f_clamped / np.sum(f_clamped)) * 3.0
                    for idx_s, s in enumerate(symbols):
                        half_kelly_weights[s] = float(f_clamped[idx_s])
                except Exception:
                    half_kelly_weights = {s: 0.18 for s in symbols}
            else:
                half_kelly_weights = {s: 0.18 for s in symbols}

            # Evaluacion por activo
            for s in symbols:
                b = bar_data[s]
                spec = SYMBOL_SPECS.get(s, {"tick": 0.01, "step": 0.01, "spread_bps": 20.0})
                cur_close = b["close"]
                cur_high = b["high"]
                cur_low = b["low"]
                cur_vol = b["vol"]
                obi = obis.get(s, 0.0)

                active_qty = symbol_inventory[s]
                entry_px = symbol_entry_px[s]
                spread_usd = cur_close * (spec["spread_bps"] / 10_000.0)

                # SIN POSICION: Cotizar Bids si no hay Kill Switch activo ni Contagio
                if active_qty == 0.0:
                    if kill_switch_active:
                        continue
                    if s != "BTC" and bid_shield_active:
                        continue

                    # El capital operativo disponible es el minimo entre el equity actual y el tope
                    effective_cap = min(portfolio.current_equity_usd, portfolio.base_capital_usd if mode == "STATIC_5P" else MAX_ACTIVE_CAPITAL_CAP_USD)
                    if effective_cap <= 1.0:
                        continue

                    w = half_kelly_weights.get(s, 0.18)
                    target_notional = min(effective_cap * w, effective_cap * 0.20)

                    if spec["step"] * cur_close > target_notional:
                        order_qty = max(0.0001, round(target_notional / cur_close, 4))
                    else:
                        st = spec["step"]
                        order_qty = max(st, round(round(target_notional / cur_close / st) * st, 6))

                    if mode == "DYNAMIC_MULTI_RATCHET":
                        # Escalera 3 niveles L2
                        delta_star = spread_usd * 0.5 * (1.0 + max(0.0, -obi * 0.25))
                        p_l1 = round((cur_close - delta_star * 1.0) / spec["tick"]) * spec["tick"]
                        p_l2 = round((cur_close - delta_star * 2.0) / spec["tick"]) * spec["tick"]
                        p_l3 = round((cur_close - delta_star * 3.5) / spec["tick"]) * spec["tick"]

                        max_fill_capacity = cur_vol * 0.15  # Límite 15% volumen de barra
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
                            bounce_ratio = (cur_close - cur_low) / max(b["high"] - cur_low, 1e-8)
                            if cur_low <= p_l3 and bounce_ratio < 0.20:
                                adverse_selection_events += 1
                                adverse_slip = (spread_usd * 0.35) + spec["tick"]
                                avg_px = min(avg_px - adverse_slip, cur_low)

                            symbol_inventory[s] = filled_qty
                            symbol_entry_px[s] = avg_px
                            symbol_entry_bar[s] = bar_idx
                            symbol_levels_filled[s] = tiers_filled
                            portfolio.active_positions[s] = filled_qty * avg_px
                    else:
                        # Modo STATIC_5P: 1 solo nivel
                        bid_discount = spread_usd * (0.50 + max(0.0, -obi * 0.25))
                        bid_px = round((cur_close - bid_discount) / spec["tick"]) * spec["tick"]
                        max_fill_capacity = cur_vol * 0.15
                        fill_qty = min(order_qty, max_fill_capacity)
                        if cur_low <= bid_px and fill_qty > 0:
                            symbol_inventory[s] = fill_qty
                            symbol_entry_px[s] = bid_px
                            symbol_entry_bar[s] = bar_idx
                            symbol_levels_filled[s] = 1
                            portfolio.active_positions[s] = fill_qty * bid_px

                # CON POSICION: Evaluar TP / SL
                elif active_qty > 0.0:
                    bars_held = bar_idx - symbol_entry_bar[s]
                    tiers_count = symbol_levels_filled.get(s, 1)

                    v_hist = hist_returns[s][max(0, bar_idx - 20):bar_idx]
                    rolling_sigma = float(np.mean(v_hist)) if v_hist else 0.004

                    if mode == "DYNAMIC_MULTI_RATCHET" and bars_held >= 1:
                        target_spread = spread_usd * 0.75
                    else:
                        target_spread = spread_usd

                    target_ask = round((entry_px + target_spread) / spec["tick"]) * spec["tick"]
                    structural_stop_dist = max(1.5 * rolling_sigma * entry_px, spread_usd * 1.5)
                    structural_stop_px = entry_px - structural_stop_dist

                    # Caso A: Stop-Loss Estructural (1.5 sigma)
                    if cur_low <= structural_stop_px:
                        exit_px = structural_stop_px
                        gross = (exit_px - entry_px) * active_qty
                        fee = (entry_px + exit_px) * active_qty * 0.00020
                        net = gross - fee

                        new_eq = max(0.01, portfolio.current_equity_usd + net)
                        portfolio.current_equity_usd = new_eq

                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)

                        losing_trades += tiers_count
                        total_rotations += tiers_count
                        gross_loss += abs(net)

                        symbol_inventory[s] = 0.0
                        symbol_entry_px[s] = 0.0
                        portfolio.active_positions.pop(s, None)

                    # Caso B: Take Profit Maker
                    elif cur_high >= target_ask:
                        exit_px = target_ask
                        gross = (exit_px - entry_px) * active_qty
                        fee = (entry_px + exit_px) * active_qty * 0.00015
                        net = gross - fee

                        new_eq = max(0.01, portfolio.current_equity_usd + net)
                        portfolio.current_equity_usd = new_eq
                        if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                            portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)

                        winning_trades += tiers_count
                        total_rotations += tiers_count
                        gross_profit += net

                        symbol_inventory[s] = 0.0
                        symbol_entry_px[s] = 0.0
                        portfolio.active_positions.pop(s, None)

                        if mode == "DYNAMIC_MULTI_RATCHET":
                            swept = step_up_ratchet(
                                portfolio,
                                threshold_pct=0.02,
                                vault_ratio=0.15,
                                reinvest_ratio=0.85,
                                max_capital_cap_usd=MAX_ACTIVE_CAPITAL_CAP_USD,
                            )
                            if swept > 0.0:
                                swept_vault_usd += swept
                                ratchet_sweeps_count += 1

                    # Caso C: Salida por tiempo adverso (> 8 barras = 2h)
                    elif bars_held > 8 and cur_close < entry_px:
                        exit_px = cur_close
                        gross = (exit_px - entry_px) * active_qty
                        fee = (entry_px + exit_px) * active_qty * 0.00015
                        net = gross - fee

                        new_eq = max(0.01, portfolio.current_equity_usd + net)
                        portfolio.current_equity_usd = new_eq

                        ret = net / max(portfolio.current_equity_usd, 1.0)
                        returns_series.append(ret)

                        total_rotations += tiers_count
                        if net > 0:
                            winning_trades += tiers_count
                            gross_profit += net
                        else:
                            losing_trades += tiers_count
                            gross_loss += abs(net)

                        symbol_inventory[s] = 0.0
                        symbol_entry_px[s] = 0.0
                        portfolio.active_positions.pop(s, None)

                        if mode == "DYNAMIC_MULTI_RATCHET":
                            swept = step_up_ratchet(
                                portfolio,
                                threshold_pct=0.02,
                                vault_ratio=0.15,
                                reinvest_ratio=0.85,
                                max_capital_cap_usd=MAX_ACTIVE_CAPITAL_CAP_USD,
                            )
                            if swept > 0.0:
                                swept_vault_usd += swept
                                ratchet_sweeps_count += 1

            # MTM Max Drawdown
            unrealized_pnl = sum(
                symbol_inventory[s] * (bar_data[s]["close"] - symbol_entry_px[s])
                for s in symbols
                if symbol_inventory[s] > 0.0
            )
            total_wealth_mtm = portfolio.current_equity_usd + swept_vault_usd + unrealized_pnl
            if total_wealth_mtm > peak_wealth:
                peak_wealth = total_wealth_mtm
            dd_usd = peak_wealth - total_wealth_mtm
            dd_pct = (dd_usd / peak_wealth) * 100.0 if peak_wealth > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        final_wealth = portfolio.current_equity_usd + swept_vault_usd
        total_pnl = final_wealth - initial_capital_usd
        roi_pct = (total_pnl / initial_capital_usd) * 100.0

        win_rate = (winning_trades / total_rotations * 100.0) if total_rotations > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 1.0)

        ret_arr = np.array(returns_series, dtype=np.float64) if len(returns_series) >= 5 else np.array([0.001] * 5)
        dsr_audit = DeflatedSharpeAuditor.audit_returns(ret_arr, num_trials_n=25)

        return {
            "final_futures_usd": round(portfolio.current_equity_usd, 2),
            "swept_vault_usd": round(swept_vault_usd, 2),
            "total_wealth_usd": round(final_wealth, 2),
            "roi_pct": round(roi_pct, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_rotations": total_rotations,
            "observed_sharpe": round(dsr_audit.observed_sharpe_sr, 2),
            "dsr_score": round(dsr_audit.deflated_sharpe_ratio, 4),
            "is_overfitted": dsr_audit.is_overfitted,
            "ratchet_sweeps_count": ratchet_sweeps_count,
            "ladder_deep_fills_count": ladder_deep_fills_count,
            "adverse_selection_events": adverse_selection_events,
            "contagion_shield_activations": contagion_shield_count,
            "kill_switch_triggers": kill_switch_triggers,
        }

    def evaluate_all_periods(self, symbols: list[str]) -> list[PeriodEvaluationResult]:
        self.load_data(symbols)
        total_bars = min(len(self.datasets[s]) for s in symbols)
        print(f"\n[EVALUATOR] Barras disponibles por par: {total_bars:,}")

        periods_def = [
            {
                "id": "FULL_2Y",
                "name": "Ciclo Completo 2021-2022 (Bull + Bear)",
                "start_idx": 0,
                "end_idx": 70080,
                "start_date": "2021-01-01",
                "end_date": "2022-12-31",
            },
            {
                "id": "BULL_2021",
                "name": "Año 2021: Bull Market Histórico ($29k -> $69k)",
                "start_idx": 0,
                "end_idx": 35040,
                "start_date": "2021-01-01",
                "end_date": "2021-12-31",
            },
            {
                "id": "BEAR_2022",
                "name": "Año 2022: Great Deleveraging ($47k -> $15k)",
                "start_idx": 35040,
                "end_idx": 70080,
                "start_date": "2022-01-01",
                "end_date": "2022-12-31",
            },
            {
                "id": "CRASH_LUNA_2022",
                "name": "Stress Test: Colapso Terra/Luna & 3AC (Q2 2022)",
                "start_idx": 43680,
                "end_idx": 52416,
                "start_date": "2022-04-01",
                "end_date": "2022-06-30",
            },
            {
                "id": "CRASH_FTX_2022",
                "name": "Stress Test: Quiebra de FTX & Alameda (Q4 2022)",
                "start_idx": 61248,
                "end_idx": 70080,
                "start_date": "2022-10-01",
                "end_date": "2022-12-31",
            },
        ]

        results: list[PeriodEvaluationResult] = []

        for p in periods_def:
            p_id = p["id"]
            p_name = p["name"]
            s_idx = p["start_idx"]
            e_idx = min(p["end_idx"], total_bars)
            b_cnt = e_idx - s_idx

            print(f"\n========================================================")
            print(f"[{p_id}] {p_name}")
            print(f"  Rango: {p['start_date']} a {p['end_date']} ({b_cnt:,} barras)")
            print(f"========================================================")

            sliced = {s: self.datasets[s][s_idx:e_idx] for s in symbols}

            # Benchmarks HODL
            btc_start = sliced["BTC"][0][1]
            btc_end = sliced["BTC"][-1][4]
            btc_hodl = ((btc_end - btc_start) / btc_start) * 100.0

            eth_start = sliced["ETH"][0][1]
            eth_end = sliced["ETH"][-1][4]
            eth_hodl = ((eth_end - eth_start) / eth_start) * 100.0

            sol_start = sliced["SOL"][0][1]
            sol_end = sliced["SOL"][-1][4]
            sol_hodl = ((sol_end - sol_start) / sol_start) * 100.0

            bnb_start = sliced["BNB"][0][1]
            bnb_end = sliced["BNB"][-1][4]
            bnb_hodl = ((bnb_end - bnb_start) / bnb_start) * 100.0

            doge_start = sliced["DOGE"][0][1]
            doge_end = sliced["DOGE"][-1][4]
            doge_hodl = ((doge_end - doge_start) / doge_start) * 100.0

            eq_hodl = (btc_hodl + eth_hodl + sol_hodl + bnb_hodl + doge_hodl) / 5.0

            print(f"  [HODL BENCHMARKS] BTC: {btc_hodl:+.1f}% | ETH: {eth_hodl:+.1f}% | SOL: {sol_hodl:+.1f}% | BNB: {bnb_hodl:+.1f}% | DOGE: {doge_hodl:+.1f}% | EW Basket: {eq_hodl:+.1f}%")

            # Correr Modo A: STATIC_5P
            t0 = time.time()
            res_static = self.simulate_slice(sliced, symbols, mode="STATIC_5P", initial_capital_usd=100.0)
            t_static = time.time() - t0
            print(f"  -> STATIC_5P ({t_static:.2f}s): Riqueza ${res_static['total_wealth_usd']:,.2f} ({res_static['roi_pct']:+.1f}%) | MaxDD: {res_static['max_drawdown_pct']:.1f}% | PF: {res_static['profit_factor']:.2f} | KS Triggers: {res_static['kill_switch_triggers']}")

            # Correr Modo B: DYNAMIC_MULTI_RATCHET
            t1 = time.time()
            res_dynamic = self.simulate_slice(sliced, symbols, mode="DYNAMIC_MULTI_RATCHET", initial_capital_usd=100.0)
            t_dyn = time.time() - t1
            alpha = res_dynamic["roi_pct"] - eq_hodl
            print(f"  -> DYNAMIC_RATCHET ({t_dyn:.2f}s): Riqueza ${res_dynamic['total_wealth_usd']:,.2f} ({res_dynamic['roi_pct']:+.1f}%) | MaxDD: {res_dynamic['max_drawdown_pct']:.1f}% | PF: {res_dynamic['profit_factor']:.2f} | Sweeps: {res_dynamic['ratchet_sweeps_count']} | Alpha vs HODL: {alpha:+.1f}%")

            res_obj = PeriodEvaluationResult(
                period_id=p_id,
                period_name=p_name,
                start_date=p["start_date"],
                end_date=p["end_date"],
                bars_count=b_cnt,
                initial_capital_usd=100.0,
                static_final_futures_usd=res_static["final_futures_usd"],
                static_swept_vault_usd=res_static["swept_vault_usd"],
                static_total_wealth_usd=res_static["total_wealth_usd"],
                static_roi_pct=res_static["roi_pct"],
                static_max_drawdown_pct=res_static["max_drawdown_pct"],
                static_win_rate_pct=res_static["win_rate_pct"],
                static_profit_factor=res_static["profit_factor"],
                static_total_rotations=res_static["total_rotations"],
                static_observed_sharpe=res_static["observed_sharpe"],
                static_dsr_score=res_static["dsr_score"],
                static_is_overfitted=res_static["is_overfitted"],
                static_kill_switch_triggers=res_static["kill_switch_triggers"],
                dynamic_final_futures_usd=res_dynamic["final_futures_usd"],
                dynamic_swept_vault_usd=res_dynamic["swept_vault_usd"],
                dynamic_total_wealth_usd=res_dynamic["total_wealth_usd"],
                dynamic_roi_pct=res_dynamic["roi_pct"],
                dynamic_max_drawdown_pct=res_dynamic["max_drawdown_pct"],
                dynamic_win_rate_pct=res_dynamic["win_rate_pct"],
                dynamic_profit_factor=res_dynamic["profit_factor"],
                dynamic_total_rotations=res_dynamic["total_rotations"],
                dynamic_observed_sharpe=res_dynamic["observed_sharpe"],
                dynamic_dsr_score=res_dynamic["dsr_score"],
                dynamic_is_overfitted=res_dynamic["is_overfitted"],
                dynamic_ratchet_sweeps_count=res_dynamic["ratchet_sweeps_count"],
                dynamic_ladder_deep_fills_count=res_dynamic["ladder_deep_fills_count"],
                dynamic_adverse_selection_events=res_dynamic["adverse_selection_events"],
                dynamic_contagion_shield_activations=res_dynamic["contagion_shield_activations"],
                dynamic_kill_switch_triggers=res_dynamic["kill_switch_triggers"],
                btc_hodl_roi_pct=round(btc_hodl, 2),
                eth_hodl_roi_pct=round(eth_hodl, 2),
                sol_hodl_roi_pct=round(sol_hodl, 2),
                bnb_hodl_roi_pct=round(bnb_hodl, 2),
                doge_hodl_roi_pct=round(doge_hodl, 2),
                equal_weight_hodl_roi_pct=round(eq_hodl, 2),
                alpha_over_equal_weight_pct=round(alpha, 2),
            )
            results.append(res_obj)

        return results


def run_full_2021_2022_backtest() -> None:
    runner = Backtest20212022Runner()
    results = runner.evaluate_all_periods(CORE_5P_SYMBOLS)

    out_file = PROJECT_ROOT / "data" / "backtest_results_2021_2022.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n[EXITO] Resultados guardados exitosamente en {out_file}")


if __name__ == "__main__":
    run_full_2021_2022_backtest()
