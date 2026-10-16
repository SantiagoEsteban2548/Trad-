# backend/multi_period_backtest.py
"""
Motor de Evaluacion Retrospectiva Multi-Periodo (31D, 90D, 180D, 365D y 2A)
sobre datos reales de Binance USD(S)-M Futures (BTC/USDT).
Ejecuta la simulacion cuantitativa completa sobre cada horizonte temporal
partiendo con un capital base de 100.00 USDT y aplicando los 4 Mandate Gates:
- Gate 1: ISOLATED Margin
- Gate 2: LIMIT + GTX (Post-Only Maker)
- Gate 3: Tope Half-Kelly y Concentracion <= 20% por par (order_qty calibrado)
- Gate 4: Kill-Switch Diario (-3.0%) y Profit Sweep diario a Boveda Spot
- Auditoria: Deflated Sharpe Ratio (DSR) de Bailey & Lopez de Prado
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import FundingArbSnapshot, FundingRateArbitrageEngine
from backend.strategies.liq_cascade import LiquidationCascadeHunter
from backend.strategies.regime_config import RegimeConfig
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, L2QueueSimulator


@dataclass
class PeriodResult:
    period_label: str
    period_days: int
    bars_count: int
    start_date: str
    end_date: str
    btc_start_price: float
    btc_end_price: float
    btc_hodl_return_pct: float
    initial_capital_usd: float
    final_futures_capital_usd: float
    swept_spot_vault_usd: float
    total_final_equity_usd: float
    net_profit_usd: float
    total_return_pct: float
    alpha_vs_hodl_pct: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    total_rotations: int
    win_rate_pct: float
    profit_factor: float
    observed_sharpe_sr: float
    benchmark_null_sr0: float
    deflated_sharpe_ratio: float
    dsr_status: str
    is_overfitted: bool
    kill_switch_triggers: int


class MultiPeriodBacktester:
    """Ejecuta backtests comparativos sobre 31D, 90D, 180D, 365D y 2A."""

    PERIODS = [
        ("31D", 31),
        ("90D", 90),
        ("180D", 180),
        ("365D", 365),
        ("2A", 730),
    ]

    def __init__(self, cache_file: Path | None = None) -> None:
        self.cache_file = cache_file or Path("data/cache/binance_BTC_USDT_2y_15m.json")

    def load_dataset(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        if not self.cache_file.exists():
            raise FileNotFoundError(f"No se encontro el archivo de cache: {self.cache_file}. Ejecute scripts/download_multi_period_data.py primero.")
        print(f"[CACHE] Cargando dataset de 2 anos desde {self.cache_file}...")
        with open(self.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["candles"], data.get("funding_rates", [])

    def run_period(
        self,
        candles_subset: list[list[float]],
        period_label: str,
        days: int,
        initial_capital_usd: float = 100.0,
        target_spread_bps: float = 18.0,
        compound_growth: bool = True,
    ) -> PeriodResult:
        gatekeeper = MandateGatekeeper()
        config_path = Path("backend/config/regime_config.json")
        regime_config = (
            RegimeConfig.load_from_file(config_path)
            if config_path.exists()
            else RegimeConfig.default_normal()
        )

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
        active_inventory_qty = 0.0
        avg_entry_price = 0.0
        entry_bar_idx = 0
        is_cascade_position = False
        cascade_wick_size = 0.0

        total_rotations = 0
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0

        peak_equity = initial_capital_usd
        max_drawdown_pct = 0.0
        max_drawdown_usd = 0.0
        kill_switch_triggers = 0

        returns_series: list[float] = []
        rolling_returns: list[float] = []
        ema_obi = 0.0
        current_day_str = ""

        # funding rate tipico en Binance Futures (~0.01% por 8h)
        base_funding_rate_8h = 0.00008

        for i in range(1, len(candles_subset)):
            ts_ms, open_p, high_p, low_p, close_p, volume = candles_subset[i]
            prev_cl = candles_subset[i - 1][4]
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            now_ns = int(ts_ms * 1_000_000)

            # Volatilidad realizada rolling (30 barras)
            bar_ret = (close_p - prev_cl) / prev_cl
            rolling_returns.append(bar_ret)
            if len(rolling_returns) > 30:
                rolling_returns.pop(0)
            sigma = max(float(np.std(rolling_returns)), 0.0003)

            # OBI suavizado
            range_p = max(high_p - low_p, 0.10)
            body_p = close_p - open_p
            raw_obi = float(np.clip(body_p / range_p, -0.95, 0.95))
            ema_obi = 0.85 * ema_obi + 0.15 * raw_obi

            # Cambio de dia a 00:00 UTC
            if current_day_str != date_str:
                current_day_str = date_str
                portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            if dt.hour == 0 and dt.minute == 0 and active_inventory_qty == 0.0:
                if not compound_growth:
                    # En modo lineal: Barrido de excedente diario a Bóveda Spot
                    if portfolio.current_equity_usd > initial_capital_usd:
                        surplus = round(portfolio.current_equity_usd - initial_capital_usd, 2)
                        swept_spot_vault_usd += surplus
                        portfolio.current_equity_usd = initial_capital_usd
                        portfolio.daily_high_watermark_usd = initial_capital_usd
                else:
                    # En crecimiento compuesto: Re-inversión completa, capital base operativo escala
                    portfolio.base_capital_usd = portfolio.current_equity_usd
                    portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            # Cobro/Pago de Funding cada 8 horas (00:00, 08:00, 16:00 UTC)
            if dt.minute == 0 and dt.hour in (0, 8, 16):
                if abs(active_inventory_qty) > 0:
                    funding_cost = (active_inventory_qty * close_p) * base_funding_rate_8h
                    portfolio.current_equity_usd -= funding_cost

            # Dimensionamiento dinamico segun precio BTC para respetar Gate 3 (max 20% capital actual)
            current_cap = portfolio.current_equity_usd if compound_growth else initial_capital_usd
            max_notional_gate3 = current_cap * 0.20
            order_qty = round((max_notional_gate3 * 0.90) / close_p, 4)
            if not compound_growth:
                order_qty = min(0.0002, order_qty)
            order_qty = max(0.0001, order_qty)

            # Calculo de Spread
            spread_usd = close_p * (target_spread_bps / 10000.0)
            bid_px = round(close_p - spread_usd / 2.0, 1)

            # Deteccion de mecha de liquidacion (Motor B)
            lower_wick = min(open_p, close_p) - low_p
            is_liq_cascade = lower_wick > (spread_usd * 1.8) and volume > (np.mean(volume) if volume else 100.0)

            # Ejecucion L2
            if active_inventory_qty == 0.0:
                # Motor B: Entrada en mecha de liquidacion
                if is_liq_cascade:
                    req_liq = OrderRequest(
                        symbol="BTC/USDT:USDT",
                        side="BUY",
                        order_type="LIMIT",
                        price=low_p + 5.0,
                        qty=order_qty,
                        time_in_force="GTX",
                        margin_mode="ISOLATED",
                        strategy_id="LIQ_CASCADE",
                        timestamp_ns=now_ns,
                    )
                    is_valid, _ = gatekeeper.validate_order(req_liq, portfolio)
                    if is_valid:
                        active_inventory_qty = order_qty
                        avg_entry_price = low_p + 5.0
                        entry_bar_idx = i
                        is_cascade_position = True
                        cascade_wick_size = lower_wick
                        portfolio.active_positions["BTC/USDT:USDT"] = order_qty * avg_entry_price
                    else:
                        if gatekeeper.kill_switch_engaged:
                            kill_switch_triggers += 1

                # Motor A: Avellaneda-Stoikov + OBI Shield
                elif ema_obi > -0.60 and low_p <= bid_px:
                    req_bid = OrderRequest(
                        symbol="BTC/USDT:USDT",
                        side="BUY",
                        order_type="LIMIT",
                        price=bid_px,
                        qty=order_qty,
                        time_in_force="GTX",
                        margin_mode="ISOLATED",
                        strategy_id="AVELLANEDA_OBI",
                        timestamp_ns=now_ns,
                    )
                    is_valid, _ = gatekeeper.validate_order(req_bid, portfolio)
                    if is_valid:
                        active_inventory_qty = order_qty
                        avg_entry_price = bid_px
                        entry_bar_idx = i
                        is_cascade_position = False
                        cascade_wick_size = 0.0
                        portfolio.active_positions["BTC/USDT:USDT"] = order_qty * bid_px
                    else:
                        if gatekeeper.kill_switch_engaged:
                            kill_switch_triggers += 1

            elif active_inventory_qty > 0.0:
                bars_held = i - entry_bar_idx

                if is_cascade_position:
                    target_ask = avg_entry_price + (0.40 * cascade_wick_size)
                else:
                    target_ask = avg_entry_price + spread_usd

                # 1. TP Maker alcanzado
                if high_p >= target_ask:
                    exit_px = target_ask
                    gross = (exit_px - avg_entry_price) * active_inventory_qty
                    fee = (avg_entry_price + exit_px) * active_inventory_qty * 0.00015
                    net = gross - fee

                    portfolio.current_equity_usd += net
                    if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                        portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

                    returns_series.append(net / (portfolio.current_equity_usd if compound_growth else initial_capital_usd))
                    total_rotations += 1
                    winning_trades += 1
                    gross_profit += net

                    active_inventory_qty = 0.0
                    avg_entry_price = 0.0
                    portfolio.active_positions.clear()

                # 2. Cierre por tiempo (>12 barras = 3 horas sin rellenar ask y precio adverso)
                elif bars_held > 12 and close_p < avg_entry_price:
                    exit_px = close_p
                    gross = (exit_px - avg_entry_price) * active_inventory_qty
                    fee = (avg_entry_price + exit_px) * active_inventory_qty * 0.00015
                    net = gross - fee

                    portfolio.current_equity_usd += net
                    returns_series.append(net / (portfolio.current_equity_usd if compound_growth else initial_capital_usd))
                    total_rotations += 1
                    if net > 0:
                        winning_trades += 1
                        gross_profit += net
                    else:
                        losing_trades += 1
                        gross_loss += abs(net)

                    active_inventory_qty = 0.0
                    avg_entry_price = 0.0
                    portfolio.active_positions.clear()

                # 3. Hard Stop Loss a 2.5x spread
                elif (avg_entry_price - low_p) > (spread_usd * 2.5):
                    exit_px = avg_entry_price - (spread_usd * 2.5)
                    gross = (exit_px - avg_entry_price) * active_inventory_qty
                    fee = (avg_entry_price + exit_px) * active_inventory_qty * 0.00020
                    net = gross - fee

                    portfolio.current_equity_usd += net
                    returns_series.append(net / (portfolio.current_equity_usd if compound_growth else initial_capital_usd))
                    total_rotations += 1
                    losing_trades += 1
                    gross_loss += abs(net)

                    active_inventory_qty = 0.0
                    avg_entry_price = 0.0
                    portfolio.active_positions.clear()

            # Monitoreo de equidad y drawdown
            tot_worth = portfolio.current_equity_usd + swept_spot_vault_usd
            if tot_worth > peak_equity:
                peak_equity = tot_worth

            cur_dd_usd = peak_equity - tot_worth
            cur_dd_pct = (cur_dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if cur_dd_pct > max_drawdown_pct:
                max_drawdown_pct = cur_dd_pct
                max_drawdown_usd = cur_dd_usd

        # Metricas finales del periodo
        btc_start_p = candles_subset[0][4]
        btc_end_p = candles_subset[-1][4]
        btc_hodl_pct = ((btc_end_p - btc_start_p) / btc_start_p) * 100.0

        net_profit_usd = (portfolio.current_equity_usd - initial_capital_usd) + swept_spot_vault_usd
        total_return_pct = (net_profit_usd / initial_capital_usd) * 100.0
        alpha_pct = total_return_pct - btc_hodl_pct

        win_rate = (winning_trades / total_rotations * 100.0) if total_rotations > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 1.0)

        # Auditoria DSR
        ret_arr = np.array(returns_series, dtype=np.float64) if len(returns_series) >= 3 else np.array([0.001, 0.001, 0.001])
        dsr_audit = DeflatedSharpeAuditor.audit_returns(ret_arr, num_trials_n=25)

        start_dt = datetime.fromtimestamp(candles_subset[0][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        end_dt = datetime.fromtimestamp(candles_subset[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        return PeriodResult(
            period_label=period_label,
            period_days=days,
            bars_count=len(candles_subset),
            start_date=start_dt,
            end_date=end_dt,
            btc_start_price=round(btc_start_p, 2),
            btc_end_price=round(btc_end_p, 2),
            btc_hodl_return_pct=round(btc_hodl_pct, 2),
            initial_capital_usd=round(initial_capital_usd, 2),
            final_futures_capital_usd=round(portfolio.current_equity_usd, 2),
            swept_spot_vault_usd=round(swept_spot_vault_usd, 2),
            total_final_equity_usd=round(portfolio.current_equity_usd + swept_spot_vault_usd, 2),
            net_profit_usd=round(net_profit_usd, 2),
            total_return_pct=round(total_return_pct, 2),
            alpha_vs_hodl_pct=round(alpha_pct, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            max_drawdown_usd=round(max_drawdown_usd, 2),
            total_rotations=total_rotations,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            observed_sharpe_sr=round(dsr_audit.observed_sharpe_sr, 3),
            benchmark_null_sr0=round(dsr_audit.benchmark_null_sr0, 3),
            deflated_sharpe_ratio=round(dsr_audit.deflated_sharpe_ratio, 4),
            dsr_status=dsr_audit.status,
            is_overfitted=dsr_audit.is_overfitted,
            kill_switch_triggers=kill_switch_triggers,
        )

    def run_all(self, compound_growth: bool = True) -> list[PeriodResult]:
        candles, _ = self.load_dataset()
        results: list[PeriodResult] = []

        mode_str = "CRECIMIENTO COMPUESTO (REINVERSION DINAMICA)" if compound_growth else "LINEAL (BARRIDO A SPOT)"
        print(f"\n=================================================================")
        print(f"      EVALUACION MULTI-PERIODO RETROSPECTIVA (BINANCE FUTURES)   ")
        print(f"      MODO: {mode_str}")
        print(f"=================================================================")

        for label, days in self.PERIODS:
            bars_needed = days * 96  # 96 velas de 15m por dia
            if bars_needed > len(candles):
                subset = candles
            else:
                subset = candles[-bars_needed:]

            print(f"\n>>> Evaluando {label} ({days} dias, {len(subset):,} velas de 15m)...")
            res = self.run_period(
                candles_subset=subset,
                period_label=label,
                days=days,
                initial_capital_usd=100.0,
                target_spread_bps=18.0,
                compound_growth=compound_growth,
            )
            results.append(res)

            print(f"    Rango: {res.start_date} -> {res.end_date}")
            print(f"    BTC/USDT Buy & Hold: {res.btc_hodl_return_pct:+.2f}% (${res.btc_start_price:,.0f} -> ${res.btc_end_price:,.0f})")
            print(f"    Bot Retorno Total:   {res.total_return_pct:+.2f}% (+${res.net_profit_usd:.2f} USDT) | Equity Final: ${res.total_final_equity_usd:.2f}")
            print(f"    Alpha generado:      {res.alpha_vs_hodl_pct:+.2f}% sobre BTC")
            print(f"    Max Drawdown:        {res.max_drawdown_pct:.2f}% (${res.max_drawdown_usd:.2f} USDT)")
            print(f"    Rotaciones:          {res.total_rotations:,} | Win Rate: {res.win_rate_pct:.1f}% | Profit Factor: {res.profit_factor:.2f}")
            print(f"    DSR Score:           {res.deflated_sharpe_ratio * 100:.1f}% [{res.dsr_status}] (SR: {res.observed_sharpe_sr:.3f} vs SR0: {res.benchmark_null_sr0:.3f})")

        # Guardar resultados consolidados en JSON
        out_file = Path("data/multi_period_results.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n[OK] Resultados consolidados guardados en {out_file}")

        return results


def main() -> None:
    backtester = MultiPeriodBacktester()
    backtester.run_all(compound_growth=True)


if __name__ == "__main__":
    main()
