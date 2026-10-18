# backend/multi_period_backtest.py
"""
Motor de Evaluacion Retrospectiva Multi-Periodo (31D, 90D, 180D, 365D y 2A)
sobre datos reales de Binance USD(S)-M Futures (BTC/USDT).
Ejecuta la simulación cuantitativa institucional completa a través del motor unificado
InstitutionalBacktestEngine (Zero Look-Ahead & Real-Friction).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from backend.backtest.engine import BacktestRunResult, InstitutionalBacktestEngine


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
    """Ejecuta backtests comparativos honestos sobre 31D, 90D, 180D, 365D y 2A."""

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
        engine = InstitutionalBacktestEngine(
            symbol="BTC/USDT:USDT",
            initial_capital_usd=initial_capital_usd,
            target_spread_bps=target_spread_bps,
            compound_growth=compound_growth,
        )
        typed_candles = [
            (int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]))
            for c in candles_subset
        ]
        res = engine.run(typed_candles, strategy_mode="MARKET_MAKING")

        btc_start_p = float(candles_subset[0][4])
        btc_end_p = float(candles_subset[-1][4])
        btc_hodl_pct = ((btc_end_p - btc_start_p) / btc_start_p) * 100.0

        net_profit_usd = res.total_wealth_usd - initial_capital_usd
        total_return_pct = res.compounded_roi_pct
        alpha_pct = total_return_pct - btc_hodl_pct

        return PeriodResult(
            period_label=period_label,
            period_days=days,
            bars_count=len(candles_subset),
            start_date=res.start_date,
            end_date=res.end_date,
            btc_start_price=round(btc_start_p, 2),
            btc_end_price=round(btc_end_p, 2),
            btc_hodl_return_pct=round(btc_hodl_pct, 2),
            initial_capital_usd=round(initial_capital_usd, 2),
            final_futures_capital_usd=round(res.final_equity_usd, 2),
            swept_spot_vault_usd=round(res.spot_vault_swept_usd, 2),
            total_final_equity_usd=round(res.total_wealth_usd, 2),
            net_profit_usd=round(net_profit_usd, 2),
            total_return_pct=round(total_return_pct, 2),
            alpha_vs_hodl_pct=round(alpha_pct, 2),
            max_drawdown_pct=round(res.max_drawdown_pct, 2),
            max_drawdown_usd=round(res.max_drawdown_usd, 2),
            total_rotations=res.total_trades,
            win_rate_pct=round(res.win_rate_pct, 2),
            profit_factor=round(res.profit_factor, 2),
            observed_sharpe_sr=round(res.observed_daily_sharpe, 3),
            benchmark_null_sr0=0.0,
            deflated_sharpe_ratio=round(res.deflated_sharpe_ratio, 4),
            dsr_status=res.dsr_status,
            is_overfitted=res.is_overfitted,
            kill_switch_triggers=res.kill_switch_triggers,
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
            print(f"    DSR Score:           {res.deflated_sharpe_ratio * 100:.1f}% [{res.dsr_status}] (SR Diario: {res.observed_sharpe_sr:.3f})")

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
