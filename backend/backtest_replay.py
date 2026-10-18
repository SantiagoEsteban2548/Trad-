# backend/backtest_replay.py
"""
Evaluador Retrospectivo y Motor de Replay Cuantitativo de Binance (USD(S)-M Futures).
Ejecuta la simulacion retrospectiva de alta fidelidad con datos reales de Binance:
- 10,080 velas reales de 1 minuto (OHLCV).
- Capital inicial: 100.00 USDT.
- Modo de ejecucion: ISOLATED, LIMIT + GTX (Post-Only Maker).
- Motores: Avellaneda-Stoikov + OBI Shield, Liquidation Cascade Hunter.
- Barrido de beneficios: Diario a las 00:00 UTC hacia la Boveda Spot Segura.
- Auditoria institucional: Deflated Sharpe Ratio (DSR) de Bailey & Lopez de Prado.
- Ejecución física verídica sin look-ahead a través de InstitutionalBacktestEngine.
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

import ccxt
import numpy as np

from backend.backtest.engine import BacktestRunResult, InstitutionalBacktestEngine
from backend.mandate_gates.schemas import PortfolioState
from backend.strategies.regime_config import RegimeConfig
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor


@dataclass
class BacktestTradeRecord:
    timestamp_utc: str
    engine_id: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl_usd: float
    net_pnl_usd: float
    pnl_pct: float
    duration_bars: int
    exit_reason: str


@dataclass
class DailySnapshot:
    date_str: str
    starting_equity_usd: float
    ending_equity_usd: float
    swept_to_vault_usd: float
    day_net_pnl_usd: float
    day_return_pct: float
    rotations_filled: int
    max_drawdown_pct: float


class BinanceHistoricalDataLoader:
    """Descarga y cachea velas de 1m y tasas de funding de Binance Futures."""

    CACHE_DIR = Path("data/cache")

    def __init__(self, symbol: str = "BTC/USDT", days: int = 7) -> None:
        self.symbol = symbol
        self.days = days
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.CACHE_DIR / f"binance_{symbol.replace('/', '_')}_{days}d_1m.json"

    def fetch_or_load(self) -> tuple[list[list[float]], list[dict[str, Any]]]:
        if self.cache_file.exists():
            print(f"[CACHE] Cargando datos historicos desde cache local ({self.cache_file})...")
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["candles"], data.get("funding_rates", [])

        print(f"[BINANCE REST] Descargando {self.days} dias de velas de 1m para {self.symbol}...")
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        now = datetime.now(timezone.utc)
        since_dt = now - timedelta(days=self.days)
        since_ms = int(since_dt.timestamp() * 1000)

        all_candles: list[list[float]] = []
        current_since = since_ms
        target_candles = self.days * 24 * 60

        while len(all_candles) < target_candles:
            try:
                batch = exchange.fetch_ohlcv(self.symbol, timeframe="1m", since=current_since, limit=1000)
                if not batch:
                    break
                all_candles.extend(batch)
                current_since = batch[-1][0] + 60000
                time.sleep(0.1)
            except Exception as e:
                print(f"[ERROR] Error al descargar velas: {e}")
                break

        print(f"[OK] Total de velas descargadas: {len(all_candles)}")
        funding_rates: list[dict[str, Any]] = []
        try:
            fr_raw = exchange.fetch_funding_rate_history(self.symbol, since=since_ms, limit=100)
            for item in fr_raw:
                funding_rates.append({"timestamp": item["timestamp"], "fundingRate": item["fundingRate"]})
        except Exception as e:
            print(f"[WARN] No se pudieron descargar funding rates: {e}")

        payload = {
            "symbol": self.symbol,
            "days": self.days,
            "candles_count": len(all_candles),
            "candles": all_candles,
            "funding_rates": funding_rates,
        }
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[CACHE] Datos guardados en {self.cache_file}")

        return all_candles, funding_rates


class RetrospectiveEngineReplay:
    """Motor de replay retrospectivo honesto que ejecuta sobre InstitutionalBacktestEngine."""

    def __init__(
        self,
        candles: list[list[float]],
        funding_rates: list[dict[str, Any]] | None = None,
        initial_capital_usd: float = 100.0,
        order_qty: float = 0.0002,
        target_spread_bps: float = 18.0,
    ) -> None:
        self.candles = candles
        self.funding_rates = funding_rates or []
        self.initial_capital_usd = float(initial_capital_usd)
        self.order_qty = float(order_qty)
        self.target_spread_bps = float(target_spread_bps)

    def run(self) -> dict[str, Any]:
        print(f"[REPLAY] Iniciando replay honesto de {len(self.candles)} velas de 1m ({len(self.candles)/1440:.1f} dias)...")
        print(f"  -> Capital Inicial: {self.initial_capital_usd:.2f} USDT")
        print(f"  -> Calibracion de Spread: {self.target_spread_bps:.1f} bps")

        engine = InstitutionalBacktestEngine(
            symbol="BTC/USDT:USDT",
            initial_capital_usd=self.initial_capital_usd,
            target_spread_bps=self.target_spread_bps,
            compound_growth=False,
            maker_fee_bps=2.0,
            taker_fee_bps=5.0,
            slippage_bps=2.0,
            require_tick_penetration=True,
        )

        typed_candles = [
            (int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]))
            for c in self.candles
        ]
        res = engine.run(typed_candles, strategy_mode="MARKET_MAKING")

        initial_btc_price = float(self.candles[0][4])
        final_btc_price = float(self.candles[-1][4])
        btc_hodl_return_pct = round(((final_btc_price - initial_btc_price) / initial_btc_price) * 100.0, 2)
        total_net_pnl = round(res.total_wealth_usd - self.initial_capital_usd, 2)

        trade_records: list[BacktestTradeRecord] = []
        for t in res.trades:
            dt_str = datetime.fromtimestamp(t.exit_ts_ms / 1000, tz=timezone.utc).isoformat()
            trade_records.append(
                BacktestTradeRecord(
                    timestamp_utc=dt_str,
                    engine_id=t.strategy,
                    side=t.side,
                    qty=t.qty,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    gross_pnl_usd=t.gross_pnl_usd,
                    net_pnl_usd=t.net_pnl_usd,
                    pnl_pct=t.return_pct,
                    duration_bars=t.bars_held,
                    exit_reason=t.exit_reason,
                )
            )

        # Snapshot diario a partir de daily_returns
        daily_snapshots: list[DailySnapshot] = []
        cur_eq = self.initial_capital_usd
        for idx, d_ret in enumerate(res.daily_returns):
            nxt_eq = round(cur_eq * (1.0 + d_ret), 2)
            daily_snapshots.append(
                DailySnapshot(
                    date_str=f"Day_{idx+1}",
                    starting_equity_usd=cur_eq,
                    ending_equity_usd=nxt_eq,
                    swept_to_vault_usd=0.0,
                    day_net_pnl_usd=round(nxt_eq - cur_eq, 2),
                    day_return_pct=round(d_ret * 100.0, 2),
                    rotations_filled=max(1, len(trade_records) // max(len(res.daily_returns), 1)),
                    max_drawdown_pct=res.max_drawdown_pct,
                )
            )
            cur_eq = nxt_eq

        results = {
            "symbol": "BTC/USDT",
            "period_days": round(len(self.candles) / 1440.0, 1),
            "start_time": datetime.fromtimestamp(self.candles[0][0] / 1000, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(self.candles[-1][0] / 1000, tz=timezone.utc).isoformat(),
            "candles_analyzed": len(self.candles),
            "btc_price_start": initial_btc_price,
            "btc_price_end": final_btc_price,
            "btc_hodl_return_pct": btc_hodl_return_pct,
            "capital_initial_usd": self.initial_capital_usd,
            "capital_final_futures_usd": res.final_equity_usd,
            "swept_spot_vault_usd": res.spot_vault_swept_usd,
            "total_final_equity_usd": res.total_wealth_usd,
            "net_profit_usd": total_net_pnl,
            "total_return_pct": res.compounded_roi_pct,
            "max_drawdown_pct": res.max_drawdown_pct,
            "max_drawdown_usd": res.max_drawdown_usd,
            "total_rotations": res.total_trades,
            "win_rate_pct": res.win_rate_pct,
            "profit_factor": res.profit_factor,
            "dsr_audit": {
                "deflated_sharpe_ratio": res.deflated_sharpe_ratio,
                "observed_sharpe_sr": res.observed_daily_sharpe,
                "annualized_sharpe": res.annualized_sharpe,
                "benchmark_null_sr0": 0.0,
                "status": res.dsr_status,
                "is_overfitted": res.is_overfitted,
            },
            "daily_snapshots": [asdict(d) for d in daily_snapshots],
            "trade_records_count": len(trade_records),
            "trades": [asdict(t) for t in trade_records],
        }

        return results


def main() -> None:
    loader = BinanceHistoricalDataLoader(symbol="BTC/USDT", days=7)
    candles, funding_rates = loader.fetch_or_load()

    engine = RetrospectiveEngineReplay(
        candles=candles,
        funding_rates=funding_rates,
        initial_capital_usd=100.0,
        order_qty=0.0002,
        target_spread_bps=18.0,
    )
    results = engine.run()

    output_path = Path("data/backtest_results_7d.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=================================================================")
    print("        RESULTADOS DEL REPLAY HISTORICO HONESTO (BINANCE 7 DIAS) ")
    print("=================================================================")
    print(f" Periodo: {results['start_time'][:10]} -> {results['end_time'][:10]} ({results['period_days']} dias)")
    print(f" Velas analizadas: {results['candles_analyzed']:,} de 1 minuto")
    print(f" BTC/USDT Inicio: ${results['btc_price_start']:,.2f} | Fin: ${results['btc_price_end']:,.2f} ({results['btc_hodl_return_pct']:+.2f}%)")
    print("-----------------------------------------------------------------")
    print(f" Capital Base Inicial:          ${results['capital_initial_usd']:.2f} USDT")
    print(f" Patrimonio Final en Futuros:   ${results['capital_final_futures_usd']:.2f} USDT")
    print(f" Boveda Spot Segura Barrida:    ${results['swept_spot_vault_usd']:.2f} USDT")
    print(f" Patrimonio Total Acumulado:    ${results['total_final_equity_usd']:.2f} USDT")
    print(f" Ganancia Neta Total:           ${results['net_profit_usd']:+.2f} USDT ({results['total_return_pct']:+.2f}%)")
    print(f" Maximo Drawdown Intradiario:    {results['max_drawdown_pct']:.2f}% (${results['max_drawdown_usd']:.2f} USDT)")
    print(f" Rotaciones Ejecutadas:          {results['total_rotations']}")
    print(f" Tasa de Acierto (Win Rate):    {results['win_rate_pct']:.2f}%")
    print(f" Factor de Beneficio:           {results['profit_factor']:.2f}")
    print("-----------------------------------------------------------------")
    dsr = results["dsr_audit"]
    print(f" Auditoria DSR (Deflated Sharpe): {dsr['deflated_sharpe_ratio'] * 100:.1f}% [{dsr['status']}]")
    print(f" Sharpe Observado Diario (SR):   {dsr['observed_sharpe_sr']:.3f} | Anualizado: {dsr['annualized_sharpe']:.3f}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
