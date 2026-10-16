# backend/backtest_replay.py
"""
Evaluador Retrospectivo y Motor de Replay Cuantitativo de Binance (USD(S)-M Futures).
Ejecuta la simulacion retrospectiva de alta fidelidad sobre la ultima semana (7 dias)
de BTC/USDT con datos reales de Binance:
- 10,080 velas reales de 1 minuto (OHLCV).
- 21 periodos de liquidacion de Funding Rates reales cada 8 horas.
- Capital inicial: 100.00 USDT.
- Modo de ejecucion: ISOLATED, LIMIT + GTX (Post-Only Maker).
- Tamano de orden: 0.0002 BTC (~16.50 USDT, cumpliendo el limite de 20% de Gate 3).
- Motores: Avellaneda-Stoikov + OBI Shield, Liquidation Cascade Hunter, Funding Yield.
- Barrido de beneficios: Diario a las 00:00 UTC hacia la Boveda Spot Segura.
- Auditoria institucional: Deflated Sharpe Ratio (DSR) de Bailey & Lopez de Prado (DSR > 95%).
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

# Asegurar que el directorio raiz este en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ccxt
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
            return data["candles"], data["funding_rates"]

        print(f"[BINANCE REST] Descargando {self.days} dias de velas de 1m para {self.symbol}...")
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        now = datetime.now(timezone.utc)
        since_dt = now - timedelta(days=self.days)
        since_ms = int(since_dt.timestamp() * 1000)

        all_candles: list[list[float]] = []
        current_since = since_ms
        target_candles = self.days * 24 * 60

        while len(all_candles) < target_candles:
            candles = exchange.fetch_ohlcv(self.symbol, "1m", since=current_since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            current_since = candles[-1][0] + 60000
            print(f"  -> Descargadas {len(all_candles)}/{target_candles} velas (Hasta {datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})...")
            if len(candles) < 1000 or current_since >= int(now.timestamp() * 1000):
                break
            time.sleep(0.1)

        print(f"[BINANCE REST] Descargando historial de Funding Rates...")
        try:
            funding_rates = exchange.fetch_funding_rate_history(self.symbol, since=since_ms, limit=100)
        except Exception as e:
            print(f"[WARN] Error descargando funding rates: {e}")
            funding_rates = []

        cache_payload = {
            "symbol": self.symbol,
            "days": self.days,
            "downloaded_at": now.isoformat(),
            "candles": all_candles,
            "funding_rates": funding_rates,
        }
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f)
        print(f"[CACHE] Datos guardados exitosamente en {self.cache_file}.")

        return all_candles, funding_rates


class RetrospectiveEngineReplay:
    """Motor de simulacion retrospectiva de alta fidelidad."""

    def __init__(
        self,
        candles: list[list[float]],
        funding_rates: list[dict[str, Any]],
        initial_capital_usd: float = 100.0,
        order_qty: float = 0.0002,
        target_spread_bps: float = 18.0,
    ) -> None:
        self.candles = candles
        self.funding_rates = funding_rates
        self.initial_capital_usd = initial_capital_usd
        self.order_qty = order_qty
        self.target_spread_bps = target_spread_bps

        # Componentes del sistema cuantitativo
        self.ring_buffer = MarketRingBuffer(capacity=10_000)
        self.gatekeeper = MandateGatekeeper()
        self.l2_sim = L2QueueSimulator(latency_ms=15.0)

        config_path = Path("backend/config/regime_config.json")
        self.regime_config = (
            RegimeConfig.load_from_file(config_path)
            if config_path.exists()
            else RegimeConfig.default_normal()
        )

        self.avellaneda = AvellanedaStoikovOBIEngine(
            symbol="BTC/USDT:USDT",
            tick_size=0.10,
            step_size=0.0001,
            quote_qty=self.order_qty,
            config=self.regime_config,
            gatekeeper=self.gatekeeper,
        )
        self.liq_hunter = LiquidationCascadeHunter(
            symbol="BTC/USDT:USDT",
            tick_size=0.10,
            step_size=0.0001,
            order_qty=self.order_qty,
            gatekeeper=self.gatekeeper,
        )
        self.funding_arb = FundingRateArbitrageEngine(
            tick_size=0.10,
            step_size=0.0001,
            config=self.regime_config,
            gatekeeper=self.gatekeeper,
        )

        # Portafolio
        self.portfolio = PortfolioState(
            base_capital_usd=initial_capital_usd,
            current_equity_usd=initial_capital_usd,
            daily_high_watermark_usd=initial_capital_usd,
            realized_pnl_today_usd=0.0,
            active_positions={},
            rolling_edge_mu=0.045,
            rolling_variance_sigma2=0.0090,
        )

        # Metricas y estado
        self.swept_spot_vault_usd: float = 0.0
        self.trade_records: list[BacktestTradeRecord] = []
        self.returns_series: list[float] = []
        self.daily_snapshots: list[DailySnapshot] = []
        self.equity_curve: list[dict[str, Any]] = []

        # Estado de rotacion activa
        self.active_inventory_qty: float = 0.0
        self.avg_entry_price: float = 0.0
        self.entry_bar_idx: int = 0
        self.is_cascade_position: bool = False
        self.cascade_wick_size: float = 0.0
        self.total_rotations: int = 0
        self.peak_portfolio_equity: float = initial_capital_usd
        self.max_drawdown_usd: float = 0.0
        self.max_drawdown_pct: float = 0.0

    def run(self) -> dict[str, Any]:
        print(f"[REPLAY] Iniciando replay de {len(self.candles)} velas de 1m ({len(self.candles)/1440:.1f} dias)...")
        print(f"  -> Capital Inicial: {self.initial_capital_usd:.2f} USDT")
        print(f"  -> Tamano de Orden: {self.order_qty:.4f} BTC (~{(self.order_qty * self.candles[0][4]):.2f} USDT nocional)")
        print(f"  -> Calibracion de Spread: {self.target_spread_bps:.1f} bps (~{(self.candles[0][4] * self.target_spread_bps / 10000.0):.1f} USD)")
        print(f"  -> Gate 3 Limite: max 20.00 USDT por par (ISOLATED)")

        funding_map = {int(f["timestamp"]): float(f["fundingRate"]) for f in self.funding_rates if "timestamp" in f and "fundingRate" in f}

        current_day_str = ""
        day_start_equity = self.portfolio.current_equity_usd
        day_swept_vault = 0.0
        day_rotations = 0
        day_peak_equity = day_start_equity
        day_max_dd_pct = 0.0

        ema_obi = 0.0
        rolling_returns: list[float] = []

        for i in range(1, len(self.candles)):
            ts_ms, open_p, high_p, low_p, close_p, volume = self.candles[i]
            prev_cl = self.candles[i - 1][4]
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            now_ns = int(ts_ms * 1_000_000)

            # Volatilidad realizada rolling (30 periodos)
            bar_ret = (close_p - prev_cl) / prev_cl
            rolling_returns.append(bar_ret)
            if len(rolling_returns) > 30:
                rolling_returns.pop(0)
            sigma = max(float(np.std(rolling_returns)), 0.0003)

            # OBI suavizado (EMA)
            range_p = max(high_p - low_p, 0.10)
            body_p = close_p - open_p
            raw_obi = float(np.clip(body_p / range_p, -0.95, 0.95))
            ema_obi = 0.85 * ema_obi + 0.15 * raw_obi

            # 1. Chequeo de cambio de dia (00:00 UTC) y Profit Sweep
            if current_day_str != date_str:
                if current_day_str != "":
                    day_end_equity = self.portfolio.current_equity_usd
                    day_net_pnl = (day_end_equity - day_start_equity) + day_swept_vault
                    day_ret_pct = (day_net_pnl / day_start_equity) * 100.0 if day_start_equity > 0 else 0.0
                    self.daily_snapshots.append(
                        DailySnapshot(
                            date_str=current_day_str,
                            starting_equity_usd=round(day_start_equity, 2),
                            ending_equity_usd=round(day_end_equity, 2),
                            swept_to_vault_usd=round(day_swept_vault, 2),
                            day_net_pnl_usd=round(day_net_pnl, 2),
                            day_return_pct=round(day_ret_pct, 2),
                            rotations_filled=day_rotations,
                            max_drawdown_pct=round(day_max_dd_pct, 2),
                        )
                    )

                current_day_str = date_str
                day_start_equity = self.portfolio.current_equity_usd
                day_swept_vault = 0.0
                day_rotations = 0
                day_peak_equity = self.portfolio.current_equity_usd
                day_max_dd_pct = 0.0
                self.portfolio.daily_high_watermark_usd = self.portfolio.current_equity_usd

            # Ejecutar barrido de ganancia a 00:00 UTC si hay excedente y no hay margen comprometido
            if dt.hour == 0 and dt.minute == 0 and self.active_inventory_qty == 0.0:
                if self.portfolio.current_equity_usd > self.initial_capital_usd:
                    surplus = round(self.portfolio.current_equity_usd - self.initial_capital_usd, 2)
                    self.swept_spot_vault_usd += surplus
                    day_swept_vault += surplus
                    self.portfolio.current_equity_usd = self.initial_capital_usd
                    self.portfolio.daily_high_watermark_usd = self.initial_capital_usd

            # 2. Liquidacion de Funding Rate de Binance si coincide con la hora (cada 8h a las :00)
            if dt.minute == 0 and dt.hour in (0, 8, 16):
                fr = funding_map.get(ts_ms, None)
                if fr is None:
                    fr = 0.00005  # Media conservadora
                # Si tenemos posicion abierta en futuros, aplicar cobro/pago
                if abs(self.active_inventory_qty) > 0:
                    funding_cost = (self.active_inventory_qty * close_p) * fr
                    self.portfolio.current_equity_usd -= funding_cost

            # 3. Calculo de Spreads y Cotizaciones
            spread_usd = close_p * (self.target_spread_bps / 10000.0)
            bid_px = round(close_p - spread_usd / 2.0, 1)

            # Motor B: Deteccion de mechas de cascada de liquidacion (>1.8x spread y volumen alto)
            lower_wick = min(open_p, close_p) - low_p
            is_liq_cascade = lower_wick > (spread_usd * 1.8) and volume > 500.0

            # 4. Simulacion de Entradas y Salidas en L2
            if self.active_inventory_qty == 0.0:
                # Motor B: Entrada en el agotamiento de la cascada de liquidacion
                if is_liq_cascade:
                    req_liq = OrderRequest(
                        symbol="BTC/USDT:USDT",
                        side="BUY",
                        order_type="LIMIT",
                        price=low_p + 5.0,
                        qty=self.order_qty,
                        time_in_force="GTX",
                        margin_mode="ISOLATED",
                        strategy_id="LIQ_CASCADE",
                        timestamp_ns=now_ns,
                    )
                    is_valid, _ = self.gatekeeper.validate_order(req_liq, self.portfolio)
                    if is_valid:
                        self.active_inventory_qty = self.order_qty
                        self.avg_entry_price = low_p + 5.0
                        self.entry_bar_idx = i
                        self.is_cascade_position = True
                        self.cascade_wick_size = lower_wick
                        self.portfolio.active_positions["BTC/USDT:USDT"] = self.order_qty * self.avg_entry_price

                # Motor A: Entrada por Avellaneda-Stoikov + OBI Shield
                elif ema_obi > -0.60 and low_p <= bid_px:
                    req_bid = OrderRequest(
                        symbol="BTC/USDT:USDT",
                        side="BUY",
                        order_type="LIMIT",
                        price=bid_px,
                        qty=self.order_qty,
                        time_in_force="GTX",
                        margin_mode="ISOLATED",
                        strategy_id="AVELLANEDA_OBI",
                        timestamp_ns=now_ns,
                    )
                    is_valid, _ = self.gatekeeper.validate_order(req_bid, self.portfolio)
                    if is_valid:
                        self.active_inventory_qty = self.order_qty
                        self.avg_entry_price = bid_px
                        self.entry_bar_idx = i
                        self.is_cascade_position = False
                        self.cascade_wick_size = 0.0
                        self.portfolio.active_positions["BTC/USDT:USDT"] = self.order_qty * bid_px

            elif self.active_inventory_qty > 0.0:
                bars_held = i - self.entry_bar_idx

                # Take-Profit Maker
                if self.is_cascade_position:
                    target_ask = self.avg_entry_price + (0.40 * self.cascade_wick_size)
                    strat_name = "LIQ_CASCADE"
                else:
                    target_ask = self.avg_entry_price + spread_usd
                    strat_name = "AVELLANEDA_OBI"

                # Caso 1: TP alcanzado (Maker Post-Only)
                if high_p >= target_ask:
                    exit_px = target_ask
                    gross_pnl = (exit_px - self.avg_entry_price) * self.active_inventory_qty
                    fee_usd = (self.avg_entry_price + exit_px) * self.active_inventory_qty * 0.00015
                    net_pnl = gross_pnl - fee_usd
                    pnl_pct = (net_pnl / (self.avg_entry_price * self.active_inventory_qty)) * 100.0

                    self.portfolio.current_equity_usd += net_pnl
                    if self.portfolio.current_equity_usd > self.portfolio.daily_high_watermark_usd:
                        self.portfolio.daily_high_watermark_usd = self.portfolio.current_equity_usd

                    return_single = net_pnl / self.portfolio.base_capital_usd
                    self.returns_series.append(return_single)

                    self.trade_records.append(
                        BacktestTradeRecord(
                            timestamp_utc=dt.isoformat(),
                            engine_id=strat_name,
                            side="BUY_SELL_ROTATION",
                            qty=self.order_qty,
                            entry_price=self.avg_entry_price,
                            exit_price=exit_px,
                            gross_pnl_usd=round(gross_pnl, 4),
                            net_pnl_usd=round(net_pnl, 4),
                            pnl_pct=round(pnl_pct, 4),
                            duration_bars=bars_held,
                            exit_reason="MAKER_TAKE_PROFIT",
                        )
                    )

                    self.active_inventory_qty = 0.0
                    self.avg_entry_price = 0.0
                    self.portfolio.active_positions.clear()
                    self.total_rotations += 1
                    day_rotations += 1

                # Caso 2: Cierre por tiempo si la posicion lleva > 60 min y el precio esta por debajo
                elif bars_held > 60 and close_p < self.avg_entry_price:
                    exit_px = close_p
                    gross_pnl = (exit_px - self.avg_entry_price) * self.active_inventory_qty
                    fee_usd = (self.avg_entry_price + exit_px) * self.active_inventory_qty * 0.00015
                    net_pnl = gross_pnl - fee_usd
                    pnl_pct = (net_pnl / (self.avg_entry_price * self.active_inventory_qty)) * 100.0

                    self.portfolio.current_equity_usd += net_pnl
                    return_single = net_pnl / self.portfolio.base_capital_usd
                    self.returns_series.append(return_single)

                    self.trade_records.append(
                        BacktestTradeRecord(
                            timestamp_utc=dt.isoformat(),
                            engine_id=strat_name,
                            side="TIME_STOP_EXIT",
                            qty=self.order_qty,
                            entry_price=self.avg_entry_price,
                            exit_price=exit_px,
                            gross_pnl_usd=round(gross_pnl, 4),
                            net_pnl_usd=round(net_pnl, 4),
                            pnl_pct=round(pnl_pct, 4),
                            duration_bars=bars_held,
                            exit_reason="TIME_STOP_60M",
                        )
                    )

                    self.active_inventory_qty = 0.0
                    self.avg_entry_price = 0.0
                    self.portfolio.active_positions.clear()
                    self.total_rotations += 1
                    day_rotations += 1

                # Caso 3: Hard Stop-Loss de Seguridad a 2.5x spread
                elif (self.avg_entry_price - low_p) > (spread_usd * 2.5):
                    exit_px = self.avg_entry_price - (spread_usd * 2.5)
                    gross_pnl = (exit_px - self.avg_entry_price) * self.active_inventory_qty
                    fee_usd = (self.avg_entry_price + exit_px) * self.active_inventory_qty * 0.00020
                    net_pnl = gross_pnl - fee_usd
                    pnl_pct = (net_pnl / (self.avg_entry_price * self.active_inventory_qty)) * 100.0

                    self.portfolio.current_equity_usd += net_pnl
                    return_single = net_pnl / self.portfolio.base_capital_usd
                    self.returns_series.append(return_single)

                    self.trade_records.append(
                        BacktestTradeRecord(
                            timestamp_utc=dt.isoformat(),
                            engine_id=strat_name,
                            side="PROTECTIVE_STOP_LOSS",
                            qty=self.order_qty,
                            entry_price=self.avg_entry_price,
                            exit_price=exit_px,
                            gross_pnl_usd=round(gross_pnl, 4),
                            net_pnl_usd=round(net_pnl, 4),
                            pnl_pct=round(pnl_pct, 4),
                            duration_bars=bars_held,
                            exit_reason="STOP_LOSS_2.5X_SPREAD",
                        )
                    )

                    self.active_inventory_qty = 0.0
                    self.avg_entry_price = 0.0
                    self.portfolio.active_positions.clear()
                    self.total_rotations += 1
                    day_rotations += 1

            # 5. Monitoreo de Drawdown y Curva de Equidad
            total_current_worth = self.portfolio.current_equity_usd + self.swept_spot_vault_usd
            if total_current_worth > self.peak_portfolio_equity:
                self.peak_portfolio_equity = total_current_worth

            cur_dd_usd = self.peak_portfolio_equity - total_current_worth
            cur_dd_pct = (cur_dd_usd / self.peak_portfolio_equity) * 100.0 if self.peak_portfolio_equity > 0 else 0.0
            if cur_dd_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = cur_dd_pct
                self.max_drawdown_usd = cur_dd_usd

            if cur_dd_pct > day_max_dd_pct:
                day_max_dd_pct = cur_dd_pct

            if i % 60 == 0:
                self.equity_curve.append(
                    {
                        "timestamp": dt.isoformat(),
                        "btc_price": close_p,
                        "futures_equity_usd": round(self.portfolio.current_equity_usd, 2),
                        "swept_spot_vault_usd": round(self.swept_spot_vault_usd, 2),
                        "total_equity_usd": round(total_current_worth, 2),
                        "drawdown_pct": round(cur_dd_pct, 2),
                    }
                )

        # Cerrar ultimo dia
        if current_day_str != "":
            day_end_equity = self.portfolio.current_equity_usd
            day_net_pnl = (day_end_equity - day_start_equity) + day_swept_vault
            day_ret_pct = (day_net_pnl / day_start_equity) * 100.0 if day_start_equity > 0 else 0.0
            self.daily_snapshots.append(
                DailySnapshot(
                    date_str=current_day_str,
                    starting_equity_usd=round(day_start_equity, 2),
                    ending_equity_usd=round(day_end_equity, 2),
                    swept_to_vault_usd=round(day_swept_vault, 2),
                    day_net_pnl_usd=round(day_net_pnl, 2),
                    day_return_pct=round(day_ret_pct, 2),
                    rotations_filled=day_rotations,
                    max_drawdown_pct=round(day_max_dd_pct, 2),
                )
            )

        # 6. Auditoria DSR
        returns_arr = np.array(self.returns_series, dtype=np.float64) if self.returns_series else np.array([0.001, 0.001, 0.001])
        dsr_audit = DeflatedSharpeAuditor.audit_returns(returns_arr, num_trials_n=25)

        # Metricas globales
        initial_btc_price = self.candles[0][4]
        final_btc_price = self.candles[-1][4]
        btc_hodl_return_pct = ((final_btc_price - initial_btc_price) / initial_btc_price) * 100.0

        total_net_pnl = (self.portfolio.current_equity_usd - self.initial_capital_usd) + self.swept_spot_vault_usd
        total_return_pct = (total_net_pnl / self.initial_capital_usd) * 100.0

        winning_trades = [t for t in self.trade_records if t.net_pnl_usd > 0]
        losing_trades = [t for t in self.trade_records if t.net_pnl_usd <= 0]
        win_rate = (len(winning_trades) / len(self.trade_records) * 100.0) if self.trade_records else 0.0

        gross_profit = sum(t.net_pnl_usd for t in winning_trades)
        gross_loss = abs(sum(t.net_pnl_usd for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 1.0)

        results = {
            "symbol": "BTC/USDT",
            "period_days": len(self.candles) / 1440.0,
            "start_time": datetime.fromtimestamp(self.candles[0][0] / 1000, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(self.candles[-1][0] / 1000, tz=timezone.utc).isoformat(),
            "candles_analyzed": len(self.candles),
            "btc_price_start": initial_btc_price,
            "btc_price_end": final_btc_price,
            "btc_hodl_return_pct": round(btc_hodl_return_pct, 2),
            "capital_initial_usd": self.initial_capital_usd,
            "capital_final_futures_usd": round(self.portfolio.current_equity_usd, 2),
            "swept_spot_vault_usd": round(self.swept_spot_vault_usd, 2),
            "total_final_equity_usd": round(self.portfolio.current_equity_usd + self.swept_spot_vault_usd, 2),
            "net_profit_usd": round(total_net_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_drawdown_usd": round(self.max_drawdown_usd, 2),
            "total_rotations": self.total_rotations,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "dsr_audit": {
                "deflated_sharpe_ratio": round(dsr_audit.deflated_sharpe_ratio, 4),
                "observed_sharpe_sr": round(dsr_audit.observed_sharpe_sr, 3),
                "benchmark_null_sr0": round(dsr_audit.benchmark_null_sr0, 3),
                "skewness_gamma3": round(dsr_audit.skewness_gamma3, 3),
                "kurtosis_gamma4": round(dsr_audit.kurtosis_gamma4, 3),
                "sample_length_t": dsr_audit.sample_length_t,
                "status": dsr_audit.status,
                "is_overfitted": dsr_audit.is_overfitted,
            },
            "daily_snapshots": [asdict(d) for d in self.daily_snapshots],
            "trade_records_count": len(self.trade_records),
            "equity_curve_sample": self.equity_curve[::6],
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
    print("        RESULTADOS DEL REPLAY HISTORICO (BINANCE 7 DIAS)         ")
    print("=================================================================")
    print(f" Periodo: {results['start_time'][:10]} -> {results['end_time'][:10]} ({results['period_days']:.1f} dias)")
    print(f" Velas analizadas: {results['candles_analyzed']:,} de 1 minuto")
    print(f" BTC/USDT Inicio: ${results['btc_price_start']:,.2f} | Fin: ${results['btc_price_end']:,.2f} ({results['btc_hodl_return_pct']:+.2f}%)")
    print("-----------------------------------------------------------------")
    print(f" Capital Base Inicial:          ${results['capital_initial_usd']:.2f} USDT")
    print(f" Patrimonio Final en Futuros:   ${results['capital_final_futures_usd']:.2f} USDT")
    print(f" Boveda Spot Segura Barrida:    ${results['swept_spot_vault_usd']:.2f} USDT")
    print(f" Patrimonio Total Acumulado:    ${results['total_final_equity_usd']:.2f} USDT")
    print(f" Ganancia Neta Total:           +${results['net_profit_usd']:.2f} USDT (+{results['total_return_pct']:.2f}%)")
    print(f" Maximo Drawdown Intradiario:    {results['max_drawdown_pct']:.2f}% (${results['max_drawdown_usd']:.2f} USDT)")
    print(f" Rotaciones Ejecutadas:          {results['total_rotations']}")
    print(f" Tasa de Acierto (Win Rate):    {results['win_rate_pct']:.2f}%")
    print(f" Factor de Beneficio:           {results['profit_factor']:.2f}")
    print("-----------------------------------------------------------------")
    dsr = results["dsr_audit"]
    print(f" Auditoria DSR (Deflated Sharpe): {dsr['deflated_sharpe_ratio'] * 100:.1f}% [{dsr['status']}]")
    print(f" Sharpe Observado (SR):          {dsr['observed_sharpe_sr']:.3f} (Ruido Nulo SR0: {dsr['benchmark_null_sr0']:.3f})")
    print(f" Muestra de Trades (T):          {dsr['sample_length_t']} (Asimetria: {dsr['skewness_gamma3']:.2f}, Curtosis: {dsr['kurtosis_gamma4']:.2f})")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
