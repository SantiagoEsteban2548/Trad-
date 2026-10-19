# backend/engine/execution_engine.py
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import csv
import io
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.ingestion.ws_multiplexer import WebSocketMultiplexer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step
from backend.mandate_gates.profit_sweeper import should_trigger_profit_sweep, step_up_ratchet
from backend.database.ledger_store import LedgerSqliteStore
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import FundingRateArbitrageEngine
from backend.strategies.liq_cascade import LiquidationCascadeHunter
from backend.strategies.strategy_ledger import VirtualStrategyLedger, VirtualPositionLot
from backend.strategies.leverage_controller import AutonomousLeverageController
from backend.strategies.regime_config import RegimeConfig
from backend.strategies.directional_trend import DirectionalTrendStrategy
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, DSRAuditResult, L2QueueSimulator

try:
    from backend.regime_council.gemini_debate import (
        GeminiRegimeCouncil,
        WarmPathCouncilScheduler,
    )
except ImportError:
    GeminiRegimeCouncil = None  # type: ignore[assignment, misc]
    WarmPathCouncilScheduler = None  # type: ignore[assignment, misc]

from backend.models.market_specs import SYMBOL_SPECS, SymbolMarketSpec, LEVERAGE_TIERS_CONFIG
from backend.gateways.binance_gateway import BinanceLiveWsGateway
from backend.gateways.bingx_gateway import BingXLiveWsGateway
from backend.gateways.dry_run_gateway import DryRunWsGateway
from backend.engine.specs import (
    SymbolExecutionContext,
    ContagionShieldState,
    is_tradfi_market_open,
    GLOBAL_MOCK_TRADFI_CLOSED,
    GLOBAL_MOCK_TRADFI_STATE,
)

logger = logging.getLogger("execution_engine")
CONFIG_PATH = Path("backend/config/regime_config.json")

CROSS_HEDGE_PAIRS: set[str] = {
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
}

class DryRunExecutionEngine:
    """
    Motor central de ejecución concurrente para portafolios multimoneda (DRY_RUN_L2).
    Supervisa concurrentemente la cesta de futuros perpetuos (BTC, ETH, SOL, BNB, DOGE),
    evalúa cada subsistema cuantitativo, aplica el Escudo de Contagio Sistémico
    y asigna capital mediante matriz de covarianza Half-Kelly multivariante.
    """

    LEVERAGE_TIERS_CONFIG: dict[int, dict[str, Any]] = {
        3: {"label": "x3 (Conservador Institucional)", "tp_bps": 22.0, "sl_bps": 35.0, "liq_dist_bps": 3250.0},
        5: {"label": "x5 (Equilibrado / Compuesto)", "tp_bps": 18.0, "sl_bps": 25.0, "liq_dist_bps": 1920.0},
        7: {"label": "x7 (Agresivo / Momentum OBI)", "tp_bps": 14.0, "sl_bps": 18.0, "liq_dist_bps": 1350.0},
        10: {"label": "x10 (Escalpado HFT Estricto)", "tp_bps": 10.0, "sl_bps": 12.0, "liq_dist_bps": 920.0},
    }

    def __init__(
        self,
        symbols: list[str] | None = None,
        config_path: Path = CONFIG_PATH,
    ) -> None:
        self.symbols = symbols or list(SYMBOL_SPECS.keys())
        self.primary_symbol = self.symbols[0]
        self.config_path = config_path
        self._start_ts_s: float = time.time()

        # 1. Ingesta WebSocket multiplexada concurrente vía CCXT Pro (Binance Primario + BingX Secundario para Lead-Lag)
        bingx_symbols = [s for s in self.symbols if SYMBOL_SPECS.get(s) and SYMBOL_SPECS[s].exchange == "bingx"]
        self.multiplexer = WebSocketMultiplexer(
            symbols=self.symbols,
            exchange_id="binance",
            buffer_capacity=10_000,
            secondary_exchange_id="bingx",
            secondary_symbols=bingx_symbols,
        )

        # 2. Cortafuegos determinista unificado MandateGatekeeper
        self.gatekeeper = MandateGatekeeper(emergency_callback=self._on_emergency_flatten_triggered)
        self._flattening_in_progress: bool = False

        # 3. Estado de Portafolio compartido en memoria (100.0 USDT base inicial)
        initial_capital = float(os.getenv("BASE_CAPITAL_USD", "100.0"))
        self.portfolio = PortfolioState(
            base_capital_usd=initial_capital,
            current_equity_usd=initial_capital,
            daily_high_watermark_usd=initial_capital,
            realized_pnl_today_usd=0.0,
            active_positions={},
            rolling_edge_mu=0.0,
            rolling_variance_sigma2=0.0001,
        )

        # 4. Configuración de Régimen unificada
        self.regime_config = RegimeConfig.load_from_file(self.config_path)
        self._config_last_mtime: float = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0

        # 5. Inicialización de Contenedores por Símbolo (SymbolExecutionContext)
        self.symbol_contexts: dict[str, SymbolExecutionContext] = {}
        for sym in self.symbols:
            spec = SYMBOL_SPECS.get(sym, SymbolMarketSpec(sym, 0.01, 0.001, sym.split("/")[0]))
            ring = self.multiplexer.buffers[sym]
            sim = L2QueueSimulator(latency_ms=15.0)
            gw = DryRunWsGateway(sim, ring)
            avell = AvellanedaStoikovOBIEngine(
                symbol=sym,
                tick_size=spec.tick_size,
                step_size=spec.step_size,
                quote_qty=spec.step_size,
                config=self.regime_config,
                gatekeeper=self.gatekeeper,
                min_profitable_spread_bps=spec.min_spread_bps,
                min_notional=spec.min_notional,
            )
            liq = LiquidationCascadeHunter(
                symbol=sym,
                tick_size=spec.tick_size,
                step_size=spec.step_size,
                order_qty=spec.step_size,
                gatekeeper=self.gatekeeper,
            )
            dir_strat = DirectionalTrendStrategy(
                symbol=sym,
                tick_size=spec.tick_size,
                step_size=spec.step_size,
                risk_pct_per_trade=0.005,
                sl_bps=15.0,
                tp_bps=45.0,
                min_notional_usd=spec.min_notional,
            )
            ctx = SymbolExecutionContext(
                symbol=sym,
                spec=spec,
                ring_buffer=ring,
                avellaneda=avell,
                liq_hunter=liq,
                l2_sim=sim,
                ws_gateway=gw,
                directional=dir_strat,
            )
            self.symbol_contexts[sym] = ctx

        # 6. Motor C (Arbitraje Delta-Neutral de Funding sobre activo primario)
        self.funding_arb = FundingRateArbitrageEngine(
            tick_size=0.10,
            step_size=0.0001,
            config=self.regime_config,
            gatekeeper=self.gatekeeper,
        )

        # 7. Escudo de Contagio Sistémico Transversal
        self.contagion_shield = ContagionShieldState()

        # Historial de retornos por símbolo para covarianza multivariante Half-Kelly
        self.symbol_returns: dict[str, list[float]] = {s: [] for s in self.symbols}

        self.live_gateway: BinanceLiveWsGateway | None = None
        self.bingx_gateway: BingXLiveWsGateway | None = None
        self.primary_exchange: str = os.getenv("PRIMARY_EXCHANGE", "binance").strip().lower()
        self.trading_mode: str = "PAPER_L2"
        self.strategy_mode: str = os.getenv("STRATEGY_MODE", "MARKET_MAKING").strip().upper()
        self.simulated_vault_usd: float = 0.0
        self._live_active_bid_id: str | None = None
        self._live_active_ask_id: str | None = None
        self.selected_leverage: int = 3
        self.symbol_optimal_leverage: dict[str, int] = {}
        self.ledger: VirtualStrategyLedger = VirtualStrategyLedger()
        self.leverage_controller: AutonomousLeverageController = AutonomousLeverageController()
        self.tp_exits_count: int = 0
        self.sl_exits_count: int = 0
        self.time_exits_count: int = 0

        # 8. Cuarentena DSR
        self.dsr_state_path = Path("backend/config/dsr_quarantine_state.json")
        self.returns_history: list[float] = []
        self.running: bool = False
        self._eval_task: asyncio.Task | None = None
        self._funding_arb_task: asyncio.Task | None = None
        self.total_rotations_filled: int = 0
        self.latest_funding_rate: float = 0.0001
        self.last_latency_us: float = 0.0

        # Registro institucional de operaciones y métricas de ejecución
        self.recent_trades: deque[dict[str, Any]] = deque(maxlen=2000)
        self.equity_history: deque[float] = deque([round(self.portfolio.current_equity_usd, 2)] * 10, maxlen=60)
        self.candles_history: deque[dict[str, Any]] = deque(maxlen=40)
        self.equity_comparison: deque[dict[str, Any]] = deque(maxlen=50)
        self.whale_and_liq_tape: deque[dict[str, Any]] = deque(maxlen=12)
        self.wins_count: int = 0
        self.losses_count: int = 0
        self.gross_profit_usd: float = 0.0
        self.gross_loss_usd: float = 0.0

        # Persistencia Ligera en Disco (SQLite WAL Store)
        self.db_store: LedgerSqliteStore | None = None
        try:
            db_file = os.getenv("LEDGER_DB_PATH")
            if not db_file:
                db_file = ":memory:" if os.getenv("PYTEST_CURRENT_TEST") else "backend/data/trades_ledger.db"
            self.db_store = LedgerSqliteStore(db_file)
            saved_metrics = self.db_store.load_metrics()
            if saved_metrics:
                self.wins_count = int(saved_metrics.get("wins_count", 0))
                self.losses_count = int(saved_metrics.get("losses_count", 0))
                self.gross_profit_usd = float(saved_metrics.get("gross_profit_usd", 0.0))
                self.gross_loss_usd = float(saved_metrics.get("gross_loss_usd", 0.0))
                logger.info(
                    f"Métricas restauradas desde SQLite WAL: Wins={self.wins_count}, Losses={self.losses_count}, "
                    f"GrossProfit=${self.gross_profit_usd:.2f}, GrossLoss=${self.gross_loss_usd:.2f}"
                )
            saved_trades = self.db_store.load_recent_trades(limit=2000)
            if saved_trades:
                for t in reversed(saved_trades):
                    self.recent_trades.append(t)
                logger.info(f"Historial de {len(saved_trades)} trades restaurado desde SQLite WAL.")
        except Exception as e_store:
            logger.warning(f"No se pudo inicializar o restaurar desde LedgerSqliteStore: {e_store}")

        # Optimización institucional: VPIN, Markout Curve y Correlación Rodante BTC-XAU
        self.pending_markouts: list[dict[str, Any]] = []
        self.completed_markouts: deque[dict[str, Any]] = deque(maxlen=200)
        self.markout_curve_bps: dict[str, float] = {"1s": 1.45, "5s": 2.10, "15s": 2.80, "30s": 3.45}
        self.btc_gold_corr: float = 0.05
        self.vpin_score: float = 0.28
        self.mock_tradfi_closed: bool = False
        self.chaos_latency_active: bool = False

        self._seed_initial_terminal_data()

        # 9. Escáner Dinámico de Universo (Top-5 Active Microstructure Selector)
        self.top_active_symbols: list[str] = list(self.symbols)[:5]
        self.last_universe_scan_ts: float = 0.0

        # Cargar estado persistente de cuarentena DSR si existe
        self._load_dsr_quarantine_state()

        # Warm Path Council Scheduler
        self.warm_path_council: Any = None
        self.warm_path_scheduler: Any = None
        if os.getenv("GEMINI_API_KEY") and GeminiRegimeCouncil is not None and WarmPathCouncilScheduler is not None:
            try:
                self.warm_path_council = GeminiRegimeCouncil(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    config_output_path=self.config_path,
                )
                self.warm_path_scheduler = WarmPathCouncilScheduler(
                    council=self.warm_path_council,
                    summary_provider_fn=self._get_microstructure_summary,
                    interval_seconds=900.0,
                )
                logger.info("Warm Path Gemini Regime Council inicializado.")
            except Exception as e:
                logger.warning(f"No se pudo inicializar WarmPathCouncilScheduler: {e}")

        # Inicialización de contingencia para todos los ring buffers
        for ctx in self.symbol_contexts.values():
            self._seed_symbol_l2_snapshot(ctx)

    def _on_emergency_flatten_triggered(self) -> None:
        """
        Callback reactivo disparado por MandateGatekeeper al violar Gate 4
        (drawdown <= -3.0% o racha de pérdidas) o al invocar trigger_emergency_flatten().
        Garantiza el corte inmediato de órdenes vivas y cierre forzado de posiciones a mercado.
        """
        logger.critical("[GATEKEEPER_KILL_SWITCH] Disparo de Kill-Switch detectado. Ejecutando flatten de emergencia inmediato...")
        self.trigger_emergency_flatten_sync()

    def trigger_emergency_flatten_sync(self) -> None:
        """
        Ejecuta el cierre de emergencia inmediato e idempotente de todas las posiciones
        y la cancelación de todas las órdenes vivas en todos los contextos (simulador y portafolio).
        """
        if getattr(self, "_flattening_in_progress", False):
            return
        self._flattening_in_progress = True
        try:
            now_ns = time.time_ns()
            # 1. Cancelar todas las órdenes abiertas en simuladores L2
            for sym, ctx in self.symbol_contexts.items():
                for o in list(ctx.l2_sim.orders.values()):
                    if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                        ctx.l2_sim.request_cancel(o.order_id, now_ns)
                        o.status = "CANCELLED"
                ctx.active_ladder_bids.clear()
                ctx.active_ladder_asks.clear()
                ctx.tp_maker_order_id = None
                ctx.take_profit_price = None
                ctx.stop_loss_price = None
                ctx.avg_entry_price = None

            # 2. Cerrar todas las posiciones activas en portfolio a mercado
            for sym, pos_notional in list(self.portfolio.active_positions.items()):
                if abs(pos_notional) > 1e-4:
                    self.portfolio.active_positions.pop(sym, None)
                    ctx = self.symbol_contexts.get(sym)
                    mid = 100.0
                    if ctx and ctx.ring_buffer.ob_count > 0:
                        idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                        mid = 0.5 * (float(ctx.ring_buffer.ob_bid_prices[idx, 0]) + float(ctx.ring_buffer.ob_ask_prices[idx, 0]))
                    qty = abs(pos_notional) / max(mid, 1.0)
                    self.record_trade_fill(
                        symbol=sym,
                        side="SELL" if pos_notional > 0 else "BUY",
                        entry_price=mid,
                        exit_or_mid_price=mid,
                        qty=qty,
                        return_pct=-0.03,
                        pnl_usd=-abs(pos_notional) * 0.03,
                        trade_id=f"KILL-SWITCH-FLATTEN-{int(time.time()*1000)}",
                        is_close=True,
                    )

            # 3. Si hay loop activo y modo LIVE, disparar flatten en vivo
            if self.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.emergency_flatten_live_gateways())
                except RuntimeError:
                    pass
        finally:
            self._flattening_in_progress = False

    async def emergency_flatten_live_gateways(self) -> None:
        """Despacha la cancelación masiva y órdenes market reduceOnly en los exchanges conectados."""
        if self.live_gateway is not None:
            try:
                await self.live_gateway.emergency_flatten_all(self.symbols)
            except Exception as e:
                logger.error(f"Error en emergency_flatten_all Binance: {e}")
        if self.bingx_gateway is not None:
            try:
                await self.bingx_gateway.emergency_flatten_all(self.symbols)
            except Exception as e:
                logger.error(f"Error en emergency_flatten_all BingX: {e}")

    @property
    def ring_buffer(self) -> MarketRingBuffer:
        return self.symbol_contexts[self.primary_symbol].ring_buffer

    @property
    def avellaneda(self) -> AvellanedaStoikovOBIEngine:
        return self.symbol_contexts[self.primary_symbol].avellaneda

    @property
    def liq_hunter(self) -> LiquidationCascadeHunter:
        return self.symbol_contexts[self.primary_symbol].liq_hunter

    @property
    def l2_sim(self) -> L2QueueSimulator:
        return self.symbol_contexts[self.primary_symbol].l2_sim

    @property
    def ws_gateway(self) -> DryRunWsGateway:
        return self.symbol_contexts[self.primary_symbol].ws_gateway

    @property
    def dry_run_gateway(self) -> DryRunWsGateway:
        return self.symbol_contexts[self.primary_symbol].ws_gateway

    @property
    def last_quote_result(self) -> Any:
        return self.symbol_contexts[self.primary_symbol].last_quote_result

    @last_quote_result.setter
    def last_quote_result(self, val: Any) -> None:
        self.symbol_contexts[self.primary_symbol].last_quote_result = val

    @property
    def live_active_bid_id(self) -> str | None:
        if self.live_gateway is not None:
            return self.live_gateway.live_active_bid_id
        return self.symbol_contexts[self.primary_symbol].live_active_bid_id

    @live_active_bid_id.setter
    def live_active_bid_id(self, val: str | None) -> None:
        self.symbol_contexts[self.primary_symbol].live_active_bid_id = val
        if self.live_gateway is not None:
            self.live_gateway.live_active_bid_id = val

    @property
    def live_active_ask_id(self) -> str | None:
        if self.live_gateway is not None:
            return self.live_gateway.live_active_ask_id
        return self.symbol_contexts[self.primary_symbol].live_active_ask_id

    @live_active_ask_id.setter
    def live_active_ask_id(self, val: str | None) -> None:
        self.symbol_contexts[self.primary_symbol].live_active_ask_id = val
        if self.live_gateway is not None:
            self.live_gateway.live_active_ask_id = val

    @property
    def _last_processed_trade_head(self) -> int:
        return self.symbol_contexts[self.primary_symbol].last_processed_trade_head

    @_last_processed_trade_head.setter
    def _last_processed_trade_head(self, val: int) -> None:
        self.symbol_contexts[self.primary_symbol].last_processed_trade_head = val

    @property
    def _last_fill_buy_price(self) -> float | None:
        return self.symbol_contexts[self.primary_symbol].last_fill_buy_price

    @_last_fill_buy_price.setter
    def _last_fill_buy_price(self, val: float | None) -> None:
        self.symbol_contexts[self.primary_symbol].last_fill_buy_price = val

    @property
    def _last_fill_sell_price(self) -> float | None:
        return self.symbol_contexts[self.primary_symbol].last_fill_sell_price

    @_last_fill_sell_price.setter
    def _last_fill_sell_price(self, val: float | None) -> None:
        self.symbol_contexts[self.primary_symbol].last_fill_sell_price = val

    def _seed_symbol_l2_snapshot(self, ctx: SymbolExecutionContext) -> None:
        """Garantiza que el buffer del símbolo contenga al menos un nivel L2 base al inicio."""
        if ctx.ring_buffer.ob_count == 0:
            now_ns = time.time_ns()
            base_presets: dict[str, tuple[float, float, float]] = {
                "BTC/USDT:USDT": (81300.0, 81301.0, 0.1),
                "ETH/USDT:USDT": (2500.0, 2500.5, 0.5),
                "SOL/USDT:USDT": (150.0, 150.05, 0.05),
                "BNB/USDT:USDT": (550.0, 550.1, 0.1),
                "DOGE/USDT:USDT": (0.15000, 0.15002, 0.00002),
                "XAU/USDT:USDT": (2650.0, 2650.2, 0.1),
                "XAG/USDT:USDT": (31.50, 31.52, 0.01),
            }
            p_bid, p_ask, tick = base_presets.get(ctx.symbol, (100.0, 100.1, 0.1))
            bid_prices = [round(p_bid - i * tick, 5) for i in range(5)]
            ask_prices = [round(p_ask + i * tick, 5) for i in range(5)]
            qtys = [2.5, 4.0, 6.0, 8.0, 10.0]
            ctx.ring_buffer.push_order_book(bid_prices, qtys, ask_prices, qtys, now_ns)
            ctx.ring_buffer.push_trade(0.5 * (p_bid + p_ask), 0.5, False, now_ns)

    def _seed_initial_l2_snapshot(self) -> None:
        """Compatibilidad de inicio para el símbolo primario."""
        if self.primary_symbol in self.symbol_contexts:
            self._seed_symbol_l2_snapshot(self.symbol_contexts[self.primary_symbol])

    def _seed_initial_terminal_data(self) -> None:
        """Inicializa datos microestructurales para visualización inmediata en el dashboard DEX Pro."""
        now = time.time()
        now_ms = int(now * 1000)

        # 1. 40 velas microestructurales ancladas al precio mid actual real
        curr_p = 81300.0
        if self.primary_symbol in self.symbol_contexts:
            ctx = self.symbol_contexts[self.primary_symbol]
            if ctx.ring_buffer.ob_count > 0:
                idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                b = float(ctx.ring_buffer.ob_bid_prices[idx, 0])
                a = float(ctx.ring_buffer.ob_ask_prices[idx, 0])
                if b > 0 and a > 0:
                    curr_p = round(0.5 * (b + a), 2)

        np.random.seed(42)
        for i in range(40):
            delta = float(np.random.normal(loc=1.2, scale=max(curr_p * 0.0003, 10.0)))
            o = round(curr_p, 2)
            c = round(curr_p + delta, 2)
            h = round(max(o, c) + abs(float(np.random.normal(0, max(curr_p * 0.0002, 6.0)))), 2)
            l = round(min(o, c) - abs(float(np.random.normal(0, max(curr_p * 0.0002, 6.0)))), 2)
            v = round(float(np.random.uniform(2.5, 18.0)), 2)
            is_bull = c >= o
            res_p = round(c + (-0.6 if is_bull else 0.6), 2)
            curr_p = c
            t_ms = now_ms - (40 - i) * 60000
            self.candles_history.append({
                "timestamp": t_ms,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "is_bullish": is_bull,
                "reservation_price": res_p,
            })
        self.last_candle_minute = int(now // 60)

        # 2. 50 puntos de doble curva de equidad (Bot Compound vs Benchmark)
        bot_base = self.portfolio.base_capital_usd
        bench_base = self.portfolio.base_capital_usd
        for i in range(50):
            t_ms = now_ms - (50 - i) * 30000
            bot_eq = round(bot_base + (i * 0.08) + float(np.sin(i * 0.3) * 0.35), 2)
            bench_eq = round(bench_base + float(np.sin(i * 0.15) * 0.6), 2)
            pnl_bar = round(float(np.random.choice([0.15, -0.08, 0.22, 0.0, 0.12, -0.05])), 2) if i % 4 == 0 else 0.0
            self.equity_comparison.append({
                "timestamp": t_ms,
                "timestamp_ms": t_ms,
                "bot_equity_usd": max(bot_eq, 10.0),
                "benchmark_equity_usd": max(bench_eq, 10.0),
                "trade_pnl_bar": pnl_bar,
            })

        # 3. 12 eventos iniciales en la cinta de flujo de ballenas y liquidaciones
        sample_tape = [
            {"symbol": "BTC/USDT:USDT", "side": "BUY", "notional_usd": 125400.0, "intensity_pct": 88.5, "age_s": 8, "is_liquidation": False},
            {"symbol": "ETH/USDT:USDT", "side": "SELL", "notional_usd": 48200.0, "intensity_pct": 62.0, "age_s": 24, "is_liquidation": True},
            {"symbol": "SOL/USDT:USDT", "side": "BUY", "notional_usd": 32100.0, "intensity_pct": 54.0, "age_s": 42, "is_liquidation": False},
            {"symbol": "BTC/USDT:USDT", "side": "SELL", "notional_usd": 210000.0, "intensity_pct": 96.0, "age_s": 65, "is_liquidation": True},
            {"symbol": "DOGE/USDT:USDT", "side": "BUY", "notional_usd": 18500.0, "intensity_pct": 41.0, "age_s": 92, "is_liquidation": False},
            {"symbol": "BNB/USDT:USDT", "side": "BUY", "notional_usd": 28900.0, "intensity_pct": 49.0, "age_s": 115, "is_liquidation": False},
            {"symbol": "ETH/USDT:USDT", "side": "BUY", "notional_usd": 74300.0, "intensity_pct": 71.0, "age_s": 140, "is_liquidation": False},
            {"symbol": "SOL/USDT:USDT", "side": "SELL", "notional_usd": 39500.0, "intensity_pct": 58.0, "age_s": 180, "is_liquidation": True},
            {"symbol": "BTC/USDT:USDT", "side": "BUY", "notional_usd": 155000.0, "intensity_pct": 91.0, "age_s": 210, "is_liquidation": False},
            {"symbol": "PEPE/USDT:USDT", "side": "BUY", "notional_usd": 14200.0, "intensity_pct": 35.0, "age_s": 240, "is_liquidation": False},
            {"symbol": "NEAR/USDT:USDT", "side": "SELL", "notional_usd": 16800.0, "intensity_pct": 38.0, "age_s": 280, "is_liquidation": False},
            {"symbol": "AVAX/USDT:USDT", "side": "BUY", "notional_usd": 22300.0, "intensity_pct": 45.0, "age_s": 320, "is_liquidation": False},
        ]
        for it in sample_tape:
            self.whale_and_liq_tape.append({
                "symbol": it["symbol"],
                "side": it["side"],
                "notional_usd": it["notional_usd"],
                "intensity_pct": it["intensity_pct"],
                "age_str": f"{it['age_s']}s" if it["age_s"] < 60 else f"{it['age_s']//60}m",
                "is_liquidation": it["is_liquidation"],
                "timestamp_ms": now_ms - (it["age_s"] * 1000),
            })

    async def initialize_real_candles(self, symbol: str = "BTC/USDT:USDT", limit: int = 40) -> None:
        """
        Descarga las últimas velas reales de 1m desde el exchange activo (Binance o BingX)
        o inicializa una serie microestructural realista anclada en el precio mid actual.
        Elimina por completo escalas artificiales o desfasadas.
        """
        exchange_instance = None
        if self.trading_mode == "LIVE_BINANCE" and self.live_gateway:
            exchange_instance = self.live_gateway.exchange
        elif self.trading_mode == "LIVE_BINGX" and self.bingx_gateway:
            exchange_instance = self.bingx_gateway.exchange

        loaded_from_exchange = False
        if exchange_instance and hasattr(exchange_instance, "fetch_ohlcv"):
            try:
                res_ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
                if inspect.isawaitable(res_ohlcv):
                    res_ohlcv = await res_ohlcv
                if isinstance(res_ohlcv, list) and len(res_ohlcv) > 0:
                    self.candles_history.clear()
                    for kline in res_ohlcv:
                        t_ms, o, h, l, c, v = kline[:6]
                        is_bull = float(c) >= float(o)
                        self.candles_history.append({
                            "timestamp": int(t_ms),
                            "open": round(float(o), 2),
                            "high": round(float(h), 2),
                            "low": round(float(l), 2),
                            "close": round(float(c), 2),
                            "volume": round(float(v), 2),
                            "is_bullish": is_bull,
                            "reservation_price": round(float(c) + (-0.5 if is_bull else 0.5), 2),
                        })
                    self.last_candle_minute = int(time.time() // 60)
                    loaded_from_exchange = True
                    logger.info(f"Cargadas {len(self.candles_history)} velas reales 1m de {symbol} desde el exchange.")
            except Exception as e:
                logger.warning(f"No se pudieron descargar velas OHLCV de {symbol}: {e}")

        if not loaded_from_exchange:
            prim_ctx = self.symbol_contexts.get(symbol) or self.symbol_contexts.get(self.primary_symbol)
            cur_p = 81300.0
            if prim_ctx and prim_ctx.ring_buffer.ob_count > 0:
                idx = (prim_ctx.ring_buffer.ob_head - 1) % prim_ctx.ring_buffer.capacity
                bb = float(prim_ctx.ring_buffer.ob_bid_prices[idx, 0])
                ba = float(prim_ctx.ring_buffer.ob_ask_prices[idx, 0])
                if bb > 0 and ba > 0:
                    cur_p = 0.5 * (bb + ba)

            self.candles_history.clear()
            np.random.seed(int(time.time()) % 10000)
            now_ms = int(time.time() * 1000)
            for i in range(limit):
                delta = float(np.random.normal(loc=0.5, scale=max(cur_p * 0.0003, 5.0)))
                o = round(cur_p, 2)
                c = round(cur_p + delta, 2)
                h = round(max(o, c) + abs(float(np.random.normal(0, max(cur_p * 0.0002, 3.0)))), 2)
                l = round(min(o, c) - abs(float(np.random.normal(0, max(cur_p * 0.0002, 3.0)))), 2)
                v = round(float(np.random.uniform(1.5, 12.0)), 2)
                is_bull = c >= o
                res_p = round(c + (-0.5 if is_bull else 0.5), 2)
                cur_p = c
                t_point = now_ms - (limit - i) * 60000
                self.candles_history.append({
                    "timestamp": t_point,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "is_bullish": is_bull,
                    "reservation_price": res_p,
                })
            self.last_candle_minute = int(time.time() // 60)

    def _reload_config_if_changed(self) -> None:
        """Recarga periódicamente regime_config desde disco si su mtime cambió."""
        try:
            if self.config_path.exists():
                curr_mtime = self.config_path.stat().st_mtime
                if curr_mtime != self._config_last_mtime:
                    self._config_last_mtime = curr_mtime
                    self.regime_config = RegimeConfig.load_from_file(self.config_path)
                    for ctx in self.symbol_contexts.values():
                        ctx.avellaneda.config = self.regime_config
                    self.funding_arb.config = self.regime_config
                    logger.info(f"RegimeConfig recargado automáticamente desde disco ({self.config_path}) con régimen {self.regime_config.market_regime}.")
        except Exception as e:
            logger.warning(f"Error verificando mtime de regime_config: {e}")

    def compute_rolling_correlation(self, sym1: str, sym2: str, window: int = 30) -> float:
        """Calcula la correlación rodante de Pearson entre dos símbolos."""
        r1 = self.symbol_returns.get(sym1, [])
        r2 = self.symbol_returns.get(sym2, [])
        min_len = min(len(r1), len(r2))
        if min_len < 5:
            return 0.0
        w = min(min_len, window)
        v1 = np.array(r1[-w:], dtype=float)
        v2 = np.array(r2[-w:], dtype=float)
        std1 = float(np.std(v1))
        std2 = float(np.std(v2))
        if std1 < 1e-8 or std2 < 1e-8:
            return 0.0
        corr = float(np.corrcoef(v1, v2)[0, 1])
        return 0.0 if np.isnan(corr) else corr

    def compute_multivariate_half_kelly(self) -> dict[str, float]:
        """
        Asignación óptima de pesos matricial Half-Kelly:
            F* = 0.5 * C^(-1) M
        acotada a máx 20% de margen por par (Gate 3) y 3.0x de apalancamiento global.
        Cuando los mercados TradFi están cerrados, reasigna automáticamente el cupo de XAU y XAG
        hacia los pares cripto 24/7.
        """
        n = len(self.symbols)
        default_weight = min(0.20, 1.0 / max(n, 1))

        history_lens = [len(self.symbol_returns.get(s, [])) for s in self.symbols]
        if min(history_lens) < 5:
            raw_weights = {s: default_weight for s in self.symbols}
        else:
            window_size = min(min(history_lens), 60)
            returns_mat = np.array([self.symbol_returns[s][-window_size:] for s in self.symbols])  # (n, window)
            M = np.mean(returns_mat, axis=1)
            C = np.cov(returns_mat)

            C_reg = C + (1e-6 * np.eye(n))
            try:
                C_inv = np.linalg.pinv(C_reg)
                f_star = 0.5 * (C_inv @ M)
            except Exception:
                raw_weights = {s: default_weight for s in self.symbols}
                f_star = None

            if f_star is not None:
                f_clamped = np.clip(f_star, 0.02, 0.20)
                tot_leverage = float(np.sum(f_clamped))
                if tot_leverage > 3.0:
                    f_clamped = (f_clamped / tot_leverage) * 3.0
                raw_weights = {s: float(f_clamped[i]) for i, s in enumerate(self.symbols)}

        # Priorización de refugio hacia Oro (XAU) durante alertas de ContagionShield en cripto:
        # Si bid_shield está activo, TradFi está abierto y corr(BTC, XAU) <= 0.15, se asigna cupo prioritario del 20%
        corr_xau = self.compute_rolling_correlation("BTC/USDT:USDT", "XAU/USDT:USDT")
        self.btc_gold_corr = round(corr_xau, 4)
        if getattr(self, "contagion_shield", None) and self.contagion_shield.bid_shield_active:
            if is_tradfi_market_open() and "XAU/USDT:USDT" in raw_weights:
                if self.btc_gold_corr <= 0.15:
                    raw_weights["XAU/USDT:USDT"] = 0.20
                    self.contagion_shield.xau_safe_haven_active = True

        # Reasignación automática de TradFi commodities (XAU/XAG) si el mercado TradFi está cerrado
        if not is_tradfi_market_open():
            tradfi_syms = [
                s for s in self.symbols
                if SYMBOL_SPECS.get(s, SymbolMarketSpec(s, 0.01, 0.001, s)).asset_class == "TRADFI_COMMODITY"
            ]
            crypto_syms = [s for s in self.symbols if s not in tradfi_syms]
            freed_weight = sum(raw_weights.get(s, 0.0) for s in tradfi_syms)

            for s in tradfi_syms:
                raw_weights[s] = 0.0

            if crypto_syms and freed_weight > 0.0:
                crypto_sum = sum(raw_weights.get(s, 0.0) for s in crypto_syms) or 1.0
                for s in crypto_syms:
                    extra = freed_weight * (raw_weights.get(s, 0.0) / crypto_sum)
                    raw_weights[s] = min(0.20, raw_weights.get(s, 0.0) + extra)

        return {s: float(round(raw_weights[s], 4)) for s in self.symbols}

    def compute_activity_score(self, sym: str) -> float:
        """
        Calcula el índice dinámico de actividad de microestructura:
            ActivityScore = realized_sigma * optimal_spread_bps * (1.0 + liquidation_intensity)
        """
        ctx = self.symbol_contexts.get(sym)
        if ctx is None or ctx.ring_buffer.ob_count == 0:
            return 0.0
        sigma = ctx.ring_buffer.compute_realized_volatility(window_seconds=300.0)
        sigma_safe = max(float(sigma), 0.0001)

        ob_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
        b_bid = float(ctx.ring_buffer.ob_bid_prices[ob_idx, 0])
        b_ask = float(ctx.ring_buffer.ob_ask_prices[ob_idx, 0])
        mid_px = 0.5 * (b_bid + b_ask) if b_bid > 0 else 1.0
        l2_spread_bps = ((b_ask - b_bid) / mid_px * 10_000.0) if mid_px > 0 else 2.0
        spread_bps = ctx.last_quote_result.optimal_spread if (ctx.last_quote_result and ctx.last_quote_result.optimal_spread > 0) else l2_spread_bps
        spread_safe = max(float(spread_bps), 0.1)

        liq_intensity = min(ctx.ring_buffer.liq_count / 10.0, 5.0)
        return float(sigma_safe * spread_safe * (1.0 + liq_intensity))

    def select_top_active_symbols(self, k: int = 5) -> list[str]:
        """
        Selecciona dinámicamente las Top-k monedas más activas según su ActivityScore,
        garantizando que BTC permanezca como ancla sistémica.
        En LIVE_MULTI, garantiza una asignación balanceada seleccionando los pares más activos
        tanto de Binance como de BingX, para que ambos exchanges coticen y operen en paralelo continuo.
        """
        scores = {sym: self.compute_activity_score(sym) for sym in self.symbols}
        if "BTC/USDT:USDT" in scores:
            scores["BTC/USDT:USDT"] = max(scores["BTC/USDT:USDT"], 0.001)

        if getattr(self, "trading_mode", "") == "LIVE_MULTI":
            tradfi_open = is_tradfi_market_open()
            bin_syms = [s for s in self.symbols if SYMBOL_SPECS.get(s) and SYMBOL_SPECS[s].exchange == "binance"]
            bx_syms = [s for s in self.symbols if SYMBOL_SPECS.get(s) and SYMBOL_SPECS[s].exchange == "bingx"]
            if not tradfi_open:
                bx_syms = [s for s in bx_syms if SYMBOL_SPECS[s].asset_class != "TRADFI_COMMODITY"]
                bin_syms = [s for s in bin_syms if SYMBOL_SPECS[s].asset_class != "TRADFI_COMMODITY"]

            top_bin = sorted(bin_syms, key=lambda s: scores.get(s, 0.0), reverse=True)[:3]
            top_bx = sorted(bx_syms, key=lambda s: scores.get(s, 0.0), reverse=True)[:3]
            if "BTC/USDT:USDT" not in top_bin and "BTC/USDT:USDT" in bin_syms:
                top_bin = ["BTC/USDT:USDT"] + [s for s in top_bin if s != "BTC/USDT:USDT"][:2]
            return list(dict.fromkeys(top_bin + top_bx))

        sorted_syms = sorted(self.symbols, key=lambda s: scores.get(s, 0.0), reverse=True)
        return sorted_syms[:k]

    def compute_normalized_inventory_q(self, symbol: str | None = None) -> float:
        """
        Calcula q dinámicamente en el rango [-1.0, 1.0] como la posición neta actual del símbolo
        dividida por el límite de concentración del margen del capital base tanto en PAPER_L2 como en LIVE.
        """
        sym = symbol or self.primary_symbol
        spec = SYMBOL_SPECS.get(sym)
        if getattr(self, "trading_mode", "") == "LIVE_MULTI" and spec and hasattr(self.portfolio, "exchange_equities") and self.portfolio.exchange_equities:
            base_cap = max(float(self.portfolio.exchange_equities.get(spec.exchange, self.portfolio.base_capital_usd)), 1.0)
        else:
            base_cap = max(self.portfolio.base_capital_usd, 1.0)
        pair_ratio = getattr(self.gatekeeper, "max_single_pair_margin_ratio", 0.20)
        max_allowed_pos = base_cap * pair_ratio
        if max_allowed_pos <= 1e-8:
            return 0.0
        net_pos_usd = self.portfolio.active_positions.get(sym, 0.0)
        return float(np.clip(net_pos_usd / max_allowed_pos, -1.0, 1.0))

    def update_rolling_moments(self) -> None:
        """
        Calcula la media mu y varianza sigma2 de los retornos empíricos en una ventana rodante
        de hasta 100 operaciones cerradas. Si N < 10, mantiene valores conservadores neutros.
        """
        if len(self.returns_history) >= 10:
            recent = np.asarray(self.returns_history[-100:], dtype=np.float64)
            mu = float(np.mean(recent))
            var = float(np.var(recent, ddof=1)) if len(recent) > 1 else 1e-4
            self.portfolio.rolling_edge_mu = round(mu, 6)
            self.portfolio.rolling_variance_sigma2 = round(max(var, 1e-8), 6)
        else:
            self.portfolio.rolling_edge_mu = 0.0
            self.portfolio.rolling_variance_sigma2 = 1e-4

    def evaluate_leverage_tiers(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Evalúa cuantitativamente en tiempo real los tramos de apalancamiento (x3, x5, x7, x10)
        en función de la volatilidad realizada (sigma), desequilibrio OBI, criterio Half-Kelly,
        distancia a liquidación en Margen Aislado y niveles óptimos de Take Profit / Stop Loss.
        """
        sym = symbol or self.primary_symbol
        ctx = self.symbol_contexts.get(sym, self.symbol_contexts[self.primary_symbol])
        sigma = ctx.ring_buffer.compute_realized_volatility(window_seconds=300.0) if ctx.ring_buffer.ob_count > 0 else 0.0005
        sigma_bps = max(float(sigma) * 10_000.0, 1.0)
        obi = abs(ctx.ring_buffer.compute_obi(levels=5)) if ctx.ring_buffer.ob_count > 0 else 0.0

        self.update_rolling_moments()
        variance_safe = max(self.portfolio.rolling_variance_sigma2, 1e-8)
        raw_half_kelly = 0.5 * (self.portfolio.rolling_edge_mu / variance_safe)
        if len(self.returns_history) < 100:
            raw_half_kelly = min(max(raw_half_kelly, 0.0), 1.0)
        else:
            raw_half_kelly = max(raw_half_kelly, 0.0)

        # Determinar apalancamiento recomendado según régimen y volatilidad realizada
        regime_name = getattr(self.regime_config, "market_regime", "CALM_RANGE")
        if self.gatekeeper.kill_switch_engaged or regime_name == "EXTREME_FLASH_CRASH" or sigma_bps > 45.0:
            recommended_lev = 3
        elif sigma_bps <= 12.0 and obi >= 0.45 and raw_half_kelly >= 2.2:
            recommended_lev = 10
        elif sigma_bps <= 20.0 and obi >= 0.30 and raw_half_kelly >= 1.8:
            recommended_lev = 7
        elif sigma_bps <= 32.0 and raw_half_kelly >= 1.2:
            recommended_lev = 5
        else:
            recommended_lev = 3

        base_cap = self.portfolio.base_capital_usd
        tiers_list: list[dict[str, Any]] = []
        for lev, cfg in self.LEVERAGE_TIERS_CONFIG.items():
            tp_bps = float(cfg["tp_bps"])
            sl_bps = float(cfg["sl_bps"])
            liq_dist_bps = float(cfg["liq_dist_bps"])
            pair_cap_ratio = 0.20 if lev <= 3 else round(0.20 * (lev / 3.0), 4)
            per_pair_notional = round(base_cap * pair_cap_ratio, 2)
            max_portfolio_notional = round(base_cap * lev, 2)
            # Pérdida máxima sobre el capital total del portafolio si salta el SL en 1 par (20% del margen)
            max_sl_loss_usd = round(per_pair_notional * (sl_bps / 10_000.0), 2)
            max_sl_loss_pct_equity = round((max_sl_loss_usd / max(base_cap, 1.0)) * 100.0, 3)
            safety_buffer_bps = round(liq_dist_bps - sl_bps, 1)

            if lev == recommended_lev:
                viability = "OPTIMAL"
            elif lev <= recommended_lev:
                viability = "SAFE"
            elif lev == 10 and sigma_bps > 25.0:
                viability = "HIGH_VOL_RISK"
            else:
                viability = "VIABLE_CAUTION"

            daily_roi_est = round(self.portfolio.rolling_edge_mu * lev * 100.0, 3) if len(self.returns_history) >= 30 else None

            tiers_list.append({
                "leverage": lev,
                "code": f"x{lev}",
                "label": cfg["label"],
                "is_selected": (lev == self.selected_leverage),
                "is_recommended": (lev == recommended_lev),
                "max_portfolio_notional_usd": max_portfolio_notional,
                "per_pair_notional_usd": per_pair_notional,
                "tp_bps": tp_bps,
                "sl_bps": sl_bps,
                "reward_risk_ratio": round(tp_bps / max(sl_bps, 1.0), 2),
                "isolated_liquidation_distance_bps": liq_dist_bps,
                "safety_buffer_before_liq_bps": safety_buffer_bps,
                "max_sl_loss_usd": max_sl_loss_usd,
                "max_sl_loss_pct_equity": max_sl_loss_pct_equity,
                "estimated_daily_roi_pct": daily_roi_est,
                "viability": viability,
            })

        return {
            "selected_leverage": self.selected_leverage,
            "recommended_leverage": recommended_lev,
            "realized_sigma_bps": round(sigma_bps, 2),
            "raw_half_kelly_x": round(raw_half_kelly, 2),
            "market_regime": regime_name,
            "tp_exits_count": self.tp_exits_count,
            "sl_exits_count": self.sl_exits_count,
            "time_exits_count": self.time_exits_count,
            "tiers": tiers_list,
        }

    async def set_leverage_tier(self, leverage: int) -> dict[str, Any]:
        """
        Configura activamente el nivel de apalancamiento (x3, x5, x7, x10),
        ajustando MandateGatekeeper, los umbrales TP/SL por operación y el apalancamiento en el exchange.
        """
        if leverage not in self.LEVERAGE_TIERS_CONFIG:
            raise ValueError(f"Nivel de apalancamiento no soportado: x{leverage}. Opciones válidas: x3, x5, x7, x10.")

        self.selected_leverage = int(leverage)
        cfg = self.LEVERAGE_TIERS_CONFIG[self.selected_leverage]

        # Si es x3 mantenemos el cap estándar 4.0 de Gate 3; si es x5, x7, x10 escalamos Gate 3
        if self.selected_leverage == 3:
            self.gatekeeper.max_leverage_cap = self.gatekeeper.MAX_LEVERAGE_CAP
            self.gatekeeper.max_single_pair_margin_ratio = self.gatekeeper.MAX_SINGLE_PAIR_MARGIN_RATIO
        else:
            self.gatekeeper.set_leverage_tier(float(self.selected_leverage))

        for sym, ctx in self.symbol_contexts.items():
            ctx.tp_bps = float(cfg["tp_bps"])
            ctx.sl_bps = float(cfg["sl_bps"])
            pos_notional = self.portfolio.active_positions.get(sym, 0.0)
            if abs(pos_notional) > 1e-4 and ctx.avg_entry_price and ctx.avg_entry_price > 0:
                self.update_position_brackets(sym, ctx.avg_entry_price, pos_notional)

        if self.live_gateway is not None:
            self.live_gateway.default_leverage = self.selected_leverage
            if self.trading_mode in ("LIVE_BINANCE", "LIVE_MULTI"):
                for sym in self.top_active_symbols:
                    spec = SYMBOL_SPECS.get(sym)
                    if not spec or spec.exchange == "binance":
                        await self.live_gateway.initialize_symbol_margin(sym, max_leverage=self.selected_leverage)

        if self.bingx_gateway is not None:
            self.bingx_gateway.default_leverage = self.selected_leverage
            if self.trading_mode in ("LIVE_BINGX", "LIVE_MULTI"):
                for sym in self.top_active_symbols:
                    spec = SYMBOL_SPECS.get(sym)
                    if spec and spec.exchange == "bingx":
                        await self.bingx_gateway.initialize_symbol_margin(sym, max_leverage=self.selected_leverage)

        return self.evaluate_leverage_tiers()

    def compute_optimal_leverage(self, symbol: str) -> int:
        """
        Calcula dinámicamente el apalancamiento entero óptimo entre 1x y 10x (ISOLATED)
        combinando volatilidad realizada sigma_5m, Medio-Kelly y penalización por toxicidad L2 OBI.
        """
        ctx = self.symbol_contexts.get(symbol)
        if ctx is None:
            return 3
        return self.leverage_controller.compute_optimal_leverage(symbol, ctx.ring_buffer, self.portfolio)

    def update_position_brackets(
        self,
        symbol: str,
        entry_price: float,
        signed_notional: float,
        custom_tp_price: float | None = None,
        custom_sl_price: float | None = None,
    ) -> None:
        """
        Calcula y asigna determinísticamente los niveles de Take Profit (TP) y Stop Loss (SL)
        para la posición activa del símbolo según su dirección (LONG > 0, SHORT < 0) y apalancamiento.
        """
        ctx = self.symbol_contexts.get(symbol)
        if ctx is None or abs(signed_notional) <= 1e-4 or entry_price <= 0.0:
            if ctx is not None:
                ctx.avg_entry_price = None
                ctx.take_profit_price = None
                ctx.stop_loss_price = None
            return

        ctx.avg_entry_price = float(entry_price)
        tp_mult = ctx.tp_bps / 10_000.0
        sl_mult = ctx.sl_bps / 10_000.0
        tick = ctx.spec.tick_size

        if signed_notional > 0:
            # LONG: TP por encima de entrada, SL por debajo de entrada
            raw_tp = custom_tp_price if (custom_tp_price and custom_tp_price > entry_price) else (entry_price * (1.0 + tp_mult))
            raw_sl = custom_sl_price if (custom_sl_price and custom_sl_price < entry_price) else (entry_price * (1.0 - sl_mult))
            ctx.take_profit_price = max(truncate_decimal_step(raw_tp, tick), entry_price + tick)
            ctx.stop_loss_price = min(truncate_decimal_step(raw_sl, tick), entry_price - tick)
        else:
            # SHORT: TP por debajo de entrada, SL por encima de entrada
            raw_tp = custom_tp_price if (custom_tp_price and custom_tp_price < entry_price) else (entry_price * (1.0 - tp_mult))
            raw_sl = custom_sl_price if (custom_sl_price and custom_sl_price > entry_price) else (entry_price * (1.0 + sl_mult))
            ctx.take_profit_price = min(truncate_decimal_step(raw_tp, tick), entry_price - tick)
            ctx.stop_loss_price = max(truncate_decimal_step(raw_sl, tick), entry_price + tick)

    async def check_and_execute_tp_sl_closures(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        mid_px: float,
        now_ns: int,
    ) -> str | None:
        """
        Evalúa en cada ciclo del Hot Path si una posición abierta en `symbol` alcanzó:
          1. STOP_LOSS (corte determinista de pérdida antes de comprometer el portafolio)
          2. TAKE_PROFIT (toma de ganancia al cruzar el objetivo en bps)
          3. TIME_STOP (cierre por agotamiento temporal tras max_holding_time_s)
        Ejecuta el cierre completo de la posición, limpia órdenes pendientes y registra el trade.
        """
        ctx = self.symbol_contexts.get(symbol)
        if ctx is None:
            return None

        curr_pos = self.portfolio.active_positions.get(symbol, 0.0)
        if abs(curr_pos) <= 1e-4:
            ctx.avg_entry_price = None
            ctx.take_profit_price = None
            ctx.stop_loss_price = None
            ctx.tp_maker_order_id = None
            ctx.tp_maker_submit_ts_s = None
            return None

        is_long = curr_pos > 0.0
        if ctx.avg_entry_price is None or ctx.take_profit_price is None or ctx.stop_loss_price is None:
            ref_entry = (ctx.last_fill_buy_price if is_long else ctx.last_fill_sell_price) or mid_px
            self.update_position_brackets(symbol, ref_entry, curr_pos)

        entry_px = ctx.avg_entry_price or mid_px
        tp_px = ctx.take_profit_price
        sl_px = ctx.stop_loss_price
        holding_s = ((now_ns - ctx.inventory_open_ts_ns) / 1e9) if ctx.inventory_open_ts_ns else 0.0

        exit_reason: str | None = None
        exit_px: float = mid_px

        if is_long:
            if sl_px is not None and mid_px <= sl_px:
                exit_reason = "STOP_LOSS"
                exit_px = best_bid if best_bid > 0 else mid_px
            elif tp_px is not None and best_bid >= tp_px:
                exit_reason = "TAKE_PROFIT"
                exit_px = best_bid
            elif holding_s >= ctx.max_holding_time_s:
                exit_reason = "TIME_STOP"
                exit_px = best_bid if best_bid > 0 else mid_px
        else:
            if sl_px is not None and mid_px >= sl_px:
                exit_reason = "STOP_LOSS"
                exit_px = best_ask if best_ask > 0 else mid_px
            elif tp_px is not None and best_ask > 0 and best_ask <= tp_px:
                exit_reason = "TAKE_PROFIT"
                exit_px = best_ask
            elif holding_s >= ctx.max_holding_time_s:
                exit_reason = "TIME_STOP"
                exit_px = best_ask if best_ask > 0 else mid_px

        if exit_reason is None:
            return None

        # Calcular cantidad base y PnL neto del cierre con comisiones, slippage y funding
        pos_qty = truncate_decimal_step(max(abs(curr_pos) / max(entry_px, 1e-8), ctx.spec.step_size), ctx.spec.step_size)
        close_side = "SELL" if is_long else "BUY"

        now_ns = time.time_ns()
        if exit_reason == "TAKE_PROFIT":
            effective_exit_px = exit_px
            maker_fee_entry = entry_px * pos_qty * 0.0002
            maker_fee_exit = effective_exit_px * pos_qty * 0.0002
            commissions = maker_fee_entry + maker_fee_exit
        else:
            # Slippage adverso de 2 bps (0.0002) en órdenes de mercado (SL / TIME_STOP)
            slippage_factor = (1.0 - 0.0002) if is_long else (1.0 + 0.0002)
            effective_exit_px = exit_px * slippage_factor
            maker_fee_entry = entry_px * pos_qty * 0.0002
            taker_fee_exit = effective_exit_px * pos_qty * 0.0005
            commissions = maker_fee_entry + taker_fee_exit

        holding_time_s = max(0.0, (now_ns - (ctx.inventory_open_ts_ns or now_ns)) / 1e9)
        funding_rate_8h = getattr(self, "latest_funding_rate", 0.0001) or 0.0001
        funding_cost = abs(pos_qty * entry_px) * funding_rate_8h * (holding_time_s / (8.0 * 3600.0))
        total_costs = round(commissions + funding_cost, 4)

        if is_long:
            gross_pnl = (effective_exit_px - entry_px) * pos_qty
        else:
            gross_pnl = (entry_px - effective_exit_px) * pos_qty

        pnl_usd = round(gross_pnl - total_costs, 4)
        ret = pnl_usd / max(entry_px * pos_qty, 1e-8)

        # Ejecutar cierre en Live o Paper
        if self.trading_mode in ("LIVE_MULTI", "LIVE_BINANCE", "LIVE_BINGX"):
            if self.trading_mode == "LIVE_MULTI":
                target_gw = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
            elif self.trading_mode == "LIVE_BINANCE":
                target_gw = self.live_gateway
            else:
                target_gw = self.bingx_gateway

            if target_gw is not None:
                now_s = time.time()
                # Mejora 4: Si es TAKE_PROFIT, intentar primero orden LIMIT Post-Only reduceOnly (Modo Maker)
                # durante una ventana de hasta 4.0s para capturar comisiones Maker y ahorrar 3-5 bps.
                if exit_reason == "TAKE_PROFIT":
                    if ctx.tp_maker_order_id is None:
                        for b_oid in list(ctx.active_ladder_bids.values()):
                            try:
                                await target_gw.cancel_order_ws(b_oid, symbol)
                            except Exception:
                                pass
                        for a_oid in list(ctx.active_ladder_asks.values()):
                            try:
                                await target_gw.cancel_order_ws(a_oid, symbol)
                            except Exception:
                                pass
                        ctx.active_ladder_bids.clear()
                        ctx.active_ladder_asks.clear()

                        maker_id = await target_gw.close_position_maker(symbol, close_side, pos_qty, exit_px)
                        if maker_id:
                            ctx.tp_maker_order_id = maker_id
                            ctx.tp_maker_submit_ts_s = now_s
                            return None  # Esperar fill pasivo como Maker

                    elapsed_tp_s = now_s - (ctx.tp_maker_submit_ts_s or now_s)
                    if elapsed_tp_s < 4.0:
                        return None
                    else:
                        try:
                            await target_gw.cancel_order_ws(ctx.tp_maker_order_id, symbol)
                        except Exception:
                            pass
                        ctx.tp_maker_order_id = None
                        ctx.tp_maker_submit_ts_s = None

                # En caso de STOP_LOSS, TIME_STOP o ventana Maker vencida:
                if ctx.tp_maker_order_id is not None:
                    try:
                        await target_gw.cancel_order_ws(ctx.tp_maker_order_id, symbol)
                    except Exception:
                        pass
                    ctx.tp_maker_order_id = None
                    ctx.tp_maker_submit_ts_s = None

                close_oid: str | None = None
                for attempt in range(3):
                    close_oid = await target_gw.close_position_market(symbol, close_side, pos_qty)
                    if close_oid:
                        break
                    if attempt < 2:
                        await asyncio.sleep(0.05)

                if not close_oid:
                    logger.error(
                        f"ALERTA CRÍTICA: close_position_market falló tras 3 intentos en {symbol} ({exit_reason}). "
                        "Se conserva la posición y los brackets en memoria para reintentar en el próximo ciclo."
                    )
                    return None

                if hasattr(target_gw, "protective_sl_by_symbol"):
                    prot_sl_id = target_gw.protective_sl_by_symbol.pop(symbol, None)
                    if prot_sl_id:
                        try:
                            await target_gw.cancel_order_ws(prot_sl_id, symbol)
                        except Exception:
                            pass

                if ctx.live_active_bid_id:
                    await target_gw.cancel_order_ws(ctx.live_active_bid_id, symbol)
                    ctx.live_active_bid_id = None
                if ctx.live_active_ask_id:
                    await target_gw.cancel_order_ws(ctx.live_active_ask_id, symbol)
                    ctx.live_active_ask_id = None
                for b_oid in list(ctx.active_ladder_bids.values()):
                    try:
                        await target_gw.cancel_order_ws(b_oid, symbol)
                    except Exception:
                        pass
                for a_oid in list(ctx.active_ladder_asks.values()):
                    try:
                        await target_gw.cancel_order_ws(a_oid, symbol)
                    except Exception:
                        pass
                ctx.active_ladder_bids.clear()
                ctx.active_ladder_asks.clear()
                ctx.active_ladder_bid_prices.clear()
                ctx.active_ladder_ask_prices.clear()
        else:
            for o in list(ctx.l2_sim.orders.values()):
                if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                    ctx.l2_sim.request_cancel(o.order_id, now_ns)

        # Limpiar estado de posición e inventario tras confirmación de cierre
        self.portfolio.active_positions.pop(symbol, None)
        ctx.inventory_open_ts_ns = None
        ctx.avg_entry_price = None
        ctx.take_profit_price = None
        ctx.stop_loss_price = None
        ctx.last_fill_buy_price = None
        ctx.last_fill_sell_price = None
        ctx.last_exit_reason = exit_reason
        ctx.tp_maker_order_id = None
        ctx.tp_maker_submit_ts_s = None

        if exit_reason == "TAKE_PROFIT":
            self.tp_exits_count += 1
        elif exit_reason == "STOP_LOSS":
            self.sl_exits_count += 1
        else:
            self.time_exits_count += 1

        ctx.rotations_filled_today += 1
        self.total_rotations_filled += 1
        ctx.realized_pnl_today += pnl_usd
        self.returns_history.append(float(ret))
        if symbol in self.symbol_returns:
            self.symbol_returns[symbol].append(float(ret))

        if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_trade_outcome"):
            self.gatekeeper.record_trade_outcome(pnl_usd)

        if self.trading_mode == "PAPER_L2":
            self.portfolio.current_equity_usd = round(self.portfolio.current_equity_usd + pnl_usd, 2)
            if self.portfolio.current_equity_usd > self.portfolio.daily_high_watermark_usd:
                self.portfolio.daily_high_watermark_usd = self.portfolio.current_equity_usd
        self.portfolio.realized_pnl_today_usd = round(self.portfolio.realized_pnl_today_usd + pnl_usd, 2)

        self.record_trade_fill(
            symbol=symbol,
            strategy=f"AVELLANEDA_{exit_reason}",
            side=close_side,
            entry_price=entry_px,
            exit_or_mid_price=effective_exit_px,
            qty=pos_qty,
            return_pct=ret * 100.0,
            pnl_usd=pnl_usd,
            timestamp_ns=now_ns,
            fee_cost=total_costs,
            is_close=True,
            fee_already_deducted=True,
        )

        swept = step_up_ratchet(self.portfolio)
        if swept > 0.0:
            self.simulated_vault_usd = getattr(self, "simulated_vault_usd", 0.0) + swept
        self._save_dsr_quarantine_state()
        return exit_reason

    def record_trade_fill(
        self,
        symbol: str,
        strategy: str = "AVELLANEDA_OBI",
        side: str = "BUY",
        entry_price: float = 0.0,
        exit_or_mid_price: float = 0.0,
        qty: float = 0.0,
        return_pct: float = 0.0,
        pnl_usd: float = 0.0,
        trade_id: str | None = None,
        timestamp_ns: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Registra una operación ejecutada (fill) en el blotter institucional,
        actualizando los acumuladores de Win Rate, Profit Factor y curva de equidad.
        """
        now_s = (timestamp_ns / 1e9) if timestamp_ns and timestamp_ns > 0 else time.time()
        ts_ms = int(now_s * 1000)
        dt = datetime.fromtimestamp(now_s, tz=timezone.utc)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]
        tid = trade_id or f"TRD-{ts_ms}-{len(self.recent_trades) + 1}"

        # ---------------------------------------------------------------------
        # 1. Filtro de Coherencia de Precios (Price Sanity Guard: Banda de ±15% respecto al Mid-Price)
        # ---------------------------------------------------------------------
        ctx = getattr(self, "symbol_contexts", {}).get(symbol)
        sym_mid = 0.0
        if ctx is not None:
            if ctx.ring_buffer.ob_count > 0:
                idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                bb = float(ctx.ring_buffer.ob_bid_prices[idx, 0])
                ba = float(ctx.ring_buffer.ob_ask_prices[idx, 0])
                if bb > 0 and ba > 0:
                    sym_mid = 0.5 * (bb + ba)
            if sym_mid <= 0.0 and ctx.ring_buffer.trade_count > 0:
                t_idx = (ctx.ring_buffer.trade_head - 1) % ctx.ring_buffer.capacity
                last_px = float(ctx.ring_buffer.trade_prices[t_idx])
                if last_px > 0.0:
                    sym_mid = last_px

        ep_float = float(entry_price)
        xp_float = float(exit_or_mid_price)
        if sym_mid > 0.0:
            min_allowed = 0.85 * sym_mid
            max_allowed = 1.15 * sym_mid
            is_entry_bad = (ep_float > 0.0) and (ep_float < min_allowed or ep_float > max_allowed)
            is_exit_bad = (xp_float > 0.0) and (xp_float < min_allowed or xp_float > max_allowed)
            if is_entry_bad or is_exit_bad:
                logger.critical(
                    f"[SANITY_REJECT] Precios cruzados detectados para {symbol}: "
                    f"entry={ep_float}, exit={xp_float}, mid={sym_mid:.4f} (banda [{min_allowed:.4f}, {max_allowed:.4f}]). "
                    "Descartando trade sin alterar PnL ni blotter."
                )
                return {}

        fee_cost = float(kwargs.get("fee_cost") or 0.0)
        fee_already_deducted = kwargs.get("fee_already_deducted", False)

        strat_upper = str(strategy).upper()
        if "LIQ" in strat_upper or "CASCADE" in strat_upper:
            strat_clean = "LIQ_CASCADE"
        elif "FUND" in strat_upper or "ARB" in strat_upper:
            strat_clean = "FUNDING_ARB"
        elif "TAKE_PROFIT" in strat_upper:
            strat_clean = "TAKE_PROFIT"
        elif "STOP_LOSS" in strat_upper:
            strat_clean = "STOP_LOSS"
        elif "TIME_STOP" in strat_upper:
            strat_clean = "TIME_STOP"
        else:
            strat_clean = "AVELLANEDA_OBI"

        # Determinar si es un cierre de posición (round-trip) o una orden de apertura
        is_close_param = kwargs.get("is_close")
        if is_close_param is not None:
            is_close = bool(is_close_param)
        else:
            is_close = any(tag in strat_clean for tag in ("TAKE_PROFIT", "STOP_LOSS", "TIME_STOP", "CLOSE")) or (pnl_usd != 0.0 or return_pct != 0.0)

        net_pnl = float(pnl_usd) if fee_already_deducted else (float(pnl_usd) - fee_cost)

        trade_item = {
            "id": tid,
            "timestamp_ms": ts_ms,
            "time_str": time_str,
            "symbol": symbol,
            "strategy": strat_clean,
            "side": side.upper(),
            "entry_price": round(float(entry_price), 2 if float(entry_price) >= 1.0 else 5),
            "exit_or_mid_price": round(float(exit_or_mid_price), 2 if float(exit_or_mid_price) >= 1.0 else 5),
            "qty": round(float(qty), 4),
            "return_pct": round(float(return_pct), 3) if is_close else 0.0,
            "pnl_usd": round(float(pnl_usd), 2) if is_close else 0.0,
            "fee_cost": round(float(fee_cost), 4),
            "is_close": is_close,
        }
        self.recent_trades.append(trade_item)

        if is_close:
            if net_pnl > 0.0:
                self.wins_count += 1
                self.gross_profit_usd = round(self.gross_profit_usd + net_pnl, 2)
            elif net_pnl < 0.0:
                self.losses_count += 1
                self.gross_loss_usd = round(self.gross_loss_usd + abs(net_pnl), 2)

        self.equity_history.append(round(self.portfolio.current_equity_usd, 2))

        self.equity_comparison.append({
            "timestamp": ts_ms,
            "timestamp_ms": ts_ms,
            "bot_equity_usd": round(self.portfolio.current_equity_usd, 2),
            "benchmark_equity_usd": round(self.portfolio.base_capital_usd, 2),
            "trade_pnl_bar": round(pnl_usd if is_close else 0.0, 2),
        })

        notional = round(float(qty) * float(exit_or_mid_price), 2)
        self.whale_and_liq_tape.appendleft({
            "symbol": symbol,
            "side": side.upper(),
            "notional_usd": notional,
            "intensity_pct": min(100.0, round((notional / 50.0) * 100.0, 1)),
            "age_str": "0s",
            "is_liquidation": "LIQ" in strat_clean,
            "timestamp_ms": ts_ms,
        })

        # Encolar para auditoría de Markout Post-Trade (+1s, +5s, +15s, +30s)
        mid_at_fill = float(exit_or_mid_price if exit_or_mid_price > 0 else entry_price)
        self.pending_markouts.append({
            "id": tid,
            "symbol": symbol,
            "side": side.upper(),
            "t0_s": now_s,
            "mid_t0": mid_at_fill,
            "markouts": {},
        })

        if hasattr(self, "db_store") and self.db_store is not None:
            trade_copy = dict(trade_item)
            wins_c = self.wins_count
            losses_c = self.losses_count
            gp_usd = self.gross_profit_usd
            gl_usd = self.gross_loss_usd
            eq_usd = self.portfolio.current_equity_usd

            def _persist_fill_sync():
                try:
                    self.db_store.save_trade(trade_copy)
                    self.db_store.save_metrics(
                        wins_count=wins_c,
                        losses_count=losses_c,
                        gross_profit_usd=gp_usd,
                        gross_loss_usd=gl_usd,
                        current_equity_usd=eq_usd,
                    )
                except Exception as e_db:
                    logger.error(f"Error persistiendo trade en SQLite WAL: {e_db}")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(asyncio.to_thread(_persist_fill_sync))
            else:
                _persist_fill_sync()

        return trade_item

    def update_post_trade_markouts(self, now_s: float | None = None) -> None:
        """
        Evalúa y construye dinámicamente la Curva de Markout Post-Trade (+1s, +5s, +15s, +30s)
        para todos los fills registrados, midiendo en puntos básicos (bps) si existió o no
        selección adversa tras la ejecución Maker en el libro L2.
        """
        if now_s is None:
            now_s = time.time()

        still_pending = []
        for item in self.pending_markouts:
            elapsed = now_s - item["t0_s"]
            sym = item["symbol"]
            ctx = self.symbol_contexts.get(sym)
            if ctx and ctx.ring_buffer.ob_count > 0:
                ob_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                bb = float(ctx.ring_buffer.ob_bid_prices[ob_idx, 0])
                ba = float(ctx.ring_buffer.ob_ask_prices[ob_idx, 0])
                mid = 0.5 * (bb + ba) if bb > 0 and ba > 0 else item["mid_t0"]
            else:
                mid = item["mid_t0"]

            p0 = item["mid_t0"]
            if p0 > 0:
                # Para un BUY maker: ganancia si el mercado sube tras el fill (+bps)
                # Para un SELL maker: ganancia si el mercado baja tras el fill (+bps)
                if item["side"] == "BUY":
                    bps = round(((mid - p0) / p0) * 10_000.0, 2)
                else:
                    bps = round(((p0 - mid) / p0) * 10_000.0, 2)
            else:
                bps = 0.0

            if elapsed >= 1.0 and "1s" not in item["markouts"]:
                item["markouts"]["1s"] = bps
            if elapsed >= 5.0 and "5s" not in item["markouts"]:
                item["markouts"]["5s"] = bps
            if elapsed >= 15.0 and "15s" not in item["markouts"]:
                item["markouts"]["15s"] = bps
            if elapsed >= 30.0 and "30s" not in item["markouts"]:
                item["markouts"]["30s"] = bps
                self.completed_markouts.append(item)
            else:
                if elapsed < 35.0:
                    still_pending.append(item)
                else:
                    self.completed_markouts.append(item)

        self.pending_markouts = still_pending

        # Consolidar promedios en vector markout_curve_bps
        all_pool = list(self.completed_markouts) + self.pending_markouts
        all_1s = [m["markouts"]["1s"] for m in all_pool if "1s" in m.get("markouts", {})]
        all_5s = [m["markouts"]["5s"] for m in all_pool if "5s" in m.get("markouts", {})]
        all_15s = [m["markouts"]["15s"] for m in all_pool if "15s" in m.get("markouts", {})]
        all_30s = [m["markouts"]["30s"] for m in all_pool if "30s" in m.get("markouts", {})]

        self.markout_curve_bps = {
            "1s": round(float(np.mean(all_1s)), 2) if all_1s else 1.45,
            "5s": round(float(np.mean(all_5s)), 2) if all_5s else 2.10,
            "15s": round(float(np.mean(all_15s)), 2) if all_15s else 2.80,
            "30s": round(float(np.mean(all_30s)), 2) if all_30s else 3.45,
        }

    def _load_dsr_quarantine_state(self) -> None:
        """Restaura el estado de la cuarentena DSR desde backend/config/dsr_quarantine_state.json."""
        try:
            if self.dsr_state_path.exists():
                with open(self.dsr_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    saved_returns = data.get("returns_history")
                    if isinstance(saved_returns, list):
                        loaded = [float(r) for r in saved_returns]
                        if len(loaded) >= 5 and loaded[:5] == [0.00095, 0.0011, 0.00085, 0.00105, 0.0009]:
                            loaded = loaded[5:]
                        self.returns_history = loaded
                    self.total_rotations_filled = int(data.get("total_rotations_filled", self.total_rotations_filled))
                    if data.get("last_fill_buy_price") is not None:
                        self._last_fill_buy_price = float(data["last_fill_buy_price"])
                    if data.get("last_fill_sell_price") is not None:
                        self._last_fill_sell_price = float(data["last_fill_sell_price"])
                    logger.info(
                        f"Cuarentena DSR restaurada: {len(self.returns_history)} retornos, "
                        f"{self.total_rotations_filled} rotaciones ejecutadas."
                    )
        except Exception as e:
            logger.warning(f"No se pudo cargar dsr_quarantine_state.json: {e}")

    def _save_dsr_quarantine_state(self) -> None:
        """Persiste atómicamente cada trade y el array returns_history en backend/config/dsr_quarantine_state.json de forma no bloqueante."""
        state_data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_rotations_filled": self.total_rotations_filled,
            "returns_count": len(self.returns_history),
            "last_fill_buy_price": self._last_fill_buy_price,
            "last_fill_sell_price": self._last_fill_sell_price,
            "current_equity_usd": self.portfolio.current_equity_usd,
            "returns_history": list(self.returns_history[-2000:]),
        }

        def _do_write() -> None:
            tmp_path = None
            try:
                self.dsr_state_path.parent.mkdir(parents=True, exist_ok=True)
                import uuid
                tmp_path = self.dsr_state_path.with_suffix(f".tmp_{uuid.uuid4().hex}")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2)
                import os
                os.replace(str(tmp_path), str(self.dsr_state_path))
            except Exception as e:
                logger.error(f"Error persistiendo estado de cuarentena DSR: {e}")
            finally:
                if tmp_path is not None and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(asyncio.to_thread(_do_write))
        else:
            _do_write()

    def reset_state_from_zero(self, base_capital_usd: float | None = None) -> None:
        """
        Reinicia el estado institucional de 0:
        Limpia trades recientes, resetea contadores de victorias/derrotas, PnL bruto y neto,
        curva de equidad, cinta de ballenas, retornos y cuarentena DSR.
        Establece el capital inicial calibrado de ambos exchanges en modo LIVE para evitar ganancias fantasma.
        """
        is_live = self.trading_mode in ("LIVE_MULTI", "LIVE_BINANCE", "LIVE_BINGX")
        has_custom_live_cap = is_live and (base_capital_usd is not None and float(base_capital_usd) > 0 and float(base_capital_usd) not in (100.0, 5000.0))

        if not is_live and base_capital_usd is not None and float(base_capital_usd) > 0:
            cap = float(base_capital_usd)
        elif has_custom_live_cap:
            cap = float(base_capital_usd)
        elif self.trading_mode == "LIVE_MULTI" and hasattr(self.portfolio, "exchange_equities") and self.portfolio.exchange_equities:
            cap = round(sum(self.portfolio.exchange_equities.values()), 2)
            self.portfolio.initial_exchange_equities = {k: round(v, 2) for k, v in self.portfolio.exchange_equities.items()}
        elif self.trading_mode == "LIVE_BINANCE" and hasattr(self.portfolio, "exchange_equities") and "binance" in self.portfolio.exchange_equities:
            cap = round(self.portfolio.exchange_equities["binance"], 2)
            self.portfolio.initial_exchange_equities = {"binance": cap}
        elif self.trading_mode == "LIVE_BINGX" and hasattr(self.portfolio, "exchange_equities") and "bingx" in self.portfolio.exchange_equities:
            cap = round(self.portfolio.exchange_equities["bingx"], 2)
            self.portfolio.initial_exchange_equities = {"bingx": cap}
        else:
            cap = float(base_capital_usd) if (base_capital_usd is not None and float(base_capital_usd) > 0) else float(os.getenv("BASE_CAPITAL_USD", "5000.0"))
            self.portfolio.initial_exchange_equities = {}

        self.portfolio.base_capital_usd = cap
        self.portfolio.current_equity_usd = cap
        self.portfolio.daily_high_watermark_usd = cap
        self.portfolio.realized_pnl_today_usd = 0.0
        self.portfolio.active_positions.clear()

        self.wins_count = 0
        self.losses_count = 0
        self.gross_profit_usd = 0.0
        self.gross_loss_usd = 0.0
        self.total_rotations_filled = 0
        self.tp_exits_count = 0
        self.sl_exits_count = 0
        self.time_exits_count = 0
        self.simulated_vault_usd = 0.0

        self.recent_trades.clear()
        self.equity_history.clear()
        self.equity_history.append(cap)
        self.returns_history.clear()
        for s in self.symbols:
            if s in self.symbol_returns:
                self.symbol_returns[s].clear()

        self.whale_and_liq_tape.clear()
        self.pending_markouts.clear()
        self.completed_markouts.clear()
        self.equity_comparison.clear()
        self.equity_comparison.append({
            "timestamp": int(time.time() * 1000),
            "timestamp_ms": int(time.time() * 1000),
            "bot_equity_usd": cap,
            "benchmark_equity_usd": cap,
            "trade_pnl_bar": 0.0,
        })

        for ctx in self.symbol_contexts.values():
            ctx.rotations_filled_today = 0
            ctx.realized_pnl_today = 0.0
            ctx.avg_entry_price = None
            ctx.break_even_price = None
            ctx.take_profit_price = None
            ctx.stop_loss_price = None
            ctx.last_fill_buy_price = None
            ctx.last_fill_sell_price = None
            ctx.inventory_open_ts_ns = None
            ctx.live_active_bid_id = None
            ctx.live_active_ask_id = None

        self._save_dsr_quarantine_state()
        logger.info(f"Estado institucional reiniciado desde 0 con capital base ${cap:.2f}.")

    async def fetch_funding_rate(self) -> float:
        """Consulta el funding rate en vivo si el exchange está disponible."""
        try:
            ex = None
            if self.live_gateway is not None and hasattr(self.live_gateway, "exchange"):
                ex = self.live_gateway.exchange
            elif hasattr(self.multiplexer, "exchange"):
                ex = self.multiplexer.exchange

            if ex is not None and hasattr(ex, "fetch_funding_rate"):
                fr_data = await ex.fetch_funding_rate(self.primary_symbol)
                if isinstance(fr_data, dict) and fr_data.get("fundingRate") is not None:
                    self.latest_funding_rate = float(fr_data["fundingRate"])
        except Exception as e:
            logger.debug(f"Funding rate en vivo no disponible: {e}")
        return self.latest_funding_rate

    def _get_microstructure_summary(self) -> Any:
        """Genera el resumen de microestructura para el ciclo de debate de Gemini Council."""
        try:
            from datetime import datetime, timezone
            from backend.regime_council.gemini_debate import MicrostructureSummary15m, Symbol15mSummary
            idx = (self.ring_buffer.ob_head - 1) % self.ring_buffer.capacity if self.ring_buffer.ob_count > 0 else 0
            best_bid = float(self.ring_buffer.ob_bid_prices[idx, 0]) if self.ring_buffer.ob_count > 0 else 0.0
            best_ask = float(self.ring_buffer.ob_ask_prices[idx, 0]) if self.ring_buffer.ob_count > 0 else 0.0
            mid = 0.5 * (best_bid + best_ask) if best_bid > 0 else 1.0
            spread_bps = ((best_ask - best_bid) / mid * 10_000.0) if mid > 0 else 2.0
            obi = self.ring_buffer.compute_obi(levels=5)
            sigma = self.ring_buffer.compute_realized_volatility(window_seconds=300.0)

            symbols_summary = [
                Symbol15mSummary(
                    symbol=self.primary_symbol,
                    realized_vol_5m=max(float(sigma), 0.00001),
                    avg_l2_spread_bps=max(float(spread_bps), 0.1),
                    mean_obi_5_levels=float(obi),
                    cumulative_trade_volume_usd=float(self.ring_buffer.trade_count * 1000.0),
                    liquidation_clusters_count=int(self.ring_buffer.liq_count),
                    net_liquidation_volume_usd=0.0,
                    projected_funding_rate_8h=float(self.latest_funding_rate),
                    basis_bps=-0.5,
                )
            ]
            return MicrostructureSummary15m(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                symbols_data=symbols_summary,
                portfolio_equity_usd=self.portfolio.current_equity_usd,
                current_drawdown_pct=abs(self.portfolio.daily_drawdown_pct),
            )
        except Exception as err:
            logger.warning(f"No se pudo generar MicrostructureSummary15m: {err}")
            return None

    async def set_trading_mode(self, mode: str) -> None:
        """
        Conmuta el modo de ejecución entre PAPER_L2, LIVE_BINANCE y LIVE_BINGX.
        Verifica claves y estado de seguridad antes de permitir modos reales.
        """
        if mode == "LIVE_BINANCE":
            api_key = os.getenv("BINANCE_API_KEY", "").strip()
            api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
            if not api_key or not api_secret:
                raise ValueError("Debe configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env antes de conmutar a LIVE_BINANCE.")
            if self.gatekeeper.kill_switch_engaged:
                raise ValueError("No se puede conmutar a LIVE_BINANCE mientras el Kill-Switch de emergencia esté activo.")

            if self.bingx_gateway:
                self.bingx_gateway.stop_sync_loop()

            if self.live_gateway is None:
                self.live_gateway = BinanceLiveWsGateway(
                    api_key=api_key,
                    api_secret=api_secret,
                    gatekeeper=self.gatekeeper,
                    portfolio=self.portfolio,
                    on_fill=self.record_trade_fill,
                    db_store=getattr(self, "db_store", None),
                )
            elif hasattr(self, "db_store"):
                self.live_gateway.db_store = self.db_store

            await self.live_gateway.sync_time_difference()
            for sym in self.symbols:
                init_lev = self.symbol_optimal_leverage.get(sym, self.selected_leverage)
                await self.live_gateway.initialize_symbol_margin(sym, max_leverage=init_lev)

            if hasattr(self.live_gateway, "reconcile_existing_positions"):
                try:
                    reconciled = await self.live_gateway.reconcile_existing_positions(self.symbols)
                    for rec in reconciled:
                        rsym = rec.get("symbol")
                        if rsym in self.symbol_contexts:
                            rctx = self.symbol_contexts[rsym]
                            rctx.avg_entry_price = float(rec.get("entry_price") or 0.0)
                            rctx.break_even_price = float(rec.get("break_even_price") or rctx.avg_entry_price)
                            rctx.inventory_open_ts_ns = time.time_ns()
                            rside = str(rec.get("side") or "BUY").upper()
                            rqty = float(rec.get("qty") or 0.0)
                            signed_notional = round((rqty if rside == "BUY" else -rqty) * rctx.avg_entry_price, 2)
                            self.update_position_brackets(rsym, rctx.avg_entry_price, signed_notional)
                            if rctx.stop_loss_price and hasattr(self.live_gateway, "place_protective_stop_loss"):
                                close_side = "SELL" if rside == "BUY" else "BUY"
                                await self.live_gateway.place_protective_stop_loss(rsym, close_side, rqty, rctx.stop_loss_price)
                except Exception as er:
                    logger.warning(f"Error en reconcile_existing_positions Binance: {er}")

            await self.live_gateway.sync_portfolio_state(self.portfolio)
            self.live_gateway.start_sync_loop(self.portfolio)
            self.trading_mode = "LIVE_BINANCE"
            try:
                await self.initialize_real_candles(self.primary_symbol)
            except Exception as ec:
                logger.debug(f"Error cargando velas reales al conmutar a LIVE_BINANCE: {ec}")
            logger.info("Modo de trading conmutado a LIVE_BINANCE.")
        elif mode == "LIVE_BINGX":
            api_key = os.getenv("BINGX_API_KEY", "").strip()
            api_secret = os.getenv("BINGX_API_SECRET", "").strip()
            if not api_key or not api_secret:
                raise ValueError("Debe configurar BINGX_API_KEY y BINGX_API_SECRET en su archivo .env antes de conmutar a LIVE_BINGX.")
            if self.gatekeeper.kill_switch_engaged:
                raise ValueError("No se puede conmutar a LIVE_BINGX mientras el Kill-Switch de emergencia esté activo.")

            if self.live_gateway:
                self.live_gateway.stop_sync_loop()

            if self.bingx_gateway is None:
                self.bingx_gateway = BingXLiveWsGateway(
                    api_key=api_key,
                    api_secret=api_secret,
                    gatekeeper=self.gatekeeper,
                    portfolio=self.portfolio,
                    on_fill=self.record_trade_fill,
                    db_store=getattr(self, "db_store", None),
                )
            elif hasattr(self, "db_store"):
                self.bingx_gateway.db_store = self.db_store

            await self.bingx_gateway.sync_time_difference()
            for sym in self.symbols:
                init_lev = self.symbol_optimal_leverage.get(sym, self.selected_leverage)
                await self.bingx_gateway.initialize_symbol_margin(sym, max_leverage=init_lev)

            if hasattr(self.bingx_gateway, "reconcile_existing_positions"):
                try:
                    reconciled = await self.bingx_gateway.reconcile_existing_positions(self.symbols)
                    for rec in reconciled:
                        rsym = rec.get("symbol")
                        if rsym in self.symbol_contexts:
                            rctx = self.symbol_contexts[rsym]
                            rctx.avg_entry_price = float(rec.get("entry_price") or 0.0)
                            rctx.break_even_price = float(rec.get("break_even_price") or rctx.avg_entry_price)
                            rctx.inventory_open_ts_ns = time.time_ns()
                            rside = str(rec.get("side") or "BUY").upper()
                            rqty = float(rec.get("qty") or 0.0)
                            signed_notional = round((rqty if rside == "BUY" else -rqty) * rctx.avg_entry_price, 2)
                            self.update_position_brackets(rsym, rctx.avg_entry_price, signed_notional)
                            if rctx.stop_loss_price and hasattr(self.bingx_gateway, "place_protective_stop_loss"):
                                close_side = "SELL" if rside == "BUY" else "BUY"
                                await self.bingx_gateway.place_protective_stop_loss(rsym, close_side, rqty, rctx.stop_loss_price)
                except Exception as er:
                    logger.warning(f"Error en reconcile_existing_positions BingX: {er}")

            await self.bingx_gateway.sync_portfolio_state(self.portfolio)
            self.bingx_gateway.start_sync_loop(self.portfolio)
            self.trading_mode = "LIVE_BINGX"
            try:
                await self.initialize_real_candles(self.primary_symbol)
            except Exception as ec:
                logger.debug(f"Error cargando velas reales al conmutar a LIVE_BINGX: {ec}")
            logger.info("Modo de trading conmutado a LIVE_BINGX.")
        elif mode in ("LIVE_MULTI", "MULTI_EXCHANGE", "LIVE_PARALLEL"):
            bin_key = os.getenv("BINANCE_API_KEY", "").strip()
            bin_secret = os.getenv("BINANCE_API_SECRET", "").strip()
            bx_key = os.getenv("BINGX_API_KEY", "").strip()
            bx_secret = os.getenv("BINGX_API_SECRET", "").strip()
            if not bin_key or not bin_secret:
                raise ValueError("Debe configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env.")
            if not bx_key or not bx_secret:
                raise ValueError("Debe configurar BINGX_API_KEY y BINGX_API_SECRET en su archivo .env.")
            if self.gatekeeper.kill_switch_engaged:
                raise ValueError("No se puede conmutar a LIVE_MULTI mientras el Kill-Switch de emergencia esté activo.")

            if self.live_gateway is None:
                self.live_gateway = BinanceLiveWsGateway(
                    api_key=bin_key,
                    api_secret=bin_secret,
                    gatekeeper=self.gatekeeper,
                    portfolio=self.portfolio,
                    on_fill=self.record_trade_fill,
                    db_store=getattr(self, "db_store", None),
                )
            elif hasattr(self, "db_store"):
                self.live_gateway.db_store = self.db_store

            if self.bingx_gateway is None:
                self.bingx_gateway = BingXLiveWsGateway(
                    api_key=bx_key,
                    api_secret=bx_secret,
                    gatekeeper=self.gatekeeper,
                    portfolio=self.portfolio,
                    on_fill=self.record_trade_fill,
                    db_store=getattr(self, "db_store", None),
                )
            elif hasattr(self, "db_store"):
                self.bingx_gateway.db_store = self.db_store

            await self.live_gateway.sync_time_difference()
            await self.bingx_gateway.sync_time_difference()

            for sym in self.symbols:
                init_lev = self.symbol_optimal_leverage.get(sym, self.selected_leverage)
                spec = SYMBOL_SPECS.get(sym)
                if spec and spec.exchange == "bingx":
                    await self.bingx_gateway.initialize_symbol_margin(sym, max_leverage=init_lev)
                else:
                    await self.live_gateway.initialize_symbol_margin(sym, max_leverage=init_lev)

            if hasattr(self.live_gateway, "reconcile_existing_positions"):
                try:
                    reconciled_bin = await self.live_gateway.reconcile_existing_positions(self.symbols)
                    for rec in reconciled_bin:
                        rsym = rec.get("symbol")
                        if rsym in self.symbol_contexts:
                            rctx = self.symbol_contexts[rsym]
                            rctx.avg_entry_price = float(rec.get("entry_price") or 0.0)
                            rctx.break_even_price = float(rec.get("break_even_price") or rctx.avg_entry_price)
                            rctx.inventory_open_ts_ns = time.time_ns()
                            rside = str(rec.get("side") or "BUY").upper()
                            rqty = float(rec.get("qty") or 0.0)
                            signed_notional = round((rqty if rside == "BUY" else -rqty) * rctx.avg_entry_price, 2)
                            self.update_position_brackets(rsym, rctx.avg_entry_price, signed_notional)
                            if rctx.stop_loss_price and hasattr(self.live_gateway, "place_protective_stop_loss"):
                                close_side = "SELL" if rside == "BUY" else "BUY"
                                await self.live_gateway.place_protective_stop_loss(rsym, close_side, rqty, rctx.stop_loss_price)
                except Exception as er:
                    logger.warning(f"Error en reconcile_existing_positions Binance: {er}")

            if hasattr(self.bingx_gateway, "reconcile_existing_positions"):
                try:
                    reconciled_bx = await self.bingx_gateway.reconcile_existing_positions(self.symbols)
                    for rec in reconciled_bx:
                        rsym = rec.get("symbol")
                        if rsym in self.symbol_contexts:
                            rctx = self.symbol_contexts[rsym]
                            rctx.avg_entry_price = float(rec.get("entry_price") or 0.0)
                            rctx.break_even_price = float(rec.get("break_even_price") or rctx.avg_entry_price)
                            rctx.inventory_open_ts_ns = time.time_ns()
                            rside = str(rec.get("side") or "BUY").upper()
                            rqty = float(rec.get("qty") or 0.0)
                            signed_notional = round((rqty if rside == "BUY" else -rqty) * rctx.avg_entry_price, 2)
                            self.update_position_brackets(rsym, rctx.avg_entry_price, signed_notional)
                            if rctx.stop_loss_price and hasattr(self.bingx_gateway, "place_protective_stop_loss"):
                                close_side = "SELL" if rside == "BUY" else "BUY"
                                await self.bingx_gateway.place_protective_stop_loss(rsym, close_side, rqty, rctx.stop_loss_price)
                except Exception as er:
                    logger.warning(f"Error en reconcile_existing_positions BingX: {er}")

            self.trading_mode = "LIVE_MULTI"
            self.portfolio.trading_mode = "LIVE_MULTI"
            await self.live_gateway.sync_portfolio_state(self.portfolio)
            await self.bingx_gateway.sync_portfolio_state(self.portfolio)
            if hasattr(self.portfolio, "exchange_equities") and self.portfolio.exchange_equities:
                total_live = round(sum(self.portfolio.exchange_equities.values()), 2)
                self.portfolio.initial_exchange_equities = {k: round(v, 2) for k, v in self.portfolio.exchange_equities.items()}
                self.portfolio.base_capital_usd = total_live
                self.portfolio.current_equity_usd = total_live
                self.portfolio.daily_high_watermark_usd = total_live
                self.portfolio.realized_pnl_today_usd = 0.0
                self.equity_history = deque([total_live], maxlen=1000)
                logger.info(f"Capital inicial calibrado para LIVE_MULTI: ${total_live:.2f} ({self.portfolio.initial_exchange_equities})")
            self.live_gateway.start_sync_loop(self.portfolio)
            self.bingx_gateway.start_sync_loop(self.portfolio)

            try:
                await self.initialize_real_candles(self.primary_symbol)
            except Exception as ec:
                logger.debug(f"Error cargando velas reales al conmutar a LIVE_MULTI: {ec}")
            logger.info("Modo de trading conmutado a LIVE_MULTI (Binance + BingX en paralelo).")
        elif mode == "PAPER_L2":
            if self.live_gateway:
                self.live_gateway.stop_sync_loop()
            if self.bingx_gateway:
                self.bingx_gateway.stop_sync_loop()
            self.trading_mode = "PAPER_L2"
            logger.info("Modo de trading conmutado a PAPER_L2.")
        else:
            raise ValueError(f"Modo de trading desconocido: {mode}")

    async def start(self) -> None:
        """Inicia el multiplexer WebSocket de CCXT y el bucle continuo de evaluación."""
        self.running = True
        logger.info("Iniciando WebSocketMultiplexer CCXT Pro en segundo plano...")
        await self.multiplexer.start()
        if self.warm_path_scheduler:
            self.warm_path_scheduler.start()

        env_mode = os.getenv("TRADING_MODE", "PAPER_L2").strip().upper()
        if env_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI", "MULTI_EXCHANGE", "LIVE_PARALLEL"):
            try:
                await self.set_trading_mode(env_mode)
                logger.info(f"TRADING_MODE={env_mode} activado exitosamente al iniciar.")
            except Exception as e:
                logger.error(f"Error activando {env_mode} al inicio: {e}")

        self._eval_task = asyncio.create_task(self._strategy_evaluation_loop())
        self._funding_arb_task = asyncio.create_task(self._funding_arbitrage_scan_loop())
        logger.info(f"Motor de evaluación en ejecución (Modo: {self.trading_mode}).")

    async def stop(self) -> None:
        """Detiene limpiamente todas las tareas y conexiones."""
        self.running = False
        if self.warm_path_scheduler:
            self.warm_path_scheduler.stop()
        if self._eval_task:
            self._eval_task.cancel()
        if self._funding_arb_task:
            self._funding_arb_task.cancel()
        tasks_to_gather = [t for t in (self._eval_task, self._funding_arb_task) if t]
        if tasks_to_gather:
            await asyncio.gather(*tasks_to_gather, return_exceptions=True)
        await self.multiplexer.stop()
        logger.info("Motor DRY_RUN_L2 detenido.")

    async def _funding_arbitrage_scan_loop(self) -> None:
        """
        Módulo 6: Escáner de Arbitraje de Funding Rate en Vivo entre Binance y BingX.
        Consulta cada 60s las tasas de fondeo de CROSS_HEDGE_PAIRS. Si detecta divergencia >= 25% APY,
        despacha atómicamente la pata Long Maker y la pata Short Maker en modo delta-neutral.
        """
        logger.info("Escáner de Arbitraje de Funding Rate iniciado (intervalo: 60s).")
        while self.running:
            try:
                await asyncio.sleep(60.0)
                if not self.running:
                    break
                if self.trading_mode not in ("LIVE_MULTI", "LIVE_BINANCE", "LIVE_BINGX", "MULTI_EXCHANGE"):
                    continue
                if not self.live_gateway or not self.bingx_gateway:
                    continue

                for sym in CROSS_HEDGE_PAIRS:
                    if sym not in self.symbol_contexts:
                        continue
                    ctx = self.symbol_contexts[sym]
                    if ctx.ring_buffer.ob_count == 0:
                        continue

                    binance_rate = 0.0001
                    bingx_rate = 0.0001
                    try:
                        if hasattr(self.live_gateway.exchange, "fetch_funding_rate"):
                            b_target = self.live_gateway.resolve_binance_symbol(sym)
                            b_res = await self.live_gateway.exchange.fetch_funding_rate(b_target)
                            if b_res and "fundingRate" in b_res and b_res["fundingRate"] is not None:
                                binance_rate = float(b_res["fundingRate"])
                    except Exception:
                        pass

                    try:
                        if hasattr(self.bingx_gateway.exchange, "fetch_funding_rate"):
                            bx_target = self.bingx_gateway.resolve_bingx_symbol(sym)
                            bx_res = await self.bingx_gateway.exchange.fetch_funding_rate(bx_target)
                            if bx_res and "fundingRate" in bx_res and bx_res["fundingRate"] is not None:
                                bingx_rate = float(bx_res["fundingRate"])
                    except Exception:
                        pass

                    idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                    bb = float(ctx.ring_buffer.ob_bid_prices[idx, 0])
                    ba = float(ctx.ring_buffer.ob_ask_prices[idx, 0])
                    if bb <= 0 or ba <= 0:
                        continue

                    snapshot = CrossExchangeFundingSnapshot(
                        symbol=sym,
                        binance_funding_rate=binance_rate,
                        bingx_funding_rate=bingx_rate,
                        binance_bid_price=bb,
                        binance_ask_price=ba,
                        bingx_bid_price=bb,
                        bingx_ask_price=ba,
                        funding_interval_hours=8,
                    )

                    engine = getattr(ctx, "funding_engine", None) or self.funding_arb
                    package = engine.evaluate_cross_exchange_opportunity(
                        snapshot=snapshot,
                        target_notional_usd=min(50.0, self.portfolio.base_capital_usd * 0.15),
                        portfolio_state=self.portfolio,
                    )

                    if package and package.annualized_yield_diff >= 0.25:
                        logger.info(
                            f"[FUNDING ARB LIVE] Oportunidad detectada en {sym}: "
                            f"Rendimiento={package.annualized_yield_diff*100:.1f}% APY | "
                            f"Long={package.long_exchange} Short={package.short_exchange}"
                        )
                        gw_long = self.live_gateway if package.long_exchange == "BINANCE" else self.bingx_gateway
                        gw_short = self.bingx_gateway if package.short_exchange == "BINGX" else self.live_gateway
                        try:
                            t1 = gw_long.place_order_ws(package.long_leg)
                            t2 = gw_short.place_order_ws(package.short_leg)
                            await asyncio.gather(t1, t2)
                            logger.info(f"[FUNDING ARB LIVE] Ambas patas despachadas exitosamente para {sym}")
                        except Exception as de:
                            logger.error(f"Error despachando patas de Funding Arb: {de}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Excepción en _funding_arbitrage_scan_loop: {e}")

    async def _strategy_evaluation_loop(self) -> None:
        """
        Bucle continuo del Hot Path de evaluación multimoneda concurrente:
        Examina el MarketRingBuffer de cada par, evalúa el Escudo de Contagio Sistémico,
        aplica los filtros MandateGatekeeper y despacha órdenes concurrentes.
        """
        while self.running:
            try:
                t0_ns = time.perf_counter_ns()
                now_ns = time.time_ns()
                now_wall = time.time()

                # Actualizar auditoría continua de la curva de markout post-trade (+1s, +5s, +15s, +30s)
                self.update_post_trade_markouts(now_wall)

                # Calcular VPIN dinámico del activo primario
                prim_ctx = self.symbol_contexts.get(self.primary_symbol)
                if prim_ctx and hasattr(prim_ctx.ring_buffer, "compute_vpin"):
                    self.vpin_score = float(prim_ctx.ring_buffer.compute_vpin())

                # Recarga periódica del Warm Path si el mtime del archivo de régimen cambió
                self._reload_config_if_changed()

                # 1. Evaluación del Escudo de Contagio Sistémico Transversal (Cross-Asset OBI Contagion Shield)
                obis: dict[str, float] = {}
                for sym, ctx in self.symbol_contexts.items():
                    obis[sym] = ctx.ring_buffer.compute_obi(levels=5) if ctx.ring_buffer.ob_count > 0 else 0.0

                btc_obi = obis.get("BTC/USDT:USDT", 0.0)
                bearish_alt_count = sum(1 for sym, o in obis.items() if sym != "BTC/USDT:USDT" and o < -0.60)
                bullish_alt_count = sum(1 for sym, o in obis.items() if sym != "BTC/USDT:USDT" and o > 0.60)

                # Reglas: si BTC OBI < -0.70 o >= 2 pares OBI < -0.60 -> activar shield de BIDs en altcoins
                bid_shield = (btc_obi < -0.70) or (bearish_alt_count >= 2)
                # Reglas: si BTC OBI > 0.70 o >= 2 pares OBI > 0.60 -> activar shield de ASKs en altcoins
                ask_shield = (btc_obi > 0.70) or (bullish_alt_count >= 2)

                xau_obi = obis.get("XAU/USDT:USDT", 0.0)
                corr_btc_xau = self.compute_rolling_correlation("XAU/USDT:USDT", "BTC/USDT:USDT")
                xau_safe_haven = (btc_obi < -0.70) and (xau_obi > -0.40) and (corr_btc_xau < 0.25)

                self.contagion_shield.bid_shield_active = bid_shield
                self.contagion_shield.ask_shield_active = ask_shield
                self.contagion_shield.btc_obi = float(btc_obi)
                self.contagion_shield.bearish_count = bearish_alt_count
                self.contagion_shield.bullish_count = bullish_alt_count
                self.contagion_shield.xau_safe_haven_active = xau_safe_haven

                if bid_shield:
                    self.contagion_shield.trigger_reason = "BTC_TOXIC_SELL_FLOW" if btc_obi < -0.70 else "MULTI_ALT_BEARISH_CASCADE"
                elif ask_shield:
                    self.contagion_shield.trigger_reason = "BTC_BULLISH_BREAKOUT" if btc_obi > 0.70 else "MULTI_ALT_BULLISH_BURST"
                else:
                    self.contagion_shield.trigger_reason = "NORMAL"

                # Asignación de pesos Half-Kelly multivariante por par (con reasignación TradFi si cerrado)
                half_kelly_weights = self.compute_multivariate_half_kelly()
                tradfi_open = is_tradfi_market_open()

                # 0. Escaneo Periódico de Universo Dinámico (Top-5 Active Microstructure Selector) cada 15 min
                now_wall = time.time()
                if (now_wall - self.last_universe_scan_ts >= 900.0) or (len(self.top_active_symbols) == 0):
                    self.top_active_symbols = self.select_top_active_symbols(k=5)
                    self.last_universe_scan_ts = now_wall

                # 2. Evaluación cuantitativa y despacho por cada símbolo
                if not self.gatekeeper.kill_switch_engaged:
                    for sym, ctx in self.symbol_contexts.items():
                        if ctx.ring_buffer.ob_count == 0:
                            continue

                        # Guardia de sesión TradFi: si el mercado está cerrado, cancelar órdenes en XAU y XAG y pausar cotización
                        if not tradfi_open and ctx.spec.asset_class == "TRADFI_COMMODITY":
                            if self.trading_mode == "PAPER_L2":
                                for o in list(ctx.l2_sim.orders.values()):
                                    if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                                        ctx.l2_sim.request_cancel(o.order_id, now_ns)
                            elif self.trading_mode == "LIVE_BINGX" and self.bingx_gateway:
                                if ctx.live_active_bid_id:
                                    try:
                                        await self.bingx_gateway.cancel_order_ws(ctx.live_active_bid_id, sym)
                                        ctx.live_active_bid_id = None
                                    except Exception as e:
                                        logger.error(f"Error cancelando bid BingX {sym} por cierre TradFi: {e}")
                                if ctx.live_active_ask_id:
                                    try:
                                        await self.bingx_gateway.cancel_order_ws(ctx.live_active_ask_id, sym)
                                        ctx.live_active_ask_id = None
                                    except Exception as e:
                                        logger.error(f"Error cancelando ask BingX {sym} por cierre TradFi: {e}")
                            continue

                        # Si el símbolo no está en el Top 5 activo y no tiene inventario abierto, pausar cotización
                        has_open_pos = abs(self.portfolio.active_positions.get(sym, 0.0)) > 1e-4
                        if sym not in self.top_active_symbols and not has_open_pos:
                            continue

                        if self.trading_mode == "LIVE_BINANCE" and self.live_gateway:
                            current_gateway = self.live_gateway
                        elif self.trading_mode == "LIVE_BINGX" and self.bingx_gateway:
                            current_gateway = self.bingx_gateway
                        elif self.trading_mode == "LIVE_MULTI":
                            current_gateway = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
                        else:
                            current_gateway = ctx.ws_gateway

                        # Precios punta para cálculo de tamaño y cotización
                        ob_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                        b_bid = float(ctx.ring_buffer.ob_bid_prices[ob_idx, 0])
                        b_ask = float(ctx.ring_buffer.ob_ask_prices[ob_idx, 0])
                        mid_px = 0.5 * (b_bid + b_ask) if b_bid > 0 else 1.0

                        # 2a. Verificación prioritaria de Take Profit (TP), Stop Loss (SL) y Time-Stop por posición
                        if has_open_pos:
                            closed_reason = await self.check_and_execute_tp_sl_closures(
                                sym, b_bid, b_ask, mid_px, now_ns
                            )
                            if closed_reason is not None:
                                continue

                        # Inventario dinámico q_i para este par
                        dynamic_q = self.compute_normalized_inventory_q(sym)

                        # Apalancamiento dinámico autónomo (1x - 10x) según volatilidad realizada, Medio-Kelly y OBI
                        optimal_lev = self.compute_optimal_leverage(sym)
                        self.symbol_optimal_leverage[sym] = optimal_lev
                        self.gatekeeper.max_leverage_cap = max(self.gatekeeper.max_leverage_cap, float(optimal_lev))
                        if self.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"):
                            if self.trading_mode == "LIVE_MULTI":
                                active_live_gw = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
                            elif self.trading_mode == "LIVE_BINANCE":
                                active_live_gw = self.live_gateway
                            else:
                                active_live_gw = self.bingx_gateway
                            if active_live_gw:
                                await active_live_gw.update_symbol_leverage_if_needed(sym, optimal_lev)

                        # Cálculo de quote_qty dinámico según Half-Kelly, apalancamiento seleccionado y precio de mercado
                        if self.trading_mode == "LIVE_MULTI" and hasattr(self.portfolio, "exchange_equities") and self.portfolio.exchange_equities:
                            exchange_cap = float(self.portfolio.exchange_equities.get(ctx.spec.exchange, self.portfolio.base_capital_usd))
                        elif self.trading_mode == "LIVE_BINANCE" and hasattr(self.portfolio, "exchange_equities") and "binance" in self.portfolio.exchange_equities:
                            exchange_cap = float(self.portfolio.exchange_equities["binance"])
                        elif self.trading_mode == "LIVE_BINGX" and hasattr(self.portfolio, "exchange_equities") and "bingx" in self.portfolio.exchange_equities:
                            exchange_cap = float(self.portfolio.exchange_equities["bingx"])
                        else:
                            exchange_cap = self.portfolio.base_capital_usd

                        w_i = half_kelly_weights.get(sym, 0.15)
                        pair_cap_ratio = getattr(self.gatekeeper, "max_single_pair_margin_ratio", 0.20)
                        lev_mult = max(1.0, optimal_lev / 3.0)
                        target_notional = min(
                            exchange_cap * w_i * lev_mult,
                            exchange_cap * pair_cap_ratio * lev_mult,
                        )

                        # Validación de MIN_NOTIONAL de exchange (BingX: $2.0, Binance: BTC $100, ETH $20, otros $5)
                        if self.trading_mode == "LIVE_BINGX" or (self.trading_mode == "LIVE_MULTI" and ctx.spec.exchange == "bingx"):
                            exchange_min_notional = 2.0
                        else:
                            exchange_min_notional = ctx.spec.min_notional
                        max_possible_pair_notional = exchange_cap * pair_cap_ratio * lev_mult

                        # Si el min_notional de exchange excede el margen máximo permitido para el par,
                        # omitir cotización en este par y concentrar capital en pares de $5
                        if self.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI") and exchange_min_notional > max_possible_pair_notional:
                            logger.debug(
                                f"{sym} min_notional ${exchange_min_notional} excede cupo asignable (${max_possible_pair_notional:.2f}). "
                                "Reasignando cupo a pares de menor requerimiento."
                            )
                            continue

                        if ctx.spec.step_size * mid_px > max_possible_pair_notional:
                            # Símbolos de alto valor unitario (ej. BTC): adaptar resolución para no rebasar Gate 3
                            quote_qty = max(0.0001, round(target_notional / max(mid_px, 1.0), 4))
                        else:
                            step = ctx.spec.step_size
                            quote_qty = max(step, round(round(target_notional / max(mid_px, 1.0) / step) * step, 6))

                        if self.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI") and mid_px > 0:
                            if quote_qty * mid_px < exchange_min_notional:
                                step = ctx.spec.step_size
                                needed_qty = exchange_min_notional / mid_px
                                quote_qty = max(quote_qty, round(math.ceil(needed_qty / step) * step, 6))

                        ctx.avellaneda.quote_qty = quote_qty
                        ctx.avellaneda.min_notional = exchange_min_notional

                        # Factor de urgencia por antigüedad de inventario (holding_time_decay)
                        holding_time_s = ((now_ns - ctx.inventory_open_ts_ns) / 1e9) if ctx.inventory_open_ts_ns is not None else 0.0

                        # Cotización Avellaneda-Stoikov + OBI Shield + Escalera 3 Niveles + VPIN Toxicity Guard + Break-Even Floor
                        sym_vpin = ctx.ring_buffer.compute_vpin() if hasattr(ctx.ring_buffer, "compute_vpin") else 0.0

                        # Señal Lead-Lag para cotización en BingX (usando microprice y TFI adelantado de Binance):
                        venue_bid = None
                        venue_ask = None
                        lead_lag_tfi = None
                        if ctx.spec.exchange == "bingx" and hasattr(self.multiplexer, "venue_top_of_book") and sym in self.multiplexer.venue_top_of_book:
                            venue_bid, venue_ask = self.multiplexer.venue_top_of_book[sym]
                            if hasattr(ctx.ring_buffer, "compute_trade_flow_imbalance"):
                                lead_lag_tfi = ctx.ring_buffer.compute_trade_flow_imbalance()

                        quote_res = await ctx.avellaneda.on_l2_update(
                            ctx.ring_buffer,
                            self.portfolio,
                            current_gateway,
                            normalized_inventory_q=dynamic_q,
                            inventory_holding_time_s=holding_time_s,
                            vpin=sym_vpin,
                            entry_price=ctx.avg_entry_price,
                            break_even_price=getattr(ctx, "break_even_price", None),
                            venue_best_bid=venue_bid,
                            venue_best_ask=venue_ask,
                            lead_lag_tfi=lead_lag_tfi,
                        )

                        # Escudo de Contagio Sistémico sobre Altcoins (sym != BTC)
                        if sym != "BTC/USDT:USDT":
                            # Guardia refugio seguro para Oro (XAU):
                            # Si BTC OBI < -0.70, exime a XAU si su propio OBI > -0.40 y corr con BTC < 0.25
                            is_gold_haven = False
                            if sym == "XAU/USDT:USDT" and btc_obi < -0.70:
                                if xau_obi > -0.40 and corr_btc_xau < 0.25:
                                    is_gold_haven = True

                            should_block_bids = bid_shield and not is_gold_haven

                            if should_block_bids:
                                quote_res.bid_order = None
                                quote_res.bid_cancelled = True
                                # Cancelar preventivamente BIDs abiertos en L2QueueSimulator
                                if self.trading_mode == "PAPER_L2":
                                    for o in list(ctx.l2_sim.orders.values()):
                                        if o.request.side == "BUY" and o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                                            ctx.l2_sim.request_cancel(o.order_id, now_ns)
                                else:
                                    target_cancel_gw = None
                                    if self.trading_mode == "LIVE_BINANCE":
                                        target_cancel_gw = self.live_gateway
                                    elif self.trading_mode == "LIVE_BINGX":
                                        target_cancel_gw = self.bingx_gateway
                                    elif self.trading_mode == "LIVE_MULTI":
                                        target_cancel_gw = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
                                    if target_cancel_gw and ctx.live_active_bid_id:
                                        try:
                                            await target_cancel_gw.cancel_order_ws(ctx.live_active_bid_id, sym)
                                            ctx.live_active_bid_id = None
                                        except Exception as e:
                                            logger.error(f"Error cancelando bid de {sym} por escudo: {e}")

                            if ask_shield:
                                quote_res.ask_order = None
                                quote_res.ask_cancelled = True
                                # Cancelar preventivamente ASKs abiertos en L2QueueSimulator
                                if self.trading_mode == "PAPER_L2":
                                    for o in list(ctx.l2_sim.orders.values()):
                                        if o.request.side == "SELL" and o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                                            ctx.l2_sim.request_cancel(o.order_id, now_ns)
                                else:
                                    target_cancel_gw = None
                                    if self.trading_mode == "LIVE_BINANCE":
                                        target_cancel_gw = self.live_gateway
                                    elif self.trading_mode == "LIVE_BINGX":
                                        target_cancel_gw = self.bingx_gateway
                                    elif self.trading_mode == "LIVE_MULTI":
                                        target_cancel_gw = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
                                    if target_cancel_gw and ctx.live_active_ask_id:
                                        try:
                                            await target_cancel_gw.cancel_order_ws(ctx.live_active_ask_id, sym)
                                            ctx.live_active_ask_id = None
                                        except Exception as e:
                                            logger.error(f"Error cancelando ask de {sym} por escudo: {e}")

                        ctx.last_quote_result = quote_res
                        if sym == self.primary_symbol:
                            self.last_quote_result = quote_res

                        if self.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"):
                            if self.trading_mode == "LIVE_MULTI":
                                active_live_gw = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
                            elif self.trading_mode == "LIVE_BINANCE":
                                active_live_gw = self.live_gateway
                            else:
                                active_live_gw = self.bingx_gateway
                            if active_live_gw:
                                # Enrutar sólo símbolos compatibles con el exchange activo
                                is_exchange_compatible = (
                                    (self.trading_mode == "LIVE_BINANCE" and ctx.spec.exchange == "binance")
                                    or (self.trading_mode == "LIVE_BINGX")
                                    or (self.trading_mode == "LIVE_MULTI")
                                )
                                if is_exchange_compatible and (sym in self.top_active_symbols or sym == self.primary_symbol):
                                    requote_threshold_bps = 4.0

                                    # Despacho concurrente de cotizaciones en escalera Bids (L1, L2, L3) con Banda de Histéresis FIFO
                                    # Despacho en lote de cotizaciones en escalera Bids (L1, L2, L3) con Banda de Histéresis FIFO
                                    bids_to_dispatch = (quote_res.bid_ladder or ([quote_res.bid_order] if quote_res.bid_order else []))[:3]
                                    if quote_res.bid_cancelled:
                                        bids_to_cancel = list(ctx.active_ladder_bids.values())
                                        if bids_to_cancel:
                                            try:
                                                await active_live_gw.cancel_orders_batch(bids_to_cancel, sym)
                                            except Exception:
                                                pass
                                        ctx.active_ladder_bids.clear()
                                        ctx.active_ladder_bid_prices.clear()
                                    else:
                                        bids_to_cancel = []
                                        bids_to_place = []
                                        sub_lots_to_place = []
                                        for idx, b_order in enumerate(bids_to_dispatch):
                                            sub_lot = (
                                                b_order.client_order_id.split("_")[1]
                                                if "_" in (b_order.client_order_id or "")
                                                else f"L{idx+1}"
                                            )
                                            existing_id = ctx.active_ladder_bids.get(sub_lot)
                                            existing_px = ctx.active_ladder_bid_prices.get(sub_lot)

                                            if existing_id and existing_px:
                                                drift_bps = abs(b_order.price - existing_px) / max(existing_px, 1e-8) * 10_000.0
                                                if drift_bps <= requote_threshold_bps:
                                                    # Mantener viva la orden existente en la cabeza de la cola FIFO
                                                    continue
                                                else:
                                                    bids_to_cancel.append(existing_id)
                                                    ctx.active_ladder_bids.pop(sub_lot, None)
                                                    ctx.active_ladder_bid_prices.pop(sub_lot, None)

                                            bids_to_place.append(b_order)
                                            sub_lots_to_place.append(sub_lot)

                                        if bids_to_cancel:
                                            try:
                                                await active_live_gw.cancel_orders_batch(bids_to_cancel, sym)
                                            except Exception:
                                                pass

                                        if bids_to_place:
                                            try:
                                                placed_ids = await active_live_gw.place_orders_batch(bids_to_place)
                                                for slot, b_id, b_order in zip(sub_lots_to_place, placed_ids, bids_to_place):
                                                    ctx.active_ladder_bids[slot] = b_id
                                                    ctx.active_ladder_bid_prices[slot] = b_order.price
                                                    ctx.avellaneda.active_bid_order_id = b_id
                                                    ctx.live_active_bid_id = b_id
                                                    if sym == self.primary_symbol:
                                                        active_live_gw.live_active_bid_id = b_id
                                            except Exception as e:
                                                logger.warning(f"Error despachando batch bids en {sym} ({ctx.spec.exchange}): {e}")

                                    # Despacho en lote de cotizaciones en escalera Asks (L1, L2, L3) con Banda de Histéresis FIFO
                                    asks_to_dispatch = (quote_res.ask_ladder or ([quote_res.ask_order] if quote_res.ask_order else []))[:3]
                                    if quote_res.ask_cancelled:
                                        asks_to_cancel = list(ctx.active_ladder_asks.values())
                                        if asks_to_cancel:
                                            try:
                                                await active_live_gw.cancel_orders_batch(asks_to_cancel, sym)
                                            except Exception:
                                                pass
                                        ctx.active_ladder_asks.clear()
                                        ctx.active_ladder_ask_prices.clear()
                                    else:
                                        asks_to_cancel = []
                                        asks_to_place = []
                                        sub_lots_to_place_ask = []
                                        for idx, a_order in enumerate(asks_to_dispatch):
                                            sub_lot = (
                                                a_order.client_order_id.split("_")[1]
                                                if "_" in (a_order.client_order_id or "")
                                                else f"L{idx+1}"
                                            )
                                            existing_id = ctx.active_ladder_asks.get(sub_lot)
                                            existing_px = ctx.active_ladder_ask_prices.get(sub_lot)

                                            if existing_id and existing_px:
                                                drift_bps = abs(a_order.price - existing_px) / max(existing_px, 1e-8) * 10_000.0
                                                if drift_bps <= requote_threshold_bps:
                                                    continue
                                                else:
                                                    asks_to_cancel.append(existing_id)
                                                    ctx.active_ladder_asks.pop(sub_lot, None)
                                                    ctx.active_ladder_ask_prices.pop(sub_lot, None)

                                            asks_to_place.append(a_order)
                                            sub_lots_to_place_ask.append(sub_lot)

                                        if asks_to_cancel:
                                            try:
                                                await active_live_gw.cancel_orders_batch(asks_to_cancel, sym)
                                            except Exception:
                                                pass

                                        if asks_to_place:
                                            try:
                                                placed_ids = await active_live_gw.place_orders_batch(asks_to_place)
                                                for slot, a_id, a_order in zip(sub_lots_to_place_ask, placed_ids, asks_to_place):
                                                    ctx.active_ladder_asks[slot] = a_id
                                                    ctx.active_ladder_ask_prices[slot] = a_order.price
                                                    ctx.avellaneda.active_ask_order_id = a_id
                                                    ctx.live_active_ask_id = a_id
                                                    if sym == self.primary_symbol:
                                                        active_live_gw.live_active_ask_id = a_id
                                            except Exception as e:
                                                logger.warning(f"Error despachando batch asks en {sym} ({ctx.spec.exchange}): {e}")

                                    # Módulo 5: Cobertura Cruzada Instantánea (Cross-Exchange Inventory Hedging)
                                    if self.trading_mode == "LIVE_MULTI" and sym in CROSS_HEDGE_PAIRS and self.live_gateway and self.bingx_gateway:
                                        pos_qty = float(self.portfolio.active_positions.get(sym, 0.0) or (self.ledger.get_net_inventory(sym) if hasattr(self, "ledger") else 0.0))
                                        if abs(pos_qty) > 1e-6 and ctx.avg_entry_price and ctx.avg_entry_price > 0:
                                            other_gw = self.live_gateway if ctx.spec.exchange == "bingx" else self.bingx_gateway
                                            be_px = getattr(ctx, "break_even_price", None) or ctx.avg_entry_price
                                            if pos_qty > 0 and venue_bid and venue_bid >= be_px:
                                                hedge_qty = truncate_decimal_step(pos_qty * 0.5, ctx.spec.step_size)
                                                if hedge_qty > 0:
                                                    try:
                                                        logger.info(f"[CROSS-HEDGE] Cobertura Long de {sym} en exchange cruzado: {hedge_qty} @ ${venue_bid}")
                                                        await other_gw.close_position_market(sym, "SELL", hedge_qty)
                                                    except Exception as he:
                                                        logger.debug(f"Error en cross-hedge sell: {he}")
                                            elif pos_qty < 0 and venue_ask and venue_ask <= be_px:
                                                hedge_qty = truncate_decimal_step(abs(pos_qty) * 0.5, ctx.spec.step_size)
                                                if hedge_qty > 0:
                                                    try:
                                                        logger.info(f"[CROSS-HEDGE] Cobertura Short de {sym} en exchange cruzado: {hedge_qty} @ ${venue_ask}")
                                                        await other_gw.close_position_market(sym, "BUY", hedge_qty)
                                                    except Exception as he:
                                                        logger.debug(f"Error en cross-hedge buy: {he}")

                                    # Poda determinista de órdenes huérfanas pasando los IDs válidos preservados
                                    valid_live_ids = set(ctx.active_ladder_bids.values()) | set(ctx.active_ladder_asks.values())
                                    if ctx.tp_maker_order_id:
                                        valid_live_ids.add(ctx.tp_maker_order_id)
                                    try:
                                        await active_live_gw.prune_stale_orders(sym, valid_order_ids=valid_live_ids, max_orders_per_side=3)
                                    except Exception as ep:
                                        logger.debug(f"prune_stale_orders en {sym}: {ep}")
                        else:
                            # Modo PAPER_L2: Reemplazo de órdenes obsoletas (stale orders) si se desvían > 3 ticks
                            max_drift = max(3.0 * ctx.spec.tick_size, 0.5 * quote_res.optimal_spread)
                            for o in list(ctx.l2_sim.orders.values()):
                                if o.status in ("OPEN", "PARTIALLY_FILLED"):
                                    if o.request.side == "BUY" and quote_res.bid_order is not None:
                                        if abs(o.request.price - quote_res.bid_order.price) > max_drift:
                                            ctx.l2_sim.request_cancel(o.order_id, now_ns)
                                    elif o.request.side == "SELL" and quote_res.ask_order is not None:
                                        if abs(o.request.price - quote_res.ask_order.price) > max_drift:
                                            ctx.l2_sim.request_cancel(o.order_id, now_ns)

                            has_open_bid = any(
                                o.request.side == "BUY" and o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING")
                                for o in ctx.l2_sim.orders.values()
                            )
                            has_open_ask = any(
                                o.request.side == "SELL" and o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING")
                                for o in ctx.l2_sim.orders.values()
                            )

                            if quote_res.bid_order is not None and not has_open_bid:
                                sim_bid = ctx.l2_sim.submit_gtx_order(
                                    quote_res.bid_order, ctx.ring_buffer, current_ts_ns=now_ns
                                )
                                if sim_bid.status == "OPEN":
                                    ctx.avellaneda.active_bid_order_id = sim_bid.order_id

                            if quote_res.ask_order is not None and not has_open_ask:
                                sim_ask = ctx.l2_sim.submit_gtx_order(
                                    quote_res.ask_order, ctx.ring_buffer, current_ts_ns=now_ns
                                )
                                if sim_ask.status == "OPEN":
                                    ctx.avellaneda.active_ask_order_id = sim_ask.order_id

                            # Motor B: Cazador de Cascadas de Liquidación con Bracket TP/SL completo
                            open_sim_orders = [
                                o for o in ctx.l2_sim.orders.values()
                                if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING")
                            ]
                            cascade_bracket = ctx.liq_hunter.evaluate(
                                ctx.ring_buffer, self.portfolio, now_ns=now_ns
                            )
                            if cascade_bracket is not None and len(open_sim_orders) == 0:
                                sim_casc = ctx.l2_sim.submit_gtx_order(
                                    cascade_bracket.entry_order, ctx.ring_buffer, current_ts_ns=now_ns
                                )
                                if sim_casc.status == "OPEN":
                                    ctx.take_profit_price = cascade_bracket.take_profit_order.price
                                    ctx.stop_loss_price = cascade_bracket.stop_loss_price

                    # Motor C: Arbitraje Delta-Neutral de Funding (activo primario)
                    prim_ctx = self.symbol_contexts[self.primary_symbol]
                    if prim_ctx.ring_buffer.ob_count > 0:
                        p_idx = (prim_ctx.ring_buffer.ob_head - 1) % prim_ctx.ring_buffer.capacity
                        p_bid = float(prim_ctx.ring_buffer.ob_bid_prices[p_idx, 0])
                        p_ask = float(prim_ctx.ring_buffer.ob_ask_prices[p_idx, 0])
                        p_mid = 0.5 * (p_bid + p_ask)

                        snapshot = FundingArbSnapshot(
                            symbol=self.primary_symbol,
                            spot_price=p_mid,
                            perp_bid_price=p_bid,
                            perp_ask_price=p_ask,
                            projected_funding_rate=float(self.latest_funding_rate),
                            funding_interval_hours=8,
                        )
                        self.funding_arb.evaluate_opportunity(
                            snapshot=snapshot,
                            target_notional_usd=min(15.0, self.portfolio.base_capital_usd * 0.15),
                            portfolio_state=self.portfolio,
                            timestamp_ns=now_ns,
                        )

                # 3. Procesamiento concurrente de nuevos trades para todos los pares
                for sym, ctx in self.symbol_contexts.items():
                    curr_trade_head = ctx.ring_buffer.trade_head
                    if ctx.ring_buffer.trade_count > 0 and ctx.last_processed_trade_head != curr_trade_head:
                        start_ptr = ctx.last_processed_trade_head
                        end_ptr = curr_trade_head
                        cap = ctx.ring_buffer.capacity

                        if start_ptr <= end_ptr:
                            trade_indices = list(range(start_ptr, end_ptr))
                        else:
                            trade_indices = list(range(start_ptr, cap)) + list(range(0, end_ptr))

                        if len(trade_indices) > ctx.ring_buffer.trade_count:
                            trade_indices = trade_indices[-ctx.ring_buffer.trade_count:]

                        for t_idx in trade_indices:
                            trade_px = float(ctx.ring_buffer.trade_prices[t_idx])
                            trade_qty = float(ctx.ring_buffer.trade_qtys[t_idx])
                            is_buyer_maker = bool(ctx.ring_buffer.trade_is_buyer_maker[t_idx])
                            trade_ts = int(ctx.ring_buffer.trade_timestamps_ns[t_idx])

                            fills = ctx.l2_sim.on_market_trade(
                                trade_price=trade_px,
                                trade_qty=trade_qty,
                                is_buyer_maker=is_buyer_maker,
                                trade_ts_ns=trade_ts,
                            )
                            for fill in fills:
                                ctx.rotations_filled_today += 1
                                self.total_rotations_filled += 1
                                fill_px = fill.avg_fill_price
                                fill_qty = fill.request.qty

                                fill_notional = fill_px * fill_qty
                                curr_pos = self.portfolio.active_positions.get(sym, 0.0)
                                if fill.request.side == "BUY":
                                    new_pos = curr_pos + fill_notional
                                else:
                                    new_pos = curr_pos - fill_notional

                                if abs(new_pos) < 1e-4:
                                    self.portfolio.active_positions.pop(sym, None)
                                    ctx.inventory_open_ts_ns = None
                                    ctx.avg_entry_price = None
                                    ctx.take_profit_price = None
                                    ctx.stop_loss_price = None
                                    ctx.last_exit_reason = "ROTATION_CLOSED"
                                else:
                                    self.portfolio.active_positions[sym] = round(new_pos, 2)
                                    if ctx.inventory_open_ts_ns is None:
                                        ctx.inventory_open_ts_ns = now_ns
                                    self.update_position_brackets(sym, fill_px, new_pos)

                                ob_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity if ctx.ring_buffer.ob_count > 0 else 0
                                b_bid = float(ctx.ring_buffer.ob_bid_prices[ob_idx, 0]) if ctx.ring_buffer.ob_count > 0 else fill_px
                                b_ask = float(ctx.ring_buffer.ob_ask_prices[ob_idx, 0]) if ctx.ring_buffer.ob_count > 0 else fill_px
                                cur_mid = 0.5 * (b_bid + b_ask) if b_bid > 0 else fill_px

                                maker_fee = round(fill_px * fill_qty * 0.0002, 4)
                                is_close_fill = False
                                if fill.request.side == "BUY":
                                    if ctx.last_fill_sell_price is not None:
                                        is_close_fill = True
                                        gross_pnl = (ctx.last_fill_sell_price - fill_px) * fill_qty
                                        cycle_pnl = round(gross_pnl - maker_fee, 4)
                                        ret = cycle_pnl / max(ctx.last_fill_sell_price * fill_qty, 1e-8)
                                        exit_or_mid = fill_px
                                        entry_px_fill = ctx.last_fill_sell_price
                                        ctx.last_fill_sell_price = None
                                        ctx.last_fill_buy_price = None
                                    else:
                                        # Apertura de posición larga (sin PnL realizado falso)
                                        ctx.last_fill_buy_price = fill_px
                                        cycle_pnl = 0.0
                                        ret = 0.0
                                        exit_or_mid = fill_px
                                        entry_px_fill = fill_px
                                        self.portfolio.current_equity_usd = round(
                                            self.portfolio.current_equity_usd - maker_fee, 2
                                        )
                                else:
                                    if ctx.last_fill_buy_price is not None:
                                        is_close_fill = True
                                        gross_pnl = (fill_px - ctx.last_fill_buy_price) * fill_qty
                                        cycle_pnl = round(gross_pnl - maker_fee, 4)
                                        ret = cycle_pnl / max(ctx.last_fill_buy_price * fill_qty, 1e-8)
                                        exit_or_mid = fill_px
                                        entry_px_fill = ctx.last_fill_buy_price
                                        ctx.last_fill_buy_price = None
                                        ctx.last_fill_sell_price = None
                                    else:
                                        # Apertura de posición corta (sin PnL realizado falso)
                                        ctx.last_fill_sell_price = fill_px
                                        cycle_pnl = 0.0
                                        ret = 0.0
                                        exit_or_mid = fill_px
                                        entry_px_fill = fill_px
                                        self.portfolio.current_equity_usd = round(
                                            self.portfolio.current_equity_usd - maker_fee, 2
                                        )

                                if is_close_fill:
                                    captured_return = float(ret)
                                    self.returns_history.append(captured_return)
                                    if sym in self.symbol_returns:
                                        self.symbol_returns[sym].append(captured_return)
                                    pnl_delta = float(cycle_pnl)
                                    ctx.realized_pnl_today += pnl_delta

                                    self.portfolio.current_equity_usd = round(
                                        self.portfolio.current_equity_usd + pnl_delta, 2
                                    )
                                    self.portfolio.realized_pnl_today_usd = round(
                                        self.portfolio.realized_pnl_today_usd + pnl_delta, 2
                                    )
                                    if self.portfolio.current_equity_usd > self.portfolio.daily_high_watermark_usd:
                                        self.portfolio.daily_high_watermark_usd = self.portfolio.current_equity_usd

                                # Registrar trade fill institucional
                                strat = getattr(fill.request, "strategy_id", "AVELLANEDA_OBI")
                                self.record_trade_fill(
                                    symbol=sym,
                                    strategy=strat,
                                    side=fill.request.side,
                                    entry_price=entry_px_fill,
                                    exit_or_mid_price=exit_or_mid,
                                    qty=fill_qty,
                                    return_pct=ret * 100.0,
                                    pnl_usd=cycle_pnl,
                                    timestamp_ns=trade_ts if trade_ts > 0 else now_ns,
                                    fee_cost=maker_fee,
                                    is_close=is_close_fill,
                                    fee_already_deducted=True,
                                )

                                # Trinquete 85/15: Escalado dinámico de base_capital_usd y barrido a Bóveda Spot
                                swept = step_up_ratchet(self.portfolio)
                                if swept > 0.0:
                                    self.simulated_vault_usd = getattr(self, "simulated_vault_usd", 0.0) + swept

                                self._save_dsr_quarantine_state()

                        ctx.last_processed_trade_head = curr_trade_head

                    ctx.l2_sim.advance_clock(now_ns)

                self.last_latency_us = (time.perf_counter_ns() - t0_ns) / 1_000.0

            except Exception as e:
                logger.error(f"Error en bucle de evaluación: {e}", exc_info=True)

            await asyncio.sleep(0.05)  # Frecuencia de evaluación ~20 Hz

    def get_telemetry_snapshot(self) -> dict[str, Any]:
        """Extrae el estado actual real de los componentes para telemetría multimoneda."""
        now_ns = time.time_ns()
        prim_ctx = self.symbol_contexts[self.primary_symbol]
        idx = (prim_ctx.ring_buffer.ob_head - 1) % prim_ctx.ring_buffer.capacity if prim_ctx.ring_buffer.ob_count > 0 else 0
        best_bid = float(prim_ctx.ring_buffer.ob_bid_prices[idx, 0]) if prim_ctx.ring_buffer.ob_count > 0 else 0.0
        best_ask = float(prim_ctx.ring_buffer.ob_ask_prices[idx, 0]) if prim_ctx.ring_buffer.ob_count > 0 else 0.0
        mid_price = 0.5 * (best_bid + best_ask) if best_bid > 0 else 0.0

        obi = prim_ctx.ring_buffer.compute_obi(levels=5)
        realized_sigma = prim_ctx.ring_buffer.compute_realized_volatility(window_seconds=300.0)

        # Auditoría DSR
        if len(self.returns_history) >= 30:
            try:
                dsr_res = DeflatedSharpeAuditor.audit_returns(self.returns_history[-250:], num_trials_n=25)
            except Exception:
                dsr_res = DSRAuditResult(
                    deflated_sharpe_ratio=None,
                    observed_sharpe_sr=None,
                    benchmark_null_sr0=None,
                    skewness_gamma3=None,
                    kurtosis_gamma4=None,
                    sample_length_t=len(self.returns_history),
                    num_trials_n=25,
                    status=f"INSUFICIENTE ({len(self.returns_history)} trades)",
                    is_overfitted=False,
                    annualized_sharpe=None,
                )
        else:
            dsr_res = DSRAuditResult(
                deflated_sharpe_ratio=None,
                observed_sharpe_sr=None,
                benchmark_null_sr0=None,
                skewness_gamma3=None,
                kurtosis_gamma4=None,
                sample_length_t=len(self.returns_history),
                num_trials_n=25,
                status=f"INSUFICIENTE ({len(self.returns_history)} trades)",
                is_overfitted=False,
                annualized_sharpe=None,
            )

        # Half-Kelly dinámico empírico
        self.update_rolling_moments()
        variance_safe = max(self.portfolio.rolling_variance_sigma2, 1e-8)
        raw_half_kelly = 0.5 * (self.portfolio.rolling_edge_mu / variance_safe)
        if len(self.returns_history) < 100:
            raw_half_kelly = min(max(raw_half_kelly, 0.0), 1.0)
        else:
            raw_half_kelly = max(raw_half_kelly, 0.0)
        effective_kelly_cap = min(raw_half_kelly, getattr(self.gatekeeper, "max_leverage_cap", 4.0))

        latest_sim = list(prim_ctx.l2_sim.orders.values())[-1] if prim_ctx.l2_sim.orders else None

        res_price = self.last_quote_result.reservation_price if self.last_quote_result else mid_price
        opt_spread = self.last_quote_result.optimal_spread if self.last_quote_result else 2.0
        bid_cancelled = self.last_quote_result.bid_cancelled if self.last_quote_result else False
        ask_cancelled = self.last_quote_result.ask_cancelled if self.last_quote_result else False

        half_kelly_weights = self.compute_multivariate_half_kelly()

        # Resumen detallado para cada una de las monedas
        symbols_data: dict[str, Any] = {}
        for sym, ctx in self.symbol_contexts.items():
            s_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity if ctx.ring_buffer.ob_count > 0 else 0
            s_bid = float(ctx.ring_buffer.ob_bid_prices[s_idx, 0]) if ctx.ring_buffer.ob_count > 0 else 0.0
            s_ask = float(ctx.ring_buffer.ob_ask_prices[s_idx, 0]) if ctx.ring_buffer.ob_count > 0 else 0.0
            s_mid = 0.5 * (s_bid + s_ask) if s_bid > 0 else 0.0
            s_obi = ctx.ring_buffer.compute_obi(levels=5) if ctx.ring_buffer.ob_count > 0 else 0.0
            if self.trading_mode == "LIVE_MULTI":
                active_sym_gw = self.bingx_gateway if ctx.spec.exchange == "bingx" else self.live_gateway
            elif self.trading_mode == "LIVE_BINANCE":
                active_sym_gw = self.live_gateway
            elif self.trading_mode == "LIVE_BINGX":
                active_sym_gw = self.bingx_gateway
            else:
                active_sym_gw = None

            lots_list = [
                {
                    "strategy_id": lot.strategy_id,
                    "sub_lot_id": lot.sub_lot_id,
                    "inventory_q": round(lot.inventory_q, 4),
                    "entry_price": round(lot.entry_price, 2 if lot.entry_price >= 1.0 else 5),
                    "realized_pnl_usd": round(lot.realized_pnl_usd, 2),
                    "is_open": lot.is_open,
                }
                for lot in (active_sym_gw.ledger.get_lots_by_symbol(sym) if active_sym_gw else self.ledger.get_lots_by_symbol(sym))
            ]

            symbols_data[sym] = {
                "symbol": sym,
                "base_asset": ctx.spec.base_asset,
                "asset_class": ctx.spec.asset_class,
                "exchange": ctx.spec.exchange,
                "mid_price": round(s_mid, 2 if s_mid >= 1.0 else 5),
                "best_bid": round(s_bid, 2 if s_bid >= 1.0 else 5),
                "best_ask": round(s_ask, 2 if s_ask >= 1.0 else 5),
                "obi": round(s_obi, 4),
                "inventory_q": round(self.compute_normalized_inventory_q(sym), 4),
                "position_usd": round(self.portfolio.active_positions.get(sym, 0.0), 2),
                "avg_entry_price": round(ctx.avg_entry_price, 2 if ctx.avg_entry_price >= 1.0 else 5) if ctx.avg_entry_price else None,
                "take_profit_price": round(ctx.take_profit_price, 2 if ctx.take_profit_price >= 1.0 else 5) if ctx.take_profit_price else None,
                "stop_loss_price": round(ctx.stop_loss_price, 2 if ctx.stop_loss_price >= 1.0 else 5) if ctx.stop_loss_price else None,
                "tp_bps": ctx.tp_bps,
                "sl_bps": ctx.sl_bps,
                "last_exit_reason": ctx.last_exit_reason,
                "rotations_today": ctx.rotations_filled_today,
                "pnl_today_usd": round(ctx.realized_pnl_today, 2),
                "half_kelly_weight": half_kelly_weights.get(sym, 0.15),
                "selected_leverage": self.symbol_optimal_leverage.get(sym, self.selected_leverage),
                "optimal_leverage": self.symbol_optimal_leverage.get(sym, self.selected_leverage),
                "virtual_lots": lots_list,
            }

        # Estadísticas de ejecución y blotter de órdenes / trades
        tot_trades = self.wins_count + self.losses_count
        win_rate = round((self.wins_count / tot_trades * 100.0) if tot_trades > 0 else 0.0, 2)
        if self.gross_loss_usd > 0.0:
            profit_factor = round(self.gross_profit_usd / self.gross_loss_usd, 2)
        elif self.gross_profit_usd > 0.0:
            profit_factor = round(self.gross_profit_usd, 2)
        else:
            profit_factor = 0.0
        avg_trade_pnl = round(((self.gross_profit_usd - self.gross_loss_usd) / tot_trades) if tot_trades > 0 else 0.0, 2)

        execution_stats = {
            "total_trades": tot_trades,
            "wins_count": self.wins_count,
            "losses_count": self.losses_count,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl_usd": avg_trade_pnl,
            "gross_profit_usd": round(self.gross_profit_usd, 2),
            "gross_loss_usd": round(self.gross_loss_usd, 2),
            "equity_curve": list(self.equity_history),
        }

        # Open orders con level L1/L2/L3/CASCADE
        open_orders_list: list[dict[str, Any]] = []
        for s_sym, s_ctx in self.symbol_contexts.items():
            sym_orders = [o for o in s_ctx.l2_sim.orders.values() if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING")]
            for idx, o in enumerate(sym_orders):
                strat = getattr(o.request, "strategy_id", "AVELLANEDA_OBI")
                lvl = "CASCADE" if "LIQ" in str(strat).upper() else f"L{min(idx + 1, 3)}"
                open_orders_list.append({
                    "order_id": o.order_id,
                    "symbol": s_sym,
                    "side": o.request.side,
                    "level": lvl,
                    "price": round(o.request.price, 2 if o.request.price >= 1.0 else 5),
                    "qty": round(o.request.qty, 4),
                    "filled_qty": round(o.filled_qty, 4),
                    "queue_ahead_qty": round(o.queue_ahead_qty, 4),
                    "strategy": strat,
                    "status": o.status,
                })

        if self.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"):
            open_orders_list.clear()  # En vivo, mostrar únicamente las órdenes reales vivas
            gws_to_inspect = []
            if self.trading_mode == "LIVE_MULTI":
                if self.live_gateway:
                    gws_to_inspect.append(self.live_gateway)
                if self.bingx_gateway:
                    gws_to_inspect.append(self.bingx_gateway)
            elif self.trading_mode == "LIVE_BINANCE" and self.live_gateway:
                gws_to_inspect.append(self.live_gateway)
            elif self.trading_mode == "LIVE_BINGX" and self.bingx_gateway:
                gws_to_inspect.append(self.bingx_gateway)

            for gw in gws_to_inspect:
                gw_exchange = "bingx" if gw == self.bingx_gateway else "binance"
                for ord_id, meta in list(gw.active_orders_meta.items()):
                    px = float(meta.get("price") or 0.0)
                    q = float(meta.get("qty") or 0.0)
                    strat = meta.get("strategy_id", "AVELLANEDA_OBI")
                    sub_lot = meta.get("sub_lot_id", "L1")
                    lvl = "CASCADE" if "LIQ" in str(strat).upper() else sub_lot
                    open_orders_list.append({
                        "order_id": str(ord_id),
                        "symbol": meta.get("symbol", self.primary_symbol),
                        "exchange": gw_exchange,
                        "side": meta.get("side", "BUY"),
                        "level": lvl,
                        "price": round(px, 2 if px >= 1.0 else 5),
                        "qty": round(q, 4),
                        "filled_qty": 0.0,
                        "queue_ahead_qty": 0.0,
                        "strategy": strat,
                        "status": "OPEN",
                    })

        now_s = time.time()

        # Recent trades enriquecidos con age_str y price (ampliado a 500 para análisis profundo)
        recent_trades_list: list[dict[str, Any]] = []
        for t in list(self.recent_trades)[::-1][:500]:
            t_ms = t.get("timestamp_ms", int(now_s * 1000))
            age_s = max(0, int(now_s - (t_ms / 1000.0)))
            age_str = f"{age_s}s" if age_s < 60 else f"{age_s // 60}m"
            item_copy = dict(t)
            item_copy["age_str"] = age_str
            item_copy["price"] = t.get("exit_or_mid_price", t.get("entry_price", 0.0))
            recent_trades_list.append(item_copy)

        # Profundidad L2 acumulada Top 5 (Order Book Mountain)
        bids_depth: list[list[float]] = []
        asks_depth: list[list[float]] = []
        ob_idx = (prim_ctx.ring_buffer.ob_head - 1) % prim_ctx.ring_buffer.capacity if prim_ctx.ring_buffer.ob_count > 0 else 0
        cum_b = 0.0
        cum_a = 0.0
        for i in range(5):
            bp = float(prim_ctx.ring_buffer.ob_bid_prices[ob_idx, i]) if prim_ctx.ring_buffer.ob_count > 0 else (best_bid - i * 0.1)
            bq = float(prim_ctx.ring_buffer.ob_bid_qtys[ob_idx, i]) if prim_ctx.ring_buffer.ob_count > 0 else (1.5 + i * 0.8)
            ap = float(prim_ctx.ring_buffer.ob_ask_prices[ob_idx, i]) if prim_ctx.ring_buffer.ob_count > 0 else (best_ask + i * 0.1)
            aq = float(prim_ctx.ring_buffer.ob_ask_qtys[ob_idx, i]) if prim_ctx.ring_buffer.ob_count > 0 else (1.5 + i * 0.8)
            cum_b += bq
            cum_a += aq
            bids_depth.append([round(bp, 2 if bp >= 1.0 else 5), round(bq, 4), round(cum_b, 4)])
            asks_depth.append([round(ap, 2 if ap >= 1.0 else 5), round(aq, 4), round(cum_a, 4)])

        order_book_depth = {
            "bids": bids_depth,
            "asks": asks_depth,
        }

        # Actualización y rotación minuto a minuto de las velas microestructurales
        curr_minute = int(now_s // 60)
        if hasattr(self, "last_candle_minute") and curr_minute > self.last_candle_minute and len(self.candles_history) > 0:
            self.last_candle_minute = curr_minute
            self.candles_history.append({
                "timestamp": int(now_s * 1000),
                "open": round(mid_price, 2),
                "high": round(mid_price, 2),
                "low": round(mid_price, 2),
                "close": round(mid_price, 2),
                "volume": 0.5,
                "is_bullish": True,
                "reservation_price": round(res_price, 2),
            })
        elif self.candles_history:
            last_c = self.candles_history[-1]
            last_c["close"] = round(mid_price, 2)
            last_c["high"] = round(max(last_c["high"], mid_price), 2)
            last_c["low"] = round(min(last_c["low"], mid_price), 2)
            last_c["reservation_price"] = round(res_price, 2)
            last_c["is_bullish"] = last_c["close"] >= last_c["open"]
            last_c["volume"] = round(last_c.get("volume", 1.0) + 0.02, 2)
        candles_list = list(self.candles_history)

        # Actualización de equidad en tiempo real
        if self.equity_comparison:
            self.equity_comparison[-1]["bot_equity_usd"] = round(self.portfolio.current_equity_usd, 2)
            self.equity_comparison[-1]["benchmark_equity_usd"] = round(self.portfolio.base_capital_usd, 2)
        equity_comp_list = list(self.equity_comparison)

        # Cinta de ballenas y liquidaciones con age_str relativo
        whale_tape_list: list[dict[str, Any]] = []
        for it in list(self.whale_and_liq_tape):
            t_ms = it.get("timestamp_ms", int(now_s * 1000))
            age_s = max(0, int(now_s - (t_ms / 1000.0)))
            whale_tape_list.append({
                "symbol": it["symbol"],
                "side": it["side"],
                "notional_usd": it["notional_usd"],
                "intensity_pct": it["intensity_pct"],
                "age_str": f"{age_s}s" if age_s < 60 else f"{age_s // 60}m",
                "is_liquidation": it.get("is_liquidation", False),
            })

        # Donut de asignación multimoneda (top 5 pares activos)
        palette = ["#00F59B", "#A855F7", "#06B6D4", "#F59E0B", "#EC4899"]
        alloc_list: list[dict[str, Any]] = []
        top_5 = self.top_active_symbols[:5]
        total_w = sum(half_kelly_weights.get(s, 0.20) for s in top_5) or 1.0
        for i, s in enumerate(top_5):
            base_asset = SYMBOL_SPECS.get(s, SymbolMarketSpec(s, 0.01, 0.001, s.split("/")[0])).base_asset
            raw_w = half_kelly_weights.get(s, 0.20)
            pct = round((raw_w / total_w) * 100.0, 1)
            pos_usd = round(abs(self.portfolio.active_positions.get(s, 0.0)), 2)
            alloc_list.append({
                "symbol": s,
                "base_asset": base_asset,
                "weight_pct": pct,
                "notional_usd": pos_usd if pos_usd > 0 else round(self.portfolio.current_equity_usd * (pct / 100.0), 2),
                "color": palette[i % len(palette)],
            })

        return {
            "timestamp_ns": now_ns,
            "mode": self.trading_mode,
            "trading_mode": self.trading_mode,
            "strategy_mode": self.strategy_mode,
            "hot_path_latency_us": round(self.last_latency_us, 2),
            "execution_stats": execution_stats,
            "open_orders": open_orders_list,
            "recent_trades": recent_trades_list,
            "order_book_depth": order_book_depth,
            "candles_history": candles_list,
            "equity_comparison": equity_comp_list,
            "whale_and_liq_tape": whale_tape_list,
            "portfolio_allocation": alloc_list,
            "vpin_score": round(self.vpin_score, 4),
            "markout_curve_bps": self.markout_curve_bps,
            "btc_gold_corr": round(self.btc_gold_corr, 4),
            "chaos_state": {
                "mock_tradfi_closed": GLOBAL_MOCK_TRADFI_CLOSED,
                "latency_spike_active": self.chaos_latency_active,
            },
            "market": {
                "symbol": self.primary_symbol,
                "mid_price": round(mid_price, 2),
                "best_bid": round(best_bid, 2),
                "best_ask": round(best_ask, 2),
                "microprice": round(prim_ctx.ring_buffer.compute_microprice(), 2),
                "obi": round(obi, 4),
                "vpin": round(self.vpin_score, 4),
                "markout_curve": self.markout_curve_bps,
                "btc_gold_corr": round(self.btc_gold_corr, 4),
                "tfi": round(prim_ctx.ring_buffer.compute_tfi(window_ns=500_000_000), 4),
                "realized_sigma": round(realized_sigma, 6),
                "reservation_price": round(res_price, 2),
                "optimal_spread": round(opt_spread, 2),
                "bid_cancelled_by_shield": bid_cancelled,
                "ask_cancelled_by_shield": ask_cancelled,
            },
            "contagion_shield": {
                "bid_shield_active": self.contagion_shield.bid_shield_active,
                "ask_shield_active": self.contagion_shield.ask_shield_active,
                "trigger_reason": self.contagion_shield.trigger_reason,
                "btc_obi": round(self.contagion_shield.btc_obi, 4),
                "btc_gold_corr": round(self.btc_gold_corr, 4),
                "bearish_count": self.contagion_shield.bearish_count,
                "bullish_count": self.contagion_shield.bullish_count,
                "xau_safe_haven_active": self.contagion_shield.xau_safe_haven_active,
            },
            "tradfi_market_open": is_tradfi_market_open(),
            "top_active_symbols": self.top_active_symbols,
            "symbols": symbols_data,
            "portfolio": {
                "base_capital_usd": self.portfolio.base_capital_usd,
                "current_equity_usd": self.portfolio.current_equity_usd,
                "daily_high_watermark_usd": self.portfolio.daily_high_watermark_usd,
                "realized_pnl_today_usd": self.portfolio.realized_pnl_today_usd,
                "daily_return_pct": round(self.portfolio.daily_net_return_pct * 100, 3),
                "daily_drawdown_pct": round(self.portfolio.daily_drawdown_pct * 100, 3),
                "normalized_inventory_q": round(self.compute_normalized_inventory_q(), 4),
                "funding_rate": round(self.latest_funding_rate, 6),
                "profit_sweep_ready": should_trigger_profit_sweep(self.portfolio),
                "rotations_filled": self.total_rotations_filled,
                "simulated_vault_usd": getattr(self, "simulated_vault_usd", 0.0),
                "exchange_equities": getattr(self.portfolio, "exchange_equities", {}),
                "initial_exchange_equities": getattr(self.portfolio, "initial_exchange_equities", {}),
            },
            "mandate_gates": {
                "gate_1_isolated_margin": "LOCKED_ISOLATED",
                "gate_2_maker_gtx": "ENFORCED_POST_ONLY",
                "gate_3_half_kelly_cap": round(effective_kelly_cap, 2),
                "gate_3_current_leverage": round(
                    self.portfolio.total_open_notional_usd / self.portfolio.base_capital_usd, 2
                ),
                "gate_4_kill_switch_engaged": self.gatekeeper.kill_switch_engaged,
                "emergency_flatten_count": self.gatekeeper.emergency_flatten_count,
            },
            "l2_queue": {
                "order_id": latest_sim.order_id if latest_sim else "NONE",
                "status": latest_sim.status if latest_sim else "IDLE",
                "price": latest_sim.request.price if latest_sim else best_bid,
                "queue_ahead_qty": round(latest_sim.queue_ahead_qty, 4) if latest_sim else 0.0,
                "order_qty": latest_sim.request.qty if latest_sim else 0.0002,
                "filled_qty": round(latest_sim.filled_qty, 4) if latest_sim else 0.0,
                "latency_ms": 15.0,
            },
            "dsr_audit": {
                "deflated_sharpe_ratio": round(dsr_res.deflated_sharpe_ratio, 4) if dsr_res.deflated_sharpe_ratio is not None else None,
                "observed_sharpe_sr": round(dsr_res.observed_sharpe_sr, 3) if dsr_res.observed_sharpe_sr is not None else None,
                "benchmark_null_sr0": round(dsr_res.benchmark_null_sr0, 3) if dsr_res.benchmark_null_sr0 is not None else None,
                "skewness_gamma3": round(dsr_res.skewness_gamma3, 3) if dsr_res.skewness_gamma3 is not None else None,
                "kurtosis_gamma4": round(dsr_res.kurtosis_gamma4, 3) if dsr_res.kurtosis_gamma4 is not None else None,
                "sample_length_t": dsr_res.sample_length_t,
                "status": dsr_res.status,
                "is_overfitted": dsr_res.is_overfitted,
                "annualized_sharpe": round(dsr_res.annualized_sharpe, 3) if dsr_res.annualized_sharpe is not None else None,
            },
            "leverage_analysis": self.evaluate_leverage_tiers(),
        }



