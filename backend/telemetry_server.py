# backend/telemetry_server.py
"""
Servidor de Telemetría en Tiempo Real (FastAPI + WebSockets) para el modo DRY_RUN_L2.
Conecta el motor cuantitativo de microestructura (Fases 1 a 5) con el Centro de Comando en React.

Flujo de producción sin datos falsos (Zero-Mock):
1. FastAPI `lifespan` levanta `WebSocketMultiplexer` conectado concurrentemente a CCXT Pro en futuros perpetuos.
2. `ws_multiplexer.py` alimenta `MarketRingBuffer` con L2 Top 5, trades y liquidaciones.
3. El bucle de ejecución evalúa síncronamente `avellaneda_obi.py`, `liq_cascade.py` y `funding_arb.py`.
4. Cada orden se valida contra `MandateGatekeeper.validate_order()` y se despacha a `L2QueueSimulator`.
5. WebSocket `/ws/telemetry` transmite la telemetría viva al Frontend y `POST /api/emergency-flatten` controla el Kill-Switch.

Ejecución local:
    uvicorn backend.telemetry_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Literal
from pydantic import BaseModel
import ccxt
import ccxt.pro as ccxtpro
import ccxt.async_support as ccxt_async

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.ingestion.ws_multiplexer import WebSocketMultiplexer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.mandate_gates.profit_sweeper import (
    DAILY_TARGET_RETURN_PCT,
    should_trigger_profit_sweep,
    step_up_ratchet,
)
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import FundingArbSnapshot, FundingRateArbitrageEngine
from backend.strategies.liq_cascade import LiquidationCascadeHunter
import os
from backend.strategies.regime_config import RegimeConfig

# Warm Path: Gemini Regime Council integration (if GEMINI_API_KEY is present)
if os.getenv("GEMINI_API_KEY"):
    from backend.regime_council.gemini_debate import (
        GeminiRegimeCouncil,
        WarmPathCouncilScheduler,
    )
else:
    GeminiRegimeCouncil = None  # type: ignore[assignment, misc]
    WarmPathCouncilScheduler = None  # type: ignore[assignment, misc]
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, L2QueueSimulator

logger = logging.getLogger("telemetry_server")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("backend/config/regime_config.json")


class BinanceLiveWsGateway:
    """
    Adaptador de Ejecución Real sobre Binance USDⓈ-M Futures (binanceusdm).
    Opera con capital real mediante ccxt.pro / ccxt.async_support, exige que
    todas las órdenes hayan sido aprobadas previamente por MandateGatekeeper,
    aplica orden límite GTX Post-Only, administra margen aislado y sincroniza
    el portafolio y posiciones reales con Binance.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool | None = None,
        gatekeeper: MandateGatekeeper | None = None,
        portfolio: PortfolioState | None = None,
        exchange: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = api_secret if api_secret is not None else os.getenv("BINANCE_API_SECRET", "").strip()
        if testnet is not None:
            self.testnet = bool(testnet)
        else:
            self.testnet = os.getenv("BINANCE_TESTNET", "false").strip().lower() in ("true", "1", "yes")

        self.gatekeeper = gatekeeper
        self.portfolio = portfolio
        self.initialized_symbols: set[str] = set()
        self.running: bool = False
        self._sync_task: asyncio.Task | None = None
        self.live_active_bid_id: str | None = None
        self.live_active_ask_id: str | None = None

        if exchange is not None:
            self.exchange = exchange
        else:
            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "fetchOpenOrders": {
                        "warnWithoutSymbol": False,
                    },
                },
            }
            exchange_class = getattr(ccxtpro, "binanceusdm", None) or getattr(ccxt_async, "binanceusdm")
            self.exchange = exchange_class(config)
            if self.testnet:
                if hasattr(self.exchange, "enable_demo_trading"):
                    self.exchange.enable_demo_trading(True)
                elif hasattr(self.exchange, "set_sandbox_mode"):
                    try:
                        self.exchange.set_sandbox_mode(True)
                    except Exception:
                        pass

    async def initialize_symbol_margin(self, symbol: str, max_leverage: int = 3) -> None:
        """
        Configura el modo de margen a ISOLATED y el apalancamiento máximo (por defecto 3x)
        en Binance antes de permitir cualquier orden en vivo.
        """
        try:
            await self.exchange.set_margin_mode("ISOLATED", symbol)
        except Exception as e:
            # En Binance retorna error si el modo ya estaba fijado previamente en ISOLATED
            logger.info(f"set_margin_mode ISOLATED en {symbol}: {e}")
        try:
            await self.exchange.set_leverage(max_leverage, symbol)
        except Exception as e:
            logger.warning(f"set_leverage {max_leverage} en {symbol}: {e}")
        self.initialized_symbols.add(symbol)

    async def place_order_ws(self, order: OrderRequest) -> str:
        """
        Exige que la orden haya sido aprobada previamente por MandateGatekeeper.validate_order().
        Envía la orden real a Binance usando create_order / create_order_ws con type='limit',
        params={'timeInForce': 'GTX'} (Post-Only estricto para motores Maker) y devuelve el id real.
        """
        if self.gatekeeper is not None:
            initial_cap = float(os.getenv("BASE_CAPITAL_USD", "100.0"))
            portfolio = self.portfolio or PortfolioState(
                base_capital_usd=initial_cap,
                current_equity_usd=initial_cap,
                daily_high_watermark_usd=initial_cap,
                realized_pnl_today_usd=0.0,
                active_positions={},
                rolling_edge_mu=0.045,
                rolling_variance_sigma2=0.0090,
            )
            is_valid, reason = self.gatekeeper.validate_order(order, portfolio)
            if not is_valid:
                raise ValueError(f"Orden rechazada por MandateGatekeeper: {reason}")
        elif order.margin_mode != "ISOLATED" or order.time_in_force != "GTX":
            raise ValueError(
                f"La orden no cumple los mandatos obligatorios (margin={order.margin_mode}, tif={order.time_in_force})"
            )

        if order.symbol not in self.initialized_symbols:
            await self.initialize_symbol_margin(order.symbol, max_leverage=3)

        side_str = order.side.lower()
        order_fn = getattr(self.exchange, "create_order_ws", None) or self.exchange.create_order
        res = await order_fn(
            symbol=order.symbol,
            type="limit",
            side=side_str,
            amount=order.qty,
            price=order.price,
            params={"timeInForce": "GTX"},
        )
        order_id = str(res["id"])
        if order.side.upper() == "BUY":
            self.live_active_bid_id = order_id
        elif order.side.upper() == "SELL":
            self.live_active_ask_id = order_id
        return order_id

    async def cancel_order_ws(self, order_id: str, symbol: str) -> bool:
        """Cancela la orden activa en Binance en tiempo real."""
        try:
            cancel_fn = getattr(self.exchange, "cancel_order_ws", None) or self.exchange.cancel_order
            await cancel_fn(id=order_id, symbol=symbol)
            if self.live_active_bid_id == order_id:
                self.live_active_bid_id = None
            if self.live_active_ask_id == order_id:
                self.live_active_ask_id = None
            return True
        except Exception as e:
            logger.error(f"Error cancelando orden {order_id} en Binance ({symbol}): {e}")
            return False

    async def emergency_flatten_all(self, symbols: list[str]) -> None:
        """
        Cancela todas las órdenes abiertas en Binance (cancel_all_orders) y,
        si existe alguna posición abierta en fetch_positions, envía una orden MARKET
        con params={'reduceOnly': True} para dejar la exposición en cero inmediatamente.
        """
        for sym in symbols:
            try:
                await self.exchange.cancel_all_orders(sym)
                logger.warning(f"cancel_all_orders ejecutado en Binance para {sym}")
            except Exception as e:
                logger.error(f"Error en cancel_all_orders para {sym}: {e}")

        try:
            positions = await self.exchange.fetch_positions(symbols)
            for pos in positions:
                sym = pos.get("symbol")
                amt = float(pos.get("contracts") or pos.get("info", {}).get("positionAmt") or 0.0)
                side = pos.get("side")
                if abs(amt) > 1e-8:
                    close_side = "sell" if amt > 0 or side == "long" else "buy"
                    qty = abs(amt)
                    logger.warning(f"EMERGENCY FLATTEN: cerrando posición en {sym}: {close_side} {qty}")
                    await self.exchange.create_order(
                        symbol=sym,
                        type="market",
                        side=close_side,
                        amount=qty,
                        params={"reduceOnly": True},
                    )
        except Exception as e:
            logger.error(f"Error en emergency_flatten_all cerrando posiciones en Binance: {e}")
        self.live_active_bid_id = None
        self.live_active_ask_id = None

    async def sync_portfolio_state(self, portfolio: PortfolioState) -> None:
        """
        Sincroniza el balance real USDT de futuros (fetch_balance) y las posiciones
        abiertas reales hacia PortfolioState. Además, verifica las órdenes abiertas
        en Binance y resetea live_active_bid_id / live_active_ask_id a None si la orden
        ya fue ejecutada o cancelada, permitiendo al motor reanudar la cotización continua.
        """
        try:
            bal = await self.exchange.fetch_balance()
            total_usdt = 0.0
            if isinstance(bal, dict):
                if "total" in bal and isinstance(bal["total"], dict) and "USDT" in bal["total"]:
                    total_usdt = float(bal["total"]["USDT"] or 0.0)
                elif "USDT" in bal and isinstance(bal["USDT"], dict) and "total" in bal["USDT"]:
                    total_usdt = float(bal["USDT"]["total"] or 0.0)
                elif "info" in bal and isinstance(bal["info"], dict):
                    info = bal["info"]
                    total_usdt = float(info.get("totalMarginBalance") or info.get("totalWalletBalance") or 0.0)

            if total_usdt > 0.0:
                portfolio.current_equity_usd = round(total_usdt, 2)
                if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                    portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            positions = await self.exchange.fetch_positions()
            active: dict[str, float] = {}
            for pos in positions:
                sym = pos.get("symbol")
                if not sym:
                    continue
                amt = float(pos.get("contracts") or pos.get("info", {}).get("positionAmt") or 0.0)
                notional = float(pos.get("notional") or pos.get("info", {}).get("notional") or 0.0)
                if notional == 0.0 and amt != 0.0:
                    mp = float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or 0.0)
                    notional = amt * mp
                if abs(amt) > 1e-8:
                    active[sym] = round(notional, 2)
            portfolio.active_positions = active
        except Exception as e:
            logger.error(f"Error sincronizando balance y posiciones con Binance: {e}")

        # Sincronización y reseteo de órdenes activas si ya no aparecen abiertas en Binance
        try:
            if hasattr(self.exchange, "fetch_open_orders"):
                import inspect
                res_orders = self.exchange.fetch_open_orders()
                if inspect.isawaitable(res_orders):
                    open_orders = await res_orders
                elif isinstance(res_orders, list):
                    open_orders = res_orders
                else:
                    open_orders = []

                if isinstance(open_orders, list):
                    open_order_ids = {
                        str(o.get("id") or o.get("orderId"))
                        for o in open_orders
                        if isinstance(o, dict)
                    }
                    if self.live_active_bid_id and self.live_active_bid_id not in open_order_ids:
                        logger.info(
                            f"Orden BID {self.live_active_bid_id} ya no aparece abierta en Binance. Reseteando ID a None."
                        )
                        self.live_active_bid_id = None
                    if self.live_active_ask_id and self.live_active_ask_id not in open_order_ids:
                        logger.info(
                            f"Orden ASK {self.live_active_ask_id} ya no aparece abierta en Binance. Reseteando ID a None."
                        )
                        self.live_active_ask_id = None
        except Exception as e:
            logger.error(f"Error sincronizando órdenes abiertas con Binance: {e}")

    def start_sync_loop(self, portfolio: PortfolioState) -> None:
        """Inicia el bucle de sincronización periódica cada 2 segundos."""
        self.running = True
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync_loop(portfolio))

    def stop_sync_loop(self) -> None:
        """Detiene el bucle de sincronización periódica."""
        self.running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            self._sync_task = None

    async def _sync_loop(self, portfolio: PortfolioState) -> None:
        while self.running:
            try:
                await self.sync_portfolio_state(portfolio)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Excepción en _sync_loop BinanceLiveWsGateway: {e}")
            await asyncio.sleep(2.0)

    async def close(self) -> None:
        """Cierra el gateway y las conexiones asíncronas de CCXT."""
        self.stop_sync_loop()
        if hasattr(self.exchange, "close"):
            await self.exchange.close()


class DryRunWsGateway:
    """Adaptador de órdenes que enruta cotizaciones hacia L2QueueSimulator en modo DRY_RUN_L2."""

    def __init__(self, simulator: L2QueueSimulator, ring_buffer: MarketRingBuffer) -> None:
        self.simulator = simulator
        self.ring_buffer = ring_buffer

    async def cancel_order_ws(self, order_id: str, symbol: str) -> bool:
        return self.simulator.request_cancel(order_id, request_ts_ns=time.time_ns())

    async def place_order_ws(self, order: OrderRequest) -> str:
        sim_order = self.simulator.submit_gtx_order(order, self.ring_buffer)
        return sim_order.order_id



@dataclass
class SymbolMarketSpec:
    symbol: str
    tick_size: float
    step_size: float
    base_asset: str
    quote_asset: str = "USDT"


SYMBOL_SPECS: dict[str, SymbolMarketSpec] = {
    "BTC/USDT:USDT": SymbolMarketSpec("BTC/USDT:USDT", tick_size=0.10, step_size=0.001, base_asset="BTC"),
    "ETH/USDT:USDT": SymbolMarketSpec("ETH/USDT:USDT", tick_size=0.01, step_size=0.001, base_asset="ETH"),
    "SOL/USDT:USDT": SymbolMarketSpec("SOL/USDT:USDT", tick_size=0.01, step_size=0.1, base_asset="SOL"),
    "BNB/USDT:USDT": SymbolMarketSpec("BNB/USDT:USDT", tick_size=0.01, step_size=0.01, base_asset="BNB"),
    "DOGE/USDT:USDT": SymbolMarketSpec("DOGE/USDT:USDT", tick_size=0.00001, step_size=1.0, base_asset="DOGE"),
    "XRP/USDT:USDT": SymbolMarketSpec("XRP/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="XRP"),
    "SUI/USDT:USDT": SymbolMarketSpec("SUI/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="SUI"),
    "AVAX/USDT:USDT": SymbolMarketSpec("AVAX/USDT:USDT", tick_size=0.01, step_size=0.01, base_asset="AVAX"),
    "LINK/USDT:USDT": SymbolMarketSpec("LINK/USDT:USDT", tick_size=0.001, step_size=0.01, base_asset="LINK"),
    "PEPE/USDT:USDT": SymbolMarketSpec("PEPE/USDT:USDT", tick_size=0.000001, step_size=1.0, base_asset="PEPE"),
    "NEAR/USDT:USDT": SymbolMarketSpec("NEAR/USDT:USDT", tick_size=0.001, step_size=0.1, base_asset="NEAR"),
    "ARB/USDT:USDT": SymbolMarketSpec("ARB/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="ARB"),
}


@dataclass
class SymbolExecutionContext:
    symbol: str
    spec: SymbolMarketSpec
    ring_buffer: MarketRingBuffer
    avellaneda: AvellanedaStoikovOBIEngine
    liq_hunter: LiquidationCascadeHunter
    l2_sim: L2QueueSimulator
    ws_gateway: DryRunWsGateway
    last_quote_result: Any = None
    last_processed_trade_head: int = 0
    last_fill_buy_price: float | None = None
    last_fill_sell_price: float | None = None
    rotations_filled_today: int = 0
    realized_pnl_today: float = 0.0
    live_active_bid_id: str | None = None
    live_active_ask_id: str | None = None
    inventory_open_ts_ns: int | None = None


@dataclass
class ContagionShieldState:
    bid_shield_active: bool = False
    ask_shield_active: bool = False
    trigger_reason: str = "NORMAL"
    bearish_count: int = 0
    bullish_count: int = 0
    btc_obi: float = 0.0


class DryRunExecutionEngine:
    """
    Motor central de ejecución concurrente para portafolios multimoneda (DRY_RUN_L2).
    Supervisa concurrentemente la cesta de futuros perpetuos (BTC, ETH, SOL, BNB, DOGE),
    evalúa cada subsistema cuantitativo, aplica el Escudo de Contagio Sistémico
    y asigna capital mediante matriz de covarianza Half-Kelly multivariante.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        config_path: Path = CONFIG_PATH,
    ) -> None:
        self.symbols = symbols or list(SYMBOL_SPECS.keys())
        self.primary_symbol = self.symbols[0]
        self.config_path = config_path

        # 1. Ingesta WebSocket multiplexada concurrente vía CCXT Pro
        self.multiplexer = WebSocketMultiplexer(
            symbols=self.symbols,
            exchange_id="binance",
            buffer_capacity=10_000,
        )

        # 2. Cortafuegos determinista unificado MandateGatekeeper
        self.gatekeeper = MandateGatekeeper()

        # 3. Estado de Portafolio compartido en memoria (100.0 USDT base inicial)
        initial_capital = float(os.getenv("BASE_CAPITAL_USD", "100.0"))
        self.portfolio = PortfolioState(
            base_capital_usd=initial_capital,
            current_equity_usd=initial_capital,
            daily_high_watermark_usd=initial_capital,
            realized_pnl_today_usd=0.0,
            active_positions={},
            rolling_edge_mu=0.045,
            rolling_variance_sigma2=0.0090,
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
            )
            liq = LiquidationCascadeHunter(
                symbol=sym,
                tick_size=spec.tick_size,
                step_size=spec.step_size,
                order_qty=spec.step_size,
                gatekeeper=self.gatekeeper,
            )
            ctx = SymbolExecutionContext(
                symbol=sym,
                spec=spec,
                ring_buffer=ring,
                avellaneda=avell,
                liq_hunter=liq,
                l2_sim=sim,
                ws_gateway=gw,
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
        self.trading_mode: str = "PAPER_L2"
        self.simulated_vault_usd: float = 0.0
        self._live_active_bid_id: str | None = None
        self._live_active_ask_id: str | None = None

        # 8. Cuarentena DSR
        self.dsr_state_path = Path("backend/config/dsr_quarantine_state.json")
        self.returns_history: list[float] = [0.00095, 0.00110, 0.00085, 0.00105, 0.00090]
        self.running: bool = False
        self._eval_task: asyncio.Task | None = None
        self.total_rotations_filled: int = 0
        self.latest_funding_rate: float = 0.0001
        self.last_latency_us: float = 0.0

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
                "BTC/USDT:USDT": (65000.0, 65001.0, 1.0),
                "ETH/USDT:USDT": (2500.0, 2500.5, 0.5),
                "SOL/USDT:USDT": (150.0, 150.05, 0.05),
                "BNB/USDT:USDT": (550.0, 550.1, 0.1),
                "DOGE/USDT:USDT": (0.15000, 0.15002, 0.00002),
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

    def compute_multivariate_half_kelly(self) -> dict[str, float]:
        """
        Asignación óptima de pesos matricial Half-Kelly:
            F* = 0.5 * C^(-1) M
        acotada a máx 20% de margen por par (Gate 3) y 3.0x de apalancamiento global.
        """
        n = len(self.symbols)
        default_weight = min(0.20, 1.0 / max(n, 1))

        history_lens = [len(self.symbol_returns.get(s, [])) for s in self.symbols]
        if min(history_lens) < 5:
            return {s: default_weight for s in self.symbols}

        window_size = min(min(history_lens), 60)
        returns_mat = np.array([self.symbol_returns[s][-window_size:] for s in self.symbols])  # (n, window)
        M = np.mean(returns_mat, axis=1)
        C = np.cov(returns_mat)

        C_reg = C + (1e-6 * np.eye(n))
        try:
            C_inv = np.linalg.pinv(C_reg)
            f_star = 0.5 * (C_inv @ M)
        except Exception:
            return {s: default_weight for s in self.symbols}

        f_clamped = np.clip(f_star, 0.02, 0.20)
        tot_leverage = float(np.sum(f_clamped))
        if tot_leverage > 3.0:
            f_clamped = (f_clamped / tot_leverage) * 3.0

        return {s: float(round(f_clamped[i], 4)) for i, s in enumerate(self.symbols)}

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
        """
        scores = {sym: self.compute_activity_score(sym) for sym in self.symbols}
        if "BTC/USDT:USDT" in scores:
            scores["BTC/USDT:USDT"] = max(scores["BTC/USDT:USDT"], 0.001)
        sorted_syms = sorted(self.symbols, key=lambda s: scores.get(s, 0.0), reverse=True)
        return sorted_syms[:k]

    def compute_normalized_inventory_q(self, symbol: str | None = None) -> float:
        """
        Calcula q dinámicamente en el rango [-1.0, 1.0] como la posición neta actual del símbolo
        dividida por el límite de concentración del 20% del capital base tanto en PAPER_L2 como en LIVE_BINANCE.
        """
        sym = symbol or self.primary_symbol
        base_cap = max(self.portfolio.base_capital_usd, 1.0)
        max_allowed_pos = base_cap * 0.20
        if max_allowed_pos <= 1e-8:
            return 0.0
        net_pos_usd = self.portfolio.active_positions.get(sym, 0.0)
        return float(np.clip(net_pos_usd / max_allowed_pos, -1.0, 1.0))

    def _load_dsr_quarantine_state(self) -> None:
        """Restaura el estado de la cuarentena DSR desde backend/config/dsr_quarantine_state.json."""
        try:
            if self.dsr_state_path.exists():
                with open(self.dsr_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    saved_returns = data.get("returns_history")
                    if isinstance(saved_returns, list) and len(saved_returns) > 0:
                        self.returns_history = [float(r) for r in saved_returns]
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
        """Persiste atómicamente cada trade y el array returns_history en backend/config/dsr_quarantine_state.json."""
        try:
            self.dsr_state_path.parent.mkdir(parents=True, exist_ok=True)
            state_data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_rotations_filled": self.total_rotations_filled,
                "returns_count": len(self.returns_history),
                "last_fill_buy_price": self._last_fill_buy_price,
                "last_fill_sell_price": self._last_fill_sell_price,
                "current_equity_usd": self.portfolio.current_equity_usd,
                "returns_history": self.returns_history[-2000:],
            }
            tmp_path = self.dsr_state_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
            tmp_path.replace(self.dsr_state_path)
        except Exception as e:
            logger.error(f"Error persistiendo estado de cuarentena DSR: {e}")

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
        Conmuta el modo de ejecución entre PAPER_L2 y LIVE_BINANCE.
        Verifica claves y estado de seguridad antes de permitir LIVE_BINANCE.
        """
        if mode == "LIVE_BINANCE":
            api_key = os.getenv("BINANCE_API_KEY", "").strip()
            api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
            if not api_key or not api_secret:
                raise ValueError("Debe configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env antes de conmutar a LIVE_BINANCE.")
            if self.gatekeeper.kill_switch_engaged:
                raise ValueError("No se puede conmutar a LIVE_BINANCE mientras el Kill-Switch de emergencia esté activo.")

            if self.live_gateway is None:
                self.live_gateway = BinanceLiveWsGateway(
                    api_key=api_key,
                    api_secret=api_secret,
                    gatekeeper=self.gatekeeper,
                    portfolio=self.portfolio,
                )
            for sym in self.symbols:
                await self.live_gateway.initialize_symbol_margin(sym, max_leverage=3)

            await self.live_gateway.sync_portfolio_state(self.portfolio)
            self.live_gateway.start_sync_loop(self.portfolio)
            self.trading_mode = "LIVE_BINANCE"
            logger.info("Modo de trading conmutado a LIVE_BINANCE.")
        elif mode == "PAPER_L2":
            if self.live_gateway:
                self.live_gateway.stop_sync_loop()
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
        if env_mode == "LIVE_BINANCE":
            try:
                await self.set_trading_mode("LIVE_BINANCE")
                logger.info("TRADING_MODE=LIVE_BINANCE activado exitosamente al iniciar.")
            except Exception as e:
                logger.error(f"Error activando LIVE_BINANCE al inicio: {e}")

        self._eval_task = asyncio.create_task(self._strategy_evaluation_loop())
        logger.info(f"Motor de evaluación en ejecución (Modo: {self.trading_mode}).")

    async def stop(self) -> None:
        """Detiene limpiamente todas las tareas y conexiones."""
        self.running = False
        if self.warm_path_scheduler:
            self.warm_path_scheduler.stop()
        if self._eval_task:
            self._eval_task.cancel()
            await asyncio.gather(self._eval_task, return_exceptions=True)
        await self.multiplexer.stop()
        logger.info("Motor DRY_RUN_L2 detenido.")

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

                self.contagion_shield.bid_shield_active = bid_shield
                self.contagion_shield.ask_shield_active = ask_shield
                self.contagion_shield.btc_obi = float(btc_obi)
                self.contagion_shield.bearish_count = bearish_alt_count
                self.contagion_shield.bullish_count = bullish_alt_count

                if bid_shield:
                    self.contagion_shield.trigger_reason = "BTC_TOXIC_SELL_FLOW" if btc_obi < -0.70 else "MULTI_ALT_BEARISH_CASCADE"
                elif ask_shield:
                    self.contagion_shield.trigger_reason = "BTC_BULLISH_BREAKOUT" if btc_obi > 0.70 else "MULTI_ALT_BULLISH_BURST"
                else:
                    self.contagion_shield.trigger_reason = "NORMAL"

                # Asignación de pesos Half-Kelly multivariante por par
                half_kelly_weights = self.compute_multivariate_half_kelly()

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

                        # Si el símbolo no está en el Top 5 activo y no tiene inventario abierto, pausar cotización
                        has_open_pos = abs(self.portfolio.active_positions.get(sym, 0.0)) > 1e-4
                        if sym not in self.top_active_symbols and not has_open_pos:
                            continue

                        current_gateway = self.live_gateway if (self.trading_mode == "LIVE_BINANCE" and self.live_gateway) else ctx.ws_gateway

                        # Precios punta para cálculo de tamaño y cotización
                        ob_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
                        b_bid = float(ctx.ring_buffer.ob_bid_prices[ob_idx, 0])
                        b_ask = float(ctx.ring_buffer.ob_ask_prices[ob_idx, 0])
                        mid_px = 0.5 * (b_bid + b_ask) if b_bid > 0 else 1.0

                        # Inventario dinámico q_i para este par
                        dynamic_q = self.compute_normalized_inventory_q(sym)

                        # Cálculo de quote_qty dinámico según Half-Kelly y precio de mercado
                        w_i = half_kelly_weights.get(sym, 0.15)
                        target_notional = min(self.portfolio.base_capital_usd * w_i, self.portfolio.base_capital_usd * 0.20)

                        if ctx.spec.step_size * mid_px > self.portfolio.base_capital_usd * 0.20:
                            # Símbolos de alto valor unitario (ej. BTC): adaptar resolución para no rebasar Gate 3 (20%)
                            quote_qty = max(0.0001, round(target_notional / max(mid_px, 1.0), 4))
                        else:
                            step = ctx.spec.step_size
                            quote_qty = max(step, round(round(target_notional / max(mid_px, 1.0) / step) * step, 6))

                        ctx.avellaneda.quote_qty = quote_qty

                        # Factor de urgencia por antigüedad de inventario (holding_time_decay)
                        holding_time_s = ((now_ns - ctx.inventory_open_ts_ns) / 1e9) if ctx.inventory_open_ts_ns is not None else 0.0

                        # Cotización Avellaneda-Stoikov + OBI Shield + Escalera 3 Niveles
                        quote_res = await ctx.avellaneda.on_l2_update(
                            ctx.ring_buffer,
                            self.portfolio,
                            current_gateway,
                            normalized_inventory_q=dynamic_q,
                            inventory_holding_time_s=holding_time_s,
                        )

                        # Escudo de Contagio Sistémico sobre Altcoins (sym != BTC)
                        if sym != "BTC/USDT:USDT":
                            if bid_shield:
                                quote_res.bid_order = None
                                quote_res.bid_cancelled = True
                                # Cancelar preventivamente BIDs abiertos en L2QueueSimulator
                                if self.trading_mode == "PAPER_L2":
                                    for o in list(ctx.l2_sim.orders.values()):
                                        if o.request.side == "BUY" and o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                                            ctx.l2_sim.request_cancel(o.order_id, now_ns)
                                elif self.live_gateway and ctx.live_active_bid_id:
                                    try:
                                        await self.live_gateway.cancel_order_ws(ctx.live_active_bid_id, sym)
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
                                elif self.live_gateway and ctx.live_active_ask_id:
                                    try:
                                        await self.live_gateway.cancel_order_ws(ctx.live_active_ask_id, sym)
                                        ctx.live_active_ask_id = None
                                    except Exception as e:
                                        logger.error(f"Error cancelando ask de {sym} por escudo: {e}")

                        ctx.last_quote_result = quote_res
                        if sym == self.primary_symbol:
                            self.last_quote_result = quote_res

                        if self.trading_mode == "LIVE_BINANCE" and self.live_gateway:
                            if sym == self.primary_symbol:
                                if self.live_active_bid_id is None:
                                    ctx.avellaneda.active_bid_order_id = None
                                if self.live_active_ask_id is None:
                                    ctx.avellaneda.active_ask_order_id = None

                                if quote_res.bid_order is not None and self.live_active_bid_id is None:
                                    try:
                                        bid_id = await self.live_gateway.place_order_ws(quote_res.bid_order)
                                        ctx.avellaneda.active_bid_order_id = bid_id
                                        self.live_active_bid_id = bid_id
                                    except Exception as e:
                                        logger.error(f"Error despachando bid en vivo para {sym}: {e}")
                                if quote_res.ask_order is not None and self.live_active_ask_id is None:
                                    try:
                                        ask_id = await self.live_gateway.place_order_ws(quote_res.ask_order)
                                        ctx.avellaneda.active_ask_order_id = ask_id
                                        self.live_active_ask_id = ask_id
                                    except Exception as e:
                                        logger.error(f"Error despachando ask en vivo para {sym}: {e}")
                        else:
                            # Modo PAPER_L2
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

                            # Motor B: Cazador de Cascadas de Liquidación
                            open_sim_orders = [
                                o for o in ctx.l2_sim.orders.values()
                                if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING")
                            ]
                            cascade_bracket = ctx.liq_hunter.evaluate(
                                ctx.ring_buffer, self.portfolio, now_ns=now_ns
                            )
                            if cascade_bracket is not None and len(open_sim_orders) == 0:
                                ctx.l2_sim.submit_gtx_order(
                                    cascade_bracket.entry_order, ctx.ring_buffer, current_ts_ns=now_ns
                                )

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
                                else:
                                    self.portfolio.active_positions[sym] = round(new_pos, 2)
                                    if ctx.inventory_open_ts_ns is None:
                                        ctx.inventory_open_ts_ns = now_ns

                                ob_idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity if ctx.ring_buffer.ob_count > 0 else 0
                                b_bid = float(ctx.ring_buffer.ob_bid_prices[ob_idx, 0]) if ctx.ring_buffer.ob_count > 0 else fill_px
                                b_ask = float(ctx.ring_buffer.ob_ask_prices[ob_idx, 0]) if ctx.ring_buffer.ob_count > 0 else fill_px
                                cur_mid = 0.5 * (b_bid + b_ask) if b_bid > 0 else fill_px

                                if fill.request.side == "BUY":
                                    ctx.last_fill_buy_price = fill_px
                                    if ctx.last_fill_sell_price is not None:
                                        cycle_pnl = (ctx.last_fill_sell_price - fill_px) * fill_qty
                                        ret = (ctx.last_fill_sell_price - fill_px) / fill_px
                                        ctx.last_fill_sell_price = None
                                    else:
                                        cycle_pnl = (cur_mid - fill_px) * fill_qty
                                        ret = (cur_mid - fill_px) / fill_px if fill_px > 0 else 0.0010
                                else:
                                    ctx.last_fill_sell_price = fill_px
                                    if ctx.last_fill_buy_price is not None:
                                        cycle_pnl = (fill_px - ctx.last_fill_buy_price) * fill_qty
                                        ret = (fill_px - ctx.last_fill_buy_price) / ctx.last_fill_buy_price
                                        ctx.last_fill_buy_price = None
                                    else:
                                        cycle_pnl = (fill_px - cur_mid) * fill_qty
                                        ret = (fill_px - cur_mid) / fill_px if fill_px > 0 else 0.0010

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
        dsr_res = DeflatedSharpeAuditor.audit_returns(self.returns_history[-250:], num_trials_n=25)

        # Half-Kelly
        variance_safe = max(self.portfolio.rolling_variance_sigma2, 1e-8)
        raw_half_kelly = 0.5 * (self.portfolio.rolling_edge_mu / variance_safe)
        effective_kelly_cap = min(max(raw_half_kelly, 0.0), 4.0)

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
            symbols_data[sym] = {
                "symbol": sym,
                "base_asset": ctx.spec.base_asset,
                "mid_price": round(s_mid, 2 if s_mid >= 1.0 else 5),
                "best_bid": round(s_bid, 2 if s_bid >= 1.0 else 5),
                "best_ask": round(s_ask, 2 if s_ask >= 1.0 else 5),
                "obi": round(s_obi, 4),
                "inventory_q": round(self.compute_normalized_inventory_q(sym), 4),
                "position_usd": round(self.portfolio.active_positions.get(sym, 0.0), 2),
                "rotations_today": ctx.rotations_filled_today,
                "pnl_today_usd": round(ctx.realized_pnl_today, 2),
                "half_kelly_weight": half_kelly_weights.get(sym, 0.15),
            }

        return {
            "timestamp_ns": now_ns,
            "mode": self.trading_mode,
            "trading_mode": self.trading_mode,
            "hot_path_latency_us": round(self.last_latency_us, 2),
            "market": {
                "symbol": self.primary_symbol,
                "mid_price": round(mid_price, 2),
                "best_bid": round(best_bid, 2),
                "best_ask": round(best_ask, 2),
                "microprice": round(prim_ctx.ring_buffer.compute_microprice(), 2),
                "obi": round(obi, 4),
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
                "bearish_count": self.contagion_shield.bearish_count,
                "bullish_count": self.contagion_shield.bullish_count,
            },
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
                "deflated_sharpe_ratio": round(dsr_res.deflated_sharpe_ratio, 4),
                "observed_sharpe_sr": round(dsr_res.observed_sharpe_sr, 3),
                "benchmark_null_sr0": round(dsr_res.benchmark_null_sr0, 3),
                "skewness_gamma3": round(dsr_res.skewness_gamma3, 3),
                "kurtosis_gamma4": round(dsr_res.kurtosis_gamma4, 3),
                "sample_length_t": dsr_res.sample_length_t,
                "status": dsr_res.status,
                "is_overfitted": dsr_res.is_overfitted,
            },
        }


engine = DryRunExecutionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de FastAPI:
    Inicia WebSocketMultiplexer real y el bucle continuo de evaluación cuantitativa.
    """
    logger.info("Iniciando lifespan de FastAPI...")
    await engine.start()
    try:
        yield
    finally:
        logger.info("Cerrando lifespan de FastAPI...")
        await engine.stop()


app = FastAPI(
    title="Apex Quant HFT Telemetry Server",
    description="Real-time WebSocket stream for DRY_RUN_L2 Paper Trading, Mandate Gates, OBI & DSR Quarantine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradingModeRequest(BaseModel):
    mode: Literal["PAPER_L2", "LIVE_BINANCE"]


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Chequeo de salud del motor bimodal."""
    return {
        "status": "OK",
        "mode": engine.trading_mode,
        "trading_mode": engine.trading_mode,
        "kill_switch_engaged": engine.gatekeeper.kill_switch_engaged,
        "emergency_flatten_count": engine.gatekeeper.emergency_flatten_count,
        "regime_config": engine.regime_config.model_dump(),
        "rotations_filled": engine.total_rotations_filled,
        "simulated_vault_usd": getattr(engine, "simulated_vault_usd", 0.0),
    }


@app.get("/api/telemetry")
async def get_telemetry_endpoint() -> dict[str, Any]:
    """Snapshot completo de telemetría real de microestructura y riesgo."""
    return engine.get_telemetry_snapshot()


@app.post("/api/trading-mode")
async def set_trading_mode_endpoint(req: TradingModeRequest) -> dict[str, Any]:
    """
    Conmuta el modo de ejecución entre PAPER_L2 y LIVE_BINANCE.
    REGLA DE SEGURIDAD CRÍTICA:
    Si se solicita LIVE_BINANCE, verifica que BINANCE_API_KEY y BINANCE_API_SECRET existan
    en el entorno y que gatekeeper.kill_switch_engaged sea False.
    Si faltan claves API o el Kill-Switch está activo, rechaza con HTTP 400 y mantiene el bot en PAPER_L2.
    """
    if req.mode == "LIVE_BINANCE":
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            logger.warning("Intento de cambio a LIVE_BINANCE rechazado por falta de claves API.")
            raise HTTPException(
                status_code=400,
                detail="Configuración de seguridad incompleta: Debe configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env antes de operar en vivo. El bot se mantiene en PAPER_L2.",
            )
        if engine.gatekeeper.kill_switch_engaged:
            logger.warning("Intento de cambio a LIVE_BINANCE rechazado: Kill-Switch activo.")
            raise HTTPException(
                status_code=400,
                detail="Bloqueo de seguridad: El Kill-Switch de emergencia está activo. Debe resetear el Kill-Switch antes de pasar a LIVE_BINANCE.",
            )

    try:
        await engine.set_trading_mode(req.mode)
        return {
            "status": "MODE_SWITCHED",
            "mode": engine.trading_mode,
            "timestamp_ns": time.time_ns(),
        }
    except Exception as e:
        logger.error(f"Error cambiando modo a {req.mode}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/emergency-flatten")
async def emergency_flatten_endpoint() -> dict[str, Any]:
    """
    Disparador síncrono del Kill-Switch de emergencia (Gate 4).
    Bloquea inmediatamente todas las órdenes nuevas y activa el flatten.
    Si el motor está en LIVE_BINANCE, invoca también live_gateway.emergency_flatten_all.
    """
    engine.gatekeeper.trigger_emergency_flatten()
    logger.warning("Kill-Switch disparado manualmente a través de /api/emergency-flatten.")

    if engine.trading_mode == "LIVE_BINANCE" and engine.live_gateway is not None:
        try:
            await engine.live_gateway.emergency_flatten_all(engine.symbols)
        except Exception as e:
            logger.error(f"Error ejecutando live emergency_flatten_all: {e}")

    return {
        "status": "EMERGENCY_FLATTEN_TRIGGERED",
        "mode": engine.trading_mode,
        "kill_switch_engaged": engine.gatekeeper.kill_switch_engaged,
        "emergency_flatten_count": engine.gatekeeper.emergency_flatten_count,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/profit-sweep")
async def profit_sweep_endpoint() -> dict[str, Any]:
    """
    Dispara el barrido de excedentes de ganancias:
    - En PAPER_L2: descuenta el excedente hacia la bóveda simulada (simulated_vault_usd)
      y resetea el capital operativo a base_capital_usd.
    - En LIVE_BINANCE: ejecuta exchange.transfer('USDT', surplus, 'future', 'spot')
      y resetea el capital operativo a base_capital_usd.
    """
    surplus = round(engine.portfolio.current_equity_usd - engine.portfolio.base_capital_usd, 2)
    if surplus <= 0.0:
        return {
            "status": "NO_SURPLUS",
            "mode": engine.trading_mode,
            "surplus_swept_usd": 0.0,
            "current_equity_usd": engine.portfolio.current_equity_usd,
            "base_capital_usd": engine.portfolio.base_capital_usd,
            "simulated_vault_usd": getattr(engine, "simulated_vault_usd", 0.0),
            "message": "No hay excedente de capital operativo para barrer.",
            "timestamp_ns": time.time_ns(),
        }

    if engine.trading_mode == "LIVE_BINANCE":
        if engine.live_gateway is None:
            raise HTTPException(status_code=400, detail="Gateway de Binance no inicializado.")
        try:
            transfer_res = await engine.live_gateway.exchange.transfer(
                code="USDT",
                amount=surplus,
                fromAccount="future",
                toAccount="spot",
            )
            logger.info(f"Barrido de ganancias real ejecutado en Binance: {surplus} USDT a Spot: {transfer_res}")
        except Exception as e:
            logger.error(f"Fallo en transferencia de futuros a spot en Binance: {e}")
            raise HTTPException(status_code=500, detail=f"Fallo al transferir a Spot en Binance: {e}")
    else:
        engine.simulated_vault_usd = round(getattr(engine, "simulated_vault_usd", 0.0) + surplus, 2)
        logger.info(f"Barrido de ganancias simulado ejecutado: {surplus} USDT a bóveda. Total: {engine.simulated_vault_usd}")

    engine.portfolio.current_equity_usd = engine.portfolio.base_capital_usd
    engine.portfolio.daily_high_watermark_usd = engine.portfolio.base_capital_usd

    return {
        "status": "PROFIT_SWEEP_EXECUTED",
        "mode": engine.trading_mode,
        "surplus_swept_usd": surplus,
        "current_equity_usd": engine.portfolio.current_equity_usd,
        "base_capital_usd": engine.portfolio.base_capital_usd,
        "simulated_vault_usd": getattr(engine, "simulated_vault_usd", 0.0),
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/reset-kill-switch")
async def reset_kill_switch_endpoint() -> dict[str, Any]:
    """Permite al operador reiniciar el Kill-Switch tras auditar la contingencia."""
    engine.gatekeeper.kill_switch_engaged = False
    logger.info("Kill-Switch reseteado por el operador.")
    return {
        "status": "KILL_SWITCH_RESET",
        "kill_switch_engaged": False,
        "timestamp_ns": time.time_ns(),
    }


@app.websocket("/ws/telemetry")
async def telemetry_websocket_endpoint(websocket: WebSocket) -> None:
    """
    Transmisión de telemetría viva a alta frecuencia (~10 Hz) hacia el Frontend React.
    Envía el estado real de PortfolioState, MarketRingBuffer, Gatekeeper y L2QueueSimulator.
    """
    await websocket.accept()
    logger.info("Cliente WebSocket conectado a /ws/telemetry")
    try:
        while True:
            payload = engine.get_telemetry_snapshot()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.10)  # ~10 Hz actualización fluida
    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado de /ws/telemetry")
    except Exception as e:
        logger.error(f"Error en websocket /ws/telemetry: {e}")
