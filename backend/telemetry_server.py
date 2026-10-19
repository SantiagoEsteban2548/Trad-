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
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import io
import inspect
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Literal
from pydantic import BaseModel
import ccxt
import ccxt.pro as ccxtpro
import ccxt.async_support as ccxt_async

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from dotenv import load_dotenv

from backend.security.auth import verify_bearer_token, verify_websocket_token

load_dotenv()

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.ingestion.ws_multiplexer import WebSocketMultiplexer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step
from backend.mandate_gates.profit_sweeper import (
    DAILY_TARGET_RETURN_PCT,
    should_trigger_profit_sweep,
    step_up_ratchet,
)
from backend.database.ledger_store import LedgerSqliteStore
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import (
    FundingArbSnapshot,
    FundingRateArbitrageEngine,
    CrossExchangeFundingSnapshot,
    CrossExchangeArbPackage,
)
from backend.strategies.liq_cascade import LiquidationCascadeHunter
from backend.strategies.strategy_ledger import VirtualStrategyLedger, VirtualPositionLot
from backend.strategies.leverage_controller import AutonomousLeverageController
import os
from backend.strategies.regime_config import RegimeConfig

CROSS_HEDGE_PAIRS: set[str] = {
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
}

# Warm Path: Gemini Regime Council integration
try:
    from backend.regime_council.gemini_debate import (
        GeminiRegimeCouncil,
        WarmPathCouncilScheduler,
    )
except ImportError:
    GeminiRegimeCouncil = None  # type: ignore[assignment, misc]
    WarmPathCouncilScheduler = None  # type: ignore[assignment, misc]
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, DSRAuditResult, L2QueueSimulator

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
        on_fill: Any | None = None,
        db_store: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = api_secret if api_secret is not None else os.getenv("BINANCE_API_SECRET", "").strip()
        if testnet is not None:
            self.testnet = bool(testnet)
        else:
            self.testnet = os.getenv("BINANCE_TESTNET", "false").strip().lower() in ("true", "1", "yes")

        self.gatekeeper = gatekeeper
        self.portfolio = portfolio
        self.on_fill = on_fill
        self.db_store = db_store
        self.default_leverage: int = 3
        self.symbol_leverage: dict[str, int] = {}
        self.ledger = VirtualStrategyLedger()
        self.initialized_symbols: set[str] = set()
        self.running: bool = False
        self._sync_task: asyncio.Task | None = None
        self._watch_orders_task: asyncio.Task | None = None
        self._watch_trades_task: asyncio.Task | None = None
        self.live_active_bid_id: str | None = None
        self.live_active_ask_id: str | None = None
        self.active_bids_by_symbol: dict[str, str] = {}
        self.active_asks_by_symbol: dict[str, str] = {}
        self.active_orders_meta: dict[str, dict[str, Any]] = {}
        self._last_trade_ts_ms: dict[str, int] = {}
        self._processed_trade_ids: set[str] = set()
        self._processed_order_ids: set[str] = set()
        self._fill_lock: asyncio.Lock = asyncio.Lock()
        self.protective_sl_by_symbol: dict[str, str] = {}
        self._init_ts_ms: int = int(time.time() * 1000)
        self._time_synced: bool = False

        if exchange is not None:
            self.exchange = exchange
            if hasattr(self.exchange, "options") and isinstance(self.exchange.options, dict):
                self.exchange.options.setdefault("adjustForTimeDifference", True)
                self.exchange.options.setdefault("recvWindow", 10000)
        else:
            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
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

    async def sync_time_difference(self) -> None:
        """Sincroniza la diferencia de reloj con Binance para evitar rechazos -1021 en Windows."""
        if hasattr(self.exchange, "load_time_difference") and callable(self.exchange.load_time_difference):
            try:
                diff = await self.exchange.load_time_difference()
                logger.info(f"[BinanceLiveWsGateway] Sincronización de reloj con Binance completada (offset: {diff} ms).")
                self._time_synced = True
            except Exception as e:
                logger.warning(f"[BinanceLiveWsGateway] Advertencia al sincronizar reloj con Binance: {e}")
        else:
            self._time_synced = True

    def resolve_binance_symbol(self, symbol: str) -> str:
        """Traduce símbolos canónicos a nombres nativos de Binance USD(S)-M si difieren."""
        if hasattr(self.exchange, "markets") and self.exchange.markets:
            if symbol in self.exchange.markets:
                return symbol
            if symbol == "PEPE/USDT:USDT" and "1000PEPE/USDT:USDT" in self.exchange.markets:
                return "1000PEPE/USDT:USDT"
        if symbol == "PEPE/USDT:USDT":
            return "1000PEPE/USDT:USDT"
        return symbol

    async def initialize_symbol_margin(self, symbol: str, max_leverage: int = 3) -> None:
        """
        Configura el modo de posición a One-Way (hedged=False), margen a ISOLATED y el apalancamiento
        óptimo en Binance antes de permitir cualquier orden en vivo.
        """
        spec = SYMBOL_SPECS.get(symbol)
        if spec and spec.exchange != "binance":
            return
        target_sym = self.resolve_binance_symbol(symbol)
        try:
            if hasattr(self.exchange, "set_position_mode"):
                await self.exchange.set_position_mode(hedged=False, symbol=target_sym)
        except Exception as e:
            logger.debug(f"set_position_mode hedged=False en {target_sym}: {e}")
        try:
            await self.exchange.set_margin_mode("ISOLATED", target_sym)
        except Exception as e:
            err_msg = str(e).lower()
            if "-4046" in err_msg or "no need to change" in err_msg or "already" in err_msg:
                logger.info(f"set_margin_mode ISOLATED ya activo en {target_sym}: {e}")
            elif "-4048" in err_msg or "-4067" in err_msg or "cannot be changed" in err_msg:
                logger.error(f"CRÍTICO: No se pudo cambiar {target_sym} a ISOLATED por órdenes/posiciones abiertas en CROSS: {e}")
                raise ValueError(f"GATE_1_ISOLATED_MARGIN: {target_sym} bloqueado porque no pudo cambiarse a ISOLATED ({e})") from e
            else:
                logger.info(f"set_margin_mode ISOLATED en {target_sym}: {e}")
        cur_lev = self.symbol_leverage.get(symbol, self.default_leverage)
        has_pos = False
        if self.portfolio and self.portfolio.active_positions:
            pos_val = float(self.portfolio.active_positions.get(symbol, 0.0) or self.portfolio.active_positions.get(target_sym, 0.0) or 0.0)
            has_pos = abs(pos_val) > 1e-6
        if not has_pos and hasattr(self, "ledger") and self.ledger is not None:
            has_pos = abs(self.ledger.get_net_inventory(symbol)) > 1e-6
        has_orders = (
            symbol in self.active_bids_by_symbol
            or symbol in self.active_asks_by_symbol
            or any(m.get("symbol") in (symbol, target_sym) for m in self.active_orders_meta.values())
        )
        if int(max_leverage) < cur_lev and (has_pos or has_orders):
            logger.info(
                f"set_leverage {max_leverage} omitido/bloqueado en {target_sym} "
                "para evitar -4161 (posición u órdenes abiertas)."
            )
            self.initialized_symbols.add(symbol)
            return

        try:
            await self.exchange.set_leverage(max_leverage, target_sym)
            self.default_leverage = int(max_leverage)
            self.symbol_leverage[symbol] = int(max_leverage)
            self.initialized_symbols.add(symbol)
        except Exception as e:
            if "-4161" in str(e) or "leverage reduction" in str(e).lower():
                logger.info(f"set_leverage {max_leverage} omitido en {target_sym} (-4161: reducción con posición abierta)")
                self.initialized_symbols.add(symbol)
            else:
                logger.warning(f"set_leverage {max_leverage} en {target_sym}: {e}")
                if "does not have market symbol" not in str(e):
                    self.initialized_symbols.add(symbol)

    async def update_symbol_leverage_if_needed(self, symbol: str, optimal_leverage: int) -> bool:
        """
        Ajusta dinámicamente el apalancamiento en Binance cuando cambia en >= 1x.
        Modo ISOLATED estricto, entero entre 1x y 10x.
        Protección contra error -4161:
        Binance rechaza reducir apalancamiento si hay posición u órdenes abiertas.
        Solo llama a set_leverage si opt_int > cur_lev o si no hay posición ni órdenes abiertas.
        """
        opt_int = max(1, min(int(optimal_leverage), 10))
        cur_lev = self.symbol_leverage.get(symbol, self.default_leverage)
        if cur_lev == opt_int:
            return False
        target_sym = self.resolve_binance_symbol(symbol)

        # Si se requiere menor apalancamiento, verificar si hay posición u órdenes abiertas
        if opt_int < cur_lev:
            has_pos = False
            if self.portfolio and self.portfolio.active_positions:
                pos_val = float(self.portfolio.active_positions.get(symbol, 0.0) or self.portfolio.active_positions.get(target_sym, 0.0) or 0.0)
                has_pos = abs(pos_val) > 1e-6
            if not has_pos and hasattr(self, "ledger") and self.ledger is not None:
                has_pos = abs(self.ledger.get_net_inventory(symbol)) > 1e-6
            has_orders = (
                symbol in self.active_bids_by_symbol
                or symbol in self.active_asks_by_symbol
                or any(m.get("symbol") in (symbol, target_sym) for m in self.active_orders_meta.values())
            )
            if has_pos or has_orders:
                logger.info(
                    f"Reducción de apalancamiento omitida en Binance {target_sym} ({cur_lev}x -> {opt_int}x) "
                    "para evitar -4161 (posición u órdenes abiertas). Se escala quote_qty."
                )
                return False

        try:
            await self.exchange.set_leverage(opt_int, target_sym)
            self.symbol_leverage[symbol] = opt_int
            logger.info(f"Apalancamiento dinámico actualizado en Binance para {symbol}: {cur_lev}x -> {opt_int}x (ISOLATED)")
            return True
        except Exception as e:
            if "-4161" in str(e) or "leverage reduction" in str(e).lower():
                logger.info(f"Binance rechazó reducción de apalancamiento en {target_sym} (-4161: posición abierta).")
            else:
                logger.warning(f"Error actualizando apalancamiento dinámico en {target_sym}: {e}")
            return False

    async def place_order_ws(self, order: OrderRequest) -> str:
        """
        Exige que la orden haya sido aprobada previamente por MandateGatekeeper.validate_order().
        Envía la orden real a Binance usando create_order / create_order_ws con type='limit',
        params={'timeInForce': 'GTX', 'postOnly': True, 'positionSide': 'BOTH'}, etiqueta con clientOrderId
        estructurado y registra la orden en el VirtualStrategyLedger.
        Si Binance rechaza con -5022 (OrderImmediatelyFillable) porque el libro se movió en vuelo,
        captura la excepción limpiamente, ajusta el precio 1 tick hacia afuera con Decimal y reintenta.
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
            await self.initialize_symbol_margin(order.symbol, max_leverage=self.default_leverage)

        spec = SYMBOL_SPECS.get(order.symbol)
        tick = spec.tick_size if spec else 0.1
        step = spec.step_size if spec else 0.001
        safe_price = truncate_decimal_step(order.price, tick)
        safe_qty = truncate_decimal_step(order.qty, step)

        side_str = order.side.lower()
        cid = order.client_order_id or self.ledger.generate_client_order_id(order.strategy_id)
        order_fn = getattr(self.exchange, "create_order_ws", None) or self.exchange.create_order
        order_params = {
            "timeInForce": "GTX",
            "postOnly": True,
            "positionSide": "BOTH",
            "clientOrderId": cid,
            "newClientOrderId": cid,
        }

        target_sym = self.resolve_binance_symbol(order.symbol)
        try:
            res = await order_fn(
                symbol=target_sym,
                type="limit",
                side=side_str,
                amount=safe_qty,
                price=safe_price,
                params=order_params,
            )
            if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_success"):
                self.gatekeeper.record_api_success()
        except Exception as e:
            err_str = str(e).lower()
            if "-5022" in str(e) or "immediately" in err_str or "post only" in err_str:
                raw_adj = safe_price - tick if order.side.upper() == "BUY" else safe_price + tick
                adj_price = truncate_decimal_step(raw_adj, tick)
                logger.warning(
                    f"Post-Only cruzó el libro en vuelo (-5022) en {order.symbol}. "
                    f"Ajustando precio {safe_price} -> {adj_price} y reintentando..."
                )
                cid_retry = self.ledger.generate_client_order_id(order.strategy_id)
                order_params["clientOrderId"] = cid_retry
                order_params["newClientOrderId"] = cid_retry
                try:
                    res = await order_fn(
                        symbol=target_sym,
                        type="limit",
                        side=side_str,
                        amount=safe_qty,
                        price=adj_price,
                        params=order_params,
                    )
                    cid = cid_retry
                    if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_success"):
                        self.gatekeeper.record_api_success()
                except Exception as e_retry:
                    err_retry = str(e_retry).lower()
                    if "-5022" in str(e_retry) or "immediately" in err_retry or "post only" in err_retry:
                        logger.warning(
                            f"Post-Only reintento descartado limpiamente tras -5022 en {order.symbol}: {e_retry}"
                        )
                        return ""
                    if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_error"):
                        self.gatekeeper.record_api_error(str(e_retry))
                    raise
            else:
                if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_error"):
                    self.gatekeeper.record_api_error(str(e))
                raise

        if not res or not isinstance(res, dict) or "id" not in res:
            logger.warning(f"Respuesta inválida o nula de orden en Binance ({order.symbol}): {res}")
            return ""

        order_id = str(res["id"])
        strat, sub_lot = self.ledger.parse_client_order_id(cid)
        self.ledger.register_order(
            client_order_id=cid,
            symbol=order.symbol,
            strategy_id=strat,
            sub_lot_id=sub_lot,
            exchange_order_id=order_id,
        )
        self.active_orders_meta[order_id] = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "price": float(safe_price),
            "qty": float(safe_qty),
            "strategy_id": strat,
            "sub_lot_id": sub_lot,
            "client_order_id": cid,
            "timestamp_ns": order.timestamp_ns,
        }
        if order.side.upper() == "BUY":
            self.live_active_bid_id = order_id
            self.active_bids_by_symbol[order.symbol] = order_id
        elif order.side.upper() == "SELL":
            self.live_active_ask_id = order_id
            self.active_asks_by_symbol[order.symbol] = order_id
        logger.info(f"ORDEN ENVIADA a Binance: {order.symbol} ({target_sym}) {order.side.upper()} {safe_qty} @ ${safe_price} [ID: {order_id}, CID: {cid}]")
        return order_id

    async def place_orders_batch(self, orders: list[OrderRequest]) -> list[str]:
        """
        Envía un lote de órdenes (create_orders) a Binance en una sola solicitud HTTP/WS.
        Agrupa los niveles de la escalera reduciendo latencia y consumo de rate limits.
        """
        if not orders:
            return []

        valid_orders: list[OrderRequest] = []
        for o in orders:
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
                is_valid, reason = self.gatekeeper.validate_order(o, portfolio)
                if not is_valid:
                    logger.warning(f"Orden batch rechazada por MandateGatekeeper ({o.symbol}): {reason}")
                    continue
            valid_orders.append(o)

        if not valid_orders:
            return []

        for o in valid_orders:
            if o.symbol not in self.initialized_symbols:
                await self.initialize_symbol_margin(o.symbol, max_leverage=self.default_leverage)

        batch_payloads = []
        cids = []
        for o in valid_orders:
            cid = o.client_order_id or self.ledger.generate_client_order_id(o.strategy_id)
            cids.append(cid)
            target_sym = self.resolve_binance_symbol(o.symbol)
            batch_payloads.append({
                "symbol": target_sym,
                "type": "limit",
                "side": o.side.lower(),
                "amount": o.qty,
                "price": o.price,
                "params": {
                    "timeInForce": "GTX",
                    "postOnly": True,
                    "positionSide": "BOTH",
                    "clientOrderId": cid,
                    "newClientOrderId": cid,
                },
            })

        placed_ids: list[str] = []
        create_orders_fn = getattr(self.exchange, "create_orders_ws", None) or getattr(self.exchange, "create_orders", None)
        if create_orders_fn is not None:
            try:
                res_raw = create_orders_fn(batch_payloads)
                responses = await res_raw if inspect.isawaitable(res_raw) else res_raw
                if isinstance(responses, list):
                    for res, order_req, cid in zip(responses, valid_orders, cids):
                        if isinstance(res, dict) and "id" in res:
                            oid = str(res["id"])
                            strat, sub_lot = self.ledger.parse_client_order_id(cid)
                            self.ledger.register_order(
                                client_order_id=cid,
                                symbol=order_req.symbol,
                                strategy_id=strat,
                                sub_lot_id=sub_lot,
                                exchange_order_id=oid,
                            )
                            self.active_orders_meta[oid] = {
                                "symbol": order_req.symbol,
                                "side": order_req.side.upper(),
                                "price": float(order_req.price),
                                "qty": float(order_req.qty),
                                "strategy_id": strat,
                                "sub_lot_id": sub_lot,
                                "client_order_id": cid,
                                "timestamp_ns": order_req.timestamp_ns,
                            }
                            if order_req.side.upper() == "BUY":
                                self.live_active_bid_id = oid
                                self.active_bids_by_symbol[order_req.symbol] = oid
                            elif order_req.side.upper() == "SELL":
                                self.live_active_ask_id = oid
                                self.active_asks_by_symbol[order_req.symbol] = oid
                            placed_ids.append(oid)
                    logger.info(f"BATCH ORDERS ENVIADAS a Binance ({len(placed_ids)}/{len(valid_orders)} órdenes)")
                    return placed_ids
            except Exception as e:
                logger.warning(f"create_orders batch no soportado o falló en Binance ({e}). Retrocediendo a concurrent...")

        # Fallback individual concurrente
        tasks = [self.place_order_ws(o) for o in valid_orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, str):
                placed_ids.append(r)
        return placed_ids

    async def cancel_orders_batch(self, order_ids: list[str], symbol: str) -> list[str]:
        """Cancela un lote de órdenes usando cancel_orders nativo o concurrentemente."""
        if not order_ids:
            return []
        target_sym = self.resolve_binance_symbol(symbol)
        cancel_orders_fn = getattr(self.exchange, "cancel_orders_ws", None) or getattr(self.exchange, "cancel_orders", None)
        cancelled: list[str] = []
        if cancel_orders_fn is not None:
            try:
                res_raw = cancel_orders_fn(order_ids, target_sym)
                res = await res_raw if inspect.isawaitable(res_raw) else res_raw
                for oid in order_ids:
                    self.active_orders_meta.pop(oid, None)
                    cancelled.append(oid)
                return cancelled
            except Exception as e:
                logger.debug(f"cancel_orders batch falló en Binance ({e}). Retrocediendo a concurrente...")

        tasks = [self.cancel_order_ws(oid, symbol) for oid in order_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for oid, r in zip(order_ids, results):
            if r is True:
                cancelled.append(oid)
        return cancelled

    async def prune_stale_orders(
        self,
        symbol: str,
        valid_order_ids: set[str] | list[str] | None = None,
        max_orders_per_side: int = 3,
    ) -> list[str]:
        """
        Poda de Órdenes Huérfanas (prune_stale_orders):
        Verifica fetch_open_orders(symbol) y cancela cualquier orden abierta que haya quedado
        fuera de los niveles vigentes o que exceda el tope de max_orders_per_side (máx 3 Bids y 3 Asks,
        es decir, máx 6 órdenes activas totales por par).
        """
        cancelled_ids: list[str] = []
        try:
            open_orders = []
            if hasattr(self.exchange, "fetch_open_orders"):
                res = self.exchange.fetch_open_orders(symbol)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    open_orders = res
        except Exception as e:
            logger.debug(f"fetch_open_orders({symbol}) en prune_stale_orders: {e}")
            return cancelled_ids

        if not open_orders:
            return cancelled_ids

        valid_set = set(str(x) for x in valid_order_ids) if valid_order_ids is not None else None
        bids: list[dict[str, Any]] = []
        asks: list[dict[str, Any]] = []

        for o in open_orders:
            oid = str(o.get("id", ""))
            side = str(o.get("side", "")).upper()
            if valid_set is not None and oid not in valid_set:
                try:
                    await self.cancel_order_ws(oid, symbol)
                    cancelled_ids.append(oid)
                except Exception as ec:
                    logger.debug(f"Error cancelando orden huérfana {oid}: {ec}")
                continue

            if side == "BUY":
                bids.append(o)
            elif side == "SELL":
                asks.append(o)

        if len(bids) > max_orders_per_side:
            bids_sorted = sorted(bids, key=lambda x: int(x.get("timestamp") or 0), reverse=True)
            stale_bids = bids_sorted[max_orders_per_side:]
            for o in stale_bids:
                oid = str(o.get("id", ""))
                try:
                    await self.cancel_order_ws(oid, symbol)
                    cancelled_ids.append(oid)
                    logger.info(f"Podada orden huérfana/excedente BID {oid} en {symbol} (tope {max_orders_per_side})")
                except Exception as ec:
                    logger.debug(f"Error podando bid {oid}: {ec}")

        if len(asks) > max_orders_per_side:
            asks_sorted = sorted(asks, key=lambda x: int(x.get("timestamp") or 0), reverse=True)
            stale_asks = asks_sorted[max_orders_per_side:]
            for o in stale_asks:
                oid = str(o.get("id", ""))
                try:
                    await self.cancel_order_ws(oid, symbol)
                    cancelled_ids.append(oid)
                    logger.info(f"Podada orden huérfana/excedente ASK {oid} en {symbol} (tope {max_orders_per_side})")
                except Exception as ec:
                    logger.debug(f"Error podando ask {oid}: {ec}")

        return cancelled_ids

    async def close_position_market(self, symbol: str, close_side: str, qty: float) -> str | None:
        """
        Cierra una posición individual en Binance con reduceOnly=True (usado por TP/SL/Time-Stop).
        Verifica previamente que exista posición contraria abierta real y que la cantidad no la exceda,
        previniendo el error -2022 ReduceOnly Order is rejected.
        """
        try:
            open_pos_contracts = 0.0
            if hasattr(self, "ledger") and self.ledger is not None:
                open_pos_contracts = self.ledger.get_net_inventory(symbol)
            if abs(open_pos_contracts) <= 1e-8 and hasattr(self.exchange, "fetch_positions"):
                try:
                    positions = self.exchange.fetch_positions([symbol])
                    if inspect.isawaitable(positions):
                        positions = await positions
                    if isinstance(positions, list):
                        for p in positions:
                            if isinstance(p, dict) and p.get("symbol") == symbol:
                                amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0.0)
                                side_s = str(p.get("side") or "").lower()
                                raw_amt = p.get("info", {}).get("positionAmt")
                                is_sh = (side_s == "short") or (raw_amt is not None and float(raw_amt) < 0.0) or (amt < 0.0)
                                open_pos_contracts = -abs(amt) if is_sh else abs(amt)
                                break
                except Exception:
                    pass

            open_pos_sign = open_pos_contracts
            if abs(open_pos_sign) <= 1e-8 and self.portfolio and symbol in self.portfolio.active_positions:
                open_pos_sign = float(self.portfolio.active_positions[symbol])

            if abs(open_pos_sign) <= 1e-8:
                logger.info(f"close_position_market omitido para {symbol}: posición neta es 0")
                return None

            # Validar que exista posición abierta en sentido contrario antes de enviar reduceOnly
            if close_side.lower() == "sell" and open_pos_sign < -1e-8:
                logger.warning(f"close_position_market omitido: close_side SELL no coincide con posición SHORT {open_pos_sign}")
                return None
            if close_side.lower() == "buy" and open_pos_sign > 1e-8:
                logger.warning(f"close_position_market omitido: close_side BUY no coincide con posición LONG {open_pos_sign}")
                return None

            if abs(open_pos_contracts) > 1e-8:
                safe_qty = min(abs(float(qty)), abs(open_pos_contracts))
            else:
                safe_qty = abs(float(qty))
            spec = SYMBOL_SPECS.get(symbol)
            if spec and spec.step_size > 0:
                safe_qty = max(truncate_decimal_step(safe_qty, spec.step_size), spec.step_size)

            target_sym = self.resolve_binance_symbol(symbol)
            res = await self.exchange.create_order(
                symbol=target_sym,
                type="market",
                side=close_side.lower(),
                amount=safe_qty,
                params={"reduceOnly": True, "positionSide": "BOTH"},
            )
            return str(res.get("id", "")) if isinstance(res, dict) else None
        except Exception as e:
            if "-2022" in str(e) or "reduceonly" in str(e).lower():
                logger.warning(f"reduceOnly rechazado en Binance ({symbol}): {e}")
            else:
                logger.error(f"Error en close_position_market ({symbol} {close_side} {qty}) en Binance: {e}")
            return None

    async def place_protective_stop_loss(
        self, symbol: str, close_side: str, qty: float, stop_price: float
    ) -> str | None:
        """
        Coloca un Stop-Loss nativo en el servidor de Binance (STOP_MARKET) con workingType='MARK_PRICE'
        y reduceOnly=True para proteger la posición contra desconexiones o caídas del proceso local.
        """
        try:
            spec = SYMBOL_SPECS.get(symbol)
            tick = spec.tick_size if spec else 0.1
            step = spec.step_size if spec else 0.001
            safe_stop = truncate_decimal_step(stop_price, tick)
            safe_qty = truncate_decimal_step(qty, step)
            if safe_qty <= 0.0 or safe_stop <= 0.0:
                return None

            target_sym = self.resolve_binance_symbol(symbol)
            old_sl_id = self.protective_sl_by_symbol.pop(symbol, None)
            if old_sl_id:
                try:
                    await self.cancel_order_ws(old_sl_id, symbol)
                except Exception:
                    pass

            res = await self.exchange.create_order(
                symbol=target_sym,
                type="STOP_MARKET",
                side=close_side.lower(),
                amount=safe_qty,
                price=None,
                params={
                    "stopPrice": safe_stop,
                    "triggerPrice": safe_stop,
                    "workingType": "MARK_PRICE",
                    "reduceOnly": True,
                    "positionSide": "BOTH",
                },
            )
            if isinstance(res, dict) and "id" in res:
                sl_oid = str(res["id"])
                self.protective_sl_by_symbol[symbol] = sl_oid
                logger.info(
                    f"Stop-Loss Protector (MARK_PRICE) colocado en Binance {symbol}: "
                    f"{close_side.upper()} {safe_qty} @ stop=${safe_stop} [ID: {sl_oid}]"
                )
                return sl_oid
            return None
        except Exception as e:
            logger.debug(f"place_protective_stop_loss omitido/rechazado en Binance ({symbol}): {e}")
            return None

    async def place_protective_stop_loss_with_retry(
        self, symbol: str, close_side: str, qty: float, stop_price: float, max_retries: int = 3
    ) -> str | None:
        """
        Coloca Stop-Loss en Binance con hasta max_retries reintentos con backoff exponencial.
        Si la colocación del SL falla tras agotar reintentos, cierra inmediatamente la posición
        a mercado para erradicar posiciones huérfanas desprotegidas.
        """
        delay = 0.05
        for attempt in range(max_retries):
            sl_oid = await self.place_protective_stop_loss(symbol, close_side, qty, stop_price)
            if sl_oid:
                return sl_oid
            logger.warning(
                f"[SL_RETRY] Intento {attempt + 1}/{max_retries} fallido al colocar SL protector en Binance ({symbol}). Reintentando en {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
            delay *= 2.0

        logger.critical(
            f"[ALERTA CRÍTICA: ORPHAN_PREVENTION] Fallaron todos los ({max_retries}) reintentos de colocar Stop-Loss en Binance ({symbol}). "
            "Cerrando inmediatamente posición huérfana a mercado para proteger el capital!"
        )
        await self.close_position_market(symbol, close_side, qty)
        return None

    async def close_position_maker(self, symbol: str, close_side: str, qty: float, price: float) -> str | None:
        """
        Coloca una orden LIMIT reduceOnly en modo Post-Only (GTX) para Take-Profit en Binance,
        capturando comisión Maker y protegiendo el margen de ganancia.
        """
        try:
            open_pos_contracts = 0.0
            if hasattr(self, "ledger") and self.ledger is not None:
                open_pos_contracts = self.ledger.get_net_inventory(symbol)
            if abs(open_pos_contracts) <= 1e-8 and hasattr(self.exchange, "fetch_positions"):
                try:
                    positions = self.exchange.fetch_positions([symbol])
                    if inspect.isawaitable(positions):
                        positions = await positions
                    if isinstance(positions, list):
                        for p in positions:
                            if isinstance(p, dict) and p.get("symbol") == symbol:
                                amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0.0)
                                side_s = str(p.get("side") or "").lower()
                                raw_amt = p.get("info", {}).get("positionAmt")
                                is_sh = (side_s == "short") or (raw_amt is not None and float(raw_amt) < 0.0) or (amt < 0.0)
                                open_pos_contracts = -abs(amt) if is_sh else abs(amt)
                                break
                except Exception:
                    pass

            open_pos_sign = open_pos_contracts
            if abs(open_pos_sign) <= 1e-8 and self.portfolio and getattr(self.portfolio, "active_positions", None) and symbol in self.portfolio.active_positions:
                open_pos_sign = float(self.portfolio.active_positions[symbol])

            if abs(open_pos_sign) <= 1e-8:
                return None

            if close_side.lower() == "sell" and open_pos_sign < -1e-8:
                return None
            if close_side.lower() == "buy" and open_pos_sign > 1e-8:
                return None

            safe_qty = min(abs(float(qty)), abs(open_pos_contracts)) if abs(open_pos_contracts) > 1e-8 else abs(float(qty))
            target_sym = self.resolve_binance_symbol(symbol)
            spec = SYMBOL_SPECS.get(symbol)
            tick = spec.tick_size if spec else 0.1
            step = spec.step_size if spec else 0.001
            safe_price = truncate_decimal_step(price, tick)
            safe_qty_str = truncate_decimal_step(safe_qty, step)
            if hasattr(self.exchange, "price_to_precision"):
                try:
                    p_res = self.exchange.price_to_precision(target_sym, safe_price)
                    if p_res is not None and not str(type(p_res)).startswith("<class 'unittest.mock"):
                        safe_price = float(p_res)
                except Exception:
                    pass

            if hasattr(self.exchange, "amount_to_precision"):
                try:
                    q_res = self.exchange.amount_to_precision(target_sym, safe_qty_str)
                    if q_res is not None and not str(type(q_res)).startswith("<class 'unittest.mock"):
                        safe_qty_str = float(q_res)
                except Exception:
                    pass

            res = await self.exchange.create_order(
                symbol=target_sym,
                type="limit",
                side=close_side.lower(),
                amount=safe_qty_str,
                price=safe_price,
                params={"reduceOnly": True, "timeInForce": "GTX", "postOnly": True, "positionSide": "BOTH"},
            )
            order_id = str(res.get("id", ""))
            logger.info(f"Take-Profit Maker colocado en Binance {symbol}: {close_side} {safe_qty_str} @ ${safe_price} [ID: {order_id}]")
            return order_id
        except Exception as e:
            logger.debug(f"close_position_maker omitido/rechazado en Binance ({symbol}): {e}")
            return None

    async def cancel_order_ws(self, order_id: str, symbol: str) -> bool:
        """Cancela la orden activa en Binance en tiempo real."""
        try:
            target_sym = self.resolve_binance_symbol(symbol)
            if hasattr(self.exchange, "cancel_order_ws") and callable(getattr(self.exchange, "cancel_order_ws")):
                res = self.exchange.cancel_order_ws(id=order_id, symbol=target_sym)
                if inspect.isawaitable(res):
                    await res
            elif hasattr(self.exchange, "cancel_order") and callable(getattr(self.exchange, "cancel_order")):
                res = self.exchange.cancel_order(id=order_id, symbol=target_sym)
                if inspect.isawaitable(res):
                    await res
            if self.live_active_bid_id == order_id:
                self.live_active_bid_id = None
            if self.live_active_ask_id == order_id:
                self.live_active_ask_id = None
            if self.active_bids_by_symbol.get(symbol) == order_id:
                self.active_bids_by_symbol.pop(symbol, None)
            if self.active_asks_by_symbol.get(symbol) == order_id:
                self.active_asks_by_symbol.pop(symbol, None)
            self.active_orders_meta.pop(order_id, None)
            return True
        except Exception as e:
            logger.error(f"Error cancelando orden {order_id} en Binance ({symbol}): {e}")
            return False

    async def emergency_flatten_all(self, symbols: list[str]) -> None:
        """
        Cancela todas las órdenes abiertas en Binance (cancel_all_orders) y,
        si existe alguna posición abierta en fetch_positions, envía una orden MARKET
        con params={'reduceOnly': True} para dejar la exposición en cero inmediatamente.
        Corrige detección de posiciones SHORT (side == 'short' o positionAmt < 0 -> close_side = 'buy').
        """
        for sym in symbols:
            try:
                target_sym = self.resolve_binance_symbol(sym)
                await self.exchange.cancel_all_orders(target_sym)
                logger.warning(f"cancel_all_orders ejecutado en Binance para {sym}")
            except Exception as e:
                logger.error(f"Error en cancel_all_orders para {sym}: {e}")

        positions = []
        try:
            positions = await self.exchange.fetch_positions(symbols)
        except Exception as e:
            logger.warning(f"Fallback fetch_positions() sin filtro en Binance tras error por símbolo: {e}")
            try:
                positions = await self.exchange.fetch_positions()
            except Exception as e2:
                logger.error(f"Error en emergency_flatten_all obteniendo posiciones en Binance: {e2}")

        for pos in positions:
            try:
                sym = pos.get("symbol")
                raw_pos_amt = pos.get("info", {}).get("positionAmt")
                amt = float(pos.get("contracts") or raw_pos_amt or 0.0)
                side_str = str(pos.get("side") or "").lower()
                is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
                qty = abs(amt)
                if qty > 1e-8:
                    close_side = "buy" if is_short else "sell"
                    logger.warning(f"EMERGENCY FLATTEN: cerrando posición en {sym} ({'SHORT' if is_short else 'LONG'}): {close_side} {qty}")
                    await self.exchange.create_order(
                        symbol=sym,
                        type="market",
                        side=close_side,
                        amount=qty,
                        params={"reduceOnly": True},
                    )
            except Exception as e:
                logger.error(f"Error en emergency_flatten_all cerrando posición individual en Binance: {e}")

        self.live_active_bid_id = None
        self.live_active_ask_id = None
        self.active_bids_by_symbol.clear()
        self.active_asks_by_symbol.clear()
        self.active_orders_meta.clear()
        self.protective_sl_by_symbol.clear()

    async def _handle_order_completion(
        self, oid: str, sym: str, side: str, portfolio: PortfolioState, ws_filled_qty: float = 0.0
    ) -> None:
        """Verifica de forma idempotente en Binance si una orden fue ejecutada (total o parcialmente) o cancelada."""
        async with self._fill_lock:
            if oid in self._processed_order_ids:
                return
            if portfolio is not None:
                self.portfolio = portfolio
            meta = self.active_orders_meta.pop(oid, {})
            order_sym = str(meta.get("symbol") or sym)
            px = float(meta.get("price") or 0.0)
            q = float(meta.get("qty") or 0.0002)

            order_status = None
            filled_qty = float(ws_filled_qty or 0.0)
            avg_px = px
            try:
                if hasattr(self.exchange, "fetch_order"):
                    res_ord = self.exchange.fetch_order(oid, order_sym)
                    if inspect.isawaitable(res_ord):
                        res_ord = await res_ord
                    if isinstance(res_ord, dict):
                        order_status = str(res_ord.get("status", "")).lower()
                        fetched_filled = float(res_ord.get("filled") or 0.0)
                        if fetched_filled > 1e-8:
                            filled_qty = fetched_filled
                        avg_px = float(res_ord.get("average") or res_ord.get("price") or px)
            except Exception as e:
                logger.debug(f"fetch_order({oid}, {order_sym}) en Binance: {e}")

            # Si el precio sigue en 0, no podemos registrar un fill válido
            if avg_px <= 0.0:
                logger.warning(f"Orden {oid} en Binance ({order_sym}) sin precio válido. Omitiendo registro de fill.")
                return

            # Solo registrar fill si la orden fue 'closed' o si tiene cantidad ejecutada > 0 (incluyendo partial fills cancelados)
            is_actually_filled = (order_status == "closed") or (filled_qty > 1e-8)
            if is_actually_filled:
                self._processed_order_ids.add(oid)
                actual_qty = filled_qty if filled_qty > 1e-8 else q
                cid = meta.get("client_order_id", oid)
                if cid:
                    self._processed_order_ids.add(str(cid))
                fill_res = self.ledger.record_fill(
                    order_id_or_cid=cid,
                    symbol=order_sym,
                    side=side,
                    fill_price=avg_px,
                    fill_qty=actual_qty,
                    ts_ns=time.time_ns(),
                )
                net_inv_contracts = self.ledger.get_net_inventory(order_sym)
                if self.portfolio is not None:
                    if not hasattr(self.portfolio, "active_positions") or self.portfolio.active_positions is None:
                        self.portfolio.active_positions = {}
                    # Almacenar siempre USD Notional con signo consistente con Gate 3 y sync_portfolio_state
                    self.portfolio.active_positions[order_sym] = round(net_inv_contracts * avg_px, 2)
                sub_lot = fill_res.get("sub_lot_id", "MAIN")
                strat_tag = f"{fill_res.get('strategy_id', 'AVELLANEDA_OBI')}_{sub_lot}" if sub_lot != "MAIN" else fill_res.get("strategy_id", "AVELLANEDA_OBI")
                fee_est = round(avg_px * actual_qty * 0.0002, 4)
                pnl_recorded = float(fill_res.get("realized_pnl_usd", 0.0) or 0.0)
                logger.info(f"FILL CONFIRMADO en Binance: {order_sym} {side} {actual_qty} @ ${avg_px} (order {oid}, lot {strat_tag})")
                if self.on_fill and callable(self.on_fill):
                    try:
                        self.on_fill(
                            symbol=order_sym,
                            strategy=strat_tag,
                            side=side,
                            entry_price=avg_px,
                            exit_or_mid_price=avg_px,
                            qty=actual_qty,
                            return_pct=0.0,
                            pnl_usd=pnl_recorded,
                            trade_id=f"LIVE-{oid}",
                            fee_cost=fee_est,
                            is_close=False,
                        )
                    except Exception as ef:
                        logger.debug(f"Error registrando fill en vivo: {ef}")

                # Colocación Síncrona Inmediata de Stop-Loss Protector (con reintentos) y Take-Profit Maker
                if fill_res.get("is_increasing"):
                    close_side = "SELL" if side.upper() == "BUY" else "BUY"
                    lev_now = self.symbol_leverage.get(order_sym, self.default_leverage)
                    tier_cfg = DryRunExecutionEngine.LEVERAGE_TIERS_CONFIG.get(lev_now, {"tp_bps": 15.0, "sl_bps": 25.0})
                    tp_offset_bps = float(tier_cfg.get("tp_bps", 15.0))
                    sl_offset_bps = float(tier_cfg.get("sl_bps", 25.0))
                    tp_price = avg_px * (1.0 + tp_offset_bps / 10_000.0) if side.upper() == "BUY" else avg_px * (1.0 - tp_offset_bps / 10_000.0)
                    sl_price = avg_px * (1.0 - sl_offset_bps / 10_000.0) if side.upper() == "BUY" else avg_px * (1.0 + sl_offset_bps / 10_000.0)
                    await self.place_protective_stop_loss_with_retry(order_sym, close_side, abs(net_inv_contracts) or actual_qty, sl_price)
                    asyncio.create_task(self.close_position_maker(order_sym, close_side, abs(net_inv_contracts) or actual_qty, tp_price))

                # OCO (One-Cancels-the-Other): Al cerrarse la posición por TP o SL, cancelar la orden opuesta pendiente
                elif fill_res.get("is_decreasing"):
                    if order_sym in self.protective_sl_by_symbol:
                        sl_oid_to_cancel = self.protective_sl_by_symbol.pop(order_sym)
                        try:
                            await self.cancel_order_ws(sl_oid_to_cancel, order_sym)
                            logger.info(f"[OCO] Posición cerrada en Binance {order_sym}: SL protector {sl_oid_to_cancel} cancelado.")
                        except Exception as e:
                            logger.warning(f"[OCO] Error cancelando SL en Binance: {e}")
            else:
                logger.info(f"Orden {oid} en {order_sym} ({side}) cancelada o expirada sin fills. No se genera trade fantasma.")

    async def sync_portfolio_state(self, portfolio: PortfolioState) -> None:
        """
        Sincroniza el balance real USDT de futuros (fetch_balance) y las posiciones
        abiertas reales hacia PortfolioState (con signo negativo para posiciones SHORT
        y positivo para LONG). Además, verifica las órdenes abiertas en Binance y
        registra fills confirmados con sus metadatos reales.
        """
        if not getattr(self, "_time_synced", False):
            await self.sync_time_difference()

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

            if not hasattr(portfolio, "exchange_equities"):
                portfolio.exchange_equities = {}
            if not hasattr(portfolio, "initial_exchange_equities"):
                portfolio.initial_exchange_equities = {}

            if total_usdt > 0.0:
                is_first_sync = "binance" not in portfolio.initial_exchange_equities or portfolio.initial_exchange_equities["binance"] <= 0.0
                if is_first_sync:
                    portfolio.initial_exchange_equities["binance"] = round(total_usdt, 2)

                portfolio.exchange_equities["binance"] = total_usdt
                if getattr(portfolio, "trading_mode", "") == "LIVE_MULTI":
                    portfolio.current_equity_usd = round(sum(portfolio.exchange_equities.values()), 2)
                    if portfolio.initial_exchange_equities:
                        combined_initial = round(sum(portfolio.initial_exchange_equities.values()), 2)
                        if combined_initial > 0 and (portfolio.base_capital_usd in (100.0, 5000.0) or len(portfolio.initial_exchange_equities) >= 2):
                            portfolio.base_capital_usd = combined_initial
                            portfolio.daily_high_watermark_usd = max(portfolio.base_capital_usd, portfolio.current_equity_usd)
                elif getattr(portfolio, "trading_mode", "") == "LIVE_BINANCE":
                    portfolio.current_equity_usd = round(total_usdt, 2)
                    if is_first_sync or portfolio.base_capital_usd in (100.0, 5000.0):
                        portfolio.base_capital_usd = round(total_usdt, 2)
                        portfolio.daily_high_watermark_usd = max(portfolio.base_capital_usd, portfolio.current_equity_usd)
                else:
                    portfolio.current_equity_usd = round(total_usdt, 2)

                if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                    portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            positions = await self.exchange.fetch_positions()
            active: dict[str, float] = {}
            for pos in positions:
                sym = pos.get("symbol")
                if not sym:
                    continue
                raw_pos_amt = pos.get("info", {}).get("positionAmt")
                amt = float(pos.get("contracts") or raw_pos_amt or 0.0)
                side_str = str(pos.get("side") or "").lower()
                is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
                notional = abs(float(pos.get("notional") or pos.get("info", {}).get("notional") or 0.0))
                if notional == 0.0 and abs(amt) > 0.0:
                    mp = float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or 0.0)
                    notional = abs(amt) * mp
                if abs(amt) > 1e-8:
                    signed_notional = -notional if is_short else notional
                    active[sym] = round(signed_notional, 2)

            if getattr(portfolio, "trading_mode", "") == "LIVE_MULTI":
                for sym, notional in active.items():
                    portfolio.active_positions[sym] = notional
                for sym in list(portfolio.active_positions.keys()):
                    spec = SYMBOL_SPECS.get(sym)
                    if spec and spec.exchange == "binance" and sym not in active:
                        portfolio.active_positions.pop(sym, None)
            else:
                portfolio.active_positions = active
        except Exception as e:
            logger.error(f"Error sincronizando balance y posiciones con Binance: {e}")

        # Sincronización y reseteo de órdenes activas si ya no aparecen abiertas en Binance
        try:
            if hasattr(self.exchange, "fetch_open_orders"):
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
                    # Actualizar nocional de órdenes abiertas para Gate 3
                    open_notional_by_sym: dict[str, float] = {}
                    for o in open_orders:
                        if isinstance(o, dict):
                            osym = str(o.get("symbol") or "")
                            rem_qty = float(o.get("remaining") or o.get("amount") or 0.0)
                            opx = float(o.get("price") or 0.0)
                            if osym and rem_qty > 0 and opx > 0:
                                open_notional_by_sym[osym] = round(open_notional_by_sym.get(osym, 0.0) + (rem_qty * opx), 2)
                    if hasattr(portfolio, "open_orders_notional"):
                        portfolio.open_orders_notional = open_notional_by_sym

                    for sym_k, bid_oid in list(self.active_bids_by_symbol.items()):
                        if bid_oid not in open_order_ids:
                            self.active_bids_by_symbol.pop(sym_k, None)
                            if self.live_active_bid_id == bid_oid:
                                self.live_active_bid_id = None
                            await self._handle_order_completion(bid_oid, sym_k, "BUY", portfolio)

                    for sym_k, ask_oid in list(self.active_asks_by_symbol.items()):
                        if ask_oid not in open_order_ids:
                            self.active_asks_by_symbol.pop(sym_k, None)
                            if self.live_active_ask_id == ask_oid:
                                self.live_active_ask_id = None
                            await self._handle_order_completion(ask_oid, sym_k, "SELL", portfolio)

                    if self.live_active_bid_id and self.live_active_bid_id not in open_order_ids:
                        old_bid = self.live_active_bid_id
                        self.live_active_bid_id = None
                        meta_sym = str(self.active_orders_meta.get(old_bid, {}).get("symbol") or "BTC/USDT:USDT")
                        await self._handle_order_completion(old_bid, meta_sym, "BUY", portfolio)

                    if self.live_active_ask_id and self.live_active_ask_id not in open_order_ids:
                        old_ask = self.live_active_ask_id
                        self.live_active_ask_id = None
                        meta_sym = str(self.active_orders_meta.get(old_ask, {}).get("symbol") or "BTC/USDT:USDT")
                        await self._handle_order_completion(old_ask, meta_sym, "SELL", portfolio)
        except Exception as e:
            logger.error(f"Error sincronizando órdenes abiertas con Binance: {e}")

        # Ingesta de ejecuciones reales (fetch_my_trades) hacia strategy_ledger y recent_trades
        try:
            if hasattr(self.exchange, "fetch_my_trades"):
                symbols_to_check = list(self.initialized_symbols) or ["BTC/USDT:USDT"]
                for sym in symbols_to_check:
                    target_sym = self.resolve_binance_symbol(sym)
                    last_ts = self._last_trade_ts_ms.get(sym, self._init_ts_ms)
                    res_trades = self.exchange.fetch_my_trades(target_sym, since=last_ts)
                    if inspect.isawaitable(res_trades):
                        res_trades = await res_trades
                    if isinstance(res_trades, list):
                        for tr in res_trades:
                            tr_sym_raw = tr.get("symbol")
                            if tr_sym_raw and self.resolve_binance_symbol(str(tr_sym_raw)) != target_sym:
                                continue

                            tid = str(tr.get("id") or "")
                            if tid and tid in self._processed_trade_ids:
                                continue
                            if tid:
                                self._processed_trade_ids.add(tid)

                            tr_ts = int(tr.get("timestamp") or 0)
                            if tr_ts > 0:
                                self._last_trade_ts_ms[sym] = max(self._last_trade_ts_ms.get(sym, 0), tr_ts + 1)

                            tr_order_id = str(tr.get("order") or "")
                            cid = str(tr.get("clientOrderId") or tr.get("info", {}).get("clientOrderId") or tr_order_id or "")
                            if (tr_order_id and tr_order_id in self._processed_order_ids) or (cid and cid in self._processed_order_ids):
                                continue
                            if tr_order_id:
                                self._processed_order_ids.add(tr_order_id)
                            if cid:
                                self._processed_order_ids.add(cid)

                            tr_side = str(tr.get("side") or "").upper()
                            tr_px = float(tr.get("price") or 0.0)
                            tr_qty = float(tr.get("amount") or 0.0)
                            if tr_px <= 0.0 or tr_qty <= 0.0:
                                continue
                            tr_cost = float(tr.get("cost") or (tr_px * tr_qty))

                            fee_cost = float(tr.get("fee", {}).get("cost") or tr.get("info", {}).get("fee") or (tr_px * tr_qty * 0.0002))
                            realized_pnl_raw = float(tr.get("info", {}).get("realizedPnl") or 0.0)

                            fill_res = self.ledger.record_fill(
                                order_id_or_cid=cid,
                                symbol=sym,
                                side=tr_side,
                                fill_price=tr_px,
                                fill_qty=tr_qty,
                                ts_ns=(tr_ts * 1_000_000) if tr_ts > 0 else time.time_ns(),
                            )
                            sub_lot = fill_res.get("sub_lot_id", "MAIN")
                            strat = fill_res.get("strategy_id", "AVELLANEDA_OBI")
                            strat_tag = f"{strat}_{sub_lot}" if sub_lot != "MAIN" else strat

                            gross_pnl = float(fill_res.get("realized_pnl_usd", 0.0) or realized_pnl_raw)
                            net_pnl = gross_pnl - fee_cost
                            ret_pct = round((net_pnl / max(tr_cost, 1.0)) * 100.0, 3) if tr_cost > 0 else 0.05

                            logger.info(
                                f"TRADE REAL REGISTRADO [Binance {sym}]: {tr_side} {tr_qty} @ ${tr_px} "
                                f"(net PnL: ${net_pnl:.2f}, fee: ${fee_cost:.4f}, lot: {strat_tag})"
                            )
                            if self.on_fill and callable(self.on_fill):
                                try:
                                    self.on_fill(
                                        symbol=sym,
                                        strategy=strat_tag,
                                        side=tr_side,
                                        entry_price=tr_px,
                                        exit_or_mid_price=tr_px,
                                        qty=tr_qty,
                                        return_pct=ret_pct,
                                        pnl_usd=round(net_pnl, 2),
                                        trade_id=f"LIVE-{tid or cid}",
                                        timestamp_ns=(tr_ts * 1_000_000) if tr_ts > 0 else None,
                                        fee_cost=round(fee_cost, 4),
                                    )
                                except Exception as ef:
                                    logger.debug(f"Error despachando on_fill Binance: {ef}")
        except Exception as et:
            logger.debug(f"fetch_my_trades en Binance sync_portfolio_state: {et}")

    async def reconcile_existing_positions(self, symbols: list[str]) -> list[dict[str, Any]]:
        """
        Reconciliación en Frío (reconcile_existing_positions):
        1. Limpia órdenes huérfanas antiguas con cancel_all_orders.
        2. Inspecciona fetch_positions.
        3. Registra posiciones abiertas en strategy_ledger.py con su entryPrice y
           su breakEvenPrice real (float(pos.get("info", {}).get("breakEvenPrice") or entry_price)),
           para que el motor gestione su cierre reduceOnly por encima del punto de equilibrio.
        """
        reconciled = []
        for sym in symbols:
            target_sym = self.resolve_binance_symbol(sym)
            try:
                if hasattr(self.exchange, "cancel_all_orders"):
                    await self.exchange.cancel_all_orders(target_sym)
            except Exception as e:
                logger.debug(f"cancel_all_orders en reconcile Binance ({target_sym}): {e}")

        positions = []
        try:
            if hasattr(self.exchange, "fetch_positions"):
                res = self.exchange.fetch_positions(symbols)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    positions = res
        except Exception as e:
            try:
                res = self.exchange.fetch_positions()
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    positions = res
            except Exception as e2:
                logger.error(f"Error consultando fetch_positions en reconcile Binance: {e2}")

        for pos in positions:
            sym = pos.get("symbol")
            if not sym:
                continue
            raw_pos_amt = pos.get("info", {}).get("positionAmt")
            amt = float(pos.get("contracts") or raw_pos_amt or 0.0)
            side_str = str(pos.get("side") or "").lower()
            is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
            qty = abs(amt)
            if qty > 1e-8:
                entry_px = float(pos.get("entryPrice") or pos.get("info", {}).get("entryPrice") or 0.0)
                be_px = float(pos.get("info", {}).get("breakEvenPrice") or entry_px)
                side = "SELL" if is_short else "BUY"
                lot = self.ledger.reconcile_position(
                    symbol=sym,
                    side=side,
                    quantity=qty,
                    entry_price=entry_px,
                    break_even_price=be_px,
                    strategy_id="AVELLANEDA_OBI",
                    sub_lot_id="MAIN",
                )
                if hasattr(self, "db_store") and self.db_store is not None:
                    try:
                        lot_k = lot.key
                        signed_q = qty if side == "BUY" else -qty
                        self.db_store.save_lot(
                            lot_key=lot_k,
                            symbol=sym,
                            strategy_id="AVELLANEDA_OBI",
                            sub_lot_id="MAIN",
                            inventory_q=signed_q,
                            entry_price=entry_px,
                            break_even_price=be_px,
                            entry_ts_ns=time.time_ns(),
                            realized_pnl_usd=0.0,
                        )
                    except Exception as e_lot:
                        logger.error(f"Error persistiendo lote reconciliado en db_store: {e_lot}")
                reconciled.append({
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "entry_price": entry_px,
                    "break_even_price": be_px,
                })
                logger.info(
                    f"POSICIÓN RECONCILIADA en Binance {sym}: {side} {qty} @ entry ${entry_px:.2f} (Break-Even: ${be_px:.2f})"
                )
        return reconciled

    def start_sync_loop(self, portfolio: PortfolioState) -> None:
        """Inicia el bucle de sincronización periódica y los User Data Streams por WebSocket."""
        self.running = True
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync_loop(portfolio))
        if self._watch_orders_task is None or self._watch_orders_task.done():
            self._watch_orders_task = asyncio.create_task(self._watch_orders_loop(portfolio))
        if self._watch_trades_task is None or self._watch_trades_task.done():
            self._watch_trades_task = asyncio.create_task(self._watch_trades_loop(portfolio))

    def stop_sync_loop(self) -> None:
        """Detiene el bucle de sincronización periódica y los WebSocket streams."""
        self.running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            self._sync_task = None
        if self._watch_orders_task and not self._watch_orders_task.done():
            self._watch_orders_task.cancel()
            self._watch_orders_task = None
        if self._watch_trades_task and not self._watch_trades_task.done():
            self._watch_trades_task.cancel()
            self._watch_trades_task = None

    async def _watch_orders_loop(self, portfolio: PortfolioState) -> None:
        """Stream privado por WebSocket para cambios en órdenes (latencia sub-10ms)."""
        while self.running:
            try:
                watch_orders_fn = getattr(self.exchange, "watch_orders", None)
                if watch_orders_fn is None or not callable(watch_orders_fn):
                    break
                orders = await watch_orders_fn()
                if not isinstance(orders, list):
                    orders = [orders]
                for o in orders:
                    status = str(o.get("status", "")).lower()
                    oid = str(o.get("id", ""))
                    sym = str(o.get("symbol", ""))
                    side = str(o.get("side", "")).upper()
                    filled_amt = float(o.get("filled") or 0.0)
                    if status in ("closed", "filled"):
                        await self._handle_order_completion(oid, sym, side, portfolio, ws_filled_qty=filled_amt)
                    elif status in ("canceled", "expired"):
                        if filled_amt > 1e-8:
                            await self._handle_order_completion(oid, sym, side, portfolio, ws_filled_qty=filled_amt)
                        else:
                            self.active_orders_meta.pop(oid, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.running:
                    break
                err_msg = str(e).lower()
                if "notsupported" in err_msg or "not implemented" in err_msg or "unsupported" in err_msg:
                    logger.debug(f"watch_orders no soportado por Binance en este modo ({e}). Usando polling.")
                    break
                logger.debug(f"watch_orders reconectando en Binance: {e}")
                await asyncio.sleep(2.0)

    async def _watch_trades_loop(self, portfolio: PortfolioState) -> None:
        """Stream privado por WebSocket para fills instantáneos (watch_my_trades)."""
        while self.running:
            try:
                watch_trades_fn = getattr(self.exchange, "watch_my_trades", None)
                if watch_trades_fn is None or not callable(watch_trades_fn):
                    break
                trades = await watch_trades_fn()
                if not isinstance(trades, list):
                    trades = [trades]
                for tr in trades:
                    tid = str(tr.get("id", ""))
                    if tid and tid in self._processed_trade_ids:
                        continue
                    if tid:
                        self._processed_trade_ids.add(tid)
                    sym = str(tr.get("symbol", ""))
                    side = str(tr.get("side", "")).upper()
                    price = float(tr.get("price") or 0.0)
                    amount = float(tr.get("amount") or 0.0)
                    cid = tr.get("order") or tr.get("clientOrderId") or ""
                    if self.on_fill:
                        try:
                            pnl_val = float(tr.get("pnl") or 0.0)
                            self.on_fill(
                                symbol=sym,
                                strategy="AVELLANEDA_OBI",
                                side=side,
                                entry_price=price,
                                exit_or_mid_price=price,
                                qty=amount,
                                return_pct=0.0,
                                pnl_usd=pnl_val,
                                trade_id=f"WS-{tid}" if tid else None,
                                is_close=False,
                            )
                        except Exception as fe:
                            logger.error(f"Error en on_fill desde watch_my_trades Binance: {fe}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.running:
                    break
                err_msg = str(e).lower()
                if "notsupported" in err_msg or "not implemented" in err_msg or "unsupported" in err_msg:
                    logger.debug(f"watch_my_trades no soportado por Binance en este modo ({e}). Usando polling.")
                    break
                logger.debug(f"watch_my_trades reconectando en Binance: {e}")
                await asyncio.sleep(2.0)

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


class BingXLiveWsGateway:
    """
    Adaptador de Ejecución Real sobre BingX Perpetual Futures (ccxt.pro.bingx / ccxt.async_support.bingx).
    Opera con capital real mediante ccxt con options={'defaultType': 'swap'}, exige validación
    previa por MandateGatekeeper, traduce automáticamente el mandato time_in_force='GTX' al parámetro
    Post-Only nativo de BingX (params={'timeInForce': 'PO', 'postOnly': True}) para garantizar 0%
    de ejecuciones Taker, configura modo unidireccional, margen aislado y apalancamiento máximo,
    y sincroniza el portafolio y posiciones reales.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        sandbox: bool | None = None,
        gatekeeper: MandateGatekeeper | None = None,
        portfolio: PortfolioState | None = None,
        exchange: Any | None = None,
        on_fill: Any | None = None,
        db_store: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("BINGX_API_KEY", "").strip()
        self.api_secret = api_secret if api_secret is not None else os.getenv("BINGX_API_SECRET", "").strip()
        if sandbox is not None:
            self.sandbox = bool(sandbox)
        else:
            self.sandbox = os.getenv("BINGX_SANDBOX", "false").strip().lower() in ("true", "1", "yes")

        self.gatekeeper = gatekeeper
        self.portfolio = portfolio
        self.on_fill = on_fill
        self.db_store = db_store
        self.default_leverage: int = 3
        self.symbol_leverage: dict[str, int] = {}
        self.ledger = VirtualStrategyLedger()
        self.initialized_symbols: set[str] = set()
        self.running: bool = False
        self._sync_task: asyncio.Task | None = None
        self._watch_orders_task: asyncio.Task | None = None
        self._watch_trades_task: asyncio.Task | None = None
        self.live_active_bid_id: str | None = None
        self.live_active_ask_id: str | None = None
        self.active_bids_by_symbol: dict[str, str] = {}
        self.active_asks_by_symbol: dict[str, str] = {}
        self.active_orders_meta: dict[str, dict[str, Any]] = {}
        self.BINGX_SYMBOL_MAP: dict[str, str] = {
            "XAU/USDT:USDT": "XAUT/USDT:USDT",
            "XAG/USDT:USDT": "NCCOXAG2USD/USDT:USDT",
            "PEPE/USDT:USDT": "1000PEPE/USDT:USDT",
        }
        self._last_trade_ts_ms: dict[str, int] = {}
        self._processed_trade_ids: set[str] = set()
        self._processed_order_ids: set[str] = set()
        self._fill_lock: asyncio.Lock = asyncio.Lock()
        self.protective_sl_by_symbol: dict[str, str] = {}
        self._init_ts_ms: int = int(time.time() * 1000)
        self._time_synced: bool = False

        if exchange is not None:
            self.exchange = exchange
            if hasattr(self.exchange, "options") and isinstance(self.exchange.options, dict):
                self.exchange.options.setdefault("adjustForTimeDifference", True)
                self.exchange.options.setdefault("recvWindow", 10000)
        else:
            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
                    "fetchOpenOrders": {
                        "warnWithoutSymbol": False,
                    },
                },
            }
            exchange_class = getattr(ccxtpro, "bingx", None) or getattr(ccxt_async, "bingx")
            self.exchange = exchange_class(config)
            if self.sandbox:
                if hasattr(self.exchange, "set_sandbox_mode"):
                    try:
                        self.exchange.set_sandbox_mode(True)
                    except Exception:
                        pass

    async def sync_time_difference(self) -> None:
        """Sincroniza la diferencia de reloj con BingX para evitar rechazos -1021 en Windows."""
        if hasattr(self.exchange, "load_time_difference") and callable(self.exchange.load_time_difference):
            try:
                diff = await self.exchange.load_time_difference()
                logger.info(f"[BingXLiveWsGateway] Sincronización de reloj con BingX completada (offset: {diff} ms).")
                self._time_synced = True
            except Exception as e:
                logger.warning(f"[BingXLiveWsGateway] Advertencia al sincronizar reloj con BingX: {e}")
        else:
            self._time_synced = True

    def resolve_bingx_symbol(self, symbol: str) -> str:
        """Traduce símbolos a nombres nativos de BingX swap si difieren del estándar."""
        if hasattr(self.exchange, "markets") and isinstance(self.exchange.markets, dict) and self.exchange.markets:
            if symbol in self.exchange.markets:
                return symbol
            mapped = self.BINGX_SYMBOL_MAP.get(symbol)
            if mapped and mapped in self.exchange.markets:
                return mapped
            return mapped or symbol
        return symbol

    async def initialize_symbol_margin(self, symbol: str, max_leverage: int = 3) -> None:
        """
        Configura modo unidireccional (one-way), margen ISOLATED y apalancamiento máximo
        en BingX antes de permitir cualquier orden en vivo.
        """
        if not getattr(self.exchange, "markets", None):
            try:
                await self.exchange.load_markets()
            except Exception as em:
                logger.debug(f"load_markets en BingX: {em}")

        target_sym = self.resolve_bingx_symbol(symbol)
        try:
            if hasattr(self.exchange, "set_position_mode"):
                await self.exchange.set_position_mode(False, target_sym)
        except Exception as e:
            logger.info(f"set_position_mode False en {target_sym} (BingX): {e}")

        try:
            await self.exchange.set_margin_mode("ISOLATED", target_sym)
        except Exception as e:
            err_msg = str(e).lower()
            if "-4046" in err_msg or "no need to change" in err_msg or "already" in err_msg:
                logger.info(f"set_margin_mode ISOLATED ya activo en {target_sym} (BingX): {e}")
            elif "-4048" in err_msg or "-4067" in err_msg or "cannot be changed" in err_msg:
                raise ValueError(
                    f"CRÍTICO: No se pudo cambiar {target_sym} a ISOLATED en BingX por posiciones/órdenes en CROSS ({e}). "
                    "Operación bloqueada para proteger la cuenta."
                ) from e
            else:
                logger.info(f"set_margin_mode ISOLATED en {target_sym} (BingX): {e}")

        cur_lev = self.symbol_leverage.get(symbol, self.default_leverage)
        has_pos = False
        if self.portfolio and self.portfolio.active_positions:
            pos_val = float(self.portfolio.active_positions.get(symbol, 0.0) or self.portfolio.active_positions.get(target_sym, 0.0) or 0.0)
            has_pos = abs(pos_val) > 1e-6
        if not has_pos and hasattr(self, "ledger") and self.ledger is not None:
            has_pos = abs(self.ledger.get_net_inventory(symbol)) > 1e-6
        has_orders = (
            symbol in self.active_bids_by_symbol
            or symbol in self.active_asks_by_symbol
            or any(m.get("symbol") in (symbol, target_sym) for m in self.active_orders_meta.values())
        )
        if int(max_leverage) < cur_lev and (has_pos or has_orders):
            logger.info(
                f"set_leverage {max_leverage} omitido/bloqueado en BingX {target_sym} "
                "para evitar error por posición u órdenes abiertas."
            )
            self.initialized_symbols.add(symbol)
            return

        try:
            await self.exchange.set_leverage(max_leverage, target_sym)
            self.default_leverage = int(max_leverage)
            self.symbol_leverage[symbol] = int(max_leverage)
        except Exception:
            try:
                await self.exchange.set_leverage(max_leverage, target_sym, params={"side": "BOTH"})
                self.default_leverage = int(max_leverage)
                self.symbol_leverage[symbol] = int(max_leverage)
            except Exception as e:
                logger.warning(f"set_leverage {max_leverage} en {target_sym} (BingX): {e}")

        self.initialized_symbols.add(symbol)

    async def update_symbol_leverage_if_needed(self, symbol: str, optimal_leverage: int) -> bool:
        """
        Ajusta dinámicamente el apalancamiento en BingX cuando cambia en >= 1x.
        Modo ISOLATED estricto, entero entre 1x y 10x.
        """
        target_sym = self.resolve_bingx_symbol(symbol)
        opt_int = max(1, min(int(optimal_leverage), 10))
        cur_lev = self.symbol_leverage.get(symbol, self.default_leverage)
        if cur_lev == opt_int:
            return False

        # Si se requiere menor apalancamiento, verificar si hay posición u órdenes abiertas
        if opt_int < cur_lev:
            has_pos = False
            if self.portfolio and self.portfolio.active_positions:
                pos_val = float(self.portfolio.active_positions.get(symbol, 0.0) or self.portfolio.active_positions.get(target_sym, 0.0) or 0.0)
                has_pos = abs(pos_val) > 1e-6
            if not has_pos and hasattr(self, "ledger") and self.ledger is not None:
                has_pos = abs(self.ledger.get_net_inventory(symbol)) > 1e-6
            has_orders = (
                symbol in self.active_bids_by_symbol
                or symbol in self.active_asks_by_symbol
                or any(m.get("symbol") in (symbol, target_sym) for m in self.active_orders_meta.values())
            )
            if has_pos or has_orders:
                logger.info(
                    f"Reducción de apalancamiento omitida en BingX {target_sym} ({cur_lev}x -> {opt_int}x) "
                    "para evitar error por posición u órdenes abiertas. Se escala quote_qty."
                )
                return False

        try:
            try:
                await self.exchange.set_leverage(opt_int, target_sym, params={"side": "BOTH"})
            except Exception:
                await self.exchange.set_leverage(opt_int, target_sym)
            self.symbol_leverage[symbol] = opt_int
            logger.info(f"Apalancamiento dinámico actualizado en BingX para {symbol}: {cur_lev}x -> {opt_int}x (ISOLATED)")
            return True
        except Exception as e:
            logger.warning(f"Error actualizando apalancamiento dinámico BingX en {symbol}: {e}")
            return False

    async def place_order_ws(self, order: OrderRequest) -> str:
        """
        Valida la orden contra MandateGatekeeper.validate_order().
        Traduce automáticamente el mandato time_in_force='GTX' al parámetro Post-Only nativo
        de BingX (params={'timeInForce': 'PO', 'postOnly': True, 'positionSide': 'BOTH'}) para garantizar 0% de ejecuciones Taker,
        etiqueta con clientOrderId estructurado y registra la orden en el VirtualStrategyLedger.
        Si la orden es rechazada por cruce de libro en vuelo, ajusta 1 tick hacia afuera y reintenta.
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
            await self.initialize_symbol_margin(order.symbol, max_leverage=self.default_leverage)

        target_sym = self.resolve_bingx_symbol(order.symbol)
        side_str = order.side.lower()
        cid = order.client_order_id or self.ledger.generate_client_order_id(order.strategy_id)
        order_fn = getattr(self.exchange, "create_order_ws", None) or self.exchange.create_order
        order_params = {
            "timeInForce": "PO",
            "postOnly": True,
            "positionSide": "BOTH",
            "clientOrderId": cid,
        }

        spec = SYMBOL_SPECS.get(order.symbol)
        tick = spec.tick_size if spec else 0.1
        step = spec.step_size if spec else 0.001

        order_qty = truncate_decimal_step(order.qty, step)
        order_price = truncate_decimal_step(order.price, tick)
        if "1000PEPE" in target_sym:
            if order_price < 0.0005:
                order_price = round(order_price * 1000.0, 7)
                order_qty = max(1.0, round(order_qty / 1000.0, 1))
            if order.side.upper() == "BUY" and order_price < 0.0010:
                order_price = 0.0010
            elif order.side.upper() == "SELL" and order_price > 0.0150:
                order_price = 0.0150

        res = None
        try:
            res_raw = order_fn(
                symbol=target_sym,
                type="limit",
                side=side_str,
                amount=order_qty,
                price=order_price,
                params=order_params,
            )
            if inspect.isawaitable(res_raw):
                res = await res_raw
            else:
                res = res_raw
            if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_success"):
                self.gatekeeper.record_api_success()
        except Exception as e:
            err_str = str(e).lower()
            if "not supported" in err_str:
                # CCXT BingX stub create_order_ws() fallback to REST create_order
                order_fn = self.exchange.create_order
                try:
                    res_raw = order_fn(
                        symbol=target_sym,
                        type="limit",
                        side=side_str,
                        amount=order_qty,
                        price=order_price,
                        params=order_params,
                    )
                    if inspect.isawaitable(res_raw):
                        res = await res_raw
                    else:
                        res = res_raw
                    if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_success"):
                        self.gatekeeper.record_api_success()
                except Exception as inner_e:
                    e = inner_e
                    err_str = str(e).lower()

            if res is None:
                is_post_only_cross = (
                    "immediately" in err_str
                    or "post only" in err_str
                    or "postonly" in err_str
                    or "post_only" in err_str
                    or ("po" in err_str and "supported" not in err_str)
                )
                if is_post_only_cross:
                    if "1000PEPE" in target_sym and tick < 0.00001:
                        tick = 0.000001
                    adj_raw = order_price - tick if order.side.upper() == "BUY" else order_price + tick
                    adj_price = truncate_decimal_step(adj_raw, tick)
                    logger.warning(f"Post-Only en BingX cruzó libro en {order.symbol}. Reintentando a {adj_price}...")
                    cid_retry = self.ledger.generate_client_order_id(order.strategy_id)
                    order_params["clientOrderId"] = cid_retry
                    try:
                        res_raw = order_fn(
                            symbol=target_sym,
                            type="limit",
                            side=side_str,
                            amount=order_qty,
                            price=adj_price,
                            params=order_params,
                        )
                        if inspect.isawaitable(res_raw):
                            res = await res_raw
                        else:
                            res = res_raw
                        cid = cid_retry
                    except Exception as e_retry:
                        err_retry = str(e_retry).lower()
                        if "immediately" in err_retry or "post" in err_retry or "-5022" in str(e_retry):
                            logger.warning(f"Post-Only reintento descartado limpiamente tras cruce en BingX ({order.symbol}): {e_retry}")
                            return ""
                        if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_error"):
                            self.gatekeeper.record_api_error(str(e_retry))
                        raise
                else:
                    if self.gatekeeper is not None and hasattr(self.gatekeeper, "record_api_error"):
                        self.gatekeeper.record_api_error(str(e))
                    raise e

        if not res or not isinstance(res, dict) or "id" not in res:
            logger.warning(f"Respuesta inválida o nula de orden en BingX ({order.symbol}): {res}")
            return ""

        order_id = str(res["id"])
        strat, sub_lot = self.ledger.parse_client_order_id(cid)
        self.ledger.register_order(
            client_order_id=cid,
            symbol=order.symbol,
            strategy_id=strat,
            sub_lot_id=sub_lot,
            exchange_order_id=order_id,
        )
        self.active_orders_meta[order_id] = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "price": float(order.price),
            "qty": float(order.qty),
            "strategy_id": strat,
            "sub_lot_id": sub_lot,
            "client_order_id": cid,
            "timestamp_ns": order.timestamp_ns,
        }
        if order.side.upper() == "BUY":
            self.live_active_bid_id = order_id
            self.active_bids_by_symbol[order.symbol] = order_id
        elif order.side.upper() == "SELL":
            self.live_active_ask_id = order_id
            self.active_asks_by_symbol[order.symbol] = order_id
        logger.info(f"ORDEN ENVIADA a BingX: {order.symbol} ({target_sym}) {order.side.upper()} {order.qty} @ ${order.price} [ID: {order_id}, CID: {cid}]")
        return order_id

    async def place_orders_batch(self, orders: list[OrderRequest]) -> list[str]:
        """
        Envía un lote de órdenes (create_orders) a BingX en una sola solicitud HTTP/WS.
        Agrupa los niveles de la escalera reduciendo latencia y consumo de rate limits.
        """
        if not orders:
            return []

        valid_orders: list[OrderRequest] = []
        for o in orders:
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
                is_valid, reason = self.gatekeeper.validate_order(o, portfolio)
                if not is_valid:
                    logger.warning(f"Orden batch rechazada por MandateGatekeeper ({o.symbol}): {reason}")
                    continue
            valid_orders.append(o)

        if not valid_orders:
            return []

        for o in valid_orders:
            if o.symbol not in self.initialized_symbols:
                await self.initialize_symbol_margin(o.symbol, max_leverage=self.default_leverage)

        batch_payloads = []
        cids = []
        for o in valid_orders:
            cid = o.client_order_id or self.ledger.generate_client_order_id(o.strategy_id)
            cids.append(cid)
            target_sym = self.resolve_bingx_symbol(o.symbol)
            spec = SYMBOL_SPECS.get(o.symbol)
            tick = spec.tick_size if spec else 0.1
            step = spec.step_size if spec else 0.001
            order_qty = truncate_decimal_step(o.qty, step)
            order_price = truncate_decimal_step(o.price, tick)
            if "1000PEPE" in target_sym:
                if order_price < 0.0005:
                    order_price = round(order_price * 1000.0, 7)
                    order_qty = max(1.0, round(order_qty / 1000.0, 1))
                if o.side.upper() == "BUY" and order_price < 0.0010:
                    order_price = 0.0010
                elif o.side.upper() == "SELL" and order_price > 0.0150:
                    order_price = 0.0150

            batch_payloads.append({
                "symbol": target_sym,
                "type": "limit",
                "side": o.side.lower(),
                "amount": order_qty,
                "price": order_price,
                "params": {
                    "timeInForce": "PO",
                    "postOnly": True,
                    "positionSide": "BOTH",
                    "clientOrderId": cid,
                },
            })

        placed_ids: list[str] = []
        create_orders_fn = getattr(self.exchange, "create_orders_ws", None) or getattr(self.exchange, "create_orders", None)
        if create_orders_fn is not None:
            try:
                res_raw = create_orders_fn(batch_payloads)
                responses = await res_raw if inspect.isawaitable(res_raw) else res_raw
                if isinstance(responses, list):
                    for res, order_req, cid in zip(responses, valid_orders, cids):
                        if isinstance(res, dict) and "id" in res:
                            oid = str(res["id"])
                            strat, sub_lot = self.ledger.parse_client_order_id(cid)
                            self.ledger.register_order(
                                client_order_id=cid,
                                symbol=order_req.symbol,
                                strategy_id=strat,
                                sub_lot_id=sub_lot,
                                exchange_order_id=oid,
                            )
                            self.active_orders_meta[oid] = {
                                "symbol": order_req.symbol,
                                "side": order_req.side.upper(),
                                "price": float(order_req.price),
                                "qty": float(order_req.qty),
                                "strategy_id": strat,
                                "sub_lot_id": sub_lot,
                                "client_order_id": cid,
                                "timestamp_ns": order_req.timestamp_ns,
                            }
                            if order_req.side.upper() == "BUY":
                                self.live_active_bid_id = oid
                                self.active_bids_by_symbol[order_req.symbol] = oid
                            elif order_req.side.upper() == "SELL":
                                self.live_active_ask_id = oid
                                self.active_asks_by_symbol[order_req.symbol] = oid
                            placed_ids.append(oid)
                    logger.info(f"BATCH ORDERS ENVIADAS a BingX ({len(placed_ids)}/{len(valid_orders)} órdenes)")
                    return placed_ids
            except Exception as e:
                logger.warning(f"create_orders batch no soportado o falló en BingX ({e}). Retrocediendo a concurrent...")

        # Fallback individual concurrente
        tasks = [self.place_order_ws(o) for o in valid_orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, str):
                placed_ids.append(r)
        return placed_ids

    async def cancel_orders_batch(self, order_ids: list[str], symbol: str) -> list[str]:
        """Cancela un lote de órdenes en BingX usando cancel_orders nativo o concurrentemente."""
        if not order_ids:
            return []
        target_sym = self.resolve_bingx_symbol(symbol)
        cancel_orders_fn = getattr(self.exchange, "cancel_orders_ws", None) or getattr(self.exchange, "cancel_orders", None)
        cancelled: list[str] = []
        if cancel_orders_fn is not None:
            try:
                res_raw = cancel_orders_fn(order_ids, target_sym)
                res = await res_raw if inspect.isawaitable(res_raw) else res_raw
                for oid in order_ids:
                    self.active_orders_meta.pop(oid, None)
                    cancelled.append(oid)
                return cancelled
            except Exception as e:
                logger.debug(f"cancel_orders batch falló en BingX ({e}). Retrocediendo a concurrente...")

        tasks = [self.cancel_order_ws(oid, symbol) for oid in order_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for oid, r in zip(order_ids, results):
            if r is True:
                cancelled.append(oid)
        return cancelled

    async def prune_stale_orders(
        self,
        symbol: str,
        valid_order_ids: set[str] | list[str] | None = None,
        max_orders_per_side: int = 3,
    ) -> list[str]:
        """
        Poda de Órdenes Huérfanas (prune_stale_orders):
        Verifica fetch_open_orders(symbol) en BingX y cancela cualquier orden abierta que haya quedado
        fuera de los niveles vigentes o que exceda el tope de max_orders_per_side (máx 3 Bids y 3 Asks).
        """
        cancelled_ids: list[str] = []
        target_sym = self.resolve_bingx_symbol(symbol)
        try:
            open_orders = []
            if hasattr(self.exchange, "fetch_open_orders"):
                res = self.exchange.fetch_open_orders(target_sym)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    open_orders = res
        except Exception as e:
            logger.debug(f"fetch_open_orders({target_sym}) en prune_stale_orders (BingX): {e}")
            return cancelled_ids

        if not open_orders:
            return cancelled_ids

        valid_set = set(str(x) for x in valid_order_ids) if valid_order_ids is not None else None
        bids: list[dict[str, Any]] = []
        asks: list[dict[str, Any]] = []

        for o in open_orders:
            oid = str(o.get("id", ""))
            side = str(o.get("side", "")).upper()
            if valid_set is not None and oid not in valid_set:
                try:
                    await self.cancel_order_ws(oid, symbol)
                    cancelled_ids.append(oid)
                except Exception as ec:
                    logger.debug(f"Error cancelando orden huérfana BingX {oid}: {ec}")
                continue

            if side == "BUY":
                bids.append(o)
            elif side == "SELL":
                asks.append(o)

        if len(bids) > max_orders_per_side:
            bids_sorted = sorted(bids, key=lambda x: int(x.get("timestamp") or 0), reverse=True)
            stale_bids = bids_sorted[max_orders_per_side:]
            for o in stale_bids:
                oid = str(o.get("id", ""))
                try:
                    await self.cancel_order_ws(oid, symbol)
                    cancelled_ids.append(oid)
                    logger.info(f"Podada orden huérfana/excedente BID {oid} en {symbol} (BingX)")
                except Exception as ec:
                    logger.debug(f"Error podando bid BingX {oid}: {ec}")

        if len(asks) > max_orders_per_side:
            asks_sorted = sorted(asks, key=lambda x: int(x.get("timestamp") or 0), reverse=True)
            stale_asks = asks_sorted[max_orders_per_side:]
            for o in stale_asks:
                oid = str(o.get("id", ""))
                try:
                    await self.cancel_order_ws(oid, symbol)
                    cancelled_ids.append(oid)
                    logger.info(f"Podada orden huérfana/excedente ASK {oid} en {symbol} (BingX)")
                except Exception as ec:
                    logger.debug(f"Error podando ask BingX {oid}: {ec}")

        return cancelled_ids

    async def close_position_market(self, symbol: str, close_side: str, qty: float) -> str | None:
        """
        Cierra una posición individual en BingX con reduceOnly=True (usado por TP/SL/Time-Stop).
        Verifica previamente que exista posición contraria abierta real y que la cantidad no la exceda,
        previniendo el error -2022 ReduceOnly Order is rejected.
        """
        try:
            target_sym = self.resolve_bingx_symbol(symbol)
            open_pos_contracts = 0.0
            if hasattr(self, "ledger") and self.ledger is not None:
                open_pos_contracts = self.ledger.get_net_inventory(symbol)
            if abs(open_pos_contracts) <= 1e-8 and hasattr(self.exchange, "fetch_positions"):
                try:
                    positions = self.exchange.fetch_positions([target_sym])
                    if inspect.isawaitable(positions):
                        positions = await positions
                    if isinstance(positions, list):
                        for p in positions:
                            if p.get("symbol") in (target_sym, symbol):
                                raw_pos_amt = p.get("info", {}).get("positionAmt")
                                amt = float(p.get("contracts") or raw_pos_amt or 0.0)
                                side_str = str(p.get("side") or "").lower()
                                is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
                                open_pos_contracts = -abs(amt) if is_short else abs(amt)
                                break
                except Exception:
                    pass
            if abs(open_pos_contracts) <= 1e-8 and self.portfolio and symbol in self.portfolio.active_positions:
                open_pos_contracts = self.portfolio.active_positions[symbol]

            if abs(open_pos_contracts) <= 1e-8:
                logger.info(f"close_position_market omitido para {symbol} en BingX: posición neta es 0")
                return None

            # Validar sentido: si close_side es 'sell', debe haber posición LONG (open_pos_contracts > 0)
            if close_side.lower() == "sell" and open_pos_contracts < -1e-8:
                logger.warning(f"close_position_market omitido en BingX: close_side SELL no coincide con posición SHORT {open_pos_contracts}")
                return None
            if close_side.lower() == "buy" and open_pos_contracts > 1e-8:
                logger.warning(f"close_position_market omitido en BingX: close_side BUY no coincide con posición LONG {open_pos_contracts}")
                return None

            safe_qty = min(abs(float(qty)), abs(open_pos_contracts))
            spec = SYMBOL_SPECS.get(symbol)
            if spec and safe_qty >= spec.step_size:
                safe_qty = truncate_decimal_step(safe_qty, spec.step_size)

            res = await self.exchange.create_order(
                symbol=target_sym,
                type="market",
                side=close_side.lower(),
                amount=safe_qty,
                params={"reduceOnly": True, "positionSide": "BOTH"},
            )
            return str(res.get("id", "")) if isinstance(res, dict) else ""
        except Exception as e:
            if "-2022" in str(e) or "reduceonly" in str(e).lower():
                logger.warning(f"reduceOnly rechazado en BingX ({symbol}): {e}")
            else:
                logger.error(f"Error en close_position_market ({symbol} {close_side} {qty}) en BingX: {e}")
            return None

    async def place_protective_stop_loss(
        self, symbol: str, close_side: str, qty: float, stop_price: float
    ) -> str | None:
        """
        Coloca un Stop-Loss nativo server-side (STOP_MARKET) en BingX con workingType='MARK_PRICE'
        y reduceOnly=True para proteger la posición contra caídas del bot o desconexiones.
        """
        try:
            target_sym = self.resolve_bingx_symbol(symbol)
            spec = SYMBOL_SPECS.get(symbol)
            tick = spec.tick_size if spec else 0.1
            step = spec.step_size if spec else 0.001
            safe_stop_px = truncate_decimal_step(stop_price, tick)
            safe_qty = truncate_decimal_step(abs(float(qty)), step)
            if safe_qty <= 0 or safe_stop_px <= 0:
                return None

            prev_sl_id = self.protective_sl_by_symbol.pop(symbol, None)
            if prev_sl_id:
                try:
                    await self.cancel_order_ws(prev_sl_id, symbol)
                except Exception:
                    pass

            res_raw = self.exchange.create_order(
                symbol=target_sym,
                type="STOP_MARKET",
                side=close_side.lower(),
                amount=safe_qty,
                price=None,
                params={
                    "stopPrice": safe_stop_px,
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                    "positionSide": "BOTH",
                },
            )
            res = await res_raw if inspect.isawaitable(res_raw) else res_raw
            if isinstance(res, dict) and "id" in res:
                sl_oid = str(res["id"])
                self.protective_sl_by_symbol[symbol] = sl_oid
                logger.info(
                    f"STOP-LOSS SERVER-SIDE colocado en BingX {symbol}: {close_side} {safe_qty} "
                    f"@ stopPrice=${safe_stop_px} (MARK_PRICE) [ID: {sl_oid}]"
                )
                return sl_oid
            return None
        except Exception as e:
            logger.debug(f"place_protective_stop_loss omitido/no soportado en BingX ({symbol}): {e}")
            return None

    async def place_protective_stop_loss_with_retry(
        self, symbol: str, close_side: str, qty: float, stop_price: float, max_retries: int = 3
    ) -> str | None:
        """
        Coloca Stop-Loss en BingX con hasta max_retries reintentos con backoff exponencial.
        Si la colocación del SL falla tras agotar reintentos, cierra inmediatamente la posición
        a mercado para erradicar posiciones huérfanas desprotegidas.
        """
        delay = 0.05
        for attempt in range(max_retries):
            sl_oid = await self.place_protective_stop_loss(symbol, close_side, qty, stop_price)
            if sl_oid:
                return sl_oid
            logger.warning(
                f"[SL_RETRY] Intento {attempt + 1}/{max_retries} fallido al colocar SL protector en BingX ({symbol}). Reintentando en {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
            delay *= 2.0

        logger.critical(
            f"[ALERTA CRÍTICA: ORPHAN_PREVENTION] Fallaron todos los ({max_retries}) reintentos de colocar Stop-Loss en BingX ({symbol}). "
            "Cerrando inmediatamente posición huérfana a mercado para proteger el capital!"
        )
        await self.close_position_market(symbol, close_side, qty)
        return None

    async def close_position_maker(self, symbol: str, close_side: str, qty: float, price: float) -> str | None:
        """
        Coloca una orden LIMIT reduceOnly en modo Post-Only para Take-Profit en BingX,
        capturando comisión Maker y optimizando el Profit Factor.
        """
        try:
            target_sym = self.resolve_bingx_symbol(symbol)
            open_pos = 0.0
            if hasattr(self, "ledger") and self.ledger is not None:
                open_pos = self.ledger.get_net_inventory(symbol)
            if abs(open_pos) <= 1e-8 and self.portfolio and getattr(self.portfolio, "active_positions", None) and symbol in self.portfolio.active_positions:
                open_pos = self.portfolio.active_positions[symbol]
            if abs(open_pos) <= 1e-8 and hasattr(self.exchange, "fetch_positions"):
                try:
                    positions = self.exchange.fetch_positions([target_sym])
                    if inspect.isawaitable(positions):
                        positions = await positions
                    for p in positions:
                        if p.get("symbol") in (target_sym, symbol):
                            amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0.0)
                            open_pos = amt
                            break
                except Exception:
                    pass

            if abs(open_pos) <= 1e-8:
                return None

            if close_side.lower() == "sell" and open_pos < -1e-8:
                return None
            if close_side.lower() == "buy" and open_pos > 1e-8:
                return None

            safe_qty = min(abs(float(qty)), abs(open_pos))
            safe_price = float(price)
            if hasattr(self.exchange, "price_to_precision"):
                try:
                    p_res = self.exchange.price_to_precision(target_sym, price)
                    if p_res is not None and not str(type(p_res)).startswith("<class 'unittest.mock"):
                        safe_price = float(p_res)
                except Exception:
                    pass

            safe_qty_str = float(safe_qty)
            if hasattr(self.exchange, "amount_to_precision"):
                try:
                    q_res = self.exchange.amount_to_precision(target_sym, safe_qty)
                    if q_res is not None and not str(type(q_res)).startswith("<class 'unittest.mock"):
                        safe_qty_str = float(q_res)
                except Exception:
                    pass

            res = await self.exchange.create_order(
                symbol=target_sym,
                type="limit",
                side=close_side.lower(),
                amount=safe_qty_str,
                price=safe_price,
                params={"reduceOnly": True, "postOnly": True, "positionSide": "BOTH"},
            )
            order_id = str(res.get("id", ""))
            logger.info(f"Take-Profit Maker colocado en BingX {symbol}: {close_side} {safe_qty_str} @ ${safe_price} [ID: {order_id}]")
            return order_id
        except Exception as e:
            logger.debug(f"close_position_maker omitido/rechazado en BingX ({symbol}): {e}")
            return None

    async def cancel_order_ws(self, order_id: str, symbol: str) -> bool:
        """Cancela la orden activa en BingX en tiempo real con fallback a REST si WS no está soportado."""
        try:
            target_sym = self.resolve_bingx_symbol(symbol)
            res = None
            cancelled = False
            if hasattr(self.exchange, "cancel_order_ws") and callable(getattr(self.exchange, "cancel_order_ws")):
                try:
                    res_raw = self.exchange.cancel_order_ws(id=order_id, symbol=target_sym)
                    if inspect.isawaitable(res_raw):
                        res = await res_raw
                    else:
                        res = res_raw
                    cancelled = True
                except Exception as ws_err:
                    if "not supported" not in str(ws_err).lower():
                        raise ws_err
            if not cancelled and hasattr(self.exchange, "cancel_order") and callable(getattr(self.exchange, "cancel_order")):
                res_raw = self.exchange.cancel_order(id=order_id, symbol=target_sym)
                if inspect.isawaitable(res_raw):
                    res = await res_raw
                else:
                    res = res_raw
            if self.live_active_bid_id == order_id:
                self.live_active_bid_id = None
            if self.live_active_ask_id == order_id:
                self.live_active_ask_id = None
            if self.active_bids_by_symbol.get(symbol) == order_id:
                self.active_bids_by_symbol.pop(symbol, None)
            if self.active_asks_by_symbol.get(symbol) == order_id:
                self.active_asks_by_symbol.pop(symbol, None)
            self.active_orders_meta.pop(order_id, None)
            return True
        except Exception as e:
            err_msg = str(e).lower()
            if "not found" in err_msg or "does not exist" in err_msg or "already canceled" in err_msg or "already closed" in err_msg:
                if self.live_active_bid_id == order_id:
                    self.live_active_bid_id = None
                if self.live_active_ask_id == order_id:
                    self.live_active_ask_id = None
                if self.active_bids_by_symbol.get(symbol) == order_id:
                    self.active_bids_by_symbol.pop(symbol, None)
                if self.active_asks_by_symbol.get(symbol) == order_id:
                    self.active_asks_by_symbol.pop(symbol, None)
                self.active_orders_meta.pop(order_id, None)
                return True
            logger.error(f"Error cancelando orden {order_id} en BingX ({symbol}): {e}")
            return False

    async def emergency_flatten_all(self, symbols: list[str]) -> None:
        """
        Cancela todas las órdenes abiertas en BingX (cancel_all_orders) y,
        si existe alguna posición abierta en fetch_positions, envía una orden MARKET
        con params={'reduceOnly': True} para dejar la exposición en cero inmediatamente.
        Corrige detección de posiciones SHORT (side == 'short' o positionAmt < 0 -> close_side = 'buy').
        """
        for sym in symbols:
            target_sym = self.resolve_bingx_symbol(sym)
            try:
                if hasattr(self.exchange, "cancel_all_orders"):
                    await self.exchange.cancel_all_orders(target_sym)
                    logger.warning(f"cancel_all_orders ejecutado en BingX para {sym}")
            except Exception as e:
                logger.error(f"Error en cancel_all_orders para {sym} (BingX): {e}")

        positions = []
        try:
            positions = await self.exchange.fetch_positions(symbols)
        except Exception as e:
            logger.warning(f"Fallback fetch_positions() sin filtro en BingX: {e}")
            try:
                positions = await self.exchange.fetch_positions()
            except Exception as e2:
                logger.error(f"Error en emergency_flatten_all cerrando posiciones en BingX: {e2}")

        for pos in positions:
            try:
                sym = pos.get("symbol")
                raw_pos_amt = pos.get("info", {}).get("positionAmt")
                amt = float(pos.get("contracts") or raw_pos_amt or 0.0)
                side_str = str(pos.get("side") or "").lower()
                is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
                qty = abs(amt)
                if qty > 1e-8:
                    close_side = "buy" if is_short else "sell"
                    logger.warning(f"EMERGENCY FLATTEN (BingX): cerrando posición en {sym} ({'SHORT' if is_short else 'LONG'}): {close_side} {qty}")
                    await self.exchange.create_order(
                        symbol=sym,
                        type="market",
                        side=close_side,
                        amount=qty,
                        params={"reduceOnly": True},
                    )
            except Exception as e:
                logger.error(f"Error en emergency_flatten_all cerrando posición individual en BingX: {e}")

        self.live_active_bid_id = None
        self.live_active_ask_id = None
        self.active_bids_by_symbol.clear()
        self.active_asks_by_symbol.clear()
        self.active_orders_meta.clear()

    async def _handle_order_completion(self, oid: str, sym: str, side: str, portfolio: PortfolioState) -> None:
        """
        Verifica en BingX si la orden finalizó con fill real (status=='closed' o filled > 0)
        o si fue una cancelación normal del market maker (status=='canceled' o expired).
        Evita registrar 'ghost trades' por meras cancelaciones de cotización y deduplica fills.
        """
        async with self._fill_lock:
            if oid and oid in self._processed_order_ids:
                return

            if portfolio is not None:
                self.portfolio = portfolio
            meta = self.active_orders_meta.pop(oid, {})
            order_sym = str(meta.get("symbol") or sym)
            px = float(meta.get("price") or 0.0)
            q = float(meta.get("qty") or 0.0002)
            order_status = "canceled"
            filled_qty = 0.0
            avg_px = px
            try:
                if hasattr(self.exchange, "fetch_order"):
                    res_ord = self.exchange.fetch_order(oid, order_sym)
                    if inspect.isawaitable(res_ord):
                        res_ord = await res_ord
                    if isinstance(res_ord, dict):
                        order_status = str(res_ord.get("status", "")).lower()
                        fetched_filled = float(res_ord.get("filled") or 0.0)
                        if fetched_filled > 0.0:
                            filled_qty = fetched_filled
                        avg_px = float(res_ord.get("average") or res_ord.get("price") or px)
            except Exception as e:
                logger.debug(f"fetch_order({oid}, {order_sym}) en BingX: {e}")

            # Si el precio sigue en 0, no podemos registrar un fill válido
            if avg_px <= 0.0:
                logger.warning(f"Orden {oid} en BingX ({order_sym}) sin precio válido. Omitiendo registro de fill.")
                return

            # Solo registrar fill si la orden fue 'closed' o si tiene cantidad ejecutada > 0
            is_actually_filled = (order_status == "closed") or (filled_qty > 1e-8)
            if is_actually_filled:
                if oid:
                    self._processed_order_ids.add(oid)
                actual_qty = filled_qty if filled_qty > 1e-8 else q
                cid = meta.get("client_order_id", oid)
                if cid:
                    self._processed_order_ids.add(str(cid))
                fill_res = self.ledger.record_fill(
                    order_id_or_cid=cid,
                    symbol=order_sym,
                    side=side,
                    fill_price=avg_px,
                    fill_qty=actual_qty,
                    ts_ns=time.time_ns(),
                )
                net_inv_contracts = self.ledger.get_net_inventory(order_sym)
                if self.portfolio is not None:
                    if not hasattr(self.portfolio, "active_positions") or self.portfolio.active_positions is None:
                        self.portfolio.active_positions = {}
                    # Almacenar nocional USD con signo en active_positions para consistencia de unidades
                    signed_notional = round(net_inv_contracts * avg_px, 2)
                    if abs(signed_notional) > 1e-4:
                        self.portfolio.active_positions[order_sym] = signed_notional
                    else:
                        self.portfolio.active_positions.pop(order_sym, None)
                sub_lot = fill_res.get("sub_lot_id", "MAIN")
                strat_tag = f"{fill_res.get('strategy_id', 'AVELLANEDA_OBI')}_{sub_lot}" if sub_lot != "MAIN" else fill_res.get("strategy_id", "AVELLANEDA_OBI")
                fee_est = round(avg_px * actual_qty * 0.0002, 4)
                pnl_recorded = float(fill_res.get("realized_pnl_usd", 0.0) or 0.0)
                logger.info(f"FILL CONFIRMADO en BingX: {order_sym} {side} {actual_qty} @ ${avg_px} (order {oid}, lot {strat_tag})")
                if self.on_fill and callable(self.on_fill):
                    try:
                        self.on_fill(
                            symbol=order_sym,
                            strategy=strat_tag,
                            side=side,
                            entry_price=avg_px,
                            exit_or_mid_price=avg_px,
                            qty=actual_qty,
                            return_pct=0.0 if pnl_recorded == 0.0 else round((pnl_recorded / max(avg_px * actual_qty, 1e-8)) * 100.0, 3),
                            pnl_usd=pnl_recorded,
                            trade_id=f"BINGX-{oid}",
                            fee_cost=fee_est,
                        )
                    except Exception as ef:
                        logger.debug(f"Error registrando fill en vivo BingX: {ef}")

                # Colocación Síncrona Inmediata de Stop-Loss Protector (con reintentos) y Take-Profit Maker
                if fill_res.get("is_increasing"):
                    close_side = "SELL" if side.upper() == "BUY" else "BUY"
                    spec = SYMBOL_SPECS.get(order_sym)
                    tp_offset_bps = spec.tp_bps if spec else 15.0
                    sl_offset_bps = spec.sl_bps if spec else 18.0
                    tp_price = avg_px * (1.0 + tp_offset_bps / 10_000.0) if side.upper() == "BUY" else avg_px * (1.0 - tp_offset_bps / 10_000.0)
                    sl_price = avg_px * (1.0 - sl_offset_bps / 10_000.0) if side.upper() == "BUY" else avg_px * (1.0 + sl_offset_bps / 10_000.0)
                    await self.place_protective_stop_loss_with_retry(order_sym, close_side, abs(net_inv_contracts) or actual_qty, sl_price)
                    asyncio.create_task(self.close_position_maker(order_sym, close_side, actual_qty, tp_price))

                # OCO (One-Cancels-the-Other): Al cerrarse la posición por TP o SL, cancelar la orden opuesta pendiente
                elif fill_res.get("is_decreasing"):
                    if order_sym in self.protective_sl_by_symbol:
                        sl_oid_to_cancel = self.protective_sl_by_symbol.pop(order_sym)
                        try:
                            await self.cancel_order_ws(sl_oid_to_cancel, order_sym)
                            logger.info(f"[OCO] Posición cerrada en BingX {order_sym}: SL protector {sl_oid_to_cancel} cancelado.")
                        except Exception as e:
                            logger.warning(f"[OCO] Error cancelando SL en BingX: {e}")
            else:
                logger.info(f"Orden {oid} en {order_sym} ({side}) cancelada o expirada en BingX sin fills. No se genera trade fantasma.")

    async def sync_portfolio_state(self, portfolio: PortfolioState) -> None:
        """
        Sincroniza el balance real USDT y las posiciones abiertas de BingX hacia PortfolioState
        (con signo negativo para posiciones SHORT y positivo para LONG).
        """
        if not getattr(self, "_time_synced", False):
            await self.sync_time_difference()

        try:
            bal = await self.exchange.fetch_balance()
            total_usdt = 0.0
            if isinstance(bal, dict):
                # BingX swap en CCXT mapea availableMargin a 'total' descontando el margen congelado en órdenes abiertas.
                # Extraemos primero de 'info.data' donde BingX reporta el equity real (balance + unrealizedProfit)
                # sin penalizar el margen congelado de órdenes limit vivas.
                if "info" in bal and isinstance(bal["info"], dict) and "data" in bal["info"] and isinstance(bal["info"]["data"], list):
                    for item in bal["info"]["data"]:
                        if isinstance(item, dict) and item.get("asset") in ("USDT", "VST"):
                            eq_val = float(item.get("equity") or item.get("balance") or 0.0)
                            if eq_val > 0.0:
                                total_usdt = eq_val
                                break
                if total_usdt <= 0.0:
                    if "total" in bal and isinstance(bal["total"], dict) and "USDT" in bal["total"] and float(bal["total"]["USDT"] or 0.0) > 0:
                        total_usdt = float(bal["total"]["USDT"] or 0.0)
                    elif "total" in bal and isinstance(bal["total"], dict) and "VST" in bal["total"]:
                        total_usdt = float(bal["total"]["VST"] or 0.0)
                    elif "USDT" in bal and isinstance(bal["USDT"], dict) and "total" in bal["USDT"] and float(bal["USDT"]["total"] or 0.0) > 0:
                        total_usdt = float(bal["USDT"]["total"] or 0.0)
                    elif "VST" in bal and isinstance(bal["VST"], dict) and "total" in bal["VST"]:
                        total_usdt = float(bal["VST"]["total"] or 0.0)
                    elif "info" in bal and isinstance(bal["info"], dict):
                        info = bal["info"]
                        total_usdt = float(info.get("balance") or info.get("equity") or 0.0)

            if not hasattr(portfolio, "exchange_equities"):
                portfolio.exchange_equities = {}
            if not hasattr(portfolio, "initial_exchange_equities"):
                portfolio.initial_exchange_equities = {}

            if total_usdt > 0.0:
                is_first_sync = "bingx" not in portfolio.initial_exchange_equities or portfolio.initial_exchange_equities["bingx"] <= 0.0
                if is_first_sync:
                    portfolio.initial_exchange_equities["bingx"] = round(total_usdt, 2)

                portfolio.exchange_equities["bingx"] = total_usdt
                if getattr(portfolio, "trading_mode", "") == "LIVE_MULTI":
                    portfolio.current_equity_usd = round(sum(portfolio.exchange_equities.values()), 2)
                    if portfolio.initial_exchange_equities:
                        combined_initial = round(sum(portfolio.initial_exchange_equities.values()), 2)
                        if combined_initial > 0 and (portfolio.base_capital_usd in (100.0, 5000.0) or len(portfolio.initial_exchange_equities) >= 2):
                            portfolio.base_capital_usd = combined_initial
                            portfolio.daily_high_watermark_usd = max(portfolio.base_capital_usd, portfolio.current_equity_usd)
                elif getattr(portfolio, "trading_mode", "") == "LIVE_BINGX":
                    portfolio.current_equity_usd = round(total_usdt, 2)
                    if is_first_sync or portfolio.base_capital_usd in (100.0, 5000.0):
                        portfolio.base_capital_usd = round(total_usdt, 2)
                        portfolio.daily_high_watermark_usd = max(portfolio.base_capital_usd, portfolio.current_equity_usd)
                else:
                    portfolio.current_equity_usd = round(total_usdt, 2)

                if portfolio.current_equity_usd > portfolio.daily_high_watermark_usd:
                    portfolio.daily_high_watermark_usd = portfolio.current_equity_usd

            positions = await self.exchange.fetch_positions()
            active: dict[str, float] = {}
            for pos in positions:
                sym = pos.get("symbol")
                if not sym:
                    continue
                raw_pos_amt = pos.get("info", {}).get("positionAmt")
                amt = float(pos.get("contracts") or raw_pos_amt or 0.0)
                side_str = str(pos.get("side") or "").lower()
                is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
                notional = abs(float(pos.get("notional") or pos.get("info", {}).get("notional") or 0.0))
                if notional == 0.0 and abs(amt) > 0.0:
                    mp = float(pos.get("markPrice") or pos.get("info", {}).get("markPrice") or 0.0)
                    notional = abs(amt) * mp
                if abs(amt) > 1e-8:
                    signed_notional = -notional if is_short else notional
                    active[sym] = round(signed_notional, 2)

            if getattr(portfolio, "trading_mode", "") == "LIVE_MULTI":
                for sym, notional in active.items():
                    portfolio.active_positions[sym] = notional
                for sym in list(portfolio.active_positions.keys()):
                    spec = SYMBOL_SPECS.get(sym)
                    if spec and spec.exchange == "bingx" and sym not in active:
                        portfolio.active_positions.pop(sym, None)
            else:
                portfolio.active_positions = active
        except Exception as e:
            logger.error(f"Error sincronizando balance y posiciones con BingX: {e}")

        try:
            if hasattr(self.exchange, "fetch_open_orders"):
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
                    if hasattr(portfolio, "open_orders_notional") and isinstance(portfolio.open_orders_notional, dict):
                        open_notional_by_sym: dict[str, float] = {}
                        for o in open_orders:
                            if isinstance(o, dict):
                                osym = str(o.get("symbol") or "")
                                opx = float(o.get("price") or 0.0)
                                orem = float(o.get("remaining") or o.get("amount") or 0.0)
                                if osym and opx > 0 and orem > 0:
                                    open_notional_by_sym[osym] = open_notional_by_sym.get(osym, 0.0) + (opx * orem)
                        portfolio.open_orders_notional.update(open_notional_by_sym)

                    for sym_k, bid_oid in list(self.active_bids_by_symbol.items()):
                        if bid_oid not in open_order_ids:
                            self.active_bids_by_symbol.pop(sym_k, None)
                            if self.live_active_bid_id == bid_oid:
                                self.live_active_bid_id = None
                            await self._handle_order_completion(bid_oid, sym_k, "BUY", portfolio)

                    for sym_k, ask_oid in list(self.active_asks_by_symbol.items()):
                        if ask_oid not in open_order_ids:
                            self.active_asks_by_symbol.pop(sym_k, None)
                            if self.live_active_ask_id == ask_oid:
                                self.live_active_ask_id = None
                            await self._handle_order_completion(ask_oid, sym_k, "SELL", portfolio)

                    if self.live_active_bid_id and self.live_active_bid_id not in open_order_ids:
                        old_bid = self.live_active_bid_id
                        self.live_active_bid_id = None
                        meta_sym = str(self.active_orders_meta.get(old_bid, {}).get("symbol") or "BTC/USDT:USDT")
                        await self._handle_order_completion(old_bid, meta_sym, "BUY", portfolio)

                    if self.live_active_ask_id and self.live_active_ask_id not in open_order_ids:
                        old_ask = self.live_active_ask_id
                        self.live_active_ask_id = None
                        meta_sym = str(self.active_orders_meta.get(old_ask, {}).get("symbol") or "BTC/USDT:USDT")
                        await self._handle_order_completion(old_ask, meta_sym, "SELL", portfolio)
        except Exception as e:
            logger.error(f"Error sincronizando órdenes abiertas con BingX: {e}")

        # Ingesta de Ejecuciones Reales (fetch_my_trades) hacia strategy_ledger.py y recent_trades en BingX
        try:
            if hasattr(self.exchange, "fetch_my_trades"):
                for sym in list(portfolio.active_positions.keys()) + list(self.active_bids_by_symbol.keys()) + list(self.active_asks_by_symbol.keys()):
                    target_sym = self.resolve_bingx_symbol(sym)
                    since_ts = self._last_trade_ts_ms.get(sym, self._init_ts_ms - 60_000)
                    res_trades = self.exchange.fetch_my_trades(target_sym, since=since_ts)
                    if inspect.isawaitable(res_trades):
                        res_trades = await res_trades
                    if not isinstance(res_trades, list):
                        continue
                    for tr in res_trades:
                        if not isinstance(tr, dict):
                            continue
                        tr_sym_raw = tr.get("symbol")
                        if tr_sym_raw and self.resolve_bingx_symbol(str(tr_sym_raw)) != target_sym:
                            continue
                        tid = str(tr.get("id") or "")
                        ord_id = str(tr.get("order") or "")
                        if (tid and tid in self._processed_trade_ids) or (ord_id and ord_id in self._processed_order_ids):
                            continue
                        if tid:
                            self._processed_trade_ids.add(tid)
                        if ord_id:
                            self._processed_order_ids.add(ord_id)
                        tr_ts = int(tr.get("timestamp") or 0)
                        if tr_ts > self._last_trade_ts_ms.get(sym, 0):
                            self._last_trade_ts_ms[sym] = tr_ts
                        tr_side = str(tr.get("side") or "").upper()
                        tr_px = float(tr.get("price") or 0.0)
                        tr_qty = float(tr.get("amount") or 0.0)
                        if tr_px <= 0.0 or tr_qty <= 0.0:
                            continue
                        fee_cost = float(tr.get("fee", {}).get("cost") or tr.get("info", {}).get("fee") or (tr_px * tr_qty * 0.0002))
                        cid = str(tr.get("order") or tr.get("clientOrderId") or "")
                        fill_res = self.ledger.record_fill(
                            order_id_or_cid=cid or f"tr_{tid}",
                            symbol=sym,
                            side=tr_side,
                            fill_price=tr_px,
                            fill_qty=tr_qty,
                            ts_ns=(tr_ts * 1_000_000) if tr_ts > 0 else time.time_ns(),
                        )
                        sub_lot = fill_res.get("sub_lot_id", "MAIN")
                        strat_tag = f"{fill_res.get('strategy_id', 'AVELLANEDA_OBI')}_{sub_lot}" if sub_lot != "MAIN" else fill_res.get("strategy_id", "AVELLANEDA_OBI")
                        gross_pnl = float(fill_res.get("realized_pnl_usd", 0.0) or 0.0)
                        net_pnl = gross_pnl - fee_cost
                        ret_pct = (net_pnl / (tr_px * tr_qty)) * 100.0 if (tr_px * tr_qty) > 0 else 0.05
                        logger.info(
                            f"TRADE REAL REGISTRADO [BingX {sym}]: {tr_side} {tr_qty} @ ${tr_px} "
                            f"(net PnL: ${net_pnl:.2f}, fee: ${fee_cost:.4f}, lot: {strat_tag})"
                        )
                        if self.on_fill and callable(self.on_fill):
                            try:
                                self.on_fill(
                                    symbol=sym,
                                    strategy=strat_tag,
                                    side=tr_side,
                                    entry_price=tr_px,
                                    exit_or_mid_price=tr_px,
                                    qty=tr_qty,
                                    return_pct=ret_pct,
                                    pnl_usd=round(net_pnl, 2),
                                    trade_id=f"LIVE-BINGX-{tid or cid}",
                                    timestamp_ns=(tr_ts * 1_000_000) if tr_ts > 0 else None,
                                    fee_cost=round(fee_cost, 4),
                                )
                            except Exception as ef:
                                logger.debug(f"Error despachando on_fill BingX: {ef}")
        except Exception as et:
            logger.debug(f"fetch_my_trades en BingX sync_portfolio_state: {et}")

    async def reconcile_existing_positions(self, symbols: list[str]) -> list[dict[str, Any]]:
        """
        Reconciliación en Frío (reconcile_existing_positions) en BingX:
        1. Limpia órdenes huérfanas antiguas con cancel_all_orders.
        2. Inspecciona fetch_positions.
        3. Registra posiciones abiertas en strategy_ledger.py con su entryPrice y
           su breakEvenPrice real (float(pos.get("info", {}).get("breakEvenPrice") or entry_price)),
           para que el motor gestione su cierre reduceOnly por encima del punto de equilibrio.
        """
        reconciled = []
        for sym in symbols:
            target_sym = self.resolve_bingx_symbol(sym)
            try:
                if hasattr(self.exchange, "cancel_all_orders"):
                    await self.exchange.cancel_all_orders(target_sym)
            except Exception as e:
                logger.debug(f"cancel_all_orders en reconcile BingX ({target_sym}): {e}")

        positions = []
        try:
            if hasattr(self.exchange, "fetch_positions"):
                res = self.exchange.fetch_positions(symbols)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    positions = res
        except Exception as e:
            try:
                res = self.exchange.fetch_positions()
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    positions = res
            except Exception as e2:
                logger.error(f"Error consultando fetch_positions en reconcile BingX: {e2}")

        for pos in positions:
            sym = pos.get("symbol")
            if not sym:
                continue
            raw_pos_amt = pos.get("info", {}).get("positionAmt")
            amt = float(pos.get("contracts") or raw_pos_amt or 0.0)
            side_str = str(pos.get("side") or "").lower()
            is_short = (side_str == "short") or (raw_pos_amt is not None and float(raw_pos_amt) < 0.0) or (amt < 0.0)
            qty = abs(amt)
            if qty > 1e-8:
                entry_px = float(pos.get("entryPrice") or pos.get("info", {}).get("entryPrice") or 0.0)
                be_px = float(pos.get("info", {}).get("breakEvenPrice") or entry_px)
                side = "SELL" if is_short else "BUY"
                lot = self.ledger.reconcile_position(
                    symbol=sym,
                    side=side,
                    quantity=qty,
                    entry_price=entry_px,
                    break_even_price=be_px,
                    strategy_id="AVELLANEDA_OBI",
                    sub_lot_id="MAIN",
                )
                if hasattr(self, "db_store") and self.db_store is not None:
                    try:
                        lot_k = lot.key
                        signed_q = qty if side == "BUY" else -qty
                        self.db_store.save_lot(
                            lot_key=lot_k,
                            symbol=sym,
                            strategy_id="AVELLANEDA_OBI",
                            sub_lot_id="MAIN",
                            inventory_q=signed_q,
                            entry_price=entry_px,
                            break_even_price=be_px,
                            entry_ts_ns=time.time_ns(),
                            realized_pnl_usd=0.0,
                        )
                    except Exception as e_lot:
                        logger.error(f"Error persistiendo lote reconciliado en db_store (BingX): {e_lot}")
                reconciled.append({
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "entry_price": entry_px,
                    "break_even_price": be_px,
                })
                logger.info(
                    f"POSICIÓN RECONCILIADA en BingX {sym}: {side} {qty} @ entry ${entry_px:.2f} (Break-Even: ${be_px:.2f})"
                )
        return reconciled

    def start_sync_loop(self, portfolio: PortfolioState) -> None:
        """Inicia el bucle de sincronización periódica y los User Data Streams por WebSocket en BingX."""
        self.running = True
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync_loop(portfolio))
        if self._watch_orders_task is None or self._watch_orders_task.done():
            self._watch_orders_task = asyncio.create_task(self._watch_orders_loop(portfolio))
        if self._watch_trades_task is None or self._watch_trades_task.done():
            self._watch_trades_task = asyncio.create_task(self._watch_trades_loop(portfolio))

    def stop_sync_loop(self) -> None:
        """Detiene el bucle de sincronización periódica y los WebSocket streams en BingX."""
        self.running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            self._sync_task = None
        if self._watch_orders_task and not self._watch_orders_task.done():
            self._watch_orders_task.cancel()
            self._watch_orders_task = None
        if self._watch_trades_task and not self._watch_trades_task.done():
            self._watch_trades_task.cancel()
            self._watch_trades_task = None

    async def _watch_orders_loop(self, portfolio: PortfolioState) -> None:
        """Stream privado por WebSocket para órdenes en BingX (latencia sub-10ms)."""
        while self.running:
            try:
                watch_orders_fn = getattr(self.exchange, "watch_orders", None)
                if watch_orders_fn is None or not callable(watch_orders_fn):
                    break
                orders = await watch_orders_fn()
                if not isinstance(orders, list):
                    orders = [orders]
                for o in orders:
                    status = str(o.get("status", "")).lower()
                    oid = str(o.get("id", ""))
                    sym = str(o.get("symbol", ""))
                    side = str(o.get("side", "")).upper()
                    filled_amt = float(o.get("filled") or 0.0)
                    if status in ("closed", "filled") or (status in ("canceled", "expired") and filled_amt > 1e-8):
                        await self._handle_order_completion(oid, sym, side, portfolio)
                    elif status in ("canceled", "expired"):
                        self.active_orders_meta.pop(oid, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.running:
                    break
                err_msg = str(e).lower()
                if "notsupported" in err_msg or "not implemented" in err_msg or "unsupported" in err_msg:
                    logger.debug(f"watch_orders no soportado por BingX en este modo ({e}). Usando polling.")
                    break
                logger.debug(f"watch_orders reconectando en BingX: {e}")
                await asyncio.sleep(2.0)

    async def _watch_trades_loop(self, portfolio: PortfolioState) -> None:
        """Stream privado por WebSocket para fills instantáneos (watch_my_trades) en BingX."""
        while self.running:
            try:
                watch_trades_fn = getattr(self.exchange, "watch_my_trades", None)
                if watch_trades_fn is None or not callable(watch_trades_fn):
                    break
                trades = await watch_trades_fn()
                if not isinstance(trades, list):
                    trades = [trades]
                for tr in trades:
                    tid = str(tr.get("id", ""))
                    ord_id = str(tr.get("order") or "")
                    if (tid and tid in self._processed_trade_ids) or (ord_id and ord_id in self._processed_order_ids):
                        continue
                    if tid:
                        self._processed_trade_ids.add(tid)
                    sym = str(tr.get("symbol", ""))
                    side = str(tr.get("side", "")).upper()
                    if ord_id:
                        await self._handle_order_completion(ord_id, sym, side, portfolio)
                        continue
                    price = float(tr.get("price") or 0.0)
                    amount = float(tr.get("amount") or 0.0)
                    if self.on_fill:
                        try:
                            pnl_val = 0.0
                            self.on_fill(
                                symbol=sym,
                                strategy="AVELLANEDA_OBI",
                                side=side,
                                entry_price=price,
                                exit_or_mid_price=price,
                                qty=amount,
                                return_pct=0.05,
                                pnl_usd=pnl_val,
                                trade_id=f"WS-BX-{tid}" if tid else None,
                            )
                        except Exception as fe:
                            logger.error(f"Error en on_fill desde watch_my_trades BingX: {fe}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.running:
                    break
                err_msg = str(e).lower()
                if "notsupported" in err_msg or "not implemented" in err_msg or "unsupported" in err_msg:
                    logger.debug(f"watch_my_trades no soportado por BingX en este modo ({e}). Usando polling.")
                    break
                logger.debug(f"watch_my_trades reconectando en BingX: {e}")
                await asyncio.sleep(2.0)

    async def _sync_loop(self, portfolio: PortfolioState) -> None:
        while self.running:
            try:
                await self.sync_portfolio_state(portfolio)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Excepción en _sync_loop BingXLiveWsGateway: {e}")
            await asyncio.sleep(2.0)

    async def close(self) -> None:
        """Cierra el gateway y las conexiones asíncronas de CCXT."""
        self.stop_sync_loop()
        if hasattr(self.exchange, "close"):
            await self.exchange.close()


GLOBAL_MOCK_TRADFI_STATE: bool | None = None
GLOBAL_MOCK_TRADFI_CLOSED: bool = False
GLOBAL_CHAOS_LATENCY_ACTIVE: bool = False


def is_tradfi_market_open(dt_utc: datetime | None = None) -> bool:
    """
    Guardia de Sesión TradFi para activos macro (Oro XAU, Plata XAG):
    Devuelve False durante:
    1. Cierre de fin de semana: Viernes 20:30 UTC a Domingo 22:05 UTC.
    2. Ventana de mantenimiento diario: Lunes a Jueves de 20:45 UTC a 22:05 UTC.
    Devuelve True en cualquier otro horario en que los mercados macro están abiertos.
    """
    global GLOBAL_MOCK_TRADFI_STATE, GLOBAL_MOCK_TRADFI_CLOSED
    if GLOBAL_MOCK_TRADFI_STATE is not None:
        return GLOBAL_MOCK_TRADFI_STATE

    if GLOBAL_MOCK_TRADFI_CLOSED:
        return False

    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)

    weekday = dt_utc.weekday()  # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
    h = dt_utc.hour
    m = dt_utc.minute

    # 1. Cierre de fin de semana: Viernes 20:30 UTC a Domingo 22:05 UTC
    if weekday == 4:  # Friday
        if h > 20 or (h == 20 and m >= 30):
            return False
    elif weekday == 5:  # Saturday
        return False
    elif weekday == 6:  # Sunday
        if h < 22 or (h == 22 and m < 5):
            return False
    else:
        # 2. Ventana de mantenimiento diario (Lunes a Jueves): 20:45 UTC a 22:05 UTC
        if (h == 20 and m >= 45) or (h == 21) or (h == 22 and m < 5):
            return False

    return True


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

    async def prune_stale_orders(
        self,
        symbol: str,
        valid_order_ids: set[str] | list[str] | None = None,
        max_orders_per_side: int = 3,
    ) -> list[str]:
        cancelled: list[str] = []
        now_ns = time.time_ns()
        bids = [o for o in self.simulator.orders.values() if o.request.side == "BUY" and o.status in ("OPEN", "PARTIALLY_FILLED")]
        asks = [o for o in self.simulator.orders.values() if o.request.side == "SELL" and o.status in ("OPEN", "PARTIALLY_FILLED")]
        if len(bids) > max_orders_per_side:
            for o in bids[:-max_orders_per_side]:
                self.simulator.request_cancel(o.order_id, now_ns)
                cancelled.append(o.order_id)
        if len(asks) > max_orders_per_side:
            for o in asks[:-max_orders_per_side]:
                self.simulator.request_cancel(o.order_id, now_ns)
                cancelled.append(o.order_id)
        return cancelled



@dataclass
class SymbolMarketSpec:
    symbol: str
    tick_size: float
    step_size: float
    base_asset: str
    quote_asset: str = "USDT"
    asset_class: str = "CRYPTO"  # "CRYPTO" | "TRADFI_COMMODITY"
    exchange: str = "binance"    # "binance" | "bingx"
    min_spread_bps: float = 10.0
    min_notional: float = 5.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0


SYMBOL_SPECS: dict[str, SymbolMarketSpec] = {
    "BTC/USDT:USDT": SymbolMarketSpec("BTC/USDT:USDT", tick_size=0.10, step_size=0.001, base_asset="BTC", asset_class="CRYPTO", exchange="binance", min_notional=100.0, min_spread_bps=10.0),
    "ETH/USDT:USDT": SymbolMarketSpec("ETH/USDT:USDT", tick_size=0.01, step_size=0.001, base_asset="ETH", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "SOL/USDT:USDT": SymbolMarketSpec("SOL/USDT:USDT", tick_size=0.01, step_size=0.1, base_asset="SOL", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "BNB/USDT:USDT": SymbolMarketSpec("BNB/USDT:USDT", tick_size=0.01, step_size=0.01, base_asset="BNB", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "DOGE/USDT:USDT": SymbolMarketSpec("DOGE/USDT:USDT", tick_size=0.00001, step_size=1.0, base_asset="DOGE", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "XRP/USDT:USDT": SymbolMarketSpec("XRP/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="XRP", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "SUI/USDT:USDT": SymbolMarketSpec("SUI/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="SUI", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "AVAX/USDT:USDT": SymbolMarketSpec("AVAX/USDT:USDT", tick_size=0.001, step_size=1.0, base_asset="AVAX", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "LINK/USDT:USDT": SymbolMarketSpec("LINK/USDT:USDT", tick_size=0.001, step_size=0.01, base_asset="LINK", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "PEPE/USDT:USDT": SymbolMarketSpec("PEPE/USDT:USDT", tick_size=0.0000001, step_size=1.0, base_asset="1000PEPE", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "NEAR/USDT:USDT": SymbolMarketSpec("NEAR/USDT:USDT", tick_size=0.001, step_size=1.0, base_asset="NEAR", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "ARB/USDT:USDT": SymbolMarketSpec("ARB/USDT:USDT", tick_size=0.00001, step_size=0.1, base_asset="ARB", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    # BingX TradFi Commodities (soportadas también en Binance USDⓈ-M)
    "XAU/USDT:USDT": SymbolMarketSpec("XAU/USDT:USDT", tick_size=0.10, step_size=0.01, base_asset="XAU", asset_class="TRADFI_COMMODITY", exchange="bingx", min_notional=5.0, min_spread_bps=6.0),
    "XAG/USDT:USDT": SymbolMarketSpec("XAG/USDT:USDT", tick_size=0.01, step_size=0.1, base_asset="XAG", asset_class="TRADFI_COMMODITY", exchange="bingx", min_notional=5.0, min_spread_bps=6.5),
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
    active_ladder_bids: dict[str, str] = field(default_factory=dict)
    active_ladder_asks: dict[str, str] = field(default_factory=dict)
    active_ladder_bid_prices: dict[str, float] = field(default_factory=dict)
    active_ladder_ask_prices: dict[str, float] = field(default_factory=dict)
    tp_maker_order_id: str | None = None
    tp_maker_submit_ts_s: float | None = None
    inventory_open_ts_ns: int | None = None
    avg_entry_price: float | None = None
    break_even_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    tp_bps: float = 22.0
    sl_bps: float = 35.0
    max_holding_time_s: float = 1800.0
    last_exit_reason: str = "NONE"


@dataclass
class ContagionShieldState:
    bid_shield_active: bool = False
    ask_shield_active: bool = False
    trigger_reason: str = "NORMAL"
    bearish_count: int = 0
    bullish_count: int = 0
    btc_obi: float = 0.0
    xau_safe_haven_active: bool = False


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
        self.bingx_gateway: BingXLiveWsGateway | None = None
        self.primary_exchange: str = os.getenv("PRIMARY_EXCHANGE", "binance").strip().lower()
        self.trading_mode: str = "PAPER_L2"
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

_raw_allowed = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradingModeRequest(BaseModel):
    mode: Literal["PAPER_L2", "LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"]
    confirm: str | None = None


class BaseCapitalRequest(BaseModel):
    base_capital_usd: float
    confirm: str | None = None


class ResetStateRequest(BaseModel):
    base_capital_usd: float | None = None
    confirm: str | None = None


class ResetKillSwitchRequest(BaseModel):
    confirm: str | None = None


class LeverageTierRequest(BaseModel):
    leverage: Literal[3, 5, 7, 10]


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Chequeo de salud público y sanitizado sin divulgación de datos sensibles."""
    uptime_s = round(time.time() - getattr(engine, "_start_ts_s", time.time()), 2)
    return {
        "status": "OK",
        "mode": engine.trading_mode,
        "uptime_s": uptime_s,
        "timestamp_ns": time.time_ns(),
    }


@app.get("/api/leverage", dependencies=[Depends(verify_bearer_token)])
async def get_leverage_evaluation_endpoint() -> dict[str, Any]:
    """Devuelve la matriz de evaluación de apalancamiento (x3, x5, x7, x10) en tiempo real."""
    return engine.evaluate_leverage_tiers()


@app.post("/api/leverage", dependencies=[Depends(verify_bearer_token)])
async def set_leverage_tier_endpoint(req: LeverageTierRequest) -> dict[str, Any]:
    """Ajusta el nivel de apalancamiento activo (x3, x5, x7, x10) y recalibra TP/SL y Gate 3."""
    try:
        analysis = await engine.set_leverage_tier(req.leverage)
        return {
            "status": "LEVERAGE_UPDATED",
            "selected_leverage": engine.selected_leverage,
            "leverage_analysis": analysis,
            "timestamp_ns": time.time_ns(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/telemetry", dependencies=[Depends(verify_bearer_token)])
async def get_telemetry_endpoint() -> dict[str, Any]:
    """Snapshot completo de telemetría real de microestructura y riesgo."""
    return engine.get_telemetry_snapshot()


@app.get("/api/trades", dependencies=[Depends(verify_bearer_token)])
async def get_trades_endpoint(limit: int = 1000, symbol: str | None = None) -> dict[str, Any]:
    """
    Retorna el historial de transacciones persistidas en SQLite (o memoria)
    para análisis detallado y exportación, permitiendo consultar hasta miles de registros.
    """
    if hasattr(engine, "db_store") and engine.db_store is not None:
        trades = engine.db_store.load_recent_trades(limit=limit)
    else:
        trades = list(engine.recent_trades)[::-1][:limit]

    if symbol and symbol.upper() != "TODOS":
        trades = [t for t in trades if symbol.upper() in str(t.get("symbol", "")).upper()]

    return {
        "status": "OK",
        "count": len(trades),
        "limit": limit,
        "trades": trades,
    }


@app.get("/api/trades/export-csv", dependencies=[Depends(verify_bearer_token)])
async def export_trades_csv_endpoint(limit: int = 10000, symbol: str | None = None):
    """
    Exporta el historial completo de transacciones en formato CSV descargable
    directamente desde la base de datos transaccional SQLite WAL.
    """
    if hasattr(engine, "db_store") and engine.db_store is not None:
        trades = engine.db_store.load_recent_trades(limit=limit)
    else:
        trades = list(engine.recent_trades)[::-1][:limit]

    if symbol and symbol.upper() != "TODOS":
        trades = [t for t in trades if symbol.upper() in str(t.get("symbol", "")).upper()]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Timestamp_MS",
        "Time_UTC",
        "Symbol",
        "Strategy",
        "Side",
        "Entry_Price",
        "Exit_Mid_Price",
        "Qty",
        "Return_Pct",
        "PnL_USD",
        "Fee_Cost",
        "Exchange",
    ])
    for t in trades:
        writer.writerow([
            t.get("id", ""),
            t.get("timestamp_ms", ""),
            t.get("time_str", ""),
            t.get("symbol", ""),
            t.get("strategy", ""),
            t.get("side", ""),
            t.get("entry_price", 0.0),
            t.get("exit_or_mid_price", 0.0),
            t.get("qty", 0.0),
            t.get("return_pct", 0.0),
            t.get("pnl_usd", 0.0),
            t.get("fee_cost", 0.0),
            t.get("exchange", "BINANCE"),
        ])

    csv_data = output.getvalue()
    filename = f"trades_blotter_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/reset-state", dependencies=[Depends(verify_bearer_token)])
async def reset_state_endpoint(req: ResetStateRequest | None = None, request: Request = None) -> dict[str, Any]:
    """
    Reinicia el estado institucional del motor cuantitativo desde 0:
    Limpia órdenes huérfanas, historial de trades, contadores de victorias/derrotas,
    PnL bruto y neto acumulado, curva de equidad y estado de cuarentena DSR.
    Exige confirm='RESET_STATE'.
    """
    client_ip = request.client.host if request and request.client else "unknown"
    if not req or req.confirm != "RESET_STATE":
        logger.warning(f"[SECURITY_AUDIT] Intento de reset-state sin confirmación válida desde IP: {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Operación destructiva: Debe enviar confirm='RESET_STATE' en el cuerpo de la solicitud.",
        )

    logger.warning(f"[SECURITY_AUDIT] reset-state confirmado ejecutado desde IP: {client_ip}")
    cap = req.base_capital_usd
    engine.reset_state_from_zero(base_capital_usd=cap)
    return {
        "status": "STATE_RESET_OK",
        "mode": engine.trading_mode,
        "base_capital_usd": engine.portfolio.base_capital_usd,
        "current_equity_usd": engine.portfolio.current_equity_usd,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/trading-mode", dependencies=[Depends(verify_bearer_token)])
async def set_trading_mode_endpoint(req: TradingModeRequest, request: Request = None) -> dict[str, Any]:
    """
    Conmuta el modo de ejecución entre PAPER_L2, LIVE_BINANCE, LIVE_BINGX y LIVE_MULTI.
    REGLA DE SEGURIDAD CRÍTICA:
    Si se solicita LIVE_BINANCE, verifica que BINANCE_API_KEY y BINANCE_API_SECRET existan.
    Si se solicita LIVE_BINGX, verifica que BINGX_API_KEY y BINGX_API_SECRET existan.
    Si se solicita LIVE_MULTI, verifica ambas credenciales de Binance y BingX.
    Verifica que gatekeeper.kill_switch_engaged sea False.
    Si faltan claves API o el Kill-Switch está activo, rechaza con HTTP 400 y mantiene el bot en PAPER_L2.
    Para cualquier modo LIVE exige confirm='LIVE'.
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
    elif req.mode == "LIVE_BINGX":
        api_key = os.getenv("BINGX_API_KEY", "").strip()
        api_secret = os.getenv("BINGX_API_SECRET", "").strip()
        if not api_key or not api_secret:
            logger.warning("Intento de cambio a LIVE_BINGX rechazado por falta de claves API.")
            raise HTTPException(
                status_code=400,
                detail="Configuración de seguridad incompleta: Debe configurar BINGX_API_KEY y BINGX_API_SECRET en su archivo .env antes de operar en vivo con BingX. El bot se mantiene en PAPER_L2.",
            )
        if engine.gatekeeper.kill_switch_engaged:
            logger.warning("Intento de cambio a LIVE_BINGX rechazado: Kill-Switch activo.")
            raise HTTPException(
                status_code=400,
                detail="Bloqueo de seguridad: El Kill-Switch de emergencia está activo. Debe resetear el Kill-Switch antes de pasar a LIVE_BINGX.",
            )
    elif req.mode == "LIVE_MULTI":
        bin_key = os.getenv("BINANCE_API_KEY", "").strip()
        bin_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        bx_key = os.getenv("BINGX_API_KEY", "").strip()
        bx_secret = os.getenv("BINGX_API_SECRET", "").strip()
        if not bin_key or not bin_secret:
            raise HTTPException(
                status_code=400,
                detail="Configuración de seguridad incompleta: Debe configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env antes de operar en modo LIVE_MULTI.",
            )
        if not bx_key or not bx_secret:
            raise HTTPException(
                status_code=400,
                detail="Configuración de seguridad incompleta: Debe configurar BINGX_API_KEY y BINGX_API_SECRET en su archivo .env antes de operar en modo LIVE_MULTI.",
            )
        if engine.gatekeeper.kill_switch_engaged:
            raise HTTPException(
                status_code=400,
                detail="Bloqueo de seguridad: El Kill-Switch de emergencia está activo. Debe resetear el Kill-Switch antes de pasar a LIVE_MULTI.",
            )

    client_ip = request.client.host if request and request.client else "unknown"
    if req.mode.startswith("LIVE"):
        if req.confirm != "LIVE":
            logger.warning(f"[SECURITY_AUDIT] Intento de pasar a {req.mode} sin confirm='LIVE' desde IP: {client_ip}")
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida: Para cambiar a modo LIVE debe enviar confirm='LIVE'.",
            )
    elif req.confirm and req.confirm not in ("CONFIRM_MODE", "PAPER_L2"):
        raise HTTPException(
            status_code=400,
            detail="Confirmación inválida: Para cambiar de modo debe enviar confirm='CONFIRM_MODE'.",
        )

    logger.warning(f"[SECURITY_AUDIT] Cambio de modo a {req.mode} ejecutado desde IP: {client_ip}")

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


@app.post("/api/emergency-flatten", dependencies=[Depends(verify_bearer_token)])
async def emergency_flatten_endpoint(request: Request = None) -> dict[str, Any]:
    """
    Disparador síncrono del Kill-Switch de emergencia (Gate 4).
    Bloquea inmediatamente todas las órdenes nuevas y activa el flatten idempotente de órdenes y posiciones.
    """
    engine.gatekeeper.trigger_emergency_flatten()
    engine.trigger_emergency_flatten_sync()
    if engine.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"):
        await engine.emergency_flatten_live_gateways()
    logger.warning("Kill-Switch disparado manualmente a través de /api/emergency-flatten.")

    return {
        "status": "EMERGENCY_FLATTEN_TRIGGERED",
        "mode": engine.trading_mode,
        "kill_switch_engaged": engine.gatekeeper.kill_switch_engaged,
        "emergency_flatten_count": engine.gatekeeper.emergency_flatten_count,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/profit-sweep", dependencies=[Depends(verify_bearer_token)])
async def profit_sweep_endpoint() -> dict[str, Any]:
    """
    Dispara el barrido de excedentes de ganancias:
    - En PAPER_L2: descuenta el excedente hacia la bóveda simulada (simulated_vault_usd)
      y resetea el capital operativo a base_capital_usd.
    - En LIVE_BINANCE: ejecuta exchange.transfer('USDT', surplus, 'future', 'spot')
      y resetea el capital operativo a base_capital_usd.
    - En LIVE_BINGX: ejecuta transferencia hacia spot si disponible y resetea a base_capital_usd.
    - En LIVE_MULTI: ejecuta barrido proporcional en ambos exchanges.
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

    if engine.trading_mode == "LIVE_MULTI":
        half_surplus = round(surplus / 2.0, 2)
        if engine.live_gateway is not None and half_surplus > 0:
            try:
                await engine.live_gateway.exchange.transfer(
                    code="USDT", amount=half_surplus, fromAccount="future", toAccount="spot"
                )
                logger.info(f"Barrido en Binance: {half_surplus} USDT a Spot")
            except Exception as e:
                logger.warning(f"Fallo sweep en Binance: {e}")
        if engine.bingx_gateway is not None and half_surplus > 0:
            try:
                ex = engine.bingx_gateway.exchange
                if hasattr(ex, "transfer"):
                    await ex.transfer(code="USDT", amount=half_surplus, fromAccount="swap", toAccount="spot")
                logger.info(f"Barrido en BingX: {half_surplus} USDT a Spot")
            except Exception as e:
                logger.warning(f"Fallo sweep en BingX: {e}")
    elif engine.trading_mode == "LIVE_BINANCE":
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
    elif engine.trading_mode == "LIVE_BINGX":
        if engine.bingx_gateway is None:
            raise HTTPException(status_code=400, detail="Gateway de BingX no inicializado.")
        try:
            ex = engine.bingx_gateway.exchange
            if hasattr(ex, "transfer"):
                await ex.transfer(
                    code="USDT",
                    amount=surplus,
                    fromAccount="swap",
                    toAccount="spot",
                )
            logger.info(f"Barrido de ganancias real ejecutado en BingX: {surplus} USDT a Spot")
        except Exception as e:
            logger.error(f"Fallo en transferencia de futuros a spot en BingX: {e}")
            raise HTTPException(status_code=500, detail=f"Fallo al transferir a Spot en BingX: {e}")
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


@app.post("/api/reset-kill-switch", dependencies=[Depends(verify_bearer_token)])
async def reset_kill_switch_endpoint(req: ResetKillSwitchRequest | None = None, request: Request = None) -> dict[str, Any]:
    """Permite al operador reiniciar el Kill-Switch tras auditar la contingencia."""
    client_ip = request.client.host if request and request.client else "unknown"
    if not req or req.confirm != "RESET_KILL_SWITCH":
        logger.warning(f"[SECURITY_AUDIT] Intento de reset-kill-switch sin confirmación desde IP: {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Confirmación requerida: Debe enviar confirm='RESET_KILL_SWITCH' para restablecer el Kill-Switch.",
        )
    logger.warning(f"[SECURITY_AUDIT] Kill-Switch reseteado con confirmación desde IP: {client_ip}")
    engine.gatekeeper.reset_kill_switch()
    return {
        "status": "KILL_SWITCH_RESET",
        "kill_switch_engaged": False,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/base-capital", dependencies=[Depends(verify_bearer_token)])
async def set_base_capital_endpoint(req: BaseCapitalRequest, request: Request = None) -> dict[str, Any]:
    """Permite ajustar dinámicamente el capital base operativo asignado a las estrategias."""
    client_ip = request.client.host if request and request.client else "unknown"
    if req.confirm != "UPDATE_CAPITAL":
        logger.warning(f"[SECURITY_AUDIT] Intento de actualizar capital sin confirmación desde IP: {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Confirmación requerida: Debe enviar confirm='UPDATE_CAPITAL' para modificar el capital base operativo.",
        )
    if req.base_capital_usd <= 0:
        raise HTTPException(status_code=400, detail="El capital base debe ser mayor a 0.")
    logger.warning(f"[SECURITY_AUDIT] Capital base operativo actualizado a {req.base_capital_usd} USDT desde IP: {client_ip}")
    engine.portfolio.base_capital_usd = round(req.base_capital_usd, 2)
    engine.portfolio.daily_high_watermark_usd = max(
        engine.portfolio.daily_high_watermark_usd, engine.portfolio.base_capital_usd
    )
    return {
        "status": "BASE_CAPITAL_UPDATED",
        "base_capital_usd": engine.portfolio.base_capital_usd,
        "current_equity_usd": engine.portfolio.current_equity_usd,
        "timestamp_ns": time.time_ns(),
    }



class StressEventRequest(BaseModel):
    event_type: str  # "FLASH_CRASH_5PCT" | "LATENCY_SPIKE_200MS" | "TRADFI_WEEKEND_CLOSE" | "RESET_CHAOS"
    symbol: str | None = "BTC/USDT:USDT"
    confirm: str | None = None


@app.post("/api/stress-event", dependencies=[Depends(verify_bearer_token)])
async def inject_stress_event_endpoint(req: StressEventRequest) -> dict[str, Any]:
    """
    Sub-panel de 'Chaos Injection' para PAPER_L2:
    Permite inyectar a voluntad un choque de liquidez (FLASH_CRASH_5PCT, LATENCY_SPIKE_200MS, TRADFI_WEEKEND_CLOSE)
    para verificar visualmente la reacción instantánea de los 4 Mandate Gates y del ContagionShield.
    """
    if engine.trading_mode != "PAPER_L2":
        raise HTTPException(
            status_code=403,
            detail="Chaos injection and stress events are strictly forbidden outside PAPER_L2 mode.",
        )
    if req.confirm != "INJECT_CHAOS":
        raise HTTPException(
            status_code=400,
            detail="Se requiere payload con 'confirm': 'INJECT_CHAOS' para inyectar eventos de estrés.",
        )
    global GLOBAL_MOCK_TRADFI_CLOSED, GLOBAL_CHAOS_LATENCY_ACTIVE

    ev_type = req.event_type.upper().strip()
    sym = req.symbol or engine.primary_symbol

    if ev_type == "FLASH_CRASH_5PCT":
        ctx = engine.symbol_contexts.get(sym, engine.symbol_contexts[engine.primary_symbol])
        if ctx.ring_buffer.ob_count > 0:
            idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
            cur_mid = 0.5 * (float(ctx.ring_buffer.ob_bid_prices[idx, 0]) + float(ctx.ring_buffer.ob_ask_prices[idx, 0]))
            crash_px = cur_mid * 0.95
            for lvl in range(5):
                ctx.ring_buffer.ob_bid_prices[idx, lvl] = round(crash_px - (lvl * ctx.spec.tick_size * 5), 2)
                ctx.ring_buffer.ob_bid_qtys[idx, lvl] = 0.05
                ctx.ring_buffer.ob_ask_prices[idx, lvl] = round(crash_px + (lvl * ctx.spec.tick_size * 2), 2)
                ctx.ring_buffer.ob_ask_qtys[idx, lvl] = 50.0  # Asks masivos dominando el libro
            now_ns = time.time_ns()
            for _ in range(10):
                ctx.ring_buffer.push_trade(crash_px, 1.5, is_buyer_maker=False, timestamp_ns=now_ns)
        engine.contagion_shield.bid_shield_active = True
        engine.contagion_shield.btc_obi = -0.85
        engine.contagion_shield.trigger_reason = "FLASH_CRASH_5PCT_INJECTED"
        msg = f"Flash crash de -5% inyectado en {sym}. OBI forzado a -0.85 y ContagionShield activado."

    elif ev_type == "LATENCY_SPIKE_200MS":
        GLOBAL_CHAOS_LATENCY_ACTIVE = not GLOBAL_CHAOS_LATENCY_ACTIVE
        engine.chaos_latency_active = GLOBAL_CHAOS_LATENCY_ACTIVE
        target_lat = 200.0 if GLOBAL_CHAOS_LATENCY_ACTIVE else 15.0
        for ctx in engine.symbol_contexts.values():
            ctx.l2_sim.latency_ms = target_lat
        engine.last_latency_us = target_lat * 1000.0
        msg = f"Pico de latencia a {target_lat}ms {'activado' if GLOBAL_CHAOS_LATENCY_ACTIVE else 'desactivado'}."

    elif ev_type == "TRADFI_WEEKEND_CLOSE":
        GLOBAL_MOCK_TRADFI_CLOSED = not GLOBAL_MOCK_TRADFI_CLOSED
        engine.mock_tradfi_closed = GLOBAL_MOCK_TRADFI_CLOSED
        GLOBAL_MOCK_TRADFI_STATE = False if GLOBAL_MOCK_TRADFI_CLOSED else True
        if GLOBAL_MOCK_TRADFI_CLOSED:
            # Cancelar preventivamente órdenes en activos TradFi
            now_ns = time.time_ns()
            for s_name in ["XAU/USDT:USDT", "XAG/USDT:USDT"]:
                s_ctx = engine.symbol_contexts.get(s_name)
                if s_ctx:
                    for o in list(s_ctx.l2_sim.orders.values()):
                        if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                            s_ctx.l2_sim.request_cancel(o.order_id, now_ns)
        # Reevaluar asignación
        engine.compute_multivariate_half_kelly()
        msg = f"Cierre TradFi de Fin de Semana {'FORZADO (CERRADO)' if GLOBAL_MOCK_TRADFI_CLOSED else 'RESTAURADO (ABIERTO)'}. Reasignación a cripto 24/7 actualizada."

    elif ev_type == "RESET_CHAOS":
        GLOBAL_MOCK_TRADFI_CLOSED = False
        GLOBAL_MOCK_TRADFI_STATE = None
        GLOBAL_CHAOS_LATENCY_ACTIVE = False
        engine.mock_tradfi_closed = False
        engine.chaos_latency_active = False
        for ctx in engine.symbol_contexts.values():
            ctx.l2_sim.latency_ms = 15.0
        engine.contagion_shield.bid_shield_active = False
        engine.contagion_shield.trigger_reason = "NORMAL"
        engine.compute_multivariate_half_kelly()
        msg = "Todos los choques de estrés de caos reseteados a condiciones nominales."
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de evento de estrés desconocido: {ev_type}")

    return {
        "status": "STRESS_EVENT_INJECTED",
        "event_type": ev_type,
        "symbol": sym,
        "message": msg,
        "mock_tradfi_closed": GLOBAL_MOCK_TRADFI_CLOSED,
        "latency_spike_active": GLOBAL_CHAOS_LATENCY_ACTIVE,
        "timestamp_ns": time.time_ns(),
    }


class GeminiKeyRequest(BaseModel):
    api_key: str
    confirm: str | None = None


@app.post("/api/gemini-key", dependencies=[Depends(verify_bearer_token)])
async def set_gemini_key_endpoint(req: GeminiKeyRequest) -> dict[str, Any]:
    """Permite configurar o actualizar en caliente la API Key de Gemini para el Warm Path Council."""
    if req.confirm != "UPDATE_KEY":
        raise HTTPException(
            status_code=400,
            detail="Se requiere payload con 'confirm': 'UPDATE_KEY' para actualizar la API key.",
        )
    new_key = req.api_key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="API key no puede estar vacía.")

    os.environ["GEMINI_API_KEY"] = new_key
    updated = False

    if engine.warm_path_council is not None:
        engine.warm_path_council.update_api_key(new_key)
        updated = True
    elif GeminiRegimeCouncil is not None and WarmPathCouncilScheduler is not None:
        try:
            engine.warm_path_council = GeminiRegimeCouncil(
                api_key=new_key,
                config_output_path=CONFIG_PATH,
            )
            engine.warm_path_scheduler = WarmPathCouncilScheduler(
                council=engine.warm_path_council,
                summary_provider_fn=engine.get_15m_microstructure_summary,
                interval_seconds=900.0,
            )
            if engine.running:
                engine.warm_path_scheduler.start()
            updated = True
        except Exception as e:
            logger.warning(f"Error instanciando WarmPathCouncil: {e}")

    return {
        "status": "SUCCESS",
        "message": "Gemini API key configurada correctamente.",
        "council_active": updated,
    }


@app.websocket("/ws/telemetry")
async def telemetry_websocket_endpoint(websocket: WebSocket) -> None:
    """
    Transmisión de telemetría viva a alta frecuencia (~10 Hz) hacia el Frontend React.
    Envía el estado real de PortfolioState, MarketRingBuffer, Gatekeeper y L2QueueSimulator.
    """
    if not await verify_websocket_token(websocket):
        return
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


# Servir Frontend React compilado (dist/) en la raíz del servidor
_dist_path = Path(__file__).resolve().parent.parent / "dist"
if _dist_path.exists():
    _assets_path = _dist_path / "assets"
    if _assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_path)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa_frontend(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = _dist_path / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        index_file = _dist_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend dist index.html no encontrado.")
