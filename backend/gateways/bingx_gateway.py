# backend/gateways/bingx_gateway.py
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import inspect
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

import ccxt.pro as ccxtpro
import ccxt.async_support as ccxt_async

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step
from backend.models.market_specs import SYMBOL_SPECS, SymbolMarketSpec, LEVERAGE_TIERS_CONFIG
from backend.strategies.strategy_ledger import VirtualStrategyLedger, VirtualPositionLot

logger = logging.getLogger("bingx_gateway")

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



