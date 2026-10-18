# backend/ingestion/ws_multiplexer.py
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
import time
from typing import Any

import ccxt.pro as ccxtpro

from backend.ingestion.ring_buffer import MarketRingBuffer

# Activación de uvloop si está disponible
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = logging.getLogger("ingestion.ws_multiplexer")


class WebSocketMultiplexer:
    """
    Multiplexor de WebSocket de alta concurrencia para futuros perpetuos.
    Conecta múltiples símbolos usando ccxt.pro, maneja reconexión exponencial,
    mantiene activo el listenKey y actualiza directamente el MarketRingBuffer.
    """

    def __init__(
        self,
        symbols: list[str],
        exchange_id: str = "binance",
        exchange_config: dict[str, Any] | None = None,
        buffer_capacity: int = 10_000,
        secondary_exchange_id: str | None = None,
        secondary_exchange_config: dict[str, Any] | None = None,
        secondary_symbols: list[str] | None = None,
    ) -> None:
        self.symbols: list[str] = symbols
        self.exchange_id: str = exchange_id

        default_config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",  # Futuros perpetuos USDT-M
            },
        }
        if exchange_config:
            default_config.update(exchange_config)

        exchange_class = getattr(ccxtpro, exchange_id)
        self.exchange: ccxtpro.Exchange = exchange_class(default_config)

        # Buffers circulares en memoria preasignados por símbolo
        self.buffers: dict[str, MarketRingBuffer] = {
            symbol: MarketRingBuffer(capacity=buffer_capacity) for symbol in symbols
        }

        # Soporte para segundo exchange (e.g. BingX para arbitraje Lead-Lag)
        self.secondary_exchange_id = secondary_exchange_id
        self.secondary_symbols = secondary_symbols or []
        self.venue_top_of_book: dict[str, tuple[float, float]] = {}
        if secondary_exchange_id:
            try:
                sec_class = getattr(ccxtpro, secondary_exchange_id)
                sec_cfg: dict[str, Any] = {
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                }
                if secondary_exchange_config:
                    sec_cfg.update(secondary_exchange_config)
                self.secondary_exchange: ccxtpro.Exchange | None = sec_class(sec_cfg)
            except Exception as e:
                logger.warning(f"No se pudo inicializar secondary exchange {secondary_exchange_id}: {e}")
                self.secondary_exchange = None
        else:
            self.secondary_exchange = None

        self.running: bool = False
        self._tasks: list[asyncio.Task] = []
        self._backoff_base: float = 0.5
        self._backoff_max: float = 30.0

    async def start(self) -> None:
        """Inicia todas las suscripciones concurrentes y tareas de fondo."""
        self.running = True
        logger.info(f"Iniciando Multiplexer WS para {self.symbols} en {self.exchange_id}")

        for symbol in self.symbols:
            # 1. Bucle de Order Book L2 (Top 5)
            self._tasks.append(
                asyncio.create_task(
                    self._run_with_backoff(
                        f"OB_{symbol}", lambda s=symbol: self._order_book_stream(s)
                    )
                )
            )
            # 2. Bucle de Flujo de Trades en tiempo real
            self._tasks.append(
                asyncio.create_task(
                    self._run_with_backoff(
                        f"TRADES_{symbol}", lambda s=symbol: self._trades_stream(s)
                    )
                )
            )
            # 3. Bucle de Liquidaciones Forzosas
            self._tasks.append(
                asyncio.create_task(
                    self._run_with_backoff(
                        f"LIQS_{symbol}", lambda s=symbol: self._liquidations_stream(s)
                    )
                )
            )

        # 4. Tarea periódica de renovación de listenKey (cada 30 minutos)
        self._tasks.append(asyncio.create_task(self._listen_key_keepalive_loop()))

        # 5. Suscripción secundaria de libros L2 (BingX) para conocer best_bid/best_ask y arbitraje Lead-Lag
        if self.secondary_exchange and self.secondary_symbols:
            for s_sym in self.secondary_symbols:
                self._tasks.append(
                    asyncio.create_task(
                        self._run_with_backoff(
                            f"SEC_OB_{s_sym}", lambda s=s_sym: self._secondary_order_book_stream(s)
                        )
                    )
                )

    async def stop(self) -> None:
        """Detiene limpiamente todas las tareas y cierra la conexión WebSocket."""
        self.running = False
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.exchange.close()
        if self.secondary_exchange:
            try:
                await self.secondary_exchange.close()
            except Exception:
                pass
        logger.info("Multiplexer WS detenido correctamente.")

    async def _secondary_order_book_stream(self, symbol: str) -> None:
        watch_symbol = self._resolve_watch_symbol(symbol)
        while self.running and self.secondary_exchange:
            try:
                ob = await self.secondary_exchange.watch_order_book(watch_symbol, limit=5)
                raw_bids = ob.get("bids", [])
                raw_asks = ob.get("asks", [])
                if raw_bids and raw_asks:
                    best_bid = float(raw_bids[0][0])
                    best_ask = float(raw_asks[0][0])
                    if best_bid > 0.0 and best_ask > 0.0:
                        self.venue_top_of_book[symbol] = (best_bid, best_ask)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"secondary orderbook stream {symbol}: {e}")
                await asyncio.sleep(2.0)


    # -------------------------------------------------------------------------
    # Bucles de Transmisión por Instrumento
    # -------------------------------------------------------------------------

    def _safe_price(self, symbol: str, val: Any) -> float:
        f_val = float(val or 0.0)
        if f_val <= 0.0:
            return 0.0
        try:
            return float(self.exchange.price_to_precision(symbol, f_val))
        except Exception:
            return f_val

    def _safe_amount(self, symbol: str, val: Any) -> float:
        f_val = float(val or 0.0)
        if f_val <= 0.0:
            return 0.0
        try:
            return float(self.exchange.amount_to_precision(symbol, f_val))
        except Exception:
            return f_val

    def _resolve_watch_symbol(self, symbol: str) -> str:
        if symbol == "PEPE/USDT:USDT":
            return "1000PEPE/USDT:USDT"
        return symbol

    async def _order_book_stream(self, symbol: str) -> None:
        ring = self.buffers[symbol]
        watch_symbol = self._resolve_watch_symbol(symbol)
        while self.running:
            try:
                ob = await asyncio.wait_for(
                    self.exchange.watch_order_book(watch_symbol, limit=5),
                    timeout=15.0,
                )
                ts_ns = time.time_ns()

                raw_bids = ob.get("bids", [])
                raw_asks = ob.get("asks", [])
                if len(raw_bids) < 5 or len(raw_asks) < 5:
                    continue

                bid_p = [self._safe_price(watch_symbol, b[0]) for b in raw_bids[:5]]
                bid_q = [self._safe_amount(watch_symbol, b[1]) for b in raw_bids[:5]]
                ask_p = [self._safe_price(watch_symbol, a[0]) for a in raw_asks[:5]]
                ask_q = [self._safe_amount(watch_symbol, a[1]) for a in raw_asks[:5]]

                if any(p <= 0.0 for p in bid_p) or any(p <= 0.0 for p in ask_p):
                    continue

                ring.push_order_book(bid_p, bid_q, ask_p, ask_q, ts_ns)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout (15s) en watch_order_book {symbol}. Reconectando stream...")
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Excepción en _order_book_stream {symbol}: {e}")
                await asyncio.sleep(1.0)

    async def _trades_stream(self, symbol: str) -> None:
        ring = self.buffers[symbol]
        watch_symbol = self._resolve_watch_symbol(symbol)
        while self.running:
            try:
                trades = await asyncio.wait_for(
                    self.exchange.watch_trades(watch_symbol),
                    timeout=30.0,
                )
                ts_ns = time.time_ns()
                for trade in trades:
                    p = self._safe_price(watch_symbol, trade.get("price"))
                    q = self._safe_amount(watch_symbol, trade.get("amount"))
                    if p <= 0.0 or q <= 0.0:
                        continue
                    side_maker = trade.get("side") == "sell" or trade.get("info", {}).get("m", False)
                    ring.push_trade(p, q, side_maker, ts_ns)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout (30s) en watch_trades {symbol}. Reconectando stream...")
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Excepción en _trades_stream {symbol}: {e}")
                await asyncio.sleep(1.0)

    async def _liquidations_stream(self, symbol: str) -> None:
        ring = self.buffers[symbol]
        watch_symbol = self._resolve_watch_symbol(symbol)
        while self.running:
            try:
                if hasattr(self.exchange, "watch_liquidations"):
                    liqs = await self.exchange.watch_liquidations(watch_symbol)
                    ts_ns = time.time_ns()
                    for liq in liqs:
                        side = liq.get("side", "BUY")
                        p = float(liq.get("price", 0.0))
                        q = float(liq.get("amount", 0.0))
                        ring.push_liquidation(side, p, q, ts_ns)
                else:
                    await asyncio.sleep(1.0)
            except (NotImplementedError, ccxtpro.NotSupported):
                await asyncio.sleep(60.0)
                break

    async def _listen_key_keepalive_loop(self) -> None:
        """Mantiene activo el canal privado de usuario renovando el listenKey cada 30 minutos."""
        while self.running:
            try:
                await asyncio.sleep(1800)  # 30 minutos
                if hasattr(self.exchange, "keep_alive_listen_key"):
                    await self.exchange.keep_alive_listen_key()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error renovando listenKey: {e}")

    # -------------------------------------------------------------------------
    # Tolerancia a Fallos y Reconexión Exponencial
    # -------------------------------------------------------------------------

    async def _run_with_backoff(self, name: str, coro_fn: Callable[[], Any]) -> None:
        backoff = self._backoff_base
        while self.running:
            try:
                await coro_fn()
                backoff = self._backoff_base
            except asyncio.CancelledError:
                break
            except (ccxtpro.BadSymbol,) as err:
                logger.info(f"Símbolo no soportado en {self.exchange_id} ({name}): {err}. Pausando stream en este exchange.")
                await asyncio.sleep(300.0)
                break
            except (ccxtpro.NetworkError, ccxtpro.ExchangeError) as err:
                err_str = str(err)
                if "does not have market symbol" in err_str:
                    logger.info(f"Símbolo omitido en {self.exchange_id} ({name}): {err_str}")
                    await asyncio.sleep(300.0)
                    break
                logger.warning(
                    f"Desconexión en {name}: {err}. Reconectando en {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self._backoff_max)
            except Exception as err:
                logger.error(f"Fallo crítico en {name}: {err}. Pausa {backoff:.2f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self._backoff_max)
