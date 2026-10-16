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

    async def stop(self) -> None:
        """Detiene limpiamente todas las tareas y cierra la conexión WebSocket."""
        self.running = False
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.exchange.close()
        logger.info("Multiplexer WS detenido correctamente.")

    # -------------------------------------------------------------------------
    # Bucles de Transmisión por Instrumento
    # -------------------------------------------------------------------------

    async def _order_book_stream(self, symbol: str) -> None:
        ring = self.buffers[symbol]
        while self.running:
            ob = await self.exchange.watch_order_book(symbol, limit=5)
            ts_ns = time.time_ns()

            raw_bids = ob.get("bids", [])
            raw_asks = ob.get("asks", [])
            if len(raw_bids) < 5 or len(raw_asks) < 5:
                continue

            bid_p = [float(self.exchange.price_to_precision(symbol, b[0])) for b in raw_bids[:5]]
            bid_q = [float(self.exchange.amount_to_precision(symbol, b[1])) for b in raw_bids[:5]]
            ask_p = [float(self.exchange.price_to_precision(symbol, a[0])) for a in raw_asks[:5]]
            ask_q = [float(self.exchange.amount_to_precision(symbol, a[1])) for a in raw_asks[:5]]

            ring.push_order_book(bid_p, bid_q, ask_p, ask_q, ts_ns)

    async def _trades_stream(self, symbol: str) -> None:
        ring = self.buffers[symbol]
        while self.running:
            trades = await self.exchange.watch_trades(symbol)
            ts_ns = time.time_ns()
            for trade in trades:
                p = float(self.exchange.price_to_precision(symbol, trade["price"]))
                q = float(self.exchange.amount_to_precision(symbol, trade["amount"]))
                side_maker = trade.get("side") == "sell" or trade.get("info", {}).get("m", False)
                ring.push_trade(p, q, side_maker, ts_ns)

    async def _liquidations_stream(self, symbol: str) -> None:
        ring = self.buffers[symbol]
        while self.running:
            try:
                if hasattr(self.exchange, "watch_liquidations"):
                    liqs = await self.exchange.watch_liquidations(symbol)
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
            except (ccxtpro.NetworkError, ccxtpro.ExchangeError) as err:
                logger.warning(
                    f"Desconexión en {name}: {err}. Reconectando en {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self._backoff_max)
            except Exception as err:
                logger.error(f"Fallo crítico en {name}: {err}. Pausa {backoff:.2f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self._backoff_max)
