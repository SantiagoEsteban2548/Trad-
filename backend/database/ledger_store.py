# backend/database/ledger_store.py
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger("LedgerStore")


class LedgerSqliteStore:
    """
    Persistencia ligera en disco mediante SQLite en modo WAL (Write-Ahead Logging).
    Garantiza almacenamiento transaccional para:
    1. Operaciones históricas (trades ejecutados, PnL, fees).
    2. Sub-libros virtuales (VirtualPositionLot) entre reinicios del bot.
    3. Métricas acumuladas (wins, losses, gross profit/loss, equity).
    4. Curva de markouts post-trade (1s, 5s, 15s, 30s) para análisis de slippage.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.getenv("LEDGER_DB_PATH", "backend/data/trades_ledger.db")
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False, timeout=10.0)
            self._memory_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception as e:
            logger.debug(f"SQLite PRAGMA setup warning: {e}")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=5000;")

                # 1. Tabla de trades ejecutados
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id TEXT PRIMARY KEY,
                        timestamp_ms INTEGER NOT NULL,
                        time_str TEXT,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_or_mid_price REAL NOT NULL,
                        qty REAL NOT NULL,
                        return_pct REAL NOT NULL,
                        pnl_usd REAL NOT NULL,
                        fee_cost REAL DEFAULT 0.0,
                        exchange TEXT DEFAULT 'BINANCE'
                    );
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp_ms DESC);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_sym ON trades(symbol);")

                # 2. Tabla de lotes virtuales (VirtualPositionLot)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS virtual_lots (
                        lot_key TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        strategy_id TEXT NOT NULL,
                        sub_lot_id TEXT NOT NULL,
                        inventory_q REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        break_even_price REAL NOT NULL,
                        entry_ts_ns INTEGER NOT NULL,
                        realized_pnl_usd REAL NOT NULL,
                        take_profit_price REAL,
                        stop_loss_price REAL,
                        updated_at_ms INTEGER NOT NULL
                    );
                    """
                )

                # 3. Métricas del sistema acumuladas
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        timestamp_ms INTEGER NOT NULL,
                        wins_count INTEGER NOT NULL,
                        losses_count INTEGER NOT NULL,
                        gross_profit_usd REAL NOT NULL,
                        gross_loss_usd REAL NOT NULL,
                        current_equity_usd REAL NOT NULL
                    );
                    """
                )

                # 4. Markouts post-trade
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS markouts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        fill_price REAL NOT NULL,
                        m1s_bps REAL,
                        m5s_bps REAL,
                        m15s_bps REAL,
                        m30s_bps REAL,
                        timestamp_ms INTEGER NOT NULL
                    );
                    """
                )
                conn.commit()
                logger.info(f"LedgerSqliteStore inicializado exitosamente en {self.db_path} (WAL Mode)")

    def save_trade(self, trade: dict[str, Any], exchange: str = "BINANCE") -> None:
        """Guarda o actualiza un trade completado de forma atómica."""
        tid = str(trade.get("id") or f"TRD_{int(time.time()*1000)}")
        ts_ms = int(trade.get("timestamp_ms") or int(time.time() * 1000))
        time_str = str(trade.get("time_str") or "")
        symbol = str(trade.get("symbol") or "")
        strategy = str(trade.get("strategy") or "AVELLANEDA_OBI")
        side = str(trade.get("side") or "BUY").upper()
        entry_price = float(trade.get("entry_price") or 0.0)
        exit_or_mid_price = float(trade.get("exit_or_mid_price") or 0.0)
        qty = float(trade.get("qty") or 0.0)
        return_pct = float(trade.get("return_pct") or 0.0)
        pnl_usd = float(trade.get("pnl_usd") or 0.0)
        fee_cost = float(trade.get("fee_cost") or 0.0)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO trades (
                        id, timestamp_ms, time_str, symbol, strategy, side,
                        entry_price, exit_or_mid_price, qty, return_pct, pnl_usd, fee_cost, exchange
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        timestamp_ms=excluded.timestamp_ms,
                        time_str=excluded.time_str,
                        exit_or_mid_price=excluded.exit_or_mid_price,
                        return_pct=excluded.return_pct,
                        pnl_usd=excluded.pnl_usd,
                        fee_cost=excluded.fee_cost;
                    """,
                    (tid, ts_ms, time_str, symbol, strategy, side,
                     entry_price, exit_or_mid_price, qty, return_pct, pnl_usd, fee_cost, exchange)
                )
                conn.commit()

    def load_recent_trades(self, limit: int | None = 1000) -> list[dict[str, Any]]:
        """Carga los trades más recientes ordenados cronológicamente descendente."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if limit is not None and limit > 0:
                    cursor.execute(
                        """
                        SELECT id, timestamp_ms, time_str, symbol, strategy, side,
                               entry_price, exit_or_mid_price, qty, return_pct, pnl_usd, fee_cost, exchange
                        FROM trades
                        ORDER BY timestamp_ms DESC
                        LIMIT ?;
                        """,
                        (limit,)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, timestamp_ms, time_str, symbol, strategy, side,
                               entry_price, exit_or_mid_price, qty, return_pct, pnl_usd, fee_cost, exchange
                        FROM trades
                        ORDER BY timestamp_ms DESC;
                        """
                    )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "timestamp_ms": row["timestamp_ms"],
                        "time_str": row["time_str"],
                        "symbol": row["symbol"],
                        "strategy": row["strategy"],
                        "side": row["side"],
                        "entry_price": row["entry_price"],
                        "exit_or_mid_price": row["exit_or_mid_price"],
                        "qty": row["qty"],
                        "return_pct": row["return_pct"],
                        "pnl_usd": row["pnl_usd"],
                        "fee_cost": row["fee_cost"],
                        "exchange": row["exchange"],
                    })
                return results

    def count_trades(self) -> int:
        """Retorna el número total de transacciones guardadas en SQLite."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trades;")
                row = cursor.fetchone()
                return int(row[0]) if row else 0

    def save_lot(self, lot_key: str, symbol: str, strategy_id: str, sub_lot_id: str,
                 inventory_q: float, entry_price: float, break_even_price: float,
                 entry_ts_ns: int, realized_pnl_usd: float,
                 take_profit_price: float | None = None, stop_loss_price: float | None = None) -> None:
        """Persiste el estado actual de un lote virtual."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO virtual_lots (
                        lot_key, symbol, strategy_id, sub_lot_id, inventory_q,
                        entry_price, break_even_price, entry_ts_ns, realized_pnl_usd,
                        take_profit_price, stop_loss_price, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(lot_key) DO UPDATE SET
                        inventory_q=excluded.inventory_q,
                        entry_price=excluded.entry_price,
                        break_even_price=excluded.break_even_price,
                        entry_ts_ns=excluded.entry_ts_ns,
                        realized_pnl_usd=excluded.realized_pnl_usd,
                        take_profit_price=excluded.take_profit_price,
                        stop_loss_price=excluded.stop_loss_price,
                        updated_at_ms=excluded.updated_at_ms;
                    """,
                    (lot_key, symbol, strategy_id, sub_lot_id, inventory_q,
                     entry_price, break_even_price, entry_ts_ns, realized_pnl_usd,
                     take_profit_price, stop_loss_price, now_ms)
                )
                conn.commit()

    def load_lots(self) -> dict[str, dict[str, Any]]:
        """Recupera todos los lotes virtuales abiertos o guardados."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT lot_key, symbol, strategy_id, sub_lot_id, inventory_q,
                           entry_price, break_even_price, entry_ts_ns, realized_pnl_usd,
                           take_profit_price, stop_loss_price, updated_at_ms
                    FROM virtual_lots;
                    """
                )
                rows = cursor.fetchall()
                results = {}
                for r in rows:
                    results[r["lot_key"]] = {
                        "lot_key": r["lot_key"],
                        "symbol": r["symbol"],
                        "strategy_id": r["strategy_id"],
                        "sub_lot_id": r["sub_lot_id"],
                        "inventory_q": r["inventory_q"],
                        "entry_price": r["entry_price"],
                        "break_even_price": r["break_even_price"],
                        "entry_ts_ns": r["entry_ts_ns"],
                        "realized_pnl_usd": r["realized_pnl_usd"],
                        "take_profit_price": r["take_profit_price"],
                        "stop_loss_price": r["stop_loss_price"],
                        "updated_at_ms": r["updated_at_ms"],
                    }
                return results

    def save_metrics(self, wins_count: int, losses_count: int,
                     gross_profit_usd: float, gross_loss_usd: float,
                     current_equity_usd: float) -> None:
        """Actualiza el snapshot de métricas de ejecución del sistema."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO system_metrics (
                        id, timestamp_ms, wins_count, losses_count,
                        gross_profit_usd, gross_loss_usd, current_equity_usd
                    ) VALUES (1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        timestamp_ms=excluded.timestamp_ms,
                        wins_count=excluded.wins_count,
                        losses_count=excluded.losses_count,
                        gross_profit_usd=excluded.gross_profit_usd,
                        gross_loss_usd=excluded.gross_loss_usd,
                        current_equity_usd=excluded.current_equity_usd;
                    """,
                    (now_ms, wins_count, losses_count, gross_profit_usd, gross_loss_usd, current_equity_usd)
                )
                conn.commit()

    def load_metrics(self) -> dict[str, Any] | None:
        """Carga el último snapshot de métricas persistidas."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT timestamp_ms, wins_count, losses_count,
                           gross_profit_usd, gross_loss_usd, current_equity_usd
                    FROM system_metrics
                    WHERE id = 1;
                    """
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return {
                    "timestamp_ms": row["timestamp_ms"],
                    "wins_count": row["wins_count"],
                    "losses_count": row["losses_count"],
                    "gross_profit_usd": row["gross_profit_usd"],
                    "gross_loss_usd": row["gross_loss_usd"],
                    "current_equity_usd": row["current_equity_usd"],
                }

    def save_markout(self, markout: dict[str, Any]) -> None:
        """Registra un evento de markout concluido."""
        now_ms = int(time.time() * 1000)
        trade_id = str(markout.get("trade_id") or "")
        symbol = str(markout.get("symbol") or "")
        side = str(markout.get("side") or "BUY").upper()
        fill_price = float(markout.get("fill_price") or 0.0)
        m1s = float(markout["m1s_bps"]) if "m1s_bps" in markout and markout["m1s_bps"] is not None else None
        m5s = float(markout["m5s_bps"]) if "m5s_bps" in markout and markout["m5s_bps"] is not None else None
        m15s = float(markout["m15s_bps"]) if "m15s_bps" in markout and markout["m15s_bps"] is not None else None
        m30s = float(markout["m30s_bps"]) if "m30s_bps" in markout and markout["m30s_bps"] is not None else None

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO markouts (
                        trade_id, symbol, side, fill_price,
                        m1s_bps, m5s_bps, m15s_bps, m30s_bps, timestamp_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (trade_id, symbol, side, fill_price, m1s, m5s, m15s, m30s, now_ms)
                )
                conn.commit()

    def load_recent_markouts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Carga los últimos markouts calculados."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT trade_id, symbol, side, fill_price,
                           m1s_bps, m5s_bps, m15s_bps, m30s_bps, timestamp_ms
                    FROM markouts
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (limit,)
                )
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "trade_id": r["trade_id"],
                        "symbol": r["symbol"],
                        "side": r["side"],
                        "fill_price": r["fill_price"],
                        "m1s_bps": r["m1s_bps"],
                        "m5s_bps": r["m5s_bps"],
                        "m15s_bps": r["m15s_bps"],
                        "m30s_bps": r["m30s_bps"],
                        "timestamp_ms": r["timestamp_ms"],
                    })
                return results

    async def save_trade_async(self, trade: dict[str, Any], exchange: str = "BINANCE") -> None:
        """Versión asíncrona no bloqueante de save_trade para el Event Loop."""
        await asyncio.to_thread(self.save_trade, trade, exchange)

    async def save_metrics_async(
        self,
        wins_count: int,
        losses_count: int,
        gross_profit_usd: float,
        gross_loss_usd: float,
        current_equity_usd: float,
    ) -> None:
        """Versión asíncrona no bloqueante de save_metrics para el Event Loop."""
        await asyncio.to_thread(
            self.save_metrics,
            wins_count,
            losses_count,
            gross_profit_usd,
            gross_loss_usd,
            current_equity_usd,
        )

    async def save_lot_async(
        self,
        lot_key: str,
        symbol: str,
        strategy_id: str,
        sub_lot_id: str,
        inventory_q: float,
        entry_price: float,
        break_even_price: float,
        entry_ts_ns: int,
        realized_pnl_usd: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> None:
        """Versión asíncrona no bloqueante de save_lot para el Event Loop."""
        await asyncio.to_thread(
            self.save_lot,
            lot_key,
            symbol,
            strategy_id,
            sub_lot_id,
            inventory_q,
            entry_price,
            break_even_price,
            entry_ts_ns,
            realized_pnl_usd,
            take_profit_price,
            stop_loss_price,
        )

    async def save_markout_async(self, markout: dict[str, Any]) -> None:
        """Versión asíncrona no bloqueante de save_markout para el Event Loop."""
        await asyncio.to_thread(self.save_markout, markout)

    async def load_recent_trades_async(self, limit: int | None = 1000) -> list[dict[str, Any]]:
        """Versión asíncrona no bloqueante de load_recent_trades para el Event Loop."""
        return await asyncio.to_thread(self.load_recent_trades, limit)

    async def load_metrics_async(self) -> dict[str, Any] | None:
        """Versión asíncrona no bloqueante de load_metrics para el Event Loop."""
        return await asyncio.to_thread(self.load_metrics)

