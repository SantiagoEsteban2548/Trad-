#!/usr/bin/env python3
"""
Script de Purga y Reconciliación de Base de Datos del Ledger
Elimina trades anómalos o corruptos (cross-pricing, tests de estrés persistidos, PnL inflado)
Recalcula contadores y métricas acumuladas reales y genera un blotter limpio.
"""

import os
import sys
import csv
import sqlite3
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_corrupted_trades")


def clean_database(db_path: str, blotter_output_path: str = "trades_blotter_CLEAN.csv") -> dict:
    if not os.path.exists(db_path):
        logger.error(f"Base de datos no encontrada en: {db_path}")
        return {}

    logger.info(f"Conectando a {db_path}...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Conteo inicial
    cursor.execute("SELECT COUNT(*) FROM trades;")
    total_before = cursor.fetchone()[0]

    # 2. Identificar trades corruptos / anómalos
    purge_condition = """
        ABS(pnl_usd) > 1000.0
        OR return_pct < -50.0
        OR return_pct > 100.0
        OR id LIKE '%-2001%'
        OR (symbol IN ('NEAR/USDT:USDT', 'SUI/USDT:USDT', 'XRP/USDT:USDT', 'LINK/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT') AND (entry_price > 500.0 OR exit_or_mid_price > 500.0))
        OR (symbol = 'BTC/USDT:USDT' AND (entry_price < 1000.0 OR exit_or_mid_price < 1000.0))
    """

    cursor.execute(f"SELECT id, symbol, entry_price, exit_or_mid_price, qty, return_pct, pnl_usd, fee_cost FROM trades WHERE {purge_condition};")
    corrupt_rows = cursor.fetchall()

    logger.info(f"Total trades antes de purga: {total_before}")
    logger.info(f"Trades anómalos / corruptos identificados: {len(corrupt_rows)}")

    for row in corrupt_rows:
        logger.warning(
            f"  [PURGE] ID={row['id']} | Sym={row['symbol']} | Entry=${row['entry_price']} | Exit=${row['exit_or_mid_price']} | "
            f"PnL=${row['pnl_usd']:.2f} | Ret={row['return_pct']}%"
        )

    # 3. Eliminar trades corruptos
    cursor.execute(f"DELETE FROM trades WHERE {purge_condition};")
    deleted_count = cursor.rowcount
    conn.commit()

    # 4. Conteo y métricas recalculadas
    cursor.execute("SELECT COUNT(*) FROM trades;")
    total_after = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl_usd >= 0;")
    wins_count = int(cursor.fetchone()[0] or 0)

    cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl_usd < 0;")
    losses_count = int(cursor.fetchone()[0] or 0)

    cursor.execute("SELECT COALESCE(SUM(pnl_usd), 0.0) FROM trades WHERE pnl_usd >= 0;")
    gross_profit_usd = round(float(cursor.fetchone()[0] or 0.0), 2)

    cursor.execute("SELECT COALESCE(SUM(ABS(pnl_usd)), 0.0) FROM trades WHERE pnl_usd < 0;")
    gross_loss_usd = round(float(cursor.fetchone()[0] or 0.0), 2)

    cursor.execute("SELECT COALESCE(SUM(fee_cost), 0.0) FROM trades;")
    total_fees_usd = round(float(cursor.fetchone()[0] or 0.0), 4)

    # Obtener equity base o actual
    cursor.execute("SELECT current_equity_usd FROM system_metrics WHERE id = 1;")
    curr_eq_row = cursor.fetchone()
    current_equity = float(curr_eq_row[0]) if curr_eq_row else 100000.0

    # Ajustar equity descontando el PnL espurio
    # Si la equidad actual fue inflada por PnL corrupto, recalcularla basada en base capital + net PnL - fees
    net_realized_pnl = round(gross_profit_usd - gross_loss_usd - total_fees_usd, 2)
    # Asumiendo capital base de $100,000 USD estándar institucional
    adjusted_equity = round(100000.0 + net_realized_pnl, 2)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    cursor.execute(
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
        (now_ms, wins_count, losses_count, gross_profit_usd, gross_loss_usd, adjusted_equity),
    )
    conn.commit()

    logger.info("=" * 60)
    logger.info("MÉTRICAS POST-PURGA RECALCULADAS:")
    logger.info(f"  Trades Restantes:    {total_after} (eliminados {deleted_count})")
    logger.info(f"  Wins Count:          {wins_count}")
    logger.info(f"  Losses Count:        {losses_count}")
    win_rate = (wins_count / max(total_after, 1)) * 100.0
    logger.info(f"  Win Rate:            {win_rate:.2f}%")
    logger.info(f"  Gross Profit USD:    ${gross_profit_usd:.2f}")
    logger.info(f"  Gross Loss USD:      ${gross_loss_usd:.2f}")
    logger.info(f"  Total Fees USD:      ${total_fees_usd:.4f}")
    logger.info(f"  Net Realized PnL:    ${net_realized_pnl:.2f}")
    logger.info(f"  Adjusted Equity:     ${adjusted_equity:.2f}")
    logger.info("=" * 60)

    # 5. Generar Blotter CSV Limpio
    cursor.execute(
        """
        SELECT id, timestamp_ms, time_str, symbol, strategy, side,
               entry_price, exit_or_mid_price, qty, return_pct, pnl_usd, fee_cost, exchange
        FROM trades
        ORDER BY timestamp_ms DESC;
        """
    )
    clean_trades = cursor.fetchall()

    with open(blotter_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        for t in clean_trades:
            writer.writerow([
                t["id"],
                t["timestamp_ms"],
                t["time_str"],
                t["symbol"],
                t["strategy"],
                t["side"],
                t["entry_price"],
                t["exit_or_mid_price"],
                t["qty"],
                t["return_pct"],
                t["pnl_usd"],
                t["fee_cost"],
                t["exchange"],
            ])

    logger.info(f"Blotter limpio exportado exitosamente a: {blotter_output_path} ({len(clean_trades)} filas)")
    conn.close()

    return {
        "total_before": total_before,
        "deleted_count": deleted_count,
        "total_after": total_after,
        "wins_count": wins_count,
        "losses_count": losses_count,
        "gross_profit_usd": gross_profit_usd,
        "gross_loss_usd": gross_loss_usd,
        "net_pnl_usd": net_realized_pnl,
        "adjusted_equity": adjusted_equity,
    }


def main():
    target_db = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LEDGER_DB_PATH", "backend/data/trades_ledger.db")
    clean_database(target_db, "trades_blotter_CLEAN.csv")

    # Si existe trades_ledger_binance.db, purgarlo también
    binance_db = "backend/data/trades_ledger_binance.db"
    if os.path.exists(binance_db) and binance_db != target_db:
        clean_database(binance_db, "trades_blotter_binance_CLEAN.csv")


if __name__ == "__main__":
    main()
