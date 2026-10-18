# tests/test_price_sanity_and_ledger_isolation.py
import os
import sqlite3
import pytest

from backend.database.ledger_store import LedgerSqliteStore
from backend.strategies.strategy_ledger import VirtualStrategyLedger
from backend.telemetry_server import DryRunExecutionEngine
from scripts.clean_corrupted_trades import clean_database


def test_price_sanity_guard_rejects_outliers_and_preserves_metrics():
    """
    Verifica que el Price Sanity Guard en record_trade_fill descarte inmediatamente
    cualquier trade con precio fuera de la banda ±15% respecto al Mid-Price del símbolo,
    sin alterar PnL, wins/losses, ni el blotter de recent_trades.
    """
    engine = DryRunExecutionEngine()
    initial_wins = engine.wins_count
    initial_losses = engine.losses_count
    initial_gross_pnl = engine.gross_profit_usd
    initial_trades_len = len(engine.recent_trades)

    # BTC preset mid está en ~81,300.50. Banda ±15% es [69,105, 93,495].
    # 1. Enviar trade de NEAR a precio de BTC ($107,124) -> DEBE SER RECHAZADO
    rejected_near = engine.record_trade_fill(
        symbol="NEAR/USDT:USDT",
        strategy="AVELLANEDA_OBI",
        side="BUY",
        entry_price=107124.0,
        exit_or_mid_price=107124.0,
        qty=1.0,
        return_pct=0.05,
        pnl_usd=107119.27,
    )
    assert rejected_near == {}
    assert engine.wins_count == initial_wins
    assert engine.losses_count == initial_losses
    assert engine.gross_profit_usd == initial_gross_pnl
    assert len(engine.recent_trades) == initial_trades_len

    # 2. Enviar trade de BTC a $500 (desplome espurio fuera del 15%) -> DEBE SER RECHAZADO
    rejected_btc = engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_OBI",
        side="BUY",
        entry_price=500.0,
        exit_or_mid_price=500.0,
        qty=0.01,
        return_pct=-99.0,
        pnl_usd=-500.0,
    )
    assert rejected_btc == {}
    assert engine.losses_count == initial_losses
    assert len(engine.recent_trades) == initial_trades_len

    # 3. Enviar trade válido dentro de la banda (ej. BTC a 81,300.0) -> DEBE SER ACEPTADO
    accepted_btc = engine.record_trade_fill(
        symbol="BTC/USDT:USDT",
        strategy="AVELLANEDA_OBI",
        side="BUY",
        entry_price=81300.0,
        exit_or_mid_price=81305.0,
        qty=0.01,
        return_pct=0.05,
        pnl_usd=0.05,
        fee_cost=0.016,
    )
    assert accepted_btc != {}
    assert accepted_btc["symbol"] == "BTC/USDT:USDT"
    assert accepted_btc["pnl_usd"] == 0.05
    assert accepted_btc["fee_cost"] == 0.016
    assert engine.wins_count == initial_wins + 1
    assert len(engine.recent_trades) == initial_trades_len + 1


def test_strategy_ledger_symbol_mismatch_isolation():
    """
    Verifica que en VirtualStrategyLedger.record_fill, si un clientOrderId
    estaba asociado previamente a un lote de otro símbolo, la validación estricta
    de símbolo evite contaminar el lote del símbolo ajeno.
    """
    ledger = VirtualStrategyLedger()
    cid_btc = ledger.generate_client_order_id("AVELLANEDA_OBI", "L1")
    ledger.register_order(cid_btc, "BTC/USDT:USDT", "AVELLANEDA_OBI", "L1", exchange_order_id="ORD_100")

    # Intentar ejecutar con el mismo CID pero símbolo NEAR/USDT:USDT
    fill_res = ledger.record_fill(
        order_id_or_cid=cid_btc,
        symbol="NEAR/USDT:USDT",
        side="BUY",
        fill_price=5.0,
        fill_qty=10.0,
    )

    # El inventario de BTC no debe haber cambiado a 10 contratos
    assert ledger.get_net_inventory("BTC/USDT:USDT") == 0.0
    # El inventario de NEAR debe ser 10 contratos
    assert ledger.get_net_inventory("NEAR/USDT:USDT") == 10.0
    assert fill_res["symbol"] == "NEAR/USDT:USDT"


def test_fee_mapping_and_db_store_persistence():
    """
    Verifica que fee_cost se guarde atómicamente en LedgerSqliteStore y
    se recupere adecuadamente en load_recent_trades.
    """
    store = LedgerSqliteStore(":memory:")
    trade_data = {
        "id": "TRD_TEST_FEE_101",
        "timestamp_ms": 1700000000000,
        "time_str": "12:00:00.000",
        "symbol": "BTC/USDT:USDT",
        "strategy": "AVELLANEDA_OBI",
        "side": "BUY",
        "entry_price": 81300.0,
        "exit_or_mid_price": 81320.0,
        "qty": 0.05,
        "return_pct": 0.025,
        "pnl_usd": 1.0,
        "fee_cost": 0.0813,
        "exchange": "BINANCE",
    }
    store.save_trade(trade_data)

    trades = store.load_recent_trades(limit=10)
    assert len(trades) == 1
    loaded = trades[0]
    assert loaded["id"] == "TRD_TEST_FEE_101"
    assert loaded["fee_cost"] == 0.0813
    assert loaded["pnl_usd"] == 1.0


def test_clean_corrupted_trades_script_logic(tmp_path):
    """
    Verifica que clean_database elimine trades anómalos (PnL > 1000, return < -50%, id -2001)
    y recalcule wins_count, losses_count, gross_profit y gross_loss fielmente.
    """
    test_db = str(tmp_path / "test_ledger.db")
    test_csv = str(tmp_path / "test_blotter.csv")

    store = LedgerSqliteStore(test_db)
    # 1. Trade legítimo ganador
    store.save_trade({
        "id": "VALID_1",
        "timestamp_ms": 1000,
        "symbol": "BTC/USDT:USDT",
        "entry_price": 81300.0,
        "exit_or_mid_price": 81350.0,
        "qty": 0.1,
        "return_pct": 0.06,
        "pnl_usd": 5.0,
        "fee_cost": 0.16,
    })
    # 2. Trade legítimo perdedor
    store.save_trade({
        "id": "VALID_2",
        "timestamp_ms": 2000,
        "symbol": "ETH/USDT:USDT",
        "entry_price": 2500.0,
        "exit_or_mid_price": 2490.0,
        "qty": 0.5,
        "return_pct": -0.40,
        "pnl_usd": -5.0,
        "fee_cost": 0.25,
    })
    # 3. Trade anómalo de test de estrés (-2001)
    store.save_trade({
        "id": "TRD-1792348057132-2001",
        "timestamp_ms": 3000,
        "symbol": "XRP/USDT:USDT",
        "entry_price": 99.92,
        "exit_or_mid_price": 1.27,
        "qty": 5.5,
        "return_pct": -98.7,
        "pnl_usd": -552.45,
        "fee_cost": 0.0,
    })
    # 4. Trade con precio cruzado inflado
    store.save_trade({
        "id": "LIVE-1000000245878771",
        "timestamp_ms": 4000,
        "symbol": "NEAR/USDT:USDT",
        "entry_price": 107124.07,
        "exit_or_mid_price": 107124.07,
        "qty": 1.0,
        "return_pct": 0.05,
        "pnl_usd": 107119.27,
        "fee_cost": 0.0,
    })

    res = clean_database(test_db, test_csv)
    assert res["total_before"] == 4
    assert res["deleted_count"] == 2
    assert res["total_after"] == 2
    assert res["wins_count"] == 1
    assert res["losses_count"] == 1
    assert res["gross_profit_usd"] == 5.0
    assert res["gross_loss_usd"] == 5.0

    # Comprobar que el archivo CSV fue creado
    assert os.path.exists(test_csv)
