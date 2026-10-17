# backend/strategies/strategy_ledger.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VirtualPositionLot:
    """
    Sub-libro virtual para un lote específico de una estrategia y nivel de escalera.
    Permite contabilidad aislada de inventario, precio de entrada y PnL en One-Way Mode.
    """
    symbol: str
    strategy_id: str  # e.g. "AVELLANEDA_OBI", "LIQ_CASCADE", "FUNDING_ARB"
    sub_lot_id: str   # e.g. "L1", "L2", "L3", "MAIN"
    inventory_q: float = 0.0  # signed: >0 para Long, <0 para Short
    entry_price: float = 0.0
    entry_ts_ns: int = 0
    realized_pnl_usd: float = 0.0
    active_orders: set[str] = field(default_factory=set)
    take_profit_price: float | None = None
    stop_loss_price: float | None = None

    @property
    def key(self) -> str:
        return f"{self.symbol}::{self.strategy_id}::{self.sub_lot_id}"

    @property
    def is_open(self) -> bool:
        return abs(self.inventory_q) > 1e-8

    @property
    def is_long(self) -> bool:
        return self.inventory_q > 1e-8

    @property
    def is_short(self) -> bool:
        return self.inventory_q < -1e-8


class VirtualStrategyLedger:
    """
    Sub-Libros Virtuales por Estrategia (VirtualStrategyLedger) para evitar choques
    en Binance / BingX One-Way Mode.
    Etiqueta cada orden enviada con un prefijo en clientOrderId (AVEL_L1_, AVEL_L2_, AVEL_L3_, LIQ_, FARB_)
    y lleva la contabilidad del inventario q y del precio de entrada (entry_price) por separado
    para cada motor y para cada nivel de la escalera, de modo que cada estrategia gestione
    y cierre exclusivamente su propio lote sin sabotear a las demás.
    """

    PREFIX_MAP: dict[str, tuple[str, str]] = {
        "AV_L1": ("AVELLANEDA_OBI", "L1"),
        "AV_L2": ("AVELLANEDA_OBI", "L2"),
        "AV_L3": ("AVELLANEDA_OBI", "L3"),
        "AV_MAIN": ("AVELLANEDA_OBI", "MAIN"),
        "AVEL_L1": ("AVELLANEDA_OBI", "L1"),
        "AVEL_L2": ("AVELLANEDA_OBI", "L2"),
        "AVEL_L3": ("AVELLANEDA_OBI", "L3"),
        "AVEL": ("AVELLANEDA_OBI", "MAIN"),
        "LIQ": ("LIQ_CASCADE", "MAIN"),
        "FARB": ("FUNDING_ARB", "MAIN"),
    }

    def __init__(self) -> None:
        # Clave: f"{symbol}::{strategy_id}::{sub_lot_id}"
        self.lots: dict[str, VirtualPositionLot] = {}
        # Mapeo client_order_id -> lot.key
        self.order_to_lot: dict[str, str] = {}
        # Mapeo exchange_order_id -> client_order_id
        self.exchange_to_client_id: dict[str, str] = {}

    def get_or_create_lot(self, symbol: str, strategy_id: str, sub_lot_id: str = "MAIN") -> VirtualPositionLot:
        key = f"{symbol}::{strategy_id}::{sub_lot_id}"
        if key not in self.lots:
            self.lots[key] = VirtualPositionLot(
                symbol=symbol,
                strategy_id=strategy_id,
                sub_lot_id=sub_lot_id,
            )
        return self.lots[key]

    def generate_client_order_id(self, strategy_id: str, sub_lot_id: str = "MAIN") -> str:
        """
        Genera un clientOrderId estructurado respetando la longitud estricta (máximo 32 caracteres
        y regex ^[a-zA-Z0-9_-]{1,32}$):
        e.g. AV_L1_74829103_a1b2c3 (21 caracteres)
        """
        strat_upper = strategy_id.upper()
        if "AVELLANEDA" in strat_upper or "AVEL" in strat_upper or "AV_" in strat_upper:
            if sub_lot_id in ("L1", "L2", "L3"):
                prefix = f"AV_{sub_lot_id}"
            else:
                prefix = "AV_MAIN"
        elif "LIQ" in strat_upper or "CASCADE" in strat_upper:
            prefix = "LIQ_MAIN"
        elif "FUND" in strat_upper or "ARB" in strat_upper:
            prefix = "FARB_MAIN"
        else:
            prefix = f"{strat_upper[:3]}_{sub_lot_id[:2]}"

        ts_part = int(time.time() * 1000) % 100_000_000  # 8 dígitos
        rand_part = uuid.uuid4().hex[:6]
        cid = f"{prefix}_{ts_part}_{rand_part}"
        return cid[:32]

    def parse_client_order_id(self, client_order_id: str) -> tuple[str, str]:
        """Extrae (strategy_id, sub_lot_id) a partir del prefijo en client_order_id."""
        cid = client_order_id.upper()
        for prefix, (strat, sub_lot) in self.PREFIX_MAP.items():
            if cid.startswith(prefix):
                return strat, sub_lot
        return "AVELLANEDA_OBI", "MAIN"

    def register_order(
        self,
        client_order_id: str,
        symbol: str,
        strategy_id: str,
        sub_lot_id: str = "MAIN",
        exchange_order_id: str | None = None,
    ) -> None:
        """Registra una orden pendiente asociándola con su sub-libro virtual."""
        lot = self.get_or_create_lot(symbol, strategy_id, sub_lot_id)
        lot.active_orders.add(client_order_id)
        self.order_to_lot[client_order_id] = lot.key
        if exchange_order_id:
            oid_str = str(exchange_order_id)
            self.exchange_to_client_id[oid_str] = client_order_id
            lot.active_orders.add(oid_str)
            self.order_to_lot[oid_str] = lot.key

    def record_fill(
        self,
        order_id_or_cid: str,
        symbol: str,
        side: str,
        fill_price: float,
        fill_qty: float,
        ts_ns: int | None = None,
    ) -> dict[str, Any]:
        """
        Registra la ejecución (fill) en el sub-libro virtual correspondiente.
        Actualiza el inventario q y el precio de entrada ponderado del lote.
        Si la orden es de cierre (reduce posición), calcula el PnL realizado.
        """
        cid = self.exchange_to_client_id.get(str(order_id_or_cid), str(order_id_or_cid))
        lot_key = self.order_to_lot.get(cid)

        if not lot_key:
            strat, sub = self.parse_client_order_id(cid)
            lot = self.get_or_create_lot(symbol, strat, sub)
        else:
            lot = self.lots[lot_key]

        side_upper = side.upper()
        now_ns = ts_ns or time.time_ns()
        pnl_realized = 0.0

        signed_delta = fill_qty if side_upper == "BUY" else -fill_qty
        current_q = lot.inventory_q

        # Determinar si la operación incrementa posición o la reduce/cierra
        is_increasing = (current_q >= 0 and signed_delta > 0) or (current_q <= 0 and signed_delta < 0)

        if is_increasing:
            new_q = current_q + signed_delta
            if abs(new_q) > 1e-8:
                lot.entry_price = (
                    (lot.entry_price * abs(current_q) + fill_price * fill_qty) / abs(new_q)
                )
            else:
                lot.entry_price = fill_price
            lot.inventory_q = new_q
            if current_q == 0:
                lot.entry_ts_ns = now_ns
        else:
            # Operación de cierre (e.g. SELL para Long o BUY para Short)
            close_qty = min(abs(current_q), fill_qty)
            if current_q > 0:  # Cerrando Long
                pnl_realized = (fill_price - lot.entry_price) * close_qty
            else:  # Cerrando Short
                pnl_realized = (lot.entry_price - fill_price) * close_qty

            lot.realized_pnl_usd += pnl_realized
            new_q = current_q + signed_delta
            lot.inventory_q = new_q

            if abs(new_q) <= 1e-8:
                lot.inventory_q = 0.0
                lot.entry_price = 0.0
                lot.entry_ts_ns = 0
                lot.take_profit_price = None
                lot.stop_loss_price = None

        lot.active_orders.discard(cid)
        lot.active_orders.discard(str(order_id_or_cid))

        return {
            "lot_key": lot.key,
            "strategy_id": lot.strategy_id,
            "sub_lot_id": lot.sub_lot_id,
            "inventory_q": lot.inventory_q,
            "entry_price": lot.entry_price,
            "realized_pnl_usd": pnl_realized,
            "is_closed": abs(lot.inventory_q) <= 1e-8,
        }

    def get_net_inventory(self, symbol: str) -> float:
        """Devuelve la suma total de inventario de todos los lotes del par (debe coincidir con la posición neta en Binance)."""
        return sum(lot.inventory_q for lot in self.lots.values() if lot.symbol == symbol)

    def get_lots_by_symbol(self, symbol: str) -> list[VirtualPositionLot]:
        """Retorna todos los lotes de un símbolo."""
        return [lot for lot in self.lots.values() if lot.symbol == symbol]
