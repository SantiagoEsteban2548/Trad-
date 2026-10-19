# backend/engine/specs.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.models.market_specs import (
    SymbolMarketSpec,
    SYMBOL_SPECS,
    LEVERAGE_TIERS_CONFIG,
)
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.liq_cascade import LiquidationCascadeHunter
from backend.validation.l2_simulator_and_dsr import L2QueueSimulator
from backend.gateways.dry_run_gateway import DryRunWsGateway

# Variables globales para control y mocking de TradFi en tests y caos
GLOBAL_MOCK_TRADFI_STATE: bool | None = None
GLOBAL_MOCK_TRADFI_CLOSED: bool = False


def is_tradfi_market_open(dt_utc: datetime | None = None) -> bool:
    """
    Guardia de Sesión TradFi para activos macro (Oro XAU, Plata XAG):
    Devuelve False durante:
    1. Cierre de fin de semana: Viernes 20:30 UTC a Domingo 22:05 UTC.
    2. Ventana de mantenimiento diario: Lunes a Jueves de 20:45 UTC a 22:05 UTC.
    Devuelve True en cualquier otro horario en que los mercados macro estén abiertos.
    """
    import sys
    ts = sys.modules.get("backend.telemetry_server")
    if ts is not None and hasattr(ts, "is_tradfi_market_open"):
        ts_func = getattr(ts, "is_tradfi_market_open")
        if ts_func is not is_tradfi_market_open:
            try:
                return ts_func(dt_utc) if dt_utc is not None else ts_func()
            except TypeError:
                return ts_func()

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
    directional: Any = None


@dataclass
class ContagionShieldState:
    bid_shield_active: bool = False
    ask_shield_active: bool = False
    trigger_reason: str = "NORMAL"
    bearish_count: int = 0
    bullish_count: int = 0
    btc_obi: float = 0.0
    xau_safe_haven_active: bool = False
