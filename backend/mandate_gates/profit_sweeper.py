# backend/mandate_gates/profit_sweeper.py
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Protocol
from backend.mandate_gates.schemas import PortfolioState


DAILY_TARGET_RETURN_PCT: float = 0.022  # +2.2% rendimiento neto diario


class AsyncExchangeTransferClient(Protocol):
    """Protocolo estructural para clientes CCXT Pro / asíncronos con soporte de transferencias."""

    async def transfer(
        self,
        code: str,
        amount: float,
        fromAccount: str,
        toAccount: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def should_trigger_profit_sweep(
    state: PortfolioState, now_utc: datetime | None = None
) -> bool:
    """
    Determina si corresponde disparar el barrido de ganancias:
    1. Es las 00:00 UTC, O
    2. El rendimiento neto diario alcanza o supera el +2.2%.
    """
    current_time = now_utc or datetime.now(timezone.utc)
    is_midnight_utc = current_time.hour == 0 and current_time.minute == 0
    net_return_pct = (state.current_equity_usd - state.base_capital_usd) / state.base_capital_usd
    target_hit = net_return_pct >= (DAILY_TARGET_RETURN_PCT - 1e-12)
    return is_midnight_utc or target_hit


async def daily_profit_sweep(
    exchange_client: AsyncExchangeTransferClient,
    state: PortfolioState,
) -> float:
    """
    Tarea asíncrona de protección de ganancias (Warm Path).
    Si current_equity_usd > base_capital_usd y no existen posiciones con margen
    comprometido, transfiere el excedente (surplus) desde Futuros USDT-M ('future')
    hacia la billetera 'spot', reseteando el capital expuesto a base_capital_usd.

    Retorna el monto en USDT barrido hacia Spot (0.0 si no se realizó transferencia).
    """
    has_committed_margin = any(
        abs(notional) > 0.0 for notional in state.active_positions.values()
    )
    if has_committed_margin:
        return 0.0

    if state.current_equity_usd <= state.base_capital_usd:
        return 0.0

    d_equity = Decimal(str(state.current_equity_usd))
    d_base = Decimal(str(state.base_capital_usd))
    d_surplus = (d_equity - d_base).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    surplus = float(d_surplus)

    if surplus <= 0.0:
        return 0.0

    await exchange_client.transfer(
        code="USDT",
        amount=surplus,
        fromAccount="future",
        toAccount="spot",
    )

    state.current_equity_usd = state.base_capital_usd
    state.daily_high_watermark_usd = state.base_capital_usd

    return surplus


RATCHET_THRESHOLD_PCT: float = 0.020  # +2.0% para disparar el trinquete 85/15
VAULT_SPLIT_RATIO: float = 0.15       # 15% a Bóveda Spot
REINVEST_SPLIT_RATIO: float = 0.85    # 85% reinvertido en base_capital_usd
MAX_ACTIVE_CAPITAL_CAP_USD: float = 25_000.0  # Tope Físico de Liquidez L2


def step_up_ratchet(
    state: PortfolioState,
    threshold_pct: float = RATCHET_THRESHOLD_PCT,
    vault_ratio: float = VAULT_SPLIT_RATIO,
    reinvest_ratio: float = REINVEST_SPLIT_RATIO,
    max_capital_cap_usd: float = MAX_ACTIVE_CAPITAL_CAP_USD,
) -> float:
    """
    Trinquete de Crecimiento Compuesto Institucional 85/15 (Step-Up Ratchet):
    Cada vez que current_equity_usd supere en >= +2.0% a base_capital_usd,
    transfiere el 15% del excedente a la Bóveda Spot (simulated_vault_usd) y reinvierte
    el 85% restante sumándolo directamente a base_capital_usd.

    Tope Físico de Liquidez L2:
    Cuando base_capital_usd alcanza max_capital_cap_usd ($25,000 USD), el interés
    compuesto en futuros se detiene y el 100% de las ganancias posteriores se transfieren
    a la Bóveda Spot (simulated_vault_usd) para evitar desbordar la profundidad L2 de las altcoins.

    Retorna el monto en USDT transferido a la Bóveda Spot (0.0 si no se activó).
    """
    if state.base_capital_usd <= 0.0:
        return 0.0

    surplus = state.current_equity_usd - state.base_capital_usd
    ratio = surplus / state.base_capital_usd

    if ratio < (threshold_pct - 1e-9):
        return 0.0

    # Si ya se alcanzó o superó el tope físico de $25,000 USD: 100% a bóveda
    if state.base_capital_usd >= max_capital_cap_usd:
        vault_transfer = round(surplus, 4)
        reinvest_amount = 0.0
    else:
        nominal_reinvest = surplus * reinvest_ratio
        if state.base_capital_usd + nominal_reinvest > max_capital_cap_usd:
            reinvest_amount = round(max_capital_cap_usd - state.base_capital_usd, 4)
            vault_transfer = round(surplus - reinvest_amount, 4)
        else:
            vault_transfer = round(surplus * vault_ratio, 4)
            reinvest_amount = round(nominal_reinvest, 4)

    # Actualizar capital base compuesto acotado al tope físico
    state.base_capital_usd = round(min(state.base_capital_usd + reinvest_amount, max_capital_cap_usd), 4)
    # Ajustar equity descontando la porción transferida a la bóveda spot
    state.current_equity_usd = round(state.current_equity_usd - vault_transfer, 4)
    if state.current_equity_usd > state.daily_high_watermark_usd:
        state.daily_high_watermark_usd = state.current_equity_usd

    return vault_transfer

