"""
Módulo cuantitativo para dimensionamiento de posición por riesgo fijo institucional (Fixed-Fractional Risk Sizing).
Calcula el tamaño de lote (qty) en función estricta del riesgo de capital tolerado por trade (0.5%),
la distancia al Stop-Loss, y los límites nocionales de Gate 3 y del exchange.
"""
from __future__ import annotations
import math
from backend.mandate_gates.schemas import truncate_decimal_step


def compute_fixed_risk_position_size(
    equity_usd: float,
    entry_price: float,
    stop_loss_price: float,
    step_size: float = 0.001,
    tick_size: float = 0.1,
    risk_pct_per_trade: float = 0.005,  # 0.5% del capital por trade
    max_pair_margin_ratio: float = 0.20,
    leverage: float = 3.0,
    min_notional_usd: float = 5.0,
) -> float:
    """
    Calcula el tamaño óptimo de la posición garantizando que la pérdida máxima en SL no exceda risk_pct_per_trade.
    Fórmula:
        risk_usd = equity_usd * risk_pct_per_trade
        stop_distance = abs(entry_price - stop_loss_price)
        raw_qty = risk_usd / stop_distance
    Ajustes:
        - Si stop_distance es 0 o negativo, retorna 0.0.
        - Capea el nocional máximo a: max_notional = equity_usd * max_pair_margin_ratio * leverage
        - Trunca hacia abajo a step_size.
        - Si el nocional resultante es inferior a min_notional_usd, evalúa si es seguro aumentarlo o retorna 0.0.
    """
    if equity_usd <= 0.0 or entry_price <= 0.0 or stop_loss_price <= 0.0:
        return 0.0

    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance <= 1e-9:
        return 0.0

    risk_usd = equity_usd * risk_pct_per_trade
    raw_qty = risk_usd / stop_distance

    # Capear por Gate 3: máximo 20% del margen * leverage
    max_notional_allowed = equity_usd * max_pair_margin_ratio * max(1.0, leverage)
    max_qty_allowed = max_notional_allowed / entry_price
    capped_qty = min(raw_qty, max_qty_allowed)

    # Truncar al step_size
    qty = truncate_decimal_step(capped_qty, step_size)

    # Verificar nocional mínimo
    if qty * entry_price < (min_notional_usd - 1e-8):
        min_qty_for_notional = truncate_decimal_step(math.ceil(min_notional_usd / entry_price / step_size) * step_size, step_size)
        risk_at_min_notional = min_qty_for_notional * stop_distance
        if risk_at_min_notional <= (risk_usd * 1.5):  # Tolerancia en micro-cuentas
            qty = min_qty_for_notional
        else:
            return 0.0

    return qty
