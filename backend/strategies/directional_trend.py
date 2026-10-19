"""
Estrategia Direccional Institucional (Trend Momentum & Breakout):
- Opera exclusivamente a favor de la tendencia detectada (cruces de medias exponenciales y quiebres de rango).
- Relación Riesgo-Beneficio estricta R:R >= 1:3 (ej. SL = 15 bps, TP = 45 bps).
- Dimensionamiento de posición por riesgo fijo (máximo 0.5% del capital por trade).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from backend.strategies.risk_sizing import compute_fixed_risk_position_size
from backend.mandate_gates.schemas import truncate_decimal_step


@dataclass
class DirectionalSignal:
    symbol: str
    side: str  # "BUY" | "SELL"
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    qty: float
    risk_reward_ratio: float
    risk_usd: float


class DirectionalTrendStrategy:
    """
    Estrategia de momentum direccional de alta frecuencia institucional.
    Calcula tendencia mediante EMA rápida (8 períodos) y EMA lenta (21 períodos).
    Ejecuta con R:R 1:3 y riesgo fijado al 0.5% del equity.
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float = 0.1,
        step_size: float = 0.001,
        fast_period: int = 8,
        slow_period: int = 21,
        risk_pct_per_trade: float = 0.005,  # 0.5%
        sl_bps: float = 15.0,               # 15 bps de riesgo
        tp_bps: float = 45.0,               # 45 bps de beneficio (R:R 1:3)
        min_notional_usd: float = 5.0,
    ) -> None:
        self.symbol = symbol
        self.tick_size = tick_size
        self.step_size = step_size
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.risk_pct_per_trade = risk_pct_per_trade
        self.sl_bps = sl_bps
        self.tp_bps = tp_bps
        self.min_notional_usd = min_notional_usd

    def evaluate_signal(
        self,
        prices: list[float],
        current_equity: float,
        best_bid: float,
        best_ask: float,
        obi: float = 0.0,
    ) -> DirectionalSignal | None:
        """
        Evalúa si existe una señal direccional válida.
        Requiere al menos `slow_period` precios históricos.
        """
        if len(prices) < self.slow_period:
            return None

        arr = np.asarray(prices[-self.slow_period * 2:], dtype=np.float64)
        # EMA rápida y lenta
        weights_fast = np.exp(np.linspace(-1.0, 0.0, self.fast_period))
        weights_fast /= weights_fast.sum()
        fast_ema = float(np.convolve(arr, weights_fast, mode="valid")[-1])

        weights_slow = np.exp(np.linspace(-1.0, 0.0, self.slow_period))
        weights_slow /= weights_slow.sum()
        slow_ema = float(np.convolve(arr, weights_slow, mode="valid")[-1])

        cur_mid = 0.5 * (best_bid + best_ask)
        side: str | None = None

        # Señal LONG: Fast EMA > Slow EMA y OBI positivo o neutral
        if fast_ema > (slow_ema * 1.0001) and obi >= -0.20:
            side = "BUY"
            entry_px = best_ask
            sl_px = truncate_decimal_step(entry_px * (1.0 - (self.sl_bps / 10_000.0)), self.tick_size)
            tp_px = truncate_decimal_step(entry_px * (1.0 + (self.tp_bps / 10_000.0)), self.tick_size)
        # Señal SHORT: Fast EMA < Slow EMA y OBI negativo o neutral
        elif fast_ema < (slow_ema * 0.9999) and obi <= 0.20:
            side = "SELL"
            entry_px = best_bid
            sl_px = truncate_decimal_step(entry_px * (1.0 + (self.sl_bps / 10_000.0)), self.tick_size)
            tp_px = truncate_decimal_step(entry_px * (1.0 - (self.tp_bps / 10_000.0)), self.tick_size)

        if side is None:
            return None

        qty = compute_fixed_risk_position_size(
            equity_usd=current_equity,
            entry_price=entry_px,
            stop_loss_price=sl_px,
            step_size=self.step_size,
            tick_size=self.tick_size,
            risk_pct_per_trade=self.risk_pct_per_trade,
            min_notional_usd=self.min_notional_usd,
        )

        if qty <= 0.0:
            return None

        risk_usd = abs(entry_px - sl_px) * qty
        rr_ratio = abs(tp_px - entry_px) / max(abs(entry_px - sl_px), 1e-8)

        return DirectionalSignal(
            symbol=self.symbol,
            side=side,
            entry_price=entry_px,
            take_profit_price=tp_px,
            stop_loss_price=sl_px,
            qty=qty,
            risk_reward_ratio=round(rr_ratio, 2),
            risk_usd=round(risk_usd, 4),
        )
