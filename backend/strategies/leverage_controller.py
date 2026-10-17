# backend/strategies/leverage_controller.py
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.ingestion.ring_buffer import MarketRingBuffer
    from backend.mandate_gates.schemas import PortfolioState


class AutonomousLeverageController:
    """
    Motor de Apalancamiento Autónomo Dinámico (1x a 10x en modo ISOLATED).
    Elimina el apalancamiento estático fijo de 3x, eligiendo automáticamente
    el apalancamiento entero óptimo combinando:
      1. Volatilidad realizada sigma_5m:
         - Rango estrecho (<0.15%): apalancamiento alto (6x - 10x) para liberar margen en calma.
         - Rango moderado (0.15% - 0.60%): apalancamiento equilibrado (3x - 5x).
         - Volatilidad explosiva (>0.60%): compresión automática a 1x - 2x para proteger capital.
      2. Fracción de Medio-Kelly (mu / sigma^2) y penalización por toxicidad:
         - Escala por (1.0 - 0.5 * abs(obi)), reduciendo la exposición cuando el flujo L2 es tóxico.
    """

    MIN_LEVERAGE: int = 1
    MAX_LEVERAGE: int = 10

    @classmethod
    def compute_optimal_leverage(
        cls,
        symbol: str,
        ring_buffer: MarketRingBuffer,
        portfolio: PortfolioState | None = None,
    ) -> int:
        if ring_buffer.ob_count == 0:
            return 3

        sigma_5m = ring_buffer.compute_realized_volatility(window_seconds=300.0)
        obi = ring_buffer.compute_obi(levels=5)

        # 1. Base por régimen de volatilidad realizada 5m
        if sigma_5m < 0.0015:  # < 0.15% (Rango muy estrecho / consolidación)
            base_lev = 9.0
        elif sigma_5m < 0.0035:  # 0.15% a 0.35% (Rango moderado bajo)
            base_lev = 6.0
        elif sigma_5m <= 0.0060:  # 0.35% a 0.60% (Rango normal a alto)
            base_lev = 4.0
        else:  # > 0.60% (Volatilidad explosiva / shock)
            base_lev = 1.5

        # 2. Factor Half-Kelly si el estado de portafolio está disponible
        kelly_factor = 1.0
        if portfolio is not None:
            var_safe = max(portfolio.rolling_variance_sigma2, 1e-8)
            # Edge ratio mu / sigma2
            raw_kelly = 0.5 * (portfolio.rolling_edge_mu / var_safe)
            # Normalizar alrededor de 1.0 (en rango seguro [0.6, 1.4])
            kelly_factor = max(0.6, min(raw_kelly / 2.5, 1.4))

        # 3. Penalización por toxicidad de flujo L2 (OBI)
        # abs(obi) = 0.0 -> penalty = 1.0 (sin penalización)
        # abs(obi) = 1.0 -> penalty = 0.5 (reduce a la mitad el apalancamiento)
        toxicity_penalty = 1.0 - (0.5 * min(abs(obi), 1.0))

        # Combinación multiplicativa
        calculated_lev = base_lev * kelly_factor * toxicity_penalty
        int_lev = int(round(calculated_lev))

        return max(cls.MIN_LEVERAGE, min(int_lev, cls.MAX_LEVERAGE))
