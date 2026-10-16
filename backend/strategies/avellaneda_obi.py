# backend/strategies/avellaneda_obi.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step
from backend.strategies.regime_config import RegimeConfig, WsOrderGateway


@dataclass(slots=True)
class AvellanedaQuoteResult:
    reservation_price: float
    optimal_spread: float
    bid_order: OrderRequest | None
    ask_order: OrderRequest | None
    bid_cancelled: bool
    ask_cancelled: bool
    obi: float
    sigma: float
    bid_ladder: list[OrderRequest] | None = None
    ask_ladder: list[OrderRequest] | None = None


class AvellanedaStoikovOBIEngine:
    """
    Motor A: Market Making Asimétrico de Avellaneda-Stoikov con OBI Shield
    y Cotización Multinivel en Escalera (3-Level Ladder).
    Opera estrictamente en modo ISOLATED + LIMIT + GTX (Post-Only).
    """

    STRATEGY_ID: str = "AVELLANEDA_OBI"
    MIN_SIGMA_FLOOR: float = 1e-4

    __slots__ = (
        "symbol",
        "tick_size",
        "step_size",
        "quote_qty",
        "config",
        "gatekeeper",
        "active_bid_order_id",
        "active_ask_order_id",
    )

    def __init__(
        self,
        *,
        symbol: str,
        tick_size: float,
        step_size: float,
        quote_qty: float,
        config: RegimeConfig,
        gatekeeper: MandateGatekeeper,
    ) -> None:
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.step_size = step_size
        self.quote_qty = truncate_decimal_step(quote_qty, step_size)
        self.config = config
        self.gatekeeper = gatekeeper
        self.active_bid_order_id: str | None = None
        self.active_ask_order_id: str | None = None

    def update_regime_config(self, new_config: RegimeConfig) -> None:
        self.config = new_config

    def compute_reservation_and_spread(
        self,
        mid_price: float,
        normalized_inventory_q: float,
        sigma: float,
        inventory_holding_time_s: float = 0.0,
    ) -> tuple[float, float]:
        gamma = self.config.gamma
        # Urgencia por antigüedad de inventario (holding_time_decay):
        # Si q != 0 permanece abierto > 15 minutos (900s), incrementa gamma en 25%
        # para acercar el precio de reserva y salida al microprecio.
        if abs(normalized_inventory_q) > 1e-4 and inventory_holding_time_s >= 900.0:
            gamma = gamma * 1.25

        kappa = self.config.kappa
        tau = self.config.time_horizon_tau

        sigma_price = max(sigma, self.MIN_SIGMA_FLOOR) * mid_price
        variance_term = gamma * (sigma_price * sigma_price) * tau

        reservation_price = mid_price - (normalized_inventory_q * variance_term)
        liquidity_term = (2.0 / gamma) * math.log(1.0 + (gamma / kappa))
        optimal_spread = variance_term + liquidity_term

        return reservation_price, optimal_spread

    async def on_l2_update(
        self,
        ring_buffer: MarketRingBuffer,
        portfolio_state: PortfolioState,
        ws_gateway: WsOrderGateway,
        normalized_inventory_q: float = 0.0,
        inventory_holding_time_s: float = 0.0,
    ) -> AvellanedaQuoteResult:
        if ring_buffer.ob_count == 0:
            return AvellanedaQuoteResult(0.0, 0.0, None, None, False, False, 0.0, 0.0, [], [])

        latest_idx = (ring_buffer.ob_head - 1) % ring_buffer.capacity
        best_bid = float(ring_buffer.ob_bid_prices[latest_idx, 0])
        best_ask = float(ring_buffer.ob_ask_prices[latest_idx, 0])
        mid_price = 0.5 * (best_bid + best_ask)

        obi = ring_buffer.compute_obi(levels=5)
        raw_sigma = ring_buffer.compute_realized_volatility(window_seconds=300.0)
        sigma = max(raw_sigma, self.MIN_SIGMA_FLOOR)

        reservation_price, optimal_spread = self.compute_reservation_and_spread(
            mid_price=mid_price,
            normalized_inventory_q=normalized_inventory_q,
            sigma=sigma,
            inventory_holding_time_s=inventory_holding_time_s,
        )

        half_spread = 0.5 * optimal_spread

        bid_cancelled = False
        ask_cancelled = False
        suppress_ask = False
        vol_shift = 0.0

        if obi < -self.config.obi_shield_threshold:
            if self.active_bid_order_id is not None:
                t0_ns = time.perf_counter_ns()
                await ws_gateway.cancel_order_ws(self.active_bid_order_id, self.symbol)
                elapsed_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
                if elapsed_ms <= 20.0:
                    bid_cancelled = True
                self.active_bid_order_id = None

            vol_shift = self.config.obi_volatility_shift_mult * sigma * mid_price

        if obi > self.config.obi_shield_threshold:
            if self.active_ask_order_id is not None:
                await ws_gateway.cancel_order_ws(self.active_ask_order_id, self.symbol)
                ask_cancelled = True
                self.active_ask_order_id = None
            suppress_ask = True

        now_ns = time.time_ns()

        # 3-Level Avellaneda Ladder (Cotización Multinivel en Escalera):
        # Nivel 1: 45% del tamaño a 1.0 * delta*
        # Nivel 2: 35% del tamaño a 2.0 * delta*
        # Nivel 3: 20% del tamaño a 3.5 * delta*
        ladder_tiers = [
            (0.45, 1.0),
            (0.35, 2.0),
            (0.20, 3.5),
        ]

        bid_ladder: list[OrderRequest] = []
        for weight, mult in ladder_tiers:
            level_half_spread = half_spread * mult
            raw_level_bid = reservation_price - level_half_spread - vol_shift
            safe_level_bid = min(raw_level_bid, best_ask - self.tick_size)
            level_qty = truncate_decimal_step(max(self.quote_qty * weight, self.step_size), self.step_size)

            cand_bid = OrderRequest.build_maker_order(
                symbol=self.symbol,
                side="BUY",
                raw_price=safe_level_bid,
                raw_qty=level_qty,
                tick_size=self.tick_size,
                step_size=self.step_size,
                strategy_id=self.STRATEGY_ID,
                timestamp_ns=now_ns,
            )
            bid_ok, _ = self.gatekeeper.validate_order(cand_bid, portfolio_state)
            if bid_ok:
                bid_ladder.append(cand_bid)

        ask_ladder: list[OrderRequest] = []
        if not suppress_ask:
            for weight, mult in ladder_tiers:
                level_half_spread = half_spread * mult
                raw_level_ask = reservation_price + level_half_spread
                safe_level_ask = max(raw_level_ask, best_bid + self.tick_size)
                level_qty = truncate_decimal_step(max(self.quote_qty * weight, self.step_size), self.step_size)

                cand_ask = OrderRequest.build_maker_order(
                    symbol=self.symbol,
                    side="SELL",
                    raw_price=safe_level_ask,
                    raw_qty=level_qty,
                    tick_size=self.tick_size,
                    step_size=self.step_size,
                    strategy_id=self.STRATEGY_ID,
                    timestamp_ns=now_ns,
                )
                ask_ok, _ = self.gatekeeper.validate_order(cand_ask, portfolio_state)
                if ask_ok:
                    ask_ladder.append(cand_ask)

        bid_order = bid_ladder[0] if bid_ladder else None
        ask_order = ask_ladder[0] if ask_ladder else None

        return AvellanedaQuoteResult(
            reservation_price=reservation_price,
            optimal_spread=optimal_spread,
            bid_order=bid_order,
            ask_order=ask_order,
            bid_cancelled=bid_cancelled,
            ask_cancelled=ask_cancelled,
            obi=obi,
            sigma=sigma,
            bid_ladder=bid_ladder,
            ask_ladder=ask_ladder,
        )

