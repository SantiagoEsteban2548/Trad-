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
    MIN_PROFITABLE_SPREAD_BPS: float = 10.0
    MIN_FEE_BPS: float = 4.0  # Comisión Maker ida y vuelta estimada (2x 0.02% = 4 bps)
    MIN_PROFIT_BPS_OVER_BREAKEVEN: float = 4.0  # Exigencia mínima de mejora sobre break-even

    __slots__ = (
        "symbol",
        "tick_size",
        "step_size",
        "quote_qty",
        "config",
        "gatekeeper",
        "active_bid_order_id",
        "active_ask_order_id",
        "min_profitable_spread_bps",
        "min_notional",
        "momentum_shift_bps",
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
        min_profitable_spread_bps: float | None = None,
        min_notional: float = 5.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.step_size = step_size
        self.quote_qty = truncate_decimal_step(quote_qty, step_size)
        self.config = config
        self.gatekeeper = gatekeeper
        self.active_bid_order_id: str | None = None
        self.active_ask_order_id: str | None = None
        # Piso de spread rentable: mínimo 10.0 bps en cripto y 6.5 bps en commodities TradFi (Oro / Plata)
        if min_profitable_spread_bps is not None:
            self.min_profitable_spread_bps = float(min_profitable_spread_bps)
        elif any(c in self.symbol for c in ("XAU", "XAG", "GOLD", "SILVER")):
            self.min_profitable_spread_bps = 6.5
        else:
            self.min_profitable_spread_bps = self.MIN_PROFITABLE_SPREAD_BPS
        self.min_notional = float(min_notional)
        self.momentum_shift_bps = 5.0

    def update_regime_config(self, new_config: RegimeConfig) -> None:
        self.config = new_config

    def compute_reservation_and_spread(
        self,
        mid_price: float,
        normalized_inventory_q: float,
        sigma: float,
        inventory_holding_time_s: float = 0.0,
        vpin: float = 0.0,
    ) -> tuple[float, float]:
        gamma = self.config.gamma
        # Urgencia por antigüedad de inventario (holding_time_decay):
        # Si q != 0 permanece abierto > 15 minutos (900s), incrementa gamma en 25%
        # para acercar el precio de reserva y salida al microprecio.
        if abs(normalized_inventory_q) > 1e-4 and inventory_holding_time_s >= 900.0:
            gamma = gamma * 1.25

        # Detección de toxicidad por volumen (VPIN):
        # Si VPIN > 0.75, incrementa dinámicamente gamma en un 40% para extremar la cautela
        if vpin > 0.75:
            gamma = gamma * 1.40

        kappa = self.config.kappa
        tau = self.config.time_horizon_tau

        sigma_price = max(sigma, self.MIN_SIGMA_FLOOR) * mid_price
        variance_term = gamma * (sigma_price * sigma_price) * tau

        reservation_price = mid_price - (normalized_inventory_q * variance_term)
        liquidity_term = (2.0 / gamma) * math.log(1.0 + (gamma / kappa))
        raw_spread = variance_term + liquidity_term

        # Piso mínimo de spread rentable (mínimo 10.0 bps en cripto y 6.5 bps en TradFi)
        # Garantiza 2x comisiones Maker + margen neto positivo
        min_spread_floor = mid_price * (self.min_profitable_spread_bps / 10_000.0)
        optimal_spread = max(raw_spread, min_spread_floor)

        return reservation_price, optimal_spread

    async def on_l2_update(
        self,
        ring_buffer: MarketRingBuffer,
        portfolio_state: PortfolioState,
        ws_gateway: WsOrderGateway,
        normalized_inventory_q: float = 0.0,
        inventory_holding_time_s: float = 0.0,
        vpin: float | None = None,
        entry_price: float | None = None,
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

        calc_vpin = vpin if vpin is not None else (ring_buffer.compute_vpin() if hasattr(ring_buffer, "compute_vpin") else 0.0)

        reservation_price, optimal_spread = self.compute_reservation_and_spread(
            mid_price=mid_price,
            normalized_inventory_q=normalized_inventory_q,
            sigma=sigma,
            inventory_holding_time_s=inventory_holding_time_s,
            vpin=calc_vpin,
        )

        half_spread = 0.5 * optimal_spread

        bid_cancelled = False
        ask_cancelled = False
        suppress_ask = False
        vol_shift = 0.0

        # Estrechar el umbral del OBI Shield en un 25% si VPIN > 0.75 para proteger el libro
        effective_obi_threshold = self.config.obi_shield_threshold * (0.75 if calc_vpin > 0.75 else 1.0)

        if obi < -effective_obi_threshold:
            if self.active_bid_order_id is not None:
                t0_ns = time.perf_counter_ns()
                await ws_gateway.cancel_order_ws(self.active_bid_order_id, self.symbol)
                elapsed_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
                if elapsed_ms <= 20.0:
                    bid_cancelled = True
                self.active_bid_order_id = None

            vol_shift = self.config.obi_volatility_shift_mult * sigma * mid_price

        if obi > effective_obi_threshold:
            if self.active_ask_order_id is not None:
                await ws_gateway.cancel_order_ws(self.active_ask_order_id, self.symbol)
                ask_cancelled = True
                self.active_ask_order_id = None
            suppress_ask = True

        now_ns = time.time_ns()
        now_ms_tag = int(time.time() * 1000) % 100_000_000
        tfi = ring_buffer.compute_tfi(window_ns=500_000_000) if hasattr(ring_buffer, "compute_tfi") else 0.0

        # Validación de MIN_NOTIONAL en la Escalera de Niveles:
        # En Binance Futures, BTC exige un nocional mínimo de $50-100 USDT, ETH/LINK $20 USDT y otros $5 USDT.
        # Si el tamaño asignado a un escalón menor (20% en L3) cae por debajo del min_notional,
        # el motor fusiona automáticamente los escalones (3 niveles -> 2 niveles -> 1 nivel L1) para
        # no recibir rechazos -4164 de la API.
        total_quote_notional = self.quote_qty * mid_price
        if total_quote_notional * 0.20 >= self.min_notional:
            ladder_tiers = [
                (0.45, 1.0, "L1"),
                (0.35, 2.0, "L2"),
                (0.20, 3.5, "L3"),
            ]
        elif total_quote_notional * 0.35 >= self.min_notional:
            ladder_tiers = [
                (0.65, 1.0, "L1"),
                (0.35, 2.0, "L2"),
            ]
        else:
            ladder_tiers = [
                (1.00, 1.0, "L1"),
            ]

        # Break-Even Real y Momentum Trailing Exit:
        # Si q > 0 (LONG), la orden ASK reduce posición. Exige precio >= entry * (1 + 8 bps) [o 6 bps en TradFi].
        # Si el flujo (OBI > +0.45 o TFI > +0.50) continúa a favor, corre el TP hacia arriba (+5 bps).
        # Si q < 0 (SHORT), la orden BID reduce posición. Exige precio <= entry * (1 - 8 bps).
        # Si el flujo (OBI < -0.45 o TFI < -0.50) continúa a favor, corre el TP hacia abajo (-5 bps).
        be_bps_delta = self.MIN_FEE_BPS + self.MIN_PROFIT_BPS_OVER_BREAKEVEN  # 8.0 bps en cripto
        if any(c in self.symbol for c in ("XAU", "XAG", "GOLD", "SILVER")):
            be_bps_delta = 6.0  # 6.0 bps en TradFi commodities

        bid_ladder: list[OrderRequest] = []
        for weight, mult, sub_lot in ladder_tiers:
            level_half_spread = half_spread * mult
            raw_level_bid = reservation_price - level_half_spread - vol_shift
            safe_level_bid = max(min(raw_level_bid, best_ask - self.tick_size), self.tick_size)

            # Si estamos en SHORT, este BID es orden de cierre: imponer Break-Even real y Momentum Trailing
            if normalized_inventory_q < -1e-4 and entry_price and entry_price > 0:
                max_be_bid = entry_price * (1.0 - (be_bps_delta / 10_000.0))
                safe_level_bid = min(safe_level_bid, max_be_bid)
                # Momentum Trailing Exit a favor de la tendencia bajista si OBI < -0.45 o TFI < -0.50
                if (obi < -0.45 or tfi < -0.50) and (best_ask < entry_price or mid_price < entry_price):
                    trail_shift = mid_price * (self.momentum_shift_bps / 10_000.0)
                    safe_level_bid = max(safe_level_bid - trail_shift, self.tick_size)

            raw_qty = max(self.quote_qty * weight, self.step_size)
            if safe_level_bid > 0 and raw_qty * safe_level_bid < self.min_notional:
                needed_qty = math.ceil(self.min_notional / safe_level_bid / self.step_size) * self.step_size
                raw_qty = max(raw_qty, needed_qty)
            level_qty = truncate_decimal_step(raw_qty, self.step_size)
            cid = f"AV_{sub_lot}_{now_ms_tag}_{int(weight * 100)}"

            cand_bid = OrderRequest.build_maker_order(
                symbol=self.symbol,
                side="BUY",
                raw_price=safe_level_bid,
                raw_qty=level_qty,
                tick_size=self.tick_size,
                step_size=self.step_size,
                strategy_id=self.STRATEGY_ID,
                timestamp_ns=now_ns,
                client_order_id=cid,
            )
            bid_ok, _ = self.gatekeeper.validate_order(cand_bid, portfolio_state)
            if bid_ok:
                bid_ladder.append(cand_bid)

        ask_ladder: list[OrderRequest] = []
        if not suppress_ask:
            for weight, mult, sub_lot in ladder_tiers:
                level_half_spread = half_spread * mult
                raw_level_ask = reservation_price + level_half_spread
                safe_level_ask = max(raw_level_ask, best_bid + self.tick_size)

                # Si estamos en LONG, este ASK es orden de cierre: imponer Break-Even real y Momentum Trailing
                if normalized_inventory_q > 1e-4 and entry_price and entry_price > 0:
                    min_be_ask = entry_price * (1.0 + (be_bps_delta / 10_000.0))
                    safe_level_ask = max(safe_level_ask, min_be_ask)
                    # Momentum Trailing Exit a favor de la tendencia alcista si OBI > 0.45 o TFI > 0.50
                    if (obi > 0.45 or tfi > 0.50) and (best_bid > entry_price or mid_price > entry_price):
                        trail_shift = mid_price * (self.momentum_shift_bps / 10_000.0)
                        safe_level_ask = safe_level_ask + trail_shift

                raw_qty = max(self.quote_qty * weight, self.step_size)
                if safe_level_ask > 0 and raw_qty * safe_level_ask < self.min_notional:
                    needed_qty = math.ceil(self.min_notional / safe_level_ask / self.step_size) * self.step_size
                    raw_qty = max(raw_qty, needed_qty)
                level_qty = truncate_decimal_step(raw_qty, self.step_size)
                cid = f"AV_{sub_lot}_{now_ms_tag}_{int(weight * 100)}"

                cand_ask = OrderRequest.build_maker_order(
                    symbol=self.symbol,
                    side="SELL",
                    raw_price=safe_level_ask,
                    raw_qty=level_qty,
                    tick_size=self.tick_size,
                    step_size=self.step_size,
                    strategy_id=self.STRATEGY_ID,
                    timestamp_ns=now_ns,
                    client_order_id=cid,
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

