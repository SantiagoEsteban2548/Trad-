# backend/strategies/avellaneda_obi.py
from __future__ import annotations

import math
import time
import uuid
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
        "max_spread_bps_ladder",
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
        max_spread_bps_ladder: tuple[float, float, float] = (25.0, 50.0, 85.0),
    ) -> None:
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.step_size = step_size
        self.quote_qty = truncate_decimal_step(quote_qty, step_size)
        self.config = config
        self.gatekeeper = gatekeeper
        self.active_bid_order_id: str | None = None
        self.active_ask_order_id: str | None = None
        self.max_spread_bps_ladder = tuple(float(x) for x in max_spread_bps_ladder)
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
        break_even_price: float | None = None,
        venue_best_bid: float | None = None,
        venue_best_ask: float | None = None,
        lead_lag_tfi: float | None = None,
    ) -> AvellanedaQuoteResult:
        if ring_buffer.ob_count == 0 and (venue_best_bid is None or venue_best_ask is None):
            return AvellanedaQuoteResult(0.0, 0.0, None, None, False, False, 0.0, 0.0, [], [])

        effective_entry = break_even_price if (break_even_price is not None and break_even_price > 0) else entry_price

        if ring_buffer.ob_count > 0:
            latest_idx = (ring_buffer.ob_head - 1) % ring_buffer.capacity
            raw_bb = float(ring_buffer.ob_bid_prices[latest_idx, 0])
            raw_ba = float(ring_buffer.ob_ask_prices[latest_idx, 0])
        else:
            raw_bb = float(venue_best_bid or 1.0)
            raw_ba = float(venue_best_ask or 1.0)

        # Si el exchange de ejecución (ej. BingX) provee su libro directo, priorizarlo para no cruzar
        best_bid = float(venue_best_bid) if (venue_best_bid is not None and venue_best_bid > 0) else raw_bb
        best_ask = float(venue_best_ask) if (venue_best_ask is not None and venue_best_ask > 0) else raw_ba
        mid_price = 0.5 * (best_bid + best_ask)

        obi = ring_buffer.compute_obi(levels=5) if ring_buffer.ob_count > 0 else 0.0
        raw_sigma = ring_buffer.compute_realized_volatility(window_seconds=300.0) if ring_buffer.ob_count > 0 else 0.0005
        sigma = max(raw_sigma, self.MIN_SIGMA_FLOOR)

        calc_vpin = vpin if vpin is not None else (ring_buffer.compute_vpin() if hasattr(ring_buffer, "compute_vpin") and ring_buffer.ob_count > 0 else 0.0)

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
        suppress_bid = False
        suppress_ask = False
        vol_shift = 0.0

        # Señal Lead-Lag entre Binance (Líder) y BingX (Seguidor)
        if lead_lag_tfi is not None:
            if lead_lag_tfi > 0.60:
                # Flujo fuertemente comprador en el líder: retirar ASK para evitar selección adversa
                suppress_ask = True
            elif lead_lag_tfi < -0.60:
                # Flujo fuertemente vendedor en el líder: retirar BID para evitar ser atropellado
                suppress_bid = True

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
        tfi = ring_buffer.compute_tfi(window_ns=500_000_000) if hasattr(ring_buffer, "compute_tfi") and ring_buffer.ob_count > 0 else 0.0

        # Auto-Colapso de Escalera por MIN_NOTIONAL ($100 USDT en BTC, $20 en ETH, $5 en el resto):
        total_quote_notional = self.quote_qty * mid_price
        if (
            total_quote_notional * 0.20 >= self.min_notional
            and total_quote_notional * 0.35 >= self.min_notional
            and total_quote_notional * 0.45 >= self.min_notional
        ):
            ladder_tiers = [
                (0.45, 1.0, "L1", self.max_spread_bps_ladder[0]),
                (0.35, 2.0, "L2", self.max_spread_bps_ladder[1]),
                (0.20, 3.5, "L3", self.max_spread_bps_ladder[2]),
            ]
        else:
            ladder_tiers = [
                (1.00, 1.0, "L1", self.max_spread_bps_ladder[0]),
            ]

        # Break-Even Real y Momentum Trailing Exit:
        be_bps_delta = self.MIN_FEE_BPS + self.MIN_PROFIT_BPS_OVER_BREAKEVEN  # 8.0 bps en cripto
        if any(c in self.symbol for c in ("XAU", "XAG", "GOLD", "SILVER")):
            be_bps_delta = 6.0  # 6.0 bps en TradFi commodities

        bid_ladder: list[OrderRequest] = []
        if not suppress_bid:
            prev_bid_half_spread = 0.0
            for weight, mult, sub_lot, max_tier_bps in ladder_tiers:
                level_half_spread = half_spread * mult
                # Techo dinámico de spread (L1: 25 bps, L2: 50 bps, L3: 85 bps)
                max_half_spread_cap = 0.5 * mid_price * (max_tier_bps / 10_000.0)
                level_half_spread = min(level_half_spread, max_half_spread_cap)
                min_half_floor = 0.5 * mid_price * (self.min_profitable_spread_bps / 10_000.0)
                level_half_spread = max(level_half_spread, min_half_floor)

                # Si hay señal Lead-Lag alcista en L1 (TFI > 0.60), pegar L1 al best_bid para captura inmediata
                if sub_lot == "L1" and lead_lag_tfi is not None and lead_lag_tfi > 0.60:
                    raw_level_bid = best_bid
                else:
                    if level_half_spread <= prev_bid_half_spread:
                        level_half_spread = prev_bid_half_spread + self.tick_size
                    prev_bid_half_spread = level_half_spread
                    raw_level_bid = reservation_price - level_half_spread - vol_shift
                safe_level_bid = max(min(raw_level_bid, best_ask - self.tick_size), self.tick_size)

                # Si estamos en SHORT, este BID es orden de cierre: imponer Break-Even real y Momentum Trailing
                if normalized_inventory_q < -1e-4 and effective_entry and effective_entry > 0:
                    max_be_bid = effective_entry * (1.0 - (be_bps_delta / 10_000.0))
                    safe_level_bid = min(safe_level_bid, max_be_bid)
                    if (obi < -0.45 or tfi < -0.50) and (best_ask < effective_entry or mid_price < effective_entry):
                        trail_shift = mid_price * (self.momentum_shift_bps / 10_000.0)
                        safe_level_bid = max(safe_level_bid - trail_shift, self.tick_size)

                raw_qty = max(self.quote_qty * weight, self.step_size)
                if safe_level_bid > 0 and raw_qty * safe_level_bid < self.min_notional:
                    needed_qty = math.ceil(self.min_notional / safe_level_bid / self.step_size) * self.step_size
                    raw_qty = max(raw_qty, needed_qty)
                level_qty = truncate_decimal_step(raw_qty, self.step_size)
                cid = f"AV_{sub_lot}_B_{now_ms_tag}_{uuid.uuid4().hex[:4]}"

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
            prev_ask_half_spread = 0.0
            for weight, mult, sub_lot, max_tier_bps in ladder_tiers:
                level_half_spread = half_spread * mult
                # Techo dinámico de spread (L1: 25 bps, L2: 50 bps, L3: 85 bps)
                max_half_spread_cap = 0.5 * mid_price * (max_tier_bps / 10_000.0)
                level_half_spread = min(level_half_spread, max_half_spread_cap)
                min_half_floor = 0.5 * mid_price * (self.min_profitable_spread_bps / 10_000.0)
                level_half_spread = max(level_half_spread, min_half_floor)

                # Si hay señal Lead-Lag bajista en L1 (TFI < -0.60), pegar L1 al best_ask para captura inmediata
                if sub_lot == "L1" and lead_lag_tfi is not None and lead_lag_tfi < -0.60:
                    raw_level_ask = best_ask
                else:
                    if level_half_spread <= prev_ask_half_spread:
                        level_half_spread = prev_ask_half_spread + self.tick_size
                    prev_ask_half_spread = level_half_spread
                    raw_level_ask = reservation_price + level_half_spread
                safe_level_ask = max(raw_level_ask, best_bid + self.tick_size)

                # Si estamos en LONG, este ASK es orden de cierre: imponer Break-Even real y Momentum Trailing
                if normalized_inventory_q > 1e-4 and effective_entry and effective_entry > 0:
                    min_be_ask = effective_entry * (1.0 + (be_bps_delta / 10_000.0))
                    safe_level_ask = max(safe_level_ask, min_be_ask)
                    if (obi > 0.45 or tfi > 0.50) and (best_bid > effective_entry or mid_price > effective_entry):
                        trail_shift = mid_price * (self.momentum_shift_bps / 10_000.0)
                        safe_level_ask = safe_level_ask + trail_shift

                raw_qty = max(self.quote_qty * weight, self.step_size)
                if safe_level_ask > 0 and raw_qty * safe_level_ask < self.min_notional:
                    needed_qty = math.ceil(self.min_notional / safe_level_ask / self.step_size) * self.step_size
                    raw_qty = max(raw_qty, needed_qty)
                level_qty = truncate_decimal_step(raw_qty, self.step_size)
                cid = f"AV_{sub_lot}_S_{now_ms_tag}_{uuid.uuid4().hex[:4]}"

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

