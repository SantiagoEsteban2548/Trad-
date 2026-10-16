# backend/strategies/liq_cascade.py
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step


@dataclass(slots=True, frozen=True)
class LiquidationCascadeBracket:
    entry_order: OrderRequest
    take_profit_order: OrderRequest
    stop_loss_price: float
    reduce_only_tp: bool
    z_score: float
    vwap_1m: float
    exhaustion_ratio: float


class LiquidationCascadeHunter:
    """
    Motor B: Absorción de Cascadas de Liquidación por Agotamiento de Flujo Forzoso.
    """

    STRATEGY_ID: str = "LIQ_CASCADE"
    Z_SCORE_THRESHOLD: float = 3.2
    EXHAUSTION_RATIO_MAX: float = 0.50
    OBI_RECOVERY_FLOOR_LONG: float = -0.15
    OBI_RECOVERY_CEIL_SHORT: float = 0.15
    WICK_RETRACEMENT_PCT: float = 0.40
    STOP_LOSS_EXTRA_SIGMA: float = 1.5

    __slots__ = (
        "symbol",
        "tick_size",
        "step_size",
        "order_qty",
        "gatekeeper",
    )

    def __init__(
        self,
        *,
        symbol: str,
        tick_size: float,
        step_size: float,
        order_qty: float,
        gatekeeper: MandateGatekeeper,
    ) -> None:
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.step_size = step_size
        self.order_qty = truncate_decimal_step(order_qty, step_size)
        self.gatekeeper = gatekeeper

    @staticmethod
    def compute_500ms_exhaustion_ratio(
        ring_buffer: MarketRingBuffer, reference_ts_ns: int
    ) -> float:
        if ring_buffer.liq_count == 0:
            return 1.0

        half_sec_ns = 500_000_000
        t_minus_500ms = reference_ts_ns - half_sec_ns
        t_minus_1000ms = reference_ts_ns - (2 * half_sec_ns)

        if ring_buffer.liq_count < ring_buffer.capacity:
            ts = ring_buffer.liq_timestamps_ns[: ring_buffer.liq_count]
            qtys = ring_buffer.liq_qtys[: ring_buffer.liq_count]
        else:
            ts = ring_buffer.liq_timestamps_ns
            qtys = ring_buffer.liq_qtys

        mask_prev = (ts >= t_minus_1000ms) & (ts < t_minus_500ms)
        mask_curr = (ts >= t_minus_500ms) & (ts <= reference_ts_ns)

        vol_prev = float(np.sum(qtys[mask_prev]))
        vol_curr = float(np.sum(qtys[mask_curr]))

        if vol_prev <= 1e-12:
            return 1.0

        return vol_curr / vol_prev

    def evaluate(
        self,
        ring_buffer: MarketRingBuffer,
        portfolio_state: PortfolioState,
        now_ns: int | None = None,
    ) -> LiquidationCascadeBracket | None:
        if ring_buffer.ob_count == 0 or ring_buffer.liq_count == 0:
            return None

        cluster = ring_buffer.detect_liquidation_cluster(percentile=99.0)
        if not cluster["is_cluster"]:
            return None

        z_score = float(cluster["z_score"])
        vwap_1m = float(cluster["vwap_1m"])

        if abs(z_score) <= self.Z_SCORE_THRESHOLD or vwap_1m <= 0.0:
            return None

        latest_ob_idx = (ring_buffer.ob_head - 1) % ring_buffer.capacity
        ref_ts_ns = now_ns if now_ns is not None else int(ring_buffer.ob_timestamps_ns[latest_ob_idx])

        exhaustion_ratio = self.compute_500ms_exhaustion_ratio(ring_buffer, ref_ts_ns)
        if exhaustion_ratio > self.EXHAUSTION_RATIO_MAX:
            return None

        obi = ring_buffer.compute_obi(levels=5)
        best_bid = float(ring_buffer.ob_bid_prices[latest_ob_idx, 0])
        best_ask = float(ring_buffer.ob_ask_prices[latest_ob_idx, 0])

        last_trade_idx = (ring_buffer.trade_head - 1) % ring_buffer.capacity
        last_trade_price = float(ring_buffer.trade_prices[last_trade_idx])
        sigma_price_1m = abs(last_trade_price - vwap_1m) / abs(z_score)

        ts_order = now_ns if now_ns is not None else time.time_ns()

        if z_score < -self.Z_SCORE_THRESHOLD:
            if obi <= self.OBI_RECOVERY_FLOOR_LONG:
                return None

            entry_price = best_bid
            wick_distance = max(vwap_1m - entry_price, self.tick_size)
            raw_tp_price = entry_price + (self.WICK_RETRACEMENT_PCT * wick_distance)
            raw_sl_price = entry_price - (self.STOP_LOSS_EXTRA_SIGMA * sigma_price_1m)

            entry_order = OrderRequest.build_maker_order(
                symbol=self.symbol,
                side="BUY",
                raw_price=entry_price,
                raw_qty=self.order_qty,
                tick_size=self.tick_size,
                step_size=self.step_size,
                strategy_id=self.STRATEGY_ID,
                timestamp_ns=ts_order,
            )
            tp_order = OrderRequest.build_maker_order(
                symbol=self.symbol,
                side="SELL",
                raw_price=raw_tp_price,
                raw_qty=self.order_qty,
                tick_size=self.tick_size,
                step_size=self.step_size,
                strategy_id=self.STRATEGY_ID,
                timestamp_ns=ts_order,
            )
        else:
            if obi >= self.OBI_RECOVERY_CEIL_SHORT:
                return None

            entry_price = best_ask
            wick_distance = max(entry_price - vwap_1m, self.tick_size)
            raw_tp_price = entry_price - (self.WICK_RETRACEMENT_PCT * wick_distance)
            raw_sl_price = entry_price + (self.STOP_LOSS_EXTRA_SIGMA * sigma_price_1m)

            entry_order = OrderRequest.build_maker_order(
                symbol=self.symbol,
                side="SELL",
                raw_price=entry_price,
                raw_qty=self.order_qty,
                tick_size=self.tick_size,
                step_size=self.step_size,
                strategy_id=self.STRATEGY_ID,
                timestamp_ns=ts_order,
            )
            tp_order = OrderRequest.build_maker_order(
                symbol=self.symbol,
                side="BUY",
                raw_price=raw_tp_price,
                raw_qty=self.order_qty,
                tick_size=self.tick_size,
                step_size=self.step_size,
                strategy_id=self.STRATEGY_ID,
                timestamp_ns=ts_order,
            )

        approved, _ = self.gatekeeper.validate_order(entry_order, portfolio_state)
        if not approved:
            return None

        return LiquidationCascadeBracket(
            entry_order=entry_order,
            take_profit_order=tp_order,
            stop_loss_price=truncate_decimal_step(raw_sl_price, self.tick_size),
            reduce_only_tp=True,
            z_score=z_score,
            vwap_1m=vwap_1m,
            exhaustion_ratio=exhaustion_ratio,
        )
