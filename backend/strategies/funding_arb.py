# backend/strategies/funding_arb.py
from __future__ import annotations

from dataclasses import dataclass
import time

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step
from backend.strategies.regime_config import RegimeConfig


@dataclass(slots=True, frozen=True)
class FundingArbSnapshot:
    symbol: str
    spot_price: float
    perp_bid_price: float
    perp_ask_price: float
    projected_funding_rate: float
    funding_interval_hours: int = 8


@dataclass(slots=True, frozen=True)
class DeltaNeutralArbPackage:
    symbol: str
    annualized_yield: float
    basis_bps: float
    spot_hedge_leg: OrderRequest
    perp_futures_leg: OrderRequest


@dataclass(slots=True, frozen=True)
class CrossExchangeFundingSnapshot:
    symbol: str
    binance_funding_rate: float
    bingx_funding_rate: float
    binance_bid_price: float
    binance_ask_price: float
    bingx_bid_price: float
    bingx_ask_price: float
    funding_interval_hours: int = 8


@dataclass(slots=True, frozen=True)
class CrossExchangeArbPackage:
    symbol: str
    annualized_yield_diff: float
    rate_spread_bps: float
    long_exchange: str
    short_exchange: str
    long_leg: OrderRequest
    short_leg: OrderRequest


class FundingRateArbitrageEngine:
    """
    Motor C: Arbitraje Delta-Neutral de Tasa de Financiación y Basis Spot-Perpetuo.
    Estructura ambas patas y exige aprobación atómica en MandateGatekeeper.
    """

    STRATEGY_ID: str = "FUNDING_BASIS_ARB"

    __slots__ = (
        "tick_size",
        "step_size",
        "config",
        "gatekeeper",
    )

    def __init__(
        self,
        *,
        tick_size: float,
        step_size: float,
        config: RegimeConfig,
        gatekeeper: MandateGatekeeper,
    ) -> None:
        self.tick_size = tick_size
        self.step_size = step_size
        self.config = config
        self.gatekeeper = gatekeeper

    def update_regime_config(self, new_config: RegimeConfig) -> None:
        self.config = new_config

    @staticmethod
    def compute_metrics(snapshot: FundingArbSnapshot) -> tuple[float, float]:
        periods_per_day = 24.0 / max(snapshot.funding_interval_hours, 1)
        annualized_funding = snapshot.projected_funding_rate * periods_per_day * 365.0

        perp_mid = 0.5 * (snapshot.perp_bid_price + snapshot.perp_ask_price)
        basis_bps = ((perp_mid - snapshot.spot_price) / snapshot.spot_price) * 10_000.0

        return annualized_funding, basis_bps

    def evaluate_opportunity(
        self,
        snapshot: FundingArbSnapshot,
        target_notional_usd: float,
        portfolio_state: PortfolioState,
        timestamp_ns: int | None = None,
    ) -> DeltaNeutralArbPackage | None:
        annualized_yield, basis_bps = self.compute_metrics(snapshot)

        if abs(annualized_yield) < self.config.min_annualized_funding_yield:
            return None

        if snapshot.projected_funding_rate > 0.0 and basis_bps < self.config.min_basis_spread_bps:
            return None

        ts_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()

        raw_qty = target_notional_usd / snapshot.spot_price
        matched_qty = truncate_decimal_step(raw_qty, self.step_size)
        if matched_qty <= 0.0:
            return None

        if snapshot.projected_funding_rate > 0.0:
            spot_side = "BUY"
            perp_side = "SELL"
            perp_entry_price = snapshot.perp_ask_price
        else:
            spot_side = "SELL"
            perp_side = "BUY"
            perp_entry_price = snapshot.perp_bid_price

        spot_leg = OrderRequest(
            symbol=f"{snapshot.symbol.split(':')[0]}:SPOT",
            side=spot_side,
            order_type="LIMIT",
            price=truncate_decimal_step(snapshot.spot_price, self.tick_size),
            qty=matched_qty,
            time_in_force="GTX",
            margin_mode="ISOLATED",
            strategy_id=self.STRATEGY_ID,
            timestamp_ns=ts_ns,
        )

        perp_leg = OrderRequest.build_maker_order(
            symbol=snapshot.symbol,
            side=perp_side,
            raw_price=perp_entry_price,
            raw_qty=matched_qty,
            tick_size=self.tick_size,
            step_size=self.step_size,
            strategy_id=self.STRATEGY_ID,
            timestamp_ns=ts_ns,
        )

        ok_spot, _ = self.gatekeeper.validate_order(spot_leg, portfolio_state)
        if not ok_spot:
            return None

        simulated_positions = dict(portfolio_state.active_positions)
        simulated_positions[spot_leg.symbol] = (
            abs(simulated_positions.get(spot_leg.symbol, 0.0)) + spot_leg.notional_usd
        )
        projected_state = portfolio_state.model_copy(
            update={"active_positions": simulated_positions}
        )

        ok_perp, _ = self.gatekeeper.validate_order(perp_leg, projected_state)
        if not ok_perp:
            return None

        return DeltaNeutralArbPackage(
            symbol=snapshot.symbol,
            annualized_yield=annualized_yield,
            basis_bps=basis_bps,
            spot_hedge_leg=spot_leg,
            perp_futures_leg=perp_leg,
        )

    @staticmethod
    def compute_cross_exchange_metrics(
        snapshot: CrossExchangeFundingSnapshot,
    ) -> tuple[float, float, str, str]:
        periods_per_day = 24.0 / max(snapshot.funding_interval_hours, 1)
        rate_diff = snapshot.binance_funding_rate - snapshot.bingx_funding_rate

        if rate_diff >= 0.0:
            short_exchange = "BINANCE"
            long_exchange = "BINGX"
            net_rate = rate_diff
        else:
            short_exchange = "BINGX"
            long_exchange = "BINANCE"
            net_rate = -rate_diff

        annualized_yield_diff = net_rate * periods_per_day * 365.0
        rate_spread_bps = net_rate * 10_000.0

        return annualized_yield_diff, rate_spread_bps, long_exchange, short_exchange

    def evaluate_cross_exchange_opportunity(
        self,
        snapshot: CrossExchangeFundingSnapshot,
        target_notional_usd: float,
        portfolio_state: PortfolioState,
        timestamp_ns: int | None = None,
    ) -> CrossExchangeArbPackage | None:
        (
            annualized_yield_diff,
            rate_spread_bps,
            long_exchange,
            short_exchange,
        ) = self.compute_cross_exchange_metrics(snapshot)

        if annualized_yield_diff < self.config.min_annualized_funding_yield:
            return None

        ref_price = 0.5 * (snapshot.binance_bid_price + snapshot.bingx_bid_price)
        if ref_price <= 0.0:
            return None

        ts_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()
        raw_qty = target_notional_usd / ref_price
        matched_qty = truncate_decimal_step(raw_qty, self.step_size)
        if matched_qty <= 0.0:
            return None

        if long_exchange == "BINANCE":
            long_price = snapshot.binance_bid_price
            short_price = snapshot.bingx_ask_price
        else:
            long_price = snapshot.bingx_bid_price
            short_price = snapshot.binance_ask_price

        long_leg = OrderRequest.build_maker_order(
            symbol=f"{snapshot.symbol}:{long_exchange}",
            side="BUY",
            raw_price=long_price,
            raw_qty=matched_qty,
            tick_size=self.tick_size,
            step_size=self.step_size,
            strategy_id=self.STRATEGY_ID,
            timestamp_ns=ts_ns,
        )

        short_leg = OrderRequest.build_maker_order(
            symbol=f"{snapshot.symbol}:{short_exchange}",
            side="SELL",
            raw_price=short_price,
            raw_qty=matched_qty,
            tick_size=self.tick_size,
            step_size=self.step_size,
            strategy_id=self.STRATEGY_ID,
            timestamp_ns=ts_ns,
        )

        ok_long, _ = self.gatekeeper.validate_order(long_leg, portfolio_state)
        if not ok_long:
            return None

        simulated_positions = dict(portfolio_state.active_positions)
        simulated_positions[long_leg.symbol] = (
            abs(simulated_positions.get(long_leg.symbol, 0.0)) + long_leg.notional_usd
        )
        projected_state = portfolio_state.model_copy(
            update={"active_positions": simulated_positions}
        )

        ok_short, _ = self.gatekeeper.validate_order(short_leg, projected_state)
        if not ok_short:
            return None

        return CrossExchangeArbPackage(
            symbol=snapshot.symbol,
            annualized_yield_diff=annualized_yield_diff,
            rate_spread_bps=rate_spread_bps,
            long_exchange=long_exchange,
            short_exchange=short_exchange,
            long_leg=long_leg,
            short_leg=short_leg,
        )

