# backend/mandate_gates/gatekeeper.py
from __future__ import annotations

import time
from collections.abc import Callable
from backend.mandate_gates.schemas import OrderRequest, PortfolioState


class MandateGatekeeper:
    """
    Candado inmutable de riesgo pre-trade en el Hot Path (<0.1 ms).
    Evalúa síncronamente los 4 mandatos cuantitativos antes de permitir que una orden
    llegue al bus de ejecución WebSocket/REST.
    """

    MAKER_ONLY_STRATEGIES: frozenset[str] = frozenset({"AVELLANEDA_OBI", "LIQ_CASCADE"})
    MAX_LEVERAGE_CAP: float = 4.0
    MAX_SINGLE_PAIR_MARGIN_RATIO: float = 0.20
    DAILY_KILL_SWITCH_DRAWDOWN: float = -0.03
    MAX_CONSECUTIVE_LOSSES: int = 5
    MAX_CONSECUTIVE_API_ERRORS: int = 5
    FLOAT_EPSILON: float = 1e-12

    __slots__ = (
        "_emergency_callback",
        "kill_switch_engaged",
        "emergency_flatten_count",
        "max_leverage_cap",
        "max_single_pair_margin_ratio",
        "consecutive_losses",
        "max_consecutive_losses",
        "consecutive_api_errors",
        "rate_limit_cooldown_until_ns",
    )

    def __init__(
        self,
        emergency_callback: Callable[[], None] | None = None,
        max_leverage_cap: float | None = None,
        max_single_pair_margin_ratio: float | None = None,
        max_consecutive_losses: int | None = None,
    ) -> None:
        self._emergency_callback = emergency_callback
        self.kill_switch_engaged: bool = False
        self.emergency_flatten_count: int = 0
        self.max_leverage_cap: float = (
            float(max_leverage_cap) if max_leverage_cap is not None else self.MAX_LEVERAGE_CAP
        )
        self.max_single_pair_margin_ratio: float = (
            float(max_single_pair_margin_ratio)
            if max_single_pair_margin_ratio is not None
            else self.MAX_SINGLE_PAIR_MARGIN_RATIO
        )
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = (
            int(max_consecutive_losses) if max_consecutive_losses is not None else self.MAX_CONSECUTIVE_LOSSES
        )
        self.consecutive_api_errors: int = 0
        self.rate_limit_cooldown_until_ns: int = 0

    def record_trade_outcome(self, pnl_usd: float) -> bool:
        """
        Registra el resultado de un cierre de posición para el Circuit Breaker de racha de pérdidas.
        Retorna True si se disparó el Kill-Switch por racha de pérdidas consecutivas.
        """
        if pnl_usd < -1e-6:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trigger_emergency_flatten()
                return True
        elif pnl_usd > 1e-6:
            self.consecutive_losses = 0
        return False

    def record_api_error(self, error_str: str, cooldown_seconds: float = 60.0) -> None:
        """
        Registra errores de API o límites de tasa (HTTP 429 / 418 / -1003) y activa
        una ventana de enfriamiento (cooldown) para proteger la IP de baneos del exchange.
        """
        self.consecutive_api_errors += 1
        err_lower = str(error_str).lower()
        is_rate_limit = (
            "429" in err_lower
            or "418" in err_lower
            or "-1003" in err_lower
            or "too many requests" in err_lower
            or "rate limit" in err_lower
            or "way too many" in err_lower
        )
        if is_rate_limit or self.consecutive_api_errors >= self.MAX_CONSECUTIVE_API_ERRORS:
            self.rate_limit_cooldown_until_ns = time.time_ns() + int(cooldown_seconds * 1_000_000_000)

    def record_api_success(self) -> None:
        """Resetea el contador de errores consecutivos de API tras una respuesta exitosa."""
        self.consecutive_api_errors = 0

    def set_leverage_tier(self, leverage: float) -> None:
        """
        Ajusta el techo de apalancamiento autorizado (ej. x3, x5, x7, x10)
        y escala proporcionalmente el cupo nocional por par mantendo máx 20% del margen.
        """
        lev = max(1.0, min(float(leverage), 20.0))
        self.max_leverage_cap = lev
        # A 3x/4x base, el cupo nocional por par es 0.20 * base_cap (o escalado si lev > 4.0)
        if lev <= 4.0:
            self.max_single_pair_margin_ratio = self.MAX_SINGLE_PAIR_MARGIN_RATIO
        else:
            self.max_single_pair_margin_ratio = round(self.MAX_SINGLE_PAIR_MARGIN_RATIO * (lev / 3.0), 4)

    def trigger_emergency_flatten(self) -> None:
        """
        Activa el estado de bloqueo total (Kill-Switch) y notifica al despachador
        de emergencia en segundo plano para cerrar posiciones sin bloquear el Hot Path.
        """
        self.kill_switch_engaged = True
        self.emergency_flatten_count += 1
        if self._emergency_callback is not None:
            self._emergency_callback()

    def validate_order(
        self, order: OrderRequest, state: PortfolioState
    ) -> tuple[bool, str]:
        """
        Valida una orden en <0.1 ms contra los 4 candados de mandato.
        Retorna (True, "APPROVED") o (False, "<CODIGO_RECHAZO>: <Detalle>").
        """
        # ---------------------------------------------------------------------
        # GATE 4: Kill-Switch Diario del -3.0% desde el High-Water Mark
        # y Circuit Breaker por Racha de Pérdidas / Rate-Limit Cooldown
        # ---------------------------------------------------------------------
        hwm = state.daily_high_watermark_usd
        drawdown = (state.current_equity_usd - hwm) / hwm
        if drawdown <= (self.DAILY_KILL_SWITCH_DRAWDOWN + self.FLOAT_EPSILON):
            self.trigger_emergency_flatten()
            return (
                False,
                f"GATE_4_KILL_SWITCH: Drawdown diario ({drawdown * 100:.2f}%) <= -3.00%. "
                "Nuevas órdenes bloqueadas y flatten de emergencia disparado.",
            )

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trigger_emergency_flatten()
            return (
                False,
                f"GATE_4_LOSS_STREAK_CIRCUIT_BREAKER: {self.consecutive_losses} pérdidas consecutivas "
                f"(máximo permitido: {self.max_consecutive_losses}). Flatten de emergencia activado.",
            )

        if self.rate_limit_cooldown_until_ns > 0 and time.time_ns() < self.rate_limit_cooldown_until_ns:
            return (
                False,
                "GATE_4_API_RATE_LIMIT_COOLDOWN: Enfriamiento activo por Rate Limit (HTTP 429/418) o errores de API.",
            )

        # ---------------------------------------------------------------------
        # GATE 1: Margen Aislado Estricto (ISOLATED)
        # ---------------------------------------------------------------------
        if order.margin_mode != "ISOLATED":
            return (
                False,
                f"GATE_1_ISOLATED_MARGIN: margin_mode='{order.margin_mode}' prohibido. "
                "Solo se permite 'ISOLATED'.",
            )

        # ---------------------------------------------------------------------
        # GATE 2: Protección de Comisiones Maker (Post-Only GTX)
        # ---------------------------------------------------------------------
        if order.strategy_id in self.MAKER_ONLY_STRATEGIES:
            if order.order_type != "LIMIT" or order.time_in_force != "GTX":
                return (
                    False,
                    f"GATE_2_MAKER_PROTECTION: Estrategia '{order.strategy_id}' exige "
                    f"order_type='LIMIT' y time_in_force='GTX' (recibido: "
                    f"type='{order.order_type}', tif='{order.time_in_force}').",
                )

        # ---------------------------------------------------------------------
        # GATE 3: Techo Dinámico Half-Kelly (0.0x a max_leverage_cap) y Límite por Par
        # ---------------------------------------------------------------------
        variance_safe = max(state.rolling_variance_sigma2, 1e-8)
        l_half_kelly = 0.5 * (state.rolling_edge_mu / variance_safe)
        # Si el operador configuró un tier explícito > 4.0 (ej. 5x, 7x, 10x), escalar el techo efectivo
        scale_mult = max(1.0, self.max_leverage_cap / self.MAX_LEVERAGE_CAP)
        allowed_leverage = min(max(l_half_kelly * scale_mult, 0.0), self.max_leverage_cap)

        base_cap = state.base_capital_usd
        order_notional = order.price * order.qty

        # 3a. Concentración máxima por par individual (<= 20% del margen operativo)
        existing_pair_notional = state.active_positions.get(order.symbol, 0.0)
        open_orders_map = getattr(state, "open_orders_notional", None) or {}
        pending_pair_notional = abs(float(open_orders_map.get(order.symbol, 0.0)))
        is_reducing = (existing_pair_notional > 0 and order.side == "SELL") or (
            existing_pair_notional < 0 and order.side == "BUY"
        )
        if is_reducing:
            projected_pair_notional = max(0.0, abs(existing_pair_notional) - order_notional)
        else:
            projected_pair_notional = abs(existing_pair_notional) + pending_pair_notional + order_notional
        pair_margin_ratio = projected_pair_notional / base_cap

        if pair_margin_ratio > (self.max_single_pair_margin_ratio + self.FLOAT_EPSILON):
            return (
                False,
                f"GATE_3_PAIR_CONCENTRATION: El par '{order.symbol}' alcanzaría "
                f"{pair_margin_ratio * 100:.2f}% del margen operativo (máximo permitido: {self.max_single_pair_margin_ratio * 100:.2f}%).",
            )

        # 3b. Apalancamiento total del portafolio vs min(L_half_kelly, max_leverage_cap)
        other_pairs_notional = sum(
            abs(pos_notional) for sym, pos_notional in state.active_positions.items() if sym != order.symbol
        ) + sum(
            abs(float(ord_notional)) for sym, ord_notional in open_orders_map.items() if sym != order.symbol
        )

        projected_total_leverage = (other_pairs_notional + projected_pair_notional) / base_cap
        if not is_reducing and projected_total_leverage > (allowed_leverage + self.FLOAT_EPSILON):
            return (
                False,
                f"GATE_3_HALF_KELLY_CAP: Apalancamiento proyectado ({projected_total_leverage:.4f}x) "
                f"supera el techo permitido ({allowed_leverage:.4f}x | Half-Kelly={l_half_kelly:.4f}x).",
            )

        return True, "APPROVED"
