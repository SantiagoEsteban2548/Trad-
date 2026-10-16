# backend/mandate_gates/gatekeeper.py
from __future__ import annotations

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
    FLOAT_EPSILON: float = 1e-12

    __slots__ = (
        "_emergency_callback",
        "kill_switch_engaged",
        "emergency_flatten_count",
    )

    def __init__(self, emergency_callback: Callable[[], None] | None = None) -> None:
        self._emergency_callback = emergency_callback
        self.kill_switch_engaged: bool = False
        self.emergency_flatten_count: int = 0

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
        # Se evalúa con prioridad crítica para frenar inmediatamente toda exposición
        # si la cuenta ha sufrido un drawdown diario >= 3.0%.
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
        # GATE 3: Techo Dinámico Half-Kelly (0.0x a 4.0x) y Límite 20% por Par
        # ---------------------------------------------------------------------
        variance_safe = max(state.rolling_variance_sigma2, 1e-8)
        l_half_kelly = 0.5 * (state.rolling_edge_mu / variance_safe)
        allowed_leverage = min(max(l_half_kelly, 0.0), self.MAX_LEVERAGE_CAP)

        base_cap = state.base_capital_usd
        order_notional = order.price * order.qty

        # 3a. Concentración máxima por par individual (<= 20% del margen operativo)
        existing_pair_notional = state.active_positions.get(order.symbol, 0.0)
        if (existing_pair_notional > 0 and order.side == "SELL") or (existing_pair_notional < 0 and order.side == "BUY"):
            projected_pair_notional = max(0.0, abs(existing_pair_notional) - order_notional)
        else:
            projected_pair_notional = abs(existing_pair_notional) + order_notional
        pair_margin_ratio = projected_pair_notional / base_cap

        if pair_margin_ratio > (self.MAX_SINGLE_PAIR_MARGIN_RATIO + self.FLOAT_EPSILON):
            return (
                False,
                f"GATE_3_PAIR_CONCENTRATION: El par '{order.symbol}' alcanzaría "
                f"{pair_margin_ratio * 100:.2f}% del margen operativo (máximo permitido: 20.00%).",
            )

        # 3b. Apalancamiento total del portafolio vs min(L_half_kelly, 4.0)
        open_notional = 0.0
        for pos_notional in state.active_positions.values():
            open_notional += abs(pos_notional)

        projected_total_leverage = (open_notional + order_notional) / base_cap
        if projected_total_leverage > (allowed_leverage + self.FLOAT_EPSILON):
            return (
                False,
                f"GATE_3_HALF_KELLY_CAP: Apalancamiento proyectado ({projected_total_leverage:.4f}x) "
                f"supera el techo permitido ({allowed_leverage:.4f}x | Half-Kelly={l_half_kelly:.4f}x).",
            )

        return True, "APPROVED"
