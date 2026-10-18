# backend/validation/l2_simulator_and_dsr.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal, Sequence
import numpy as np

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.schemas import OrderRequest


OrderSimStatus = Literal[
    "PENDING_NEW",
    "OPEN",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCEL_PENDING",
    "CANCELED",
    "POST_ONLY_REJECTED",
]


@dataclass(slots=True)
class SimulatedL2Order:
    """Estado interno de una orden límite GTX en la cola FIFO del libro L2."""

    order_id: str
    request: OrderRequest
    status: OrderSimStatus
    placed_ts_ns: int
    effective_live_ts_ns: int
    queue_ahead_qty: float
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    cancel_requested_ts_ns: int | None = None
    effective_cancel_ts_ns: int | None = None


class L2QueueSimulator:
    """
    Simulador determinista de posición en cola L2 (FIFO) y latencia de red
    para validación Dry-Run / Paper Trading sin sesgo de ejecución instantánea.
    """

    PRICE_EPSILON: float = 1e-8

    __slots__ = (
        "latency_ns",
        "orders",
        "_seq",
    )

    def __init__(self, latency_ms: float = 15.0) -> None:
        if latency_ms < 0.0:
            raise ValueError("La latencia de red simulada no puede ser negativa.")
        self.latency_ns: int = int(latency_ms * 1_000_000)
        self.orders: dict[str, SimulatedL2Order] = {}
        self._seq: int = 0

    @property
    def latency_ms(self) -> float:
        return self.latency_ns / 1_000_000.0

    @latency_ms.setter
    def latency_ms(self, val: float) -> None:
        if val < 0.0:
            raise ValueError("La latencia no puede ser negativa.")
        self.latency_ns = int(val * 1_000_000)

    def submit_gtx_order(
        self,
        order: OrderRequest,
        ring_buffer: MarketRingBuffer,
        current_ts_ns: int | None = None,
    ) -> SimulatedL2Order:
        """
        Registra una orden límite Post-Only (GTX) calculando el volumen en cola
        (`queue_ahead_qty`) delante de nuestra orden en el nivel de precio P.
        """
        if order.order_type != "LIMIT" or order.time_in_force != "GTX":
            raise ValueError("L2QueueSimulator solo admite órdenes Maker LIMIT con time_in_force='GTX'.")

        self._seq += 1
        order_id = f"SIM_GTX_{self._seq}"
        now_ns = current_ts_ns if current_ts_ns is not None else order.timestamp_ns
        live_ts_ns = now_ns + self.latency_ns

        queue_ahead = 0.0
        status: OrderSimStatus = "OPEN"

        if ring_buffer.ob_count > 0:
            idx = (ring_buffer.ob_head - 1) % ring_buffer.capacity
            best_bid = float(ring_buffer.ob_bid_prices[idx, 0])
            best_ask = float(ring_buffer.ob_ask_prices[idx, 0])

            # 1. Verificación Post-Only (GTX): rechazar si cruza el libro inmediatamente
            if order.side == "BUY" and order.price >= (best_ask - self.PRICE_EPSILON):
                status = "POST_ONLY_REJECTED"
            elif order.side == "SELL" and order.price <= (best_bid + self.PRICE_EPSILON):
                status = "POST_ONLY_REJECTED"
            else:
                # 2. Calcular queue_ahead_qty buscando el nivel exacto en los 5 niveles L2
                if order.side == "BUY":
                    for lvl in range(5):
                        lvl_price = float(ring_buffer.ob_bid_prices[idx, lvl])
                        if abs(lvl_price - order.price) <= self.PRICE_EPSILON:
                            queue_ahead = float(ring_buffer.ob_bid_qtys[idx, lvl])
                            break
                else:
                    for lvl in range(5):
                        lvl_price = float(ring_buffer.ob_ask_prices[idx, lvl])
                        if abs(lvl_price - order.price) <= self.PRICE_EPSILON:
                            queue_ahead = float(ring_buffer.ob_ask_qtys[idx, lvl])
                            break

        sim_order = SimulatedL2Order(
            order_id=order_id,
            request=order,
            status=status,
            placed_ts_ns=now_ns,
            effective_live_ts_ns=live_ts_ns,
            queue_ahead_qty=queue_ahead,
        )
        self.orders[order_id] = sim_order
        return sim_order

    def request_cancel(self, order_id: str, request_ts_ns: int) -> bool:
        """
        Programa la cancelación de una orden aplicando la latencia de red (`latency_ms`).
        La orden sigue expuesta a fills adversos hasta `request_ts_ns + latency_ns`.
        """
        sim_order = self.orders.get(order_id)
        if sim_order is None:
            return False
        if sim_order.status in ("FILLED", "CANCELED", "POST_ONLY_REJECTED"):
            return False

        sim_order.cancel_requested_ts_ns = request_ts_ns
        sim_order.effective_cancel_ts_ns = request_ts_ns + self.latency_ns

        # Si la orden estaba abierta, pasa a CANCEL_PENDING durante el vuelo del paquete
        if sim_order.status == "OPEN":
            sim_order.status = "CANCEL_PENDING"
        return True

    def on_market_trade(
        self,
        *,
        trade_price: float,
        trade_qty: float,
        is_buyer_maker: bool,
        trade_ts_ns: int,
    ) -> list[SimulatedL2Order]:
        """
        Procesa un trade ejecutado en el mercado contra las órdenes activas:
        - Si `trade_ts_ns >= effective_cancel_ts_ns`, consolida primero la cancelación.
        - Si el precio atraviesa completamente nuestro nivel P, llena el total remanente.
        - Si el trade golpea exactamente nuestro nivel P del lado pasivo correspondiente,
          descuenta `queue_ahead_qty` y solo llena nuestra orden con el volumen excedente.
        """
        filled_orders: list[SimulatedL2Order] = []

        for sim_order in self.orders.values():
            if sim_order.status not in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                continue

            # 1. ¿Llegó ya la solicitud de cancelación al exchange antes de este trade?
            if (
                sim_order.effective_cancel_ts_ns is not None
                and trade_ts_ns >= sim_order.effective_cancel_ts_ns
            ):
                sim_order.status = "CANCELED"
                continue

            # 2. ¿La orden ya llegó al motor de emparejamiento del exchange?
            if trade_ts_ns < sim_order.effective_live_ts_ns:
                continue

            order_price = sim_order.request.price
            remaining_qty = sim_order.request.qty - sim_order.filled_qty

            if sim_order.request.side == "BUY":
                # A) El precio atravesó completamente nuestro nivel BUY (trade_price < P)
                if trade_price < (order_price - self.PRICE_EPSILON):
                    sim_order.queue_ahead_qty = 0.0
                    sim_order.filled_qty = sim_order.request.qty
                    sim_order.avg_fill_price = order_price
                    sim_order.status = "FILLED"
                    filled_orders.append(sim_order)

                # B) Venta a mercado agresora (is_buyer_maker=True) exactamente en nuestro nivel BUY
                elif (
                    is_buyer_maker
                    and abs(trade_price - order_price) <= self.PRICE_EPSILON
                ):
                    self._consume_queue_and_fill(sim_order, trade_qty, remaining_qty, filled_orders)

            else:  # side == "SELL"
                # A) El precio atravesó completamente nuestro nivel SELL (trade_price > P)
                if trade_price > (order_price + self.PRICE_EPSILON):
                    sim_order.queue_ahead_qty = 0.0
                    sim_order.filled_qty = sim_order.request.qty
                    sim_order.avg_fill_price = order_price
                    sim_order.status = "FILLED"
                    filled_orders.append(sim_order)

                # B) Compra a mercado agresora (is_buyer_maker=False) exactamente en nuestro nivel SELL
                elif (
                    (not is_buyer_maker)
                    and abs(trade_price - order_price) <= self.PRICE_EPSILON
                ):
                    self._consume_queue_and_fill(sim_order, trade_qty, remaining_qty, filled_orders)

        return filled_orders

    def advance_clock(self, current_ts_ns: int) -> None:
        """Consolida cancelaciones cuyo tiempo de viaje de red ya se cumplió."""
        for sim_order in self.orders.values():
            if (
                sim_order.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING")
                and sim_order.effective_cancel_ts_ns is not None
                and current_ts_ns >= sim_order.effective_cancel_ts_ns
            ):
                sim_order.status = "CANCELED"

    @staticmethod
    def _consume_queue_and_fill(
        sim_order: SimulatedL2Order,
        trade_qty: float,
        remaining_qty: float,
        filled_orders: list[SimulatedL2Order],
    ) -> None:
        """Descuenta la cola FIFO delante de la orden y asigna el excedente a nuestra orden."""
        if trade_qty <= sim_order.queue_ahead_qty:
            sim_order.queue_ahead_qty -= trade_qty
            return

        # Volumen que logra superar la cola delante de nosotros
        overflow_qty = trade_qty - sim_order.queue_ahead_qty
        sim_order.queue_ahead_qty = 0.0

        fill_amount = min(overflow_qty, remaining_qty)
        sim_order.filled_qty += fill_amount
        sim_order.avg_fill_price = sim_order.request.price

        if sim_order.filled_qty >= (sim_order.request.qty - 1e-12):
            sim_order.filled_qty = sim_order.request.qty
            sim_order.status = "FILLED"
            filled_orders.append(sim_order)
        else:
            sim_order.status = "PARTIALLY_FILLED"


# =============================================================================
# 2. AUDITOR DEL DEFLATED SHARPE RATIO (BAILEY & LÓPEZ DE PRADO)
# =============================================================================

@dataclass(slots=True, frozen=True)
class DSRAuditResult:
    deflated_sharpe_ratio: float | None
    observed_sharpe_sr: float | None
    benchmark_null_sr0: float | None
    skewness_gamma3: float | None
    kurtosis_gamma4: float | None
    sample_length_t: int
    num_trials_n: int
    status: str
    is_overfitted: bool
    annualized_sharpe: float | None = None


class TrialRegistry:
    """Registro determinista y observable del número real de configuraciones / trials evaluados."""
    _trials_count: int = 1

    @classmethod
    def register_trial(cls) -> int:
        cls._trials_count += 1
        return cls._trials_count

    @classmethod
    def set_trials_count(cls, count: int) -> None:
        cls._trials_count = max(1, int(count))

    @classmethod
    def get_trials_count(cls) -> int:
        return cls._trials_count

    @classmethod
    def reset(cls) -> None:
        cls._trials_count = 1


class DeflatedSharpeAuditor:
    """
    Implementación exacta del Deflated Sharpe Ratio (DSR) de Bailey y López de Prado (2014):
      DSR = Phi( ((SR_hat - SR_0) * sqrt(T - 1)) / sqrt(1 - gamma_3 * SR_hat + ((gamma_4 - 1) / 4) * SR_hat^2) )
    Emite OVERFITTING_WARNING si DSR < 0.95.
    """

    EULER_MASCHERONI: float = 0.5772156649015329
    MIN_DSR_THRESHOLD: float = 0.95

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Función de distribución acumulada normal estándar Phi(x)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_ppf(p: float) -> float:
        """
        Inversa de la normal estándar Phi^{-1}(p) mediante aproximación racional
        de Peter Acklam (error relativo < 1.15e-9, sin dependencias externas).
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError(f"La probabilidad p debe estar en (0, 1). Recibido: {p}")

        a = (
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        )
        b = (
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        )
        c = (
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        )
        d = (
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        )

        p_low = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            )
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    @classmethod
    def compute_expected_max_sr0(
        cls,
        num_trials_n: int,
        sample_length_t: int,
        trials_sr_variance: float | None = None,
    ) -> float:
        """
        Calcula el umbral máximo esperado de Sharpe Ratio bajo ruido puro (SR_0)
        según el número total de configuraciones evaluadas (N) y la longitud muestral T:
          SR_0 = sqrt(V) * [ (1 - gamma_E) * Phi^{-1}(1 - 1/N) + gamma_E * Phi^{-1}(1 - 1/(N*e)) ]
        """
        if num_trials_n <= 1:
            return 0.0

        v = trials_sr_variance if trials_sr_variance is not None else (1.0 / max(sample_length_t - 1, 1))
        sigma_sr = math.sqrt(max(v, 1e-12))

        # Cuantiles extremos aproximados
        p1 = 1.0 - (1.0 / num_trials_n)
        p2 = 1.0 - (1.0 / (num_trials_n * math.e))

        # Clamping de seguridad para evitar indeterminación numérica
        p1 = min(max(p1, 1e-12), 1.0 - 1e-12)
        p2 = min(max(p2, 1e-12), 1.0 - 1e-12)

        z1 = cls.norm_ppf(p1)
        z2 = cls.norm_ppf(p2)

        sr0 = sigma_sr * ((1.0 - cls.EULER_MASCHERONI) * z1 + cls.EULER_MASCHERONI * z2)
        return max(sr0, 0.0)

    @classmethod
    def compute_dsr(
        cls,
        observed_sr: float,
        sample_length_t: int,
        skewness_gamma3: float,
        kurtosis_gamma4: float,
        num_trials_n: int = 1,
        trials_sr_variance: float | None = None,
        annualization_factor: float = 365.0,
    ) -> DSRAuditResult:
        """
        Calcula el Deflated Sharpe Ratio de Bailey y López de Prado.
        `kurtosis_gamma4` corresponde a la curtosis de Pearson (donde Normal = 3.0).
        """
        if sample_length_t < 3:
            raise ValueError(f"sample_length_t debe ser >= 3. Recibido: {sample_length_t}")

        # Calcular el umbral nulo esperado SR_0 debido a la selección múltiple
        sr0 = cls.compute_expected_max_sr0(num_trials_n, sample_length_t, trials_sr_variance)

        # Denominador de la varianza asintótica de Mertens (1998)
        term1 = 1.0
        term2 = skewness_gamma3 * observed_sr
        term3 = ((kurtosis_gamma4 - 1.0) / 4.0) * (observed_sr**2)
        var_sr = term1 - term2 + term3

        if var_sr <= 1e-12:
            var_sr = 1e-12

        std_sr = math.sqrt(var_sr)
        z_stat = ((observed_sr - sr0) * math.sqrt(sample_length_t - 1)) / std_sr
        dsr_value = cls.norm_cdf(z_stat)

        is_overfitted = dsr_value < cls.MIN_DSR_THRESHOLD
        status: Literal["STATISTICALLY_SIGNIFICANT", "OVERFITTING_WARNING"] = (
            "OVERFITTING_WARNING" if is_overfitted else "STATISTICALLY_SIGNIFICANT"
        )
        annualized_sr = round(float(observed_sr * math.sqrt(annualization_factor)), 4)

        return DSRAuditResult(
            deflated_sharpe_ratio=dsr_value,
            observed_sharpe_sr=observed_sr,
            benchmark_null_sr0=sr0,
            skewness_gamma3=skewness_gamma3,
            kurtosis_gamma4=kurtosis_gamma4,
            sample_length_t=sample_length_t,
            num_trials_n=num_trials_n,
            status=status,
            is_overfitted=is_overfitted,
            annualized_sharpe=annualized_sr,
        )

    @classmethod
    def aggregate_daily_returns(
        cls,
        equity_snapshots: Sequence[tuple[int | str, float]],
    ) -> list[float]:
        """
        Agrega una serie de equidad con timestamps (ms o date_str) en retornos diarios discretos:
        R_d = (Equity_d - Equity_{d-1}) / Equity_{d-1}
        """
        daily_equity: dict[str, float] = {}
        for ts, eq in equity_snapshots:
            if isinstance(ts, (int, float)):
                d_str = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                d_str = str(ts)[:10]
            daily_equity[d_str] = float(eq)

        sorted_days = sorted(daily_equity.keys())
        if len(sorted_days) < 2:
            return []

        daily_returns: list[float] = []
        for i in range(1, len(sorted_days)):
            prev_eq = daily_equity[sorted_days[i - 1]]
            curr_eq = daily_equity[sorted_days[i]]
            if prev_eq > 1e-6:
                daily_returns.append((curr_eq - prev_eq) / prev_eq)
        return daily_returns

    @classmethod
    def audit_returns(
        cls,
        returns: Sequence[float] | np.ndarray,
        num_trials_n: int = 1,
        trials_sr_variance: float | None = None,
        annualization_factor: float = 365.0,
    ) -> DSRAuditResult:
        """
        Calcula asimetría, curtosis de Pearson, Sharpe observado y DSR
        directamente a partir de una serie temporal de retornos individuales o diarios.
        """
        arr = np.asarray(returns, dtype=np.float64)
        t = len(arr)
        if t < 3:
            raise ValueError(f"Se requieren al menos 3 retornos. Recibidos: {t}")

        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))

        if std_r <= 1e-12:
            return DSRAuditResult(
                deflated_sharpe_ratio=0.0,
                observed_sharpe_sr=0.0,
                benchmark_null_sr0=0.0,
                skewness_gamma3=0.0,
                kurtosis_gamma4=3.0,
                sample_length_t=t,
                num_trials_n=num_trials_n,
                status="OVERFITTING_WARNING",
                is_overfitted=True,
                annualized_sharpe=0.0,
            )

        observed_sr = mean_r / std_r

        # Momentos centrales superiores
        diff = arr - mean_r
        m2 = np.mean(diff**2)
        m3 = np.mean(diff**3)
        m4 = np.mean(diff**4)

        # Asimetría (skewness) gamma_3
        skewness_gamma3 = float(m3 / (m2**1.5)) if m2 > 1e-12 else 0.0

        # Curtosis de Pearson gamma_4 (donde Normal = 3.0)
        kurtosis_gamma4 = float(m4 / (m2**2)) if m2 > 1e-12 else 3.0

        return cls.compute_dsr(
            observed_sr=observed_sr,
            sample_length_t=t,
            skewness_gamma3=skewness_gamma3,
            kurtosis_gamma4=kurtosis_gamma4,
            num_trials_n=num_trials_n,
            trials_sr_variance=trials_sr_variance,
            annualization_factor=annualization_factor,
        )
