# backend/mandate_gates/schemas.py
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


SideLiteral = Literal["BUY", "SELL"]
OrderTypeLiteral = Literal["LIMIT", "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
TimeInForceLiteral = Literal["GTX", "GTC", "IOC", "FOK"]
MarginModeLiteral = Literal["ISOLATED", "CROSS"]
StrategyIdLiteral = Literal[
    "AVELLANEDA_OBI",
    "LIQ_CASCADE",
    "FUNDING_BASIS_ARB",
    "EMERGENCY_FLATTEN",
]


def truncate_decimal_step(value: float | Decimal | str, step: float | Decimal | str) -> float:
    """
    Truncado exacto hacia abajo (ROUND_DOWN) al tick_size o step_size del mercado
    utilizando aritmética Decimal pura, eliminando errores de coma flotante IEEE 754.
    """
    d_val = Decimal(str(value))
    d_step = Decimal(str(step))
    if d_step <= Decimal("0"):
        raise ValueError(f"El incremento (tick/step) debe ser estrictamente positivo. Recibido: {step}")
    quantized = (d_val // d_step) * d_step
    return float(quantized.quantize(d_step, rounding=ROUND_DOWN))


class OrderRequest(BaseModel):
    """
    Representación inmutable de una solicitud de orden hacia el motor de ejecución (Hot Path).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)

    symbol: str = Field(..., min_length=3, description="Par de mercado perpetuo, ej. BTC/USDT:USDT")
    side: SideLiteral = Field(..., description="Dirección de la orden: BUY o SELL")
    order_type: str = Field(..., description="Tipo de orden: LIMIT, MARKET, etc.")
    price: float = Field(..., gt=0.0, description="Precio de ejecución truncado al tick_size")
    qty: float = Field(..., gt=0.0, description="Cantidad en moneda base truncada al step_size")
    time_in_force: str = Field(..., description="Política de vigencia temporal: GTX (Post-Only), GTC, IOC, FOK")
    margin_mode: str = Field(..., description="Modo de margen: ISOLATED o CROSS")
    strategy_id: str = Field(..., min_length=1, description="Identificador del motor de microestructura")
    timestamp_ns: int = Field(..., gt=0, description="Timestamp monotónico o epoch en nanosegundos")
    client_order_id: str | None = Field(default=None, description="Identificador único del cliente para etiquetar estrategia y lote")
    min_notional: float | None = Field(default=None, description="Nocional mínimo del par en USDT")

    @field_validator("symbol", "order_type", "time_in_force", "margin_mode", "strategy_id")
    @classmethod
    def normalize_upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("client_order_id")
    @classmethod
    def normalize_cid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip()

    @property
    def notional_usd(self) -> float:
        """Valor nocional de la orden en USD/USDT."""
        return self.price * self.qty

    @classmethod
    def build_maker_order(
        cls,
        *,
        symbol: str,
        side: SideLiteral,
        raw_price: float | str | Decimal,
        raw_qty: float | str | Decimal,
        tick_size: float | str | Decimal,
        step_size: float | str | Decimal,
        strategy_id: str,
        timestamp_ns: int,
        margin_mode: str = "ISOLATED",
        client_order_id: str | None = None,
    ) -> OrderRequest:
        """
        Constructor auxiliar libre de errores de coma flotante que garantiza
        truncado exacto a tick_size/step_size y bandera Post-Only (GTX).
        """
        return cls(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            price=truncate_decimal_step(raw_price, tick_size),
            qty=truncate_decimal_step(raw_qty, step_size),
            time_in_force="GTX",
            margin_mode=margin_mode,
            strategy_id=strategy_id,
            timestamp_ns=timestamp_ns,
            client_order_id=client_order_id,
        )


class PortfolioState(BaseModel):
    """
    Estado financiero y estadístico en tiempo real del portafolio cuantitativo.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    base_capital_usd: float = Field(
        ..., gt=0.0, description="Capital operativo base asignado en Futuros USDT-M"
    )
    current_equity_usd: float = Field(
        ..., gt=0.0, description="Patrimonio neto actual (incluyendo PnL no realizado)"
    )
    daily_high_watermark_usd: float = Field(
        ..., gt=0.0, description="Marca de agua máxima diaria de patrimonio (High-Water Mark)"
    )
    realized_pnl_today_usd: float = Field(
        default=0.0, description="PnL neto realizado acumulado en la sesión UTC actual"
    )
    active_positions: dict[str, float] = Field(
        default_factory=dict,
        description="Mapa {symbol: notional_usd} de posiciones activas con margen comprometido",
    )
    open_orders_notional: dict[str, float] = Field(
        default_factory=dict,
        description="Mapa {symbol: notional_usd} de órdenes límite activas en el libro pendientes de ejecución",
    )
    rolling_edge_mu: float = Field(
        ..., description="Retorno esperado instantáneo (drift mu) estimado sobre microestructura"
    )
    rolling_variance_sigma2: float = Field(
        ..., ge=0.0, description="Varianza instantánea de retornos (sigma^2)"
    )
    exchange_equities: dict[str, float] = Field(
        default_factory=dict,
        description="Patrimonio neto segregado por exchange {exchange: equity_usd}",
    )
    initial_exchange_equities: dict[str, float] = Field(
        default_factory=dict,
        description="Capital inicial de referencia por exchange {exchange: equity_usd} para evitar ganancias fantasma",
    )
    trading_mode: str = Field(
        default="PAPER_L2",
        description="Modo de ejecución activo (PAPER_L2, LIVE_BINANCE, LIVE_BINGX, LIVE_MULTI)",
    )

    @property
    def total_open_notional_usd(self) -> float:
        """Suma del valor nocional absoluto de todas las posiciones abiertas."""
        return sum(abs(notional) for notional in self.active_positions.values())

    @property
    def daily_drawdown_pct(self) -> float:
        """Drawdown relativo desde el High-Water Mark diario."""
        return (self.current_equity_usd - self.daily_high_watermark_usd) / self.daily_high_watermark_usd

    @property
    def daily_net_return_pct(self) -> float:
        """Rendimiento neto porcentual respecto al capital base operativo."""
        return (self.current_equity_usd - self.base_capital_usd) / self.base_capital_usd
