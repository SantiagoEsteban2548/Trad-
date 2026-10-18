# backend/strategies/regime_config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class WsOrderGateway(Protocol):
    """
    Protocolo del bus de ejecución WebSocket asíncrono de ultra-baja latencia.
    Prohibido el uso de llamadas HTTP REST síncronas.
    """

    async def cancel_order_ws(self, order_id: str, symbol: str) -> bool: ...

    async def place_order_ws(self, payload: dict[str, Any]) -> str: ...


class RegimeConfig(BaseModel):
    """
    Snapshot en memoria RAM de `backend/config/regime_config.json`.
    Leído en 0 ns por los motores del Hot Path sin tocar el sistema de archivos.
    Soporta interoperabilidad directa con el esquema `RegimeDecision` de Gemini Council.
    """

    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    gamma: float = Field(
        default=0.15,
        gt=0.0,
        validation_alias=AliasChoices("gamma", "avellaneda_gamma"),
        description="Aversión al riesgo de Avellaneda-Stoikov",
    )
    kappa: float = Field(
        default=1.50,
        gt=0.0,
        validation_alias=AliasChoices("kappa", "order_book_intensity_kappa"),
        description="Densidad de liquidez del libro L2",
    )
    time_horizon_tau: float = Field(
        default=1.0, gt=0.0, description="Horizonte remanente normalizado (T - t)"
    )
    obi_shield_threshold: float = Field(
        default=0.60,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("obi_shield_threshold", "obi_cancel_threshold"),
        description="Umbral de flujo tóxico OBI",
    )
    obi_volatility_shift_mult: float = Field(
        default=2.5,
        gt=0.0,
        description="Múltiplo de volatilidad para desplazamiento defensivo de cotización",
    )
    min_annualized_funding_yield: float = Field(
        default=0.18,
        gt=0.0,
        description="Rendimiento anualizado mínimo (APR) para arbitraje de funding",
    )
    min_basis_spread_bps: float = Field(
        default=-5.0, description="Basis mínimo (Perp vs Spot) en puntos básicos"
    )
    market_regime: str = Field(
        default="LOW_VOL_RANGE", description="Régimen dominante diagnosticado"
    )
    approved_symbols: list[str] = Field(
        default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    )
    max_leverage_cap: float = Field(
        default=4.0, description="Tope máximo de apalancamiento"
    )

    @classmethod
    def load_from_file(
        cls, path: str | Path = "backend/config/regime_config.json"
    ) -> RegimeConfig:
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def default_normal(cls) -> RegimeConfig:
        """Retorna una configuración canónica por defecto para régimen normal/rango."""
        return cls()

DEFAULT_REGIME_CONFIG_PATH = Path("backend/config/regime_config.json")
