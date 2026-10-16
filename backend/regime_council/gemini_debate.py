# backend/regime_council/gemini_debate.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Literal

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field

from backend.strategies.regime_config import DEFAULT_REGIME_CONFIG_PATH

logger = logging.getLogger("regime_council.gemini_debate")


# =============================================================================
# 1. ESQUEMAS PYDANTIC (Entrada Microestructural y Salida Estructurada)
# =============================================================================

MarketRegimeType = Literal[
    "LOW_VOL_RANGE",
    "HIGH_VOL_LIQ_HUNT",
    "TOXIC_TREND_PAUSE",
    "FUNDING_ARB_FAVORABLE",
]


class Symbol15mSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    realized_vol_5m: float = Field(..., description="Volatilidad realizada anualizada de 5m")
    avg_l2_spread_bps: float = Field(..., description="Spread medio L2 en puntos básicos")
    mean_obi_5_levels: float = Field(..., description="OBI medio de los primeros 5 niveles [-1.0, 1.0]")
    cumulative_trade_volume_usd: float = Field(..., ge=0.0)
    liquidation_clusters_count: int = Field(..., ge=0)
    net_liquidation_volume_usd: float = Field(..., description="Volumen neto liquidado (+ = Longs, - = Shorts)")
    projected_funding_rate_8h: float = Field(..., description="Tasa de financiación proyectada de 8h")
    basis_bps: float = Field(..., description="Diferencial Spot-Perpetuo en puntos básicos")


class MicrostructureSummary15m(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    symbols_data: list[Symbol15mSummary]
    portfolio_equity_usd: float
    current_drawdown_pct: float


class RegimeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    market_regime: MarketRegimeType = Field(
        ...,
        description="Diagnóstico dominante del régimen de microestructura.",
    )
    avellaneda_gamma: float = Field(
        ...,
        ge=0.05,
        le=1.50,
        description="Parámetro de aversión al riesgo de inventario acotado estrictamente entre 0.05 y 1.50.",
    )
    order_book_intensity_kappa: float = Field(
        ...,
        gt=0.0,
        description="Densidad de liquidez del libro L2 (tasa de decaimiento lambda(delta)).",
    )
    obi_cancel_threshold: float = Field(
        ...,
        ge=0.45,
        le=0.80,
        description="Umbral de asimetría de libro L2 para disparo del OBI Shield (acotado entre 0.45 y 0.80).",
    )
    approved_symbols: list[str] = Field(
        ...,
        min_length=0,
        description="Lista de pares autorizados para operar durante la ventana de 15 minutos.",
    )
    max_leverage_cap: float = Field(
        ...,
        ge=0.0,
        le=3.5,
        description="Techo máximo absoluto de apalancamiento permitido (acotado entre 0.0 y 3.5x).",
    )
    risk_rationale: str = Field(
        ...,
        min_length=10,
        description="Justificación cuantitativa y dialéctica de la decisión aprobada.",
    )


# =============================================================================
# 2. SYSTEM PROMPTS DEL FLUJO ADVERSARIAL
# =============================================================================

SYSTEM_PROMPT_ALPHA_HUNTER = (
    "Eres un investigador cuantitativo de microestructura. Analiza el resumen JSON "
    "de volatilidad realizada, spreads medios, tasas de financiación y clusters de "
    "liquidación de los últimos 15 minutos. Propón los pares y parámetros de "
    "Avellaneda-Stoikov (gamma, kappa) óptimos para capturar entre 0.08% y 0.12% por rotación."
)

SYSTEM_PROMPT_RISK_PROSECUTOR = (
    "Eres el Director de Riesgo Cuantitativo y Auditor de Selección Adversa. Tu único "
    "trabajo es encontrar fallos fatales en la propuesta del Alpha Hunter. Busca "
    "señales de flujo tóxico unidireccional, caída de liquidez en el libro L2, "
    "ensanchamiento anormal de spreads o riesgo de cascada sin rebote. Si detectas peligro, "
    "exige subir gamma, estrechar el umbral OBI o pausar el par (TOXIC_TREND_PAUSE)."
)

SYSTEM_PROMPT_CHIEF_ARBITER = (
    "Evalúa el informe del Alpha Hunter y la refutación del Risk Prosecutor. En caso "
    "de duda estadística, prioriza siempre la preservación del capital y emite el "
    "JSON final validado contra el esquema `RegimeDecision`."
)


# =============================================================================
# 3. MOTOR DEL CONSEJO ADVERSARIAL (GEMINI 3-STEP FLOW)
# =============================================================================

class GeminiRegimeCouncil:
    PRIMARY_MODEL: str = "gemini-3.8-flash"
    FALLBACK_MODEL: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0.1

    def __init__(
        self,
        api_key: str | None = None,
        config_output_path: str | Path = DEFAULT_REGIME_CONFIG_PATH,
        primary_model: str = PRIMARY_MODEL,
        fallback_model: str = FALLBACK_MODEL,
    ) -> None:
        self.client: genai.Client = genai.Client(api_key=api_key)
        self.output_path: Path = Path(config_output_path)
        self.primary_model: str = primary_model
        self.fallback_model: str = fallback_model

    def _call_model_with_fallback(
        self,
        contents: str,
        system_instruction: str,
        response_schema: type[BaseModel] | None = None,
    ) -> tuple[str, Any]:
        cfg_kwargs: dict[str, Any] = {
            "temperature": self.TEMPERATURE,
            "system_instruction": system_instruction,
        }

        if response_schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**cfg_kwargs)

        for attempt, model_name in enumerate([self.primary_model, self.fallback_model]):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return model_name, response
            except (APIError, Exception) as exc:
                if attempt == 0:
                    logger.warning(
                        f"Fallo en modelo primario '{model_name}': {exc}. "
                        f"Intentando fallback en '{self.fallback_model}'..."
                    )
                else:
                    logger.error(f"Fallo crítico en modelo de respaldo '{model_name}': {exc}")
                    raise

        raise RuntimeError("No se pudo obtener respuesta de ningún modelo configurado.")

    def run_debate_cycle(self, summary: MicrostructureSummary15m) -> RegimeDecision:
        summary_json_str = summary.model_dump_json(indent=2)
        logger.info("Iniciando ciclo de debate adversarial de 15 minutos...")

        hunter_prompt = (
            f"Analiza la siguiente microestructura de mercado de los últimos 15 minutos:\n"
            f"{summary_json_str}\n\n"
            f"Propón los pares viables y parámetros de Avellaneda-Stoikov (gamma, kappa, spread)."
        )
        model_used, hunter_resp = self._call_model_with_fallback(
            contents=hunter_prompt,
            system_instruction=SYSTEM_PROMPT_ALPHA_HUNTER,
        )
        hunter_text = hunter_resp.text or ""

        prosecutor_prompt = (
            f"DATOS DE MERCADO:\n{summary_json_str}\n\n"
            f"PROPUESTA DEL ALPHA HUNTER:\n{hunter_text}\n\n"
            f"Identifica riesgos fatales, flujo tóxico, liquidez fantasma y exige ajustes."
        )
        model_used, prosecutor_resp = self._call_model_with_fallback(
            contents=prosecutor_prompt,
            system_instruction=SYSTEM_PROMPT_RISK_PROSECUTOR,
        )
        prosecutor_text = prosecutor_resp.text or ""

        arbiter_prompt = (
            f"RESUMEN DE MERCADO (15m):\n{summary_json_str}\n\n"
            f"PROPUESTA DEL ALPHA HUNTER:\n{hunter_text}\n\n"
            f"REFUTACIÓN DEL RISK PROSECUTOR:\n{prosecutor_text}\n\n"
            f"Emite el veredicto final preservando capital bajo duda estadística."
        )
        model_used, arbiter_resp = self._call_model_with_fallback(
            contents=arbiter_prompt,
            system_instruction=SYSTEM_PROMPT_CHIEF_ARBITER,
            response_schema=RegimeDecision,
        )

        raw_text = arbiter_resp.text or "{}"
        try:
            decision = RegimeDecision.model_validate_json(raw_text)
        except Exception:
            data = json.loads(raw_text)
            decision = RegimeDecision.model_validate(data)

        self.save_decision_atomically(decision)
        return decision

    def save_decision_atomically(self, decision: RegimeDecision) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        dir_name = self.output_path.parent

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=dir_name,
            delete=False,
            encoding="utf-8",
            suffix=".tmp",
        ) as tmp_file:
            temp_name = tmp_file.name
            json.dump(decision.model_dump(), tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(temp_name, self.output_path)
        logger.info(f"Configuración de régimen actualizada atómicamente en: {self.output_path}")


# =============================================================================
# 4. TAREA EN SEGUNDO PLANO (WARM PATH SCHEDULER: 15 MINUTOS)
# =============================================================================

class WarmPathCouncilScheduler:
    def __init__(
        self,
        council: GeminiRegimeCouncil,
        summary_provider_fn: Any,
        interval_seconds: float = 900.0,
    ) -> None:
        self.council = council
        self.summary_provider_fn = summary_provider_fn
        self.interval_seconds = interval_seconds
        self._running: bool = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="WarmPathCouncil")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while self._running:
            try:
                summary = self.summary_provider_fn()
                if summary:
                    self.council.run_debate_cycle(summary)
            except Exception as e:
                logger.error(f"Excepción en el Consejo Warm Path: {e}")

            slept = 0.0
            while self._running and slept < self.interval_seconds:
                time.sleep(1.0)
                slept += 1.0
