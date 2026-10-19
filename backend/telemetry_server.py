# backend/telemetry_server.py
"""
Servidor de Telemetría en Tiempo Real (FastAPI + WebSockets) para el modo DRY_RUN_L2.
Conecta el motor cuantitativo de microestructura (Fases 1 a 5) con el Centro de Comando en React.

Flujo de producción sin datos falsos (Zero-Mock):
1. FastAPI `lifespan` levanta `WebSocketMultiplexer` conectado concurrentemente a CCXT Pro en futuros perpetuos.
2. `ws_multiplexer.py` alimenta `MarketRingBuffer` con L2 Top 5, trades y liquidaciones.
3. El bucle de ejecución evalúa síncronamente `avellaneda_obi.py`, `liq_cascade.py` y `funding_arb.py`.
4. Cada orden se valida contra `MandateGatekeeper.validate_order()` y se despacha a `L2QueueSimulator`.
5. WebSocket `/ws/telemetry` transmite la telemetría viva al Frontend y `POST /api/emergency-flatten` controla el Kill-Switch.

Ejecución local:
    uvicorn backend.telemetry_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import io
import inspect
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Literal
from pydantic import BaseModel
import ccxt
import ccxt.pro as ccxtpro
import ccxt.async_support as ccxt_async

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from dotenv import load_dotenv

from backend.security.auth import verify_bearer_token, verify_websocket_token

load_dotenv()

from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.ingestion.ws_multiplexer import WebSocketMultiplexer
from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.schemas import OrderRequest, PortfolioState, truncate_decimal_step
from backend.mandate_gates.profit_sweeper import (
    DAILY_TARGET_RETURN_PCT,
    should_trigger_profit_sweep,
    step_up_ratchet,
)
from backend.database.ledger_store import LedgerSqliteStore
from backend.strategies.avellaneda_obi import AvellanedaStoikovOBIEngine
from backend.strategies.funding_arb import (
    FundingArbSnapshot,
    FundingRateArbitrageEngine,
    CrossExchangeFundingSnapshot,
    CrossExchangeArbPackage,
)
from backend.strategies.liq_cascade import LiquidationCascadeHunter
from backend.strategies.strategy_ledger import VirtualStrategyLedger, VirtualPositionLot
from backend.strategies.leverage_controller import AutonomousLeverageController
import os
from backend.strategies.regime_config import RegimeConfig
from backend.strategies.directional_trend import DirectionalTrendStrategy

CROSS_HEDGE_PAIRS: set[str] = {
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
}

# Warm Path: Gemini Regime Council integration
try:
    from backend.regime_council.gemini_debate import (
        GeminiRegimeCouncil,
        WarmPathCouncilScheduler,
    )
except ImportError:
    GeminiRegimeCouncil = None  # type: ignore[assignment, misc]
    WarmPathCouncilScheduler = None  # type: ignore[assignment, misc]
from backend.validation.l2_simulator_and_dsr import DeflatedSharpeAuditor, DSRAuditResult, L2QueueSimulator

logger = logging.getLogger("telemetry_server")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("backend/config/regime_config.json")



# =============================================================================
# MODULAR ARQUITECTURE RE-EXPORTS (Fase 6)
# =============================================================================
from backend.models.market_specs import (
    SymbolMarketSpec,
    SYMBOL_SPECS,
    LEVERAGE_TIERS_CONFIG,
)
from backend.gateways.binance_gateway import BinanceLiveWsGateway
from backend.gateways.bingx_gateway import BingXLiveWsGateway
from backend.gateways.dry_run_gateway import DryRunWsGateway
from backend.engine.specs import (
    SymbolExecutionContext,
    ContagionShieldState,
    is_tradfi_market_open,
    GLOBAL_MOCK_TRADFI_STATE,
    GLOBAL_MOCK_TRADFI_CLOSED,
)
from backend.engine.execution_engine import DryRunExecutionEngine

__all__ = [
    "BinanceLiveWsGateway",
    "BingXLiveWsGateway",
    "DryRunWsGateway",
    "SymbolMarketSpec",
    "SYMBOL_SPECS",
    "LEVERAGE_TIERS_CONFIG",
    "SymbolExecutionContext",
    "ContagionShieldState",
    "is_tradfi_market_open",
    "DryRunExecutionEngine",
    "engine",
    "app",
]

engine = DryRunExecutionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de FastAPI:
    Inicia WebSocketMultiplexer real y el bucle continuo de evaluación cuantitativa.
    """
    logger.info("Iniciando lifespan de FastAPI...")
    await engine.start()
    try:
        yield
    finally:
        logger.info("Cerrando lifespan de FastAPI...")
        await engine.stop()


app = FastAPI(
    title="Apex Quant HFT Telemetry Server",
    description="Real-time WebSocket stream for DRY_RUN_L2 Paper Trading, Mandate Gates, OBI & DSR Quarantine.",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_allowed = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradingModeRequest(BaseModel):
    mode: Literal["PAPER_L2", "LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"]
    confirm: str | None = None


class BaseCapitalRequest(BaseModel):
    base_capital_usd: float
    confirm: str | None = None


class ResetStateRequest(BaseModel):
    base_capital_usd: float | None = None
    confirm: str | None = None


class ResetKillSwitchRequest(BaseModel):
    confirm: str | None = None


class LeverageTierRequest(BaseModel):
    leverage: Literal[3, 5, 7, 10]


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Chequeo de salud público y sanitizado sin divulgación de datos sensibles."""
    uptime_s = round(time.time() - getattr(engine, "_start_ts_s", time.time()), 2)
    return {
        "status": "OK",
        "mode": engine.trading_mode,
        "uptime_s": uptime_s,
        "timestamp_ns": time.time_ns(),
    }


class StrategyModeRequest(BaseModel):
    strategy_mode: Literal["MARKET_MAKING", "DIRECTIONAL", "DIRECTIONAL_TREND"]


@app.get("/api/strategy-mode", dependencies=[Depends(verify_bearer_token)])
async def get_strategy_mode_endpoint() -> dict[str, Any]:
    """Retorna el modo de estrategia activo (MARKET_MAKING o DIRECTIONAL_TREND)."""
    return {
        "status": "SUCCESS",
        "strategy_mode": engine.strategy_mode,
        "available_modes": ["MARKET_MAKING", "DIRECTIONAL_TREND"],
    }


@app.post("/api/strategy-mode", dependencies=[Depends(verify_bearer_token)])
async def set_strategy_mode_endpoint(req: StrategyModeRequest) -> dict[str, Any]:
    """Conmuta dinámicamente el modo de estrategia entre Market Making y Direccional 1:3."""
    norm_mode = "DIRECTIONAL_TREND" if req.strategy_mode == "DIRECTIONAL" else req.strategy_mode
    engine.strategy_mode = norm_mode
    logger.info(f"[STRATEGY_MODE] Modo de estrategia conmutado a: {norm_mode}")
    return {"status": "SUCCESS", "strategy_mode": engine.strategy_mode}


@app.get("/api/leverage", dependencies=[Depends(verify_bearer_token)])
async def get_leverage_evaluation_endpoint() -> dict[str, Any]:
    """Devuelve la matriz de evaluación de apalancamiento (x3, x5, x7, x10) en tiempo real."""
    return engine.evaluate_leverage_tiers()


@app.post("/api/leverage", dependencies=[Depends(verify_bearer_token)])
async def set_leverage_tier_endpoint(req: LeverageTierRequest) -> dict[str, Any]:
    """Ajusta el nivel de apalancamiento activo (x3, x5, x7, x10) y recalibra TP/SL y Gate 3."""
    try:
        analysis = await engine.set_leverage_tier(req.leverage)
        return {
            "status": "LEVERAGE_UPDATED",
            "selected_leverage": engine.selected_leverage,
            "leverage_analysis": analysis,
            "timestamp_ns": time.time_ns(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/telemetry", dependencies=[Depends(verify_bearer_token)])
async def get_telemetry_endpoint() -> dict[str, Any]:
    """Snapshot completo de telemetría real de microestructura y riesgo."""
    return engine.get_telemetry_snapshot()


@app.get("/api/trades", dependencies=[Depends(verify_bearer_token)])
async def get_trades_endpoint(limit: int = 1000, symbol: str | None = None) -> dict[str, Any]:
    """
    Retorna el historial de transacciones persistidas en SQLite (o memoria)
    para análisis detallado y exportación, permitiendo consultar hasta miles de registros.
    """
    if hasattr(engine, "db_store") and engine.db_store is not None:
        trades = engine.db_store.load_recent_trades(limit=limit)
    else:
        trades = list(engine.recent_trades)[::-1][:limit]

    if symbol and symbol.upper() != "TODOS":
        trades = [t for t in trades if symbol.upper() in str(t.get("symbol", "")).upper()]

    return {
        "status": "OK",
        "count": len(trades),
        "limit": limit,
        "trades": trades,
    }


@app.get("/api/trades/export-csv", dependencies=[Depends(verify_bearer_token)])
async def export_trades_csv_endpoint(limit: int = 10000, symbol: str | None = None):
    """
    Exporta el historial completo de transacciones en formato CSV descargable
    directamente desde la base de datos transaccional SQLite WAL.
    """
    if hasattr(engine, "db_store") and engine.db_store is not None:
        trades = engine.db_store.load_recent_trades(limit=limit)
    else:
        trades = list(engine.recent_trades)[::-1][:limit]

    if symbol and symbol.upper() != "TODOS":
        trades = [t for t in trades if symbol.upper() in str(t.get("symbol", "")).upper()]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Timestamp_MS",
        "Time_UTC",
        "Symbol",
        "Strategy",
        "Side",
        "Entry_Price",
        "Exit_Mid_Price",
        "Qty",
        "Return_Pct",
        "PnL_USD",
        "Fee_Cost",
        "Exchange",
    ])
    for t in trades:
        writer.writerow([
            t.get("id", ""),
            t.get("timestamp_ms", ""),
            t.get("time_str", ""),
            t.get("symbol", ""),
            t.get("strategy", ""),
            t.get("side", ""),
            t.get("entry_price", 0.0),
            t.get("exit_or_mid_price", 0.0),
            t.get("qty", 0.0),
            t.get("return_pct", 0.0),
            t.get("pnl_usd", 0.0),
            t.get("fee_cost", 0.0),
            t.get("exchange", "BINANCE"),
        ])

    csv_data = output.getvalue()
    filename = f"trades_blotter_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/reset-state", dependencies=[Depends(verify_bearer_token)])
async def reset_state_endpoint(req: ResetStateRequest | None = None, request: Request = None) -> dict[str, Any]:
    """
    Reinicia el estado institucional del motor cuantitativo desde 0:
    Limpia órdenes huérfanas, historial de trades, contadores de victorias/derrotas,
    PnL bruto y neto acumulado, curva de equidad y estado de cuarentena DSR.
    Exige confirm='RESET_STATE'.
    """
    client_ip = request.client.host if request and request.client else "unknown"
    if not req or req.confirm != "RESET_STATE":
        logger.warning(f"[SECURITY_AUDIT] Intento de reset-state sin confirmación válida desde IP: {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Operación destructiva: Debe enviar confirm='RESET_STATE' en el cuerpo de la solicitud.",
        )

    logger.warning(f"[SECURITY_AUDIT] reset-state confirmado ejecutado desde IP: {client_ip}")
    cap = req.base_capital_usd
    engine.reset_state_from_zero(base_capital_usd=cap)
    return {
        "status": "STATE_RESET_OK",
        "mode": engine.trading_mode,
        "base_capital_usd": engine.portfolio.base_capital_usd,
        "current_equity_usd": engine.portfolio.current_equity_usd,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/trading-mode", dependencies=[Depends(verify_bearer_token)])
async def set_trading_mode_endpoint(req: TradingModeRequest, request: Request = None) -> dict[str, Any]:
    """
    Conmuta el modo de ejecución entre PAPER_L2, LIVE_BINANCE, LIVE_BINGX y LIVE_MULTI.
    REGLA DE SEGURIDAD CRÍTICA:
    Para cualquier modo LIVE exige confirm='LIVE' primero.
    Si se solicita LIVE_BINANCE, verifica que BINANCE_API_KEY y BINANCE_API_SECRET existan.
    Si se solicita LIVE_BINGX, verifica que BINGX_API_KEY y BINGX_API_SECRET existan.
    Si se solicita LIVE_MULTI, verifica ambas credenciales de Binance y BingX.
    Verifica que gatekeeper.kill_switch_engaged sea False.
    Si faltan claves API o el Kill-Switch está activo, rechaza con HTTP 400 y mantiene el bot en PAPER_L2.
    """
    client_ip = request.client.host if request and request.client else "unknown"

    if req.mode == "LIVE_BINANCE":
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            logger.warning("Intento de cambio a LIVE_BINANCE rechazado por falta de claves API.")
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y configuración incompleta: Para cambiar a modo LIVE debe enviar confirm='LIVE' y configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env antes de operar en vivo. El bot se mantiene en PAPER_L2.",
            )
        if engine.gatekeeper.kill_switch_engaged:
            logger.warning("Intento de cambio a LIVE_BINANCE rechazado: Kill-Switch activo.")
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y bloqueo de seguridad: Para cambiar a modo LIVE debe enviar confirm='LIVE'. El Kill-Switch de emergencia está activo. Debe resetear el Kill-Switch antes de pasar a LIVE_BINANCE.",
            )
    elif req.mode == "LIVE_BINGX":
        api_key = os.getenv("BINGX_API_KEY", "").strip()
        api_secret = os.getenv("BINGX_API_SECRET", "").strip()
        if not api_key or not api_secret:
            logger.warning("Intento de cambio a LIVE_BINGX rechazado por falta de claves API.")
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y configuración incompleta: Para cambiar a modo LIVE debe enviar confirm='LIVE' y configurar BINGX_API_KEY y BINGX_API_SECRET en su archivo .env antes de operar en vivo con BingX. El bot se mantiene en PAPER_L2.",
            )
        if engine.gatekeeper.kill_switch_engaged:
            logger.warning("Intento de cambio a LIVE_BINGX rechazado: Kill-Switch activo.")
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y bloqueo de seguridad: Para cambiar a modo LIVE debe enviar confirm='LIVE'. El Kill-Switch de emergencia está activo. Debe resetear el Kill-Switch antes de pasar a LIVE_BINGX.",
            )
    elif req.mode == "LIVE_MULTI":
        bin_key = os.getenv("BINANCE_API_KEY", "").strip()
        bin_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        bx_key = os.getenv("BINGX_API_KEY", "").strip()
        bx_secret = os.getenv("BINGX_API_SECRET", "").strip()
        if not bin_key or not bin_secret:
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y configuración incompleta: Para cambiar a modo LIVE debe enviar confirm='LIVE' y configurar BINANCE_API_KEY y BINANCE_API_SECRET en su archivo .env antes de operar en modo LIVE_MULTI.",
            )
        if not bx_key or not bx_secret:
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y configuración incompleta: Para cambiar a modo LIVE debe enviar confirm='LIVE' y configurar BINGX_API_KEY y BINGX_API_SECRET en su archivo .env antes de operar en modo LIVE_MULTI.",
            )
        if engine.gatekeeper.kill_switch_engaged:
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida y bloqueo de seguridad: Para cambiar a modo LIVE debe enviar confirm='LIVE'. El Kill-Switch de emergencia está activo. Debe resetear el Kill-Switch antes de pasar a LIVE_MULTI.",
            )

    if req.mode.startswith("LIVE"):
        if req.confirm != "LIVE":
            logger.warning(f"[SECURITY_AUDIT] Intento de pasar a {req.mode} sin confirm='LIVE' desde IP: {client_ip}")
            raise HTTPException(
                status_code=400,
                detail="Confirmación requerida: Para cambiar a modo LIVE debe enviar confirm='LIVE'.",
            )
    elif req.confirm and req.confirm not in ("CONFIRM_MODE", "PAPER_L2"):
        raise HTTPException(
            status_code=400,
            detail="Confirmación inválida: Para cambiar de modo debe enviar confirm='CONFIRM_MODE'.",
        )

    logger.warning(f"[SECURITY_AUDIT] Cambio de modo a {req.mode} ejecutado desde IP: {client_ip}")

    try:
        await engine.set_trading_mode(req.mode)
        return {
            "status": "MODE_SWITCHED",
            "mode": engine.trading_mode,
            "timestamp_ns": time.time_ns(),
        }
    except Exception as e:
        logger.error(f"Error cambiando modo a {req.mode}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/emergency-flatten", dependencies=[Depends(verify_bearer_token)])
async def emergency_flatten_endpoint(request: Request = None) -> dict[str, Any]:
    """
    Disparador síncrono del Kill-Switch de emergencia (Gate 4).
    Bloquea inmediatamente todas las órdenes nuevas y activa el flatten idempotente de órdenes y posiciones.
    """
    engine.gatekeeper.trigger_emergency_flatten()
    engine.trigger_emergency_flatten_sync()
    if engine.trading_mode in ("LIVE_BINANCE", "LIVE_BINGX", "LIVE_MULTI"):
        await engine.emergency_flatten_live_gateways()
    logger.warning("Kill-Switch disparado manualmente a través de /api/emergency-flatten.")

    return {
        "status": "EMERGENCY_FLATTEN_TRIGGERED",
        "mode": engine.trading_mode,
        "kill_switch_engaged": engine.gatekeeper.kill_switch_engaged,
        "emergency_flatten_count": engine.gatekeeper.emergency_flatten_count,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/profit-sweep", dependencies=[Depends(verify_bearer_token)])
async def profit_sweep_endpoint() -> dict[str, Any]:
    """
    Dispara el barrido de excedentes de ganancias:
    - En PAPER_L2: descuenta el excedente hacia la bóveda simulada (simulated_vault_usd)
      y resetea el capital operativo a base_capital_usd.
    - En LIVE_BINANCE: ejecuta exchange.transfer('USDT', surplus, 'future', 'spot')
      y resetea el capital operativo a base_capital_usd.
    - En LIVE_BINGX: ejecuta transferencia hacia spot si disponible y resetea a base_capital_usd.
    - En LIVE_MULTI: ejecuta barrido proporcional en ambos exchanges.
    """
    surplus = round(engine.portfolio.current_equity_usd - engine.portfolio.base_capital_usd, 2)
    if surplus <= 0.0:
        return {
            "status": "NO_SURPLUS",
            "mode": engine.trading_mode,
            "surplus_swept_usd": 0.0,
            "current_equity_usd": engine.portfolio.current_equity_usd,
            "base_capital_usd": engine.portfolio.base_capital_usd,
            "simulated_vault_usd": getattr(engine, "simulated_vault_usd", 0.0),
            "message": "No hay excedente de capital operativo para barrer.",
            "timestamp_ns": time.time_ns(),
        }

    if engine.trading_mode == "LIVE_MULTI":
        half_surplus = round(surplus / 2.0, 2)
        if engine.live_gateway is not None and half_surplus > 0:
            try:
                await engine.live_gateway.exchange.transfer(
                    code="USDT", amount=half_surplus, fromAccount="future", toAccount="spot"
                )
                logger.info(f"Barrido en Binance: {half_surplus} USDT a Spot")
            except Exception as e:
                logger.warning(f"Fallo sweep en Binance: {e}")
        if engine.bingx_gateway is not None and half_surplus > 0:
            try:
                ex = engine.bingx_gateway.exchange
                if hasattr(ex, "transfer"):
                    await ex.transfer(code="USDT", amount=half_surplus, fromAccount="swap", toAccount="spot")
                logger.info(f"Barrido en BingX: {half_surplus} USDT a Spot")
            except Exception as e:
                logger.warning(f"Fallo sweep en BingX: {e}")
    elif engine.trading_mode == "LIVE_BINANCE":
        if engine.live_gateway is None:
            raise HTTPException(status_code=400, detail="Gateway de Binance no inicializado.")
        try:
            transfer_res = await engine.live_gateway.exchange.transfer(
                code="USDT",
                amount=surplus,
                fromAccount="future",
                toAccount="spot",
            )
            logger.info(f"Barrido de ganancias real ejecutado en Binance: {surplus} USDT a Spot: {transfer_res}")
        except Exception as e:
            logger.error(f"Fallo en transferencia de futuros a spot en Binance: {e}")
            raise HTTPException(status_code=500, detail=f"Fallo al transferir a Spot en Binance: {e}")
    elif engine.trading_mode == "LIVE_BINGX":
        if engine.bingx_gateway is None:
            raise HTTPException(status_code=400, detail="Gateway de BingX no inicializado.")
        try:
            ex = engine.bingx_gateway.exchange
            if hasattr(ex, "transfer"):
                await ex.transfer(
                    code="USDT",
                    amount=surplus,
                    fromAccount="swap",
                    toAccount="spot",
                )
            logger.info(f"Barrido de ganancias real ejecutado en BingX: {surplus} USDT a Spot")
        except Exception as e:
            logger.error(f"Fallo en transferencia de futuros a spot en BingX: {e}")
            raise HTTPException(status_code=500, detail=f"Fallo al transferir a Spot en BingX: {e}")
    else:
        engine.simulated_vault_usd = round(getattr(engine, "simulated_vault_usd", 0.0) + surplus, 2)
        logger.info(f"Barrido de ganancias simulado ejecutado: {surplus} USDT a bóveda. Total: {engine.simulated_vault_usd}")

    engine.portfolio.current_equity_usd = engine.portfolio.base_capital_usd
    engine.portfolio.daily_high_watermark_usd = engine.portfolio.base_capital_usd

    return {
        "status": "PROFIT_SWEEP_EXECUTED",
        "mode": engine.trading_mode,
        "surplus_swept_usd": surplus,
        "current_equity_usd": engine.portfolio.current_equity_usd,
        "base_capital_usd": engine.portfolio.base_capital_usd,
        "simulated_vault_usd": getattr(engine, "simulated_vault_usd", 0.0),
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/reset-kill-switch", dependencies=[Depends(verify_bearer_token)])
async def reset_kill_switch_endpoint(req: ResetKillSwitchRequest | None = None, request: Request = None) -> dict[str, Any]:
    """Permite al operador reiniciar el Kill-Switch tras auditar la contingencia."""
    client_ip = request.client.host if request and request.client else "unknown"
    if not req or req.confirm != "RESET_KILL_SWITCH":
        logger.warning(f"[SECURITY_AUDIT] Intento de reset-kill-switch sin confirmación desde IP: {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Confirmación requerida: Debe enviar confirm='RESET_KILL_SWITCH' para restablecer el Kill-Switch.",
        )
    logger.warning(f"[SECURITY_AUDIT] Kill-Switch reseteado con confirmación desde IP: {client_ip}")
    engine.gatekeeper.reset_kill_switch()
    return {
        "status": "KILL_SWITCH_RESET",
        "kill_switch_engaged": False,
        "timestamp_ns": time.time_ns(),
    }


@app.post("/api/base-capital", dependencies=[Depends(verify_bearer_token)])
async def set_base_capital_endpoint(req: BaseCapitalRequest, request: Request = None) -> dict[str, Any]:
    """Permite ajustar dinámicamente el capital base operativo asignado a las estrategias."""
    client_ip = request.client.host if request and request.client else "unknown"
    if req.confirm != "UPDATE_CAPITAL":
        logger.warning(f"[SECURITY_AUDIT] Intento de actualizar capital sin confirmación desde IP: {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Confirmación requerida: Debe enviar confirm='UPDATE_CAPITAL' para modificar el capital base operativo.",
        )
    if req.base_capital_usd <= 0:
        raise HTTPException(status_code=400, detail="El capital base debe ser mayor a 0.")
    logger.warning(f"[SECURITY_AUDIT] Capital base operativo actualizado a {req.base_capital_usd} USDT desde IP: {client_ip}")
    engine.portfolio.base_capital_usd = round(req.base_capital_usd, 2)
    engine.portfolio.daily_high_watermark_usd = max(
        engine.portfolio.daily_high_watermark_usd, engine.portfolio.base_capital_usd
    )
    return {
        "status": "BASE_CAPITAL_UPDATED",
        "base_capital_usd": engine.portfolio.base_capital_usd,
        "current_equity_usd": engine.portfolio.current_equity_usd,
        "timestamp_ns": time.time_ns(),
    }



class StressEventRequest(BaseModel):
    event_type: str  # "FLASH_CRASH_5PCT" | "LATENCY_SPIKE_200MS" | "TRADFI_WEEKEND_CLOSE" | "RESET_CHAOS"
    symbol: str | None = "BTC/USDT:USDT"
    confirm: str | None = None


@app.post("/api/stress-event", dependencies=[Depends(verify_bearer_token)])
async def inject_stress_event_endpoint(req: StressEventRequest) -> dict[str, Any]:
    """
    Sub-panel de 'Chaos Injection' para PAPER_L2:
    Permite inyectar a voluntad un choque de liquidez (FLASH_CRASH_5PCT, LATENCY_SPIKE_200MS, TRADFI_WEEKEND_CLOSE)
    para verificar visualmente la reacción instantánea de los 4 Mandate Gates y del ContagionShield.
    """
    if engine.trading_mode != "PAPER_L2":
        raise HTTPException(
            status_code=403,
            detail="Chaos injection and stress events are strictly forbidden outside PAPER_L2 mode.",
        )
    if req.confirm != "INJECT_CHAOS":
        raise HTTPException(
            status_code=400,
            detail="Se requiere payload con 'confirm': 'INJECT_CHAOS' para inyectar eventos de estrés.",
        )
    global GLOBAL_MOCK_TRADFI_CLOSED, GLOBAL_CHAOS_LATENCY_ACTIVE

    ev_type = req.event_type.upper().strip()
    sym = req.symbol or engine.primary_symbol

    if ev_type == "FLASH_CRASH_5PCT":
        ctx = engine.symbol_contexts.get(sym, engine.symbol_contexts[engine.primary_symbol])
        if ctx.ring_buffer.ob_count > 0:
            idx = (ctx.ring_buffer.ob_head - 1) % ctx.ring_buffer.capacity
            cur_mid = 0.5 * (float(ctx.ring_buffer.ob_bid_prices[idx, 0]) + float(ctx.ring_buffer.ob_ask_prices[idx, 0]))
            crash_px = cur_mid * 0.95
            for lvl in range(5):
                ctx.ring_buffer.ob_bid_prices[idx, lvl] = round(crash_px - (lvl * ctx.spec.tick_size * 5), 2)
                ctx.ring_buffer.ob_bid_qtys[idx, lvl] = 0.05
                ctx.ring_buffer.ob_ask_prices[idx, lvl] = round(crash_px + (lvl * ctx.spec.tick_size * 2), 2)
                ctx.ring_buffer.ob_ask_qtys[idx, lvl] = 50.0  # Asks masivos dominando el libro
            now_ns = time.time_ns()
            for _ in range(10):
                ctx.ring_buffer.push_trade(crash_px, 1.5, is_buyer_maker=False, timestamp_ns=now_ns)
        engine.contagion_shield.bid_shield_active = True
        engine.contagion_shield.btc_obi = -0.85
        engine.contagion_shield.trigger_reason = "FLASH_CRASH_5PCT_INJECTED"
        msg = f"Flash crash de -5% inyectado en {sym}. OBI forzado a -0.85 y ContagionShield activado."

    elif ev_type == "LATENCY_SPIKE_200MS":
        GLOBAL_CHAOS_LATENCY_ACTIVE = not GLOBAL_CHAOS_LATENCY_ACTIVE
        engine.chaos_latency_active = GLOBAL_CHAOS_LATENCY_ACTIVE
        target_lat = 200.0 if GLOBAL_CHAOS_LATENCY_ACTIVE else 15.0
        for ctx in engine.symbol_contexts.values():
            ctx.l2_sim.latency_ms = target_lat
        engine.last_latency_us = target_lat * 1000.0
        msg = f"Pico de latencia a {target_lat}ms {'activado' if GLOBAL_CHAOS_LATENCY_ACTIVE else 'desactivado'}."

    elif ev_type == "TRADFI_WEEKEND_CLOSE":
        GLOBAL_MOCK_TRADFI_CLOSED = not GLOBAL_MOCK_TRADFI_CLOSED
        engine.mock_tradfi_closed = GLOBAL_MOCK_TRADFI_CLOSED
        GLOBAL_MOCK_TRADFI_STATE = False if GLOBAL_MOCK_TRADFI_CLOSED else True
        if GLOBAL_MOCK_TRADFI_CLOSED:
            # Cancelar preventivamente órdenes en activos TradFi
            now_ns = time.time_ns()
            for s_name in ["XAU/USDT:USDT", "XAG/USDT:USDT"]:
                s_ctx = engine.symbol_contexts.get(s_name)
                if s_ctx:
                    for o in list(s_ctx.l2_sim.orders.values()):
                        if o.status in ("OPEN", "PARTIALLY_FILLED", "CANCEL_PENDING"):
                            s_ctx.l2_sim.request_cancel(o.order_id, now_ns)
        # Reevaluar asignación
        engine.compute_multivariate_half_kelly()
        msg = f"Cierre TradFi de Fin de Semana {'FORZADO (CERRADO)' if GLOBAL_MOCK_TRADFI_CLOSED else 'RESTAURADO (ABIERTO)'}. Reasignación a cripto 24/7 actualizada."

    elif ev_type == "RESET_CHAOS":
        GLOBAL_MOCK_TRADFI_CLOSED = False
        GLOBAL_MOCK_TRADFI_STATE = None
        GLOBAL_CHAOS_LATENCY_ACTIVE = False
        engine.mock_tradfi_closed = False
        engine.chaos_latency_active = False
        for ctx in engine.symbol_contexts.values():
            ctx.l2_sim.latency_ms = 15.0
        engine.contagion_shield.bid_shield_active = False
        engine.contagion_shield.trigger_reason = "NORMAL"
        engine.compute_multivariate_half_kelly()
        msg = "Todos los choques de estrés de caos reseteados a condiciones nominales."
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de evento de estrés desconocido: {ev_type}")

    return {
        "status": "STRESS_EVENT_INJECTED",
        "event_type": ev_type,
        "symbol": sym,
        "message": msg,
        "mock_tradfi_closed": GLOBAL_MOCK_TRADFI_CLOSED,
        "latency_spike_active": GLOBAL_CHAOS_LATENCY_ACTIVE,
        "timestamp_ns": time.time_ns(),
    }


class GeminiKeyRequest(BaseModel):
    api_key: str
    confirm: str | None = None


@app.post("/api/gemini-key", dependencies=[Depends(verify_bearer_token)])
async def set_gemini_key_endpoint(req: GeminiKeyRequest) -> dict[str, Any]:
    """Permite configurar o actualizar en caliente la API Key de Gemini para el Warm Path Council."""
    if req.confirm != "UPDATE_KEY":
        raise HTTPException(
            status_code=400,
            detail="Se requiere payload con 'confirm': 'UPDATE_KEY' para actualizar la API key.",
        )
    new_key = req.api_key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="API key no puede estar vacía.")

    os.environ["GEMINI_API_KEY"] = new_key
    updated = False

    if engine.warm_path_council is not None:
        engine.warm_path_council.update_api_key(new_key)
        updated = True
    elif GeminiRegimeCouncil is not None and WarmPathCouncilScheduler is not None:
        try:
            engine.warm_path_council = GeminiRegimeCouncil(
                api_key=new_key,
                config_output_path=CONFIG_PATH,
            )
            engine.warm_path_scheduler = WarmPathCouncilScheduler(
                council=engine.warm_path_council,
                summary_provider_fn=engine.get_15m_microstructure_summary,
                interval_seconds=900.0,
            )
            if engine.running:
                engine.warm_path_scheduler.start()
            updated = True
        except Exception as e:
            logger.warning(f"Error instanciando WarmPathCouncil: {e}")

    return {
        "status": "SUCCESS",
        "message": "Gemini API key configurada correctamente.",
        "council_active": updated,
    }


@app.websocket("/ws/telemetry")
async def telemetry_websocket_endpoint(websocket: WebSocket) -> None:
    """
    Transmisión de telemetría viva a alta frecuencia (~10 Hz) hacia el Frontend React.
    Envía el estado real de PortfolioState, MarketRingBuffer, Gatekeeper y L2QueueSimulator.
    """
    if not await verify_websocket_token(websocket):
        return
    await websocket.accept()
    logger.info("Cliente WebSocket conectado a /ws/telemetry")
    try:
        while True:
            payload = engine.get_telemetry_snapshot()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.10)  # ~10 Hz actualización fluida
    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado de /ws/telemetry")
    except Exception as e:
        logger.error(f"Error en websocket /ws/telemetry: {e}")


# Servir Frontend React compilado (dist/) en la raíz del servidor
_dist_path = Path(__file__).resolve().parent.parent / "dist"
if _dist_path.exists():
    _assets_path = _dist_path / "assets"
    if _assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_path)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa_frontend(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = _dist_path / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        index_file = _dist_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend dist index.html no encontrado.")
