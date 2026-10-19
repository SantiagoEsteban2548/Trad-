"""
Módulo de seguridad institucional para Apex Quant HFT:
Autenticación Bearer token para API REST y WebSockets, sanitización y auditoría de accesos.
"""
import os
import time
import logging
from typing import Annotated
from fastapi import Request, WebSocket, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("ApexQuant.Security")

security_scheme = HTTPBearer(auto_error=False)


def get_configured_auth_token() -> str:
    """Obtiene el token de autenticación configurado en las variables de entorno."""
    return os.getenv("API_AUTH_TOKEN", "").strip()


def verify_bearer_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security_scheme)] = None,
) -> str:
    """
    Dependencia de FastAPI para validar el Bearer token en endpoints REST.
    Si API_AUTH_TOKEN está definido en el entorno, exige coincidencia exacta.
    Si no está configurado y el sistema opera en modo LIVE, bloquea con HTTP 403.
    """
    configured_token = get_configured_auth_token()

    # Si se configuró un token en el entorno, es estrictamente obligatorio
    if configured_token:
        token = credentials.credentials if credentials and credentials.credentials else request.query_params.get("token")
        if not token:
            client_ip = request.client.host if request and request.client else "unknown"
            logger.warning(f"[AUTH_REJECT] Intento de acceso sin token Bearer desde IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No autorizado: Token Bearer ausente o malformado.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if token != configured_token:
            client_ip = request.client.host if request and request.client else "unknown"
            logger.warning(f"[AUTH_REJECT] Token Bearer inválido desde IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No autorizado: Token Bearer inválido.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token

    # Si no hay token configurado en variables de entorno:
    trading_mode = os.getenv("TRADING_MODE", "PAPER_L2").upper()
    if trading_mode.startswith("LIVE"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bloqueo de seguridad: Se requiere configurar API_AUTH_TOKEN para operar en modo LIVE.",
        )

    return ""


async def verify_websocket_token(websocket: WebSocket) -> bool:
    """
    Valida el token en conexiones WebSocket mediante query param ?token= o header Authorization.
    """
    configured_token = get_configured_auth_token()
    if not configured_token:
        trading_mode = os.getenv("TRADING_MODE", "PAPER_L2").upper()
        if trading_mode.startswith("LIVE"):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return False
        return True

    # 1. Intentar desde query param (?token=...)
    token = websocket.query_params.get("token")

    # 2. Intentar desde headers
    if not token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if token != configured_token:
        client_ip = websocket.client.host if websocket.client else "unknown"
        logger.warning(f"[WS_AUTH_REJECT] Conexión WebSocket rechazada por token inválido desde IP: {client_ip}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    return True
