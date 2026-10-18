"""
scripts/start_dual_engine.py

Lanzador de producción para ejecución dual o individual de Apex Quant HFT:
  - Instancia 1: Binance USDⓈ-M Futures (Puerto 8000, TRADING_MODE=LIVE_BINANCE, todas las 14 monedas)
  - Instancia 2: BingX Swap & Commodities (Puerto 8001, TRADING_MODE=LIVE_BINGX, todas las 14 monedas)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanzador Dual Apex Quant HFT (Binance & BingX)")
    parser.add_argument(
        "--exchange",
        choices=["binance", "bingx", "both"],
        default="both",
        help="Exchange a ejecutar: 'binance' (:8000), 'bingx' (:8001) o 'both' (ambos en paralelo)",
    )
    args = parser.parse_args()

    python_executable = sys.executable

    procs: list[subprocess.Popen] = []

    if args.exchange in ("binance", "both"):
        env_bin = os.environ.copy()
        env_bin["PORT"] = "8000"
        env_bin["TARGET_EXCHANGE"] = "binance"
        env_bin["TRADING_MODE"] = "LIVE_BINANCE"
        env_bin["QUOTE_ALL_SYMBOLS"] = "true"

        print("[STARTUP] Lanzando Instancia 1: BINANCE USDⓈ-M Futures (:8000, 14 monedas)...")
        cmd_bin = [
            python_executable,
            "-m",
            "uvicorn",
            "backend.telemetry_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        p_bin = subprocess.Popen(cmd_bin, env=env_bin)
        procs.append(p_bin)

    if args.exchange in ("bingx", "both"):
        env_bx = os.environ.copy()
        env_bx["PORT"] = "8001"
        env_bx["TARGET_EXCHANGE"] = "bingx"
        env_bx["TRADING_MODE"] = "LIVE_BINGX"
        env_bx["QUOTE_ALL_SYMBOLS"] = "true"

        print("[STARTUP] Lanzando Instancia 2: BINGX Perpetual Swap & Commodities (:8001, 14 monedas)...")
        cmd_bx = [
            python_executable,
            "-m",
            "uvicorn",
            "backend.telemetry_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ]
        p_bx = subprocess.Popen(cmd_bx, env=env_bx)
        procs.append(p_bx)

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupción recibida. Deteniendo instancias...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        print("[SHUTDOWN] Instancias finalizadas.")


if __name__ == "__main__":
    main()
