# scripts/download_2021_2022_data.py
"""
Descarga y almacena en cache exactamente el ciclo historico 2021-2022
(2021-01-01 00:00:00 UTC a 2022-12-31 23:59:59 UTC, 730 dias = ~70,080 velas de 15m)
de Binance USD(S)-M Futures para la cesta de activos.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import ccxt

TARGET_SYMBOLS = [
    ("BTC/USDT:USDT", "BTC", "binance_BTC_USDT_2021_2022_15m.json"),
    ("ETH/USDT:USDT", "ETH", "binance_ETH_USDT_2021_2022_15m.json"),
    ("SOL/USDT:USDT", "SOL", "binance_SOL_USDT_2021_2022_15m.json"),
    ("BNB/USDT:USDT", "BNB", "binance_BNB_USDT_2021_2022_15m.json"),
    ("DOGE/USDT:USDT", "DOGE", "binance_DOGE_USDT_2021_2022_15m.json"),
    ("XRP/USDT:USDT", "XRP", "binance_XRP_USDT_2021_2022_15m.json"),
    ("AVAX/USDT:USDT", "AVAX", "binance_AVAX_USDT_2021_2022_15m.json"),
    ("LINK/USDT:USDT", "LINK", "binance_LINK_USDT_2021_2022_15m.json"),
    ("NEAR/USDT:USDT", "NEAR", "binance_NEAR_USDT_2021_2022_15m.json"),
    ("ADA/USDT:USDT", "ADA", "binance_ADA_USDT_2021_2022_15m.json"),
    ("DOT/USDT:USDT", "DOT", "binance_DOT_USDT_2021_2022_15m.json"),
    ("LTC/USDT:USDT", "LTC", "binance_LTC_USDT_2021_2022_15m.json"),
]

START_DT = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
START_MS = int(START_DT.timestamp() * 1000)
END_MS = int(END_DT.timestamp() * 1000)
TARGET_BARS = 730 * 96  # 70,080 barras de 15m


def download_2021_2022(symbols_list=None) -> None:
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    targets = symbols_list or TARGET_SYMBOLS
    print(f"[HISTORICAL FETCHER] Iniciando descarga 2021-2022:")
    print(f"  Inicio: {START_DT.strftime('%Y-%m-%d %H:%M UTC')} ({START_MS})")
    print(f"  Fin:    {END_DT.strftime('%Y-%m-%d %H:%M UTC')} ({END_MS})")
    print(f"  Objetivo por par: {TARGET_BARS:,} barras de 15m\n")

    for market_sym, short_sym, filename in targets:
        cache_file = cache_dir / filename
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_bars = len(cached_data.get("candles", []))
                if cached_bars >= TARGET_BARS - 100:
                    print(f"[CACHE HIT] {short_sym} ya cuenta con {cached_bars:,} velas en {filename} ({cache_file.stat().st_size / (1024*1024):.1f} MB)")
                    continue
            except Exception:
                pass

        print(f"[BINANCE REST] Descargando {market_sym} ({short_sym})...")
        candles = []
        current_since = START_MS
        t0 = time.time()
        batch = 0

        while current_since <= END_MS and len(candles) < TARGET_BARS + 500:
            try:
                chunk = exchange.fetch_ohlcv(market_sym, "15m", since=current_since, limit=1000)
                if not chunk:
                    break
                
                # Filtrar cualquier vela posterior al 2022-12-31 23:59:59
                valid_chunk = [c for c in chunk if c[0] <= END_MS]
                candles.extend(valid_chunk)
                
                # La siguiente consulta parte 15m despues de la ultima barra recibida del chunk
                current_since = chunk[-1][0] + (15 * 60 * 1000)
                batch += 1

                if batch % 15 == 0 or len(valid_chunk) < len(chunk):
                    pct = min(100.0, (len(candles) / TARGET_BARS) * 100.0)
                    last_date = datetime.fromtimestamp(chunk[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                    print(f"  [{short_sym}] {len(candles):,}/{TARGET_BARS:,} ({pct:.1f}%) — Fecha: {last_date}")

                if len(chunk) < 1000 or chunk[-1][0] >= END_MS:
                    break

                time.sleep(0.04)
            except Exception as e:
                print(f"  [WARN] Error en {short_sym} ({e}). Reintentando en 1s...")
                time.sleep(1.0)

        elapsed = time.time() - t0
        print(f"  [OK] {short_sym}: {len(candles):,} velas descargadas en {elapsed:.1f}s")

        payload = {
            "symbol": market_sym,
            "short_symbol": short_sym,
            "timeframe": "15m",
            "period": "2021_2022",
            "start_dt": START_DT.isoformat(),
            "end_dt": END_DT.isoformat(),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "candles": candles,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"  [GUARDADO] {cache_file} ({cache_file.stat().st_size / (1024*1024):.1f} MB)\n")


if __name__ == "__main__":
    download_2021_2022()
