# scripts/download_multi_period_data.py
"""
Descarga y almacena en caché 730 días (2 años) de velas de 15m y funding rates
de Binance USDⓈ-M Futures para BTC/USDT.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt


def download_2y_data() -> Path:
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "binance_BTC_USDT_2y_15m.json"

    if cache_file.exists():
        print(f"[CACHE] El archivo de 2 años ya existe en {cache_file} ({cache_file.stat().st_size / (1024*1024):.1f} MB)")
        return cache_file

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    now = datetime.now(timezone.utc)
    two_years_ago = now - timedelta(days=730)
    since_ms = int(two_years_ago.timestamp() * 1000)

    print(f"[BINANCE REST] Descargando 730 días (2 años) de velas de 15m para BTC/USDT:USDT...")
    print(f"  Desde: {two_years_ago.strftime('%Y-%m-%d %H:%M UTC')} hasta {now.strftime('%Y-%m-%d %H:%M UTC')}")
    all_candles = []
    current_since = since_ms
    target_count = 730 * 96  # 96 velas de 15m por día = 70,080 velas

    t0 = time.time()
    batch_count = 0
    while len(all_candles) < target_count:
        c = exchange.fetch_ohlcv("BTC/USDT:USDT", "15m", since=current_since, limit=1000)
        if not c:
            break
        all_candles.extend(c)
        current_since = c[-1][0] + (15 * 60 * 1000)
        batch_count += 1
        if batch_count % 10 == 0:
            pct = (len(all_candles) / target_count) * 100.0
            dt_curr = datetime.fromtimestamp(c[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  -> Progreso: {len(all_candles):,}/{target_count:,} velas ({pct:.1f}%) — Fecha actual: {dt_curr}")
        if len(c) < 1000 or current_since >= int(now.timestamp() * 1000):
            break
        time.sleep(0.04)

    elapsed = time.time() - t0
    print(f"[BINANCE REST] Descarga completada: {len(all_candles):,} velas en {elapsed:.1f}s ({len(all_candles)/elapsed:.0f} velas/seg)")

    print("[BINANCE REST] Descargando historial de tasas de financiación...")
    try:
        funding_rates = exchange.fetch_funding_rate_history("BTC/USDT:USDT", limit=100)
    except Exception as e:
        print(f"[WARN] Error descargando funding rates: {e}")
        funding_rates = []

    payload = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "15m",
        "days": 730,
        "downloaded_at": now.isoformat(),
        "candles": all_candles,
        "funding_rates": funding_rates,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[CACHE] Dataset guardado exitosamente en {cache_file} ({cache_file.stat().st_size / (1024*1024):.1f} MB)")
    return cache_file


if __name__ == "__main__":
    download_2y_data()
