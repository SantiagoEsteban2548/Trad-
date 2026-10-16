# scripts/download_all_5_assets.py
"""
Descarga y almacena en caché 730 días (2 años) de velas de 15m para la cesta completa
de 5 activos de Binance Futures: BTC, ETH, SOL, BNB, DOGE.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import ccxt

SYMBOLS = [
    ("BTC/USDT:USDT", "binance_BTC_USDT_2y_15m.json"),
    ("ETH/USDT:USDT", "binance_ETH_USDT_2y_15m.json"),
    ("SOL/USDT:USDT", "binance_SOL_USDT_2y_15m.json"),
    ("BNB/USDT:USDT", "binance_BNB_USDT_2y_15m.json"),
    ("DOGE/USDT:USDT", "binance_DOGE_USDT_2y_15m.json"),
]


def download_all_assets() -> None:
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    now = datetime.now(timezone.utc)
    two_years_ago = now - timedelta(days=730)
    since_ms = int(two_years_ago.timestamp() * 1000)
    target_count = 730 * 96  # 70,080 velas

    for symbol, filename in SYMBOLS:
        cache_file = cache_dir / filename
        if cache_file.exists():
            print(f"[CACHE] {symbol} ya existe en {cache_file} ({cache_file.stat().st_size / (1024*1024):.1f} MB)")
            continue

        print(f"\n[BINANCE REST] Descargando {symbol} (730 días de 15m)...")
        all_candles = []
        current_since = since_ms
        t0 = time.time()
        batch = 0

        while len(all_candles) < target_count:
            try:
                c = exchange.fetch_ohlcv(symbol, "15m", since=current_since, limit=1000)
                if not c:
                    break
                all_candles.extend(c)
                current_since = c[-1][0] + (15 * 60 * 1000)
                batch += 1
                if batch % 15 == 0:
                    pct = (len(all_candles) / target_count) * 100.0
                    print(f"  [{symbol}] {len(all_candles):,}/{target_count:,} ({pct:.1f}%)")
                if len(c) < 1000 or current_since >= int(now.timestamp() * 1000):
                    break
                time.sleep(0.04)
            except Exception as e:
                print(f"  [ERROR] {e}. Reintentando en 1s...")
                time.sleep(1.0)

        elapsed = time.time() - t0
        print(f"  [OK] {symbol}: {len(all_candles):,} velas descargadas en {elapsed:.1f}s")

        payload = {
            "symbol": symbol,
            "timeframe": "15m",
            "days": 730,
            "downloaded_at": now.isoformat(),
            "candles": all_candles,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"  [CACHE] Guardado en {cache_file} ({cache_file.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    download_all_assets()
