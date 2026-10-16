# scripts/check_and_download_12_assets.py
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import ccxt

SYMBOLS = [
    ("BTC", "BTC/USDT:USDT", "binance_BTC_USDT_2y_15m.json"),
    ("ETH", "ETH/USDT:USDT", "binance_ETH_USDT_2y_15m.json"),
    ("SOL", "SOL/USDT:USDT", "binance_SOL_USDT_2y_15m.json"),
    ("BNB", "BNB/USDT:USDT", "binance_BNB_USDT_2y_15m.json"),
    ("DOGE", "DOGE/USDT:USDT", "binance_DOGE_USDT_2y_15m.json"),
    ("XRP", "XRP/USDT:USDT", "binance_XRP_USDT_2y_15m.json"),
    ("SUI", "SUI/USDT:USDT", "binance_SUI_USDT_2y_15m.json"),
    ("AVAX", "AVAX/USDT:USDT", "binance_AVAX_USDT_2y_15m.json"),
    ("LINK", "LINK/USDT:USDT", "binance_LINK_USDT_2y_15m.json"),
    ("PEPE", "1000PEPE/USDT:USDT", "binance_PEPE_USDT_2y_15m.json"),
    ("NEAR", "NEAR/USDT:USDT", "binance_NEAR_USDT_2y_15m.json"),
    ("ARB", "ARB/USDT:USDT", "binance_ARB_USDT_2y_15m.json"),
]

def run():
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    markets = exchange.load_markets()

    now = datetime.now(timezone.utc)
    two_years_ago = now - timedelta(days=730)
    since_ms = int(two_years_ago.timestamp() * 1000)
    target_count = 730 * 96  # 70,080 velas

    for name, symbol, filename in SYMBOLS:
        cache_file = cache_dir / filename
        if cache_file.exists() and cache_file.stat().st_size > 100_000:
            print(f"[CACHE OK] {name} ({symbol}) ya existe: {cache_file.stat().st_size / (1024*1024):.1f} MB")
            continue

        if symbol not in markets:
            # Fallback check
            alt_sym = f"{name}/USDT:USDT"
            if alt_sym in markets:
                symbol = alt_sym

        print(f"\n[DESCARGANDO] {name} ({symbol}) desde Binance USD(S)-M...")
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
                    print(f"  [{name}] {len(all_candles):,}/{target_count:,} ({pct:.1f}%)")
                if len(c) < 1000 or current_since >= int(now.timestamp() * 1000):
                    break
                time.sleep(0.03)
            except Exception as e:
                print(f"  [ERROR] {e}. Reintentando...")
                time.sleep(1.0)

        elapsed = time.time() - t0
        print(f"  [COMPLETADO] {name}: {len(all_candles):,} velas en {elapsed:.1f}s")
        payload = {
            "symbol": symbol,
            "name": name,
            "timeframe": "15m",
            "days": 730,
            "downloaded_at": now.isoformat(),
            "candles": all_candles,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"  [GUARDADO] {cache_file}")

if __name__ == "__main__":
    run()
