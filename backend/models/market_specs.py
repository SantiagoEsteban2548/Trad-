# backend/models/market_specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SymbolMarketSpec:
    symbol: str
    tick_size: float
    step_size: float
    base_asset: str
    quote_asset: str = "USDT"
    asset_class: str = "CRYPTO"  # "CRYPTO" | "TRADFI_COMMODITY"
    exchange: str = "binance"    # "binance" | "bingx"
    min_spread_bps: float = 10.0
    min_notional: float = 5.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0


LEVERAGE_TIERS_CONFIG: dict[int, dict[str, Any]] = {
    3: {"label": "x3 (Conservador Institucional)", "tp_bps": 22.0, "sl_bps": 35.0, "liq_dist_bps": 3250.0},
    5: {"label": "x5 (Equilibrado / Compuesto)", "tp_bps": 18.0, "sl_bps": 25.0, "liq_dist_bps": 1920.0},
    7: {"label": "x7 (Agresivo / Momentum OBI)", "tp_bps": 14.0, "sl_bps": 18.0, "liq_dist_bps": 1350.0},
    10: {"label": "x10 (Escalpado HFT Estricto)", "tp_bps": 10.0, "sl_bps": 12.0, "liq_dist_bps": 920.0},
}


SYMBOL_SPECS: dict[str, SymbolMarketSpec] = {
    "BTC/USDT:USDT": SymbolMarketSpec("BTC/USDT:USDT", tick_size=0.10, step_size=0.001, base_asset="BTC", asset_class="CRYPTO", exchange="binance", min_notional=100.0, min_spread_bps=10.0),
    "ETH/USDT:USDT": SymbolMarketSpec("ETH/USDT:USDT", tick_size=0.01, step_size=0.001, base_asset="ETH", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "SOL/USDT:USDT": SymbolMarketSpec("SOL/USDT:USDT", tick_size=0.01, step_size=0.1, base_asset="SOL", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "BNB/USDT:USDT": SymbolMarketSpec("BNB/USDT:USDT", tick_size=0.01, step_size=0.01, base_asset="BNB", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "DOGE/USDT:USDT": SymbolMarketSpec("DOGE/USDT:USDT", tick_size=0.00001, step_size=1.0, base_asset="DOGE", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "XRP/USDT:USDT": SymbolMarketSpec("XRP/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="XRP", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "SUI/USDT:USDT": SymbolMarketSpec("SUI/USDT:USDT", tick_size=0.0001, step_size=0.1, base_asset="SUI", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "AVAX/USDT:USDT": SymbolMarketSpec("AVAX/USDT:USDT", tick_size=0.001, step_size=1.0, base_asset="AVAX", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "LINK/USDT:USDT": SymbolMarketSpec("LINK/USDT:USDT", tick_size=0.001, step_size=0.01, base_asset="LINK", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "PEPE/USDT:USDT": SymbolMarketSpec("PEPE/USDT:USDT", tick_size=0.0000001, step_size=1.0, base_asset="1000PEPE", asset_class="CRYPTO", exchange="bingx", min_notional=2.0, min_spread_bps=10.0),
    "NEAR/USDT:USDT": SymbolMarketSpec("NEAR/USDT:USDT", tick_size=0.001, step_size=1.0, base_asset="NEAR", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    "ARB/USDT:USDT": SymbolMarketSpec("ARB/USDT:USDT", tick_size=0.00001, step_size=0.1, base_asset="ARB", asset_class="CRYPTO", exchange="binance", min_notional=5.0, min_spread_bps=10.0),
    # BingX TradFi Commodities (soportadas también en Binance USDⓈ-M)
    "XAU/USDT:USDT": SymbolMarketSpec("XAU/USDT:USDT", tick_size=0.10, step_size=0.01, base_asset="XAU", asset_class="TRADFI_COMMODITY", exchange="bingx", min_notional=5.0, min_spread_bps=6.0),
    "XAG/USDT:USDT": SymbolMarketSpec("XAG/USDT:USDT", tick_size=0.01, step_size=0.1, base_asset="XAG", asset_class="TRADFI_COMMODITY", exchange="bingx", min_notional=5.0, min_spread_bps=6.5),
}
