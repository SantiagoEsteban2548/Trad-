import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from backend.backtest.engine import InstitutionalBacktestEngine

def run_wf():
    fp = Path("data/cache/binance_BTC_USDT_2021_2022_15m.json")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    candles = data["candles"]
    total = len(candles)
    half = total // 2
    in_sample = candles[:half]      # 2021
    out_sample = candles[half:]     # 2022

    print(f"Total candles: {total}, In-Sample (2021): {len(in_sample)}, Out-of-Sample (2022): {len(out_sample)}")

    # 1. Market Making
    eng_mm_in = InstitutionalBacktestEngine(symbol="BTC/USDT:USDT", initial_capital_usd=1000.0)
    res_mm_in = eng_mm_in.run(in_sample, strategy_mode="MARKET_MAKING")
    
    eng_mm_out = InstitutionalBacktestEngine(symbol="BTC/USDT:USDT", initial_capital_usd=1000.0)
    res_mm_out = eng_mm_out.run(out_sample, strategy_mode="MARKET_MAKING")

    # 2. Directional 1:3 Fixed Risk
    eng_dir_in = InstitutionalBacktestEngine(symbol="BTC/USDT:USDT", initial_capital_usd=1000.0, fixed_risk_pct=0.005, risk_reward_ratio=3.0)
    res_dir_in = eng_dir_in.run(in_sample, strategy_mode="DIRECTIONAL_TREND")

    eng_dir_out = InstitutionalBacktestEngine(symbol="BTC/USDT:USDT", initial_capital_usd=1000.0, fixed_risk_pct=0.005, risk_reward_ratio=3.0)
    res_dir_out = eng_dir_out.run(out_sample, strategy_mode="DIRECTIONAL_TREND")

    results = {
        "market_making": {
            "in_sample_2021": {
                "trades": res_mm_in.total_trades,
                "win_rate": res_mm_in.win_rate_pct,
                "net_pnl_usd": round(res_mm_in.final_equity_usd - res_mm_in.initial_capital_usd, 2),
                "roi_pct": res_mm_in.compounded_roi_pct,
                "max_dd_pct": res_mm_in.max_drawdown_pct,
                "sharpe": res_mm_in.annualized_sharpe,
            },
            "out_of_sample_2022": {
                "trades": res_mm_out.total_trades,
                "win_rate": res_mm_out.win_rate_pct,
                "net_pnl_usd": round(res_mm_out.final_equity_usd - res_mm_out.initial_capital_usd, 2),
                "roi_pct": res_mm_out.compounded_roi_pct,
                "max_dd_pct": res_mm_out.max_drawdown_pct,
                "sharpe": res_mm_out.annualized_sharpe,
            }
        },
        "directional_trend_1_3": {
            "in_sample_2021": {
                "trades": res_dir_in.total_trades,
                "win_rate": res_dir_in.win_rate_pct,
                "net_pnl_usd": round(res_dir_in.final_equity_usd - res_dir_in.initial_capital_usd, 2),
                "roi_pct": res_dir_in.compounded_roi_pct,
                "max_dd_pct": res_dir_in.max_drawdown_pct,
                "sharpe": res_dir_in.annualized_sharpe,
            },
            "out_of_sample_2022": {
                "trades": res_dir_out.total_trades,
                "win_rate": res_dir_out.win_rate_pct,
                "net_pnl_usd": round(res_dir_out.final_equity_usd - res_dir_out.initial_capital_usd, 2),
                "roi_pct": res_dir_out.compounded_roi_pct,
                "max_dd_pct": res_dir_out.max_drawdown_pct,
                "sharpe": res_dir_out.annualized_sharpe,
            }
        }
    }

    print("\n=== WALK-FORWARD REPORT ===")
    print(json.dumps(results, indent=2))
    with open("walk_forward_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_wf()
