import ccxt

def main():
    b = ccxt.binanceusdm()
    x = ccxt.bingx({'options': {'defaultType': 'swap'}})
    b_markets = b.load_markets()
    x_markets = x.load_markets()

    symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
        'DOGE/USDT:USDT', 'XRP/USDT:USDT', 'SUI/USDT:USDT', 'AVAX/USDT:USDT',
        'LINK/USDT:USDT', '1000PEPE/USDT:USDT', 'NEAR/USDT:USDT', 'ARB/USDT:USDT',
        'XAU/USDT:USDT', 'XAG/USDT:USDT'
    ]

    print("=== AUDIT OF ALL 14 SYMBOLS IN BINANCE & BINGX ===")
    for sym in symbols:
        print(f"\n--- {sym} ---")
        if sym in b_markets:
            bm = b_markets[sym]
            b_cost = bm.get('limits', {}).get('cost', {}).get('min')
            b_amt = bm.get('limits', {}).get('amount', {}).get('min')
            b_pprec = bm.get('precision', {}).get('price')
            b_aprec = bm.get('precision', {}).get('amount')
            b_maker = bm.get('maker')
            b_taker = bm.get('taker')
            print(f"BINANCE: min_cost={b_cost}, min_amt={b_amt}, tick_prec={b_pprec}, step_prec={b_aprec}, maker={b_maker}, taker={b_taker}")
        else:
            print("BINANCE: NOT FOUND")

        x_sym = sym
        if sym not in x_markets:
            if 'XAU' in sym:
                x_sym = 'XAUT/USDT:USDT' if 'XAUT/USDT:USDT' in x_markets else 'NCCOGOLD2USD/USDT:USDT'
            elif 'XAG' in sym:
                x_sym = 'NCCOXAG2USD/USDT:USDT'
        if x_sym in x_markets:
            xm = x_markets[x_sym]
            x_cost = xm.get('limits', {}).get('cost', {}).get('min')
            x_amt = xm.get('limits', {}).get('amount', {}).get('min')
            x_pprec = xm.get('precision', {}).get('price')
            x_aprec = xm.get('precision', {}).get('amount')
            x_maker = xm.get('maker')
            x_taker = xm.get('taker')
            print(f"BINGX ({x_sym}): min_cost={x_cost}, min_amt={x_amt}, tick_prec={x_pprec}, step_prec={x_aprec}, maker={x_maker}, taker={x_taker}")
        else:
            print(f"BINGX ({x_sym}): NOT FOUND")

if __name__ == '__main__':
    main()
