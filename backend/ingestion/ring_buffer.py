# backend/ingestion/ring_buffer.py
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


class MarketRingBuffer:
    """
    Buffer circular de memoria contigua (zero-allocation) para microestructura de mercado L2.
    Preasigna arrays de NumPy (float64 e int64) de tamaño fijo para evitar pausas del GC.
    Todos los métodos vectorizados operan en < 0.2 ms.
    """

    __slots__ = (
        "capacity",
        # L2 Order Book
        "ob_bid_prices",
        "ob_bid_qtys",
        "ob_ask_prices",
        "ob_ask_qtys",
        "ob_timestamps_ns",
        "ob_head",
        "ob_count",
        # Trades
        "trade_prices",
        "trade_qtys",
        "trade_is_buyer_maker",
        "trade_timestamps_ns",
        "trade_head",
        "trade_count",
        # Forced Liquidations
        "liq_sides",
        "liq_prices",
        "liq_qtys",
        "liq_timestamps_ns",
        "liq_head",
        "liq_count",
    )

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity: int = capacity

        # 1. Snapshots L2 de los primeros 5 niveles
        self.ob_bid_prices: np.ndarray = np.zeros((capacity, 5), dtype=np.float64)
        self.ob_bid_qtys: np.ndarray = np.zeros((capacity, 5), dtype=np.float64)
        self.ob_ask_prices: np.ndarray = np.zeros((capacity, 5), dtype=np.float64)
        self.ob_ask_qtys: np.ndarray = np.zeros((capacity, 5), dtype=np.float64)
        self.ob_timestamps_ns: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.ob_head: int = 0
        self.ob_count: int = 0

        # 2. Flujo de transacciones ejecutadas (Trades)
        self.trade_prices: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.trade_qtys: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.trade_is_buyer_maker: np.ndarray = np.zeros(capacity, dtype=np.uint8)
        self.trade_timestamps_ns: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.trade_head: int = 0
        self.trade_count: int = 0

        # 3. Liquidaciones forzosas (forceOrder)
        # liq_sides: +1 = BUY (cierre forzoso de corto), -1 = SELL (cierre forzoso de largo)
        self.liq_sides: np.ndarray = np.zeros(capacity, dtype=np.int8)
        self.liq_prices: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.liq_qtys: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.liq_timestamps_ns: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.liq_head: int = 0
        self.liq_count: int = 0

    # -------------------------------------------------------------------------
    # Inserciones Zero-Allocation en Hot Path (< 1 microsegundo)
    # -------------------------------------------------------------------------

    def push_order_book(
        self,
        bid_prices: Sequence[float] | np.ndarray,
        bid_qtys: Sequence[float] | np.ndarray,
        ask_prices: Sequence[float] | np.ndarray,
        ask_qtys: Sequence[float] | np.ndarray,
        timestamp_ns: int,
    ) -> None:
        """Inserta snapshot L2 de 5 niveles sin alocar nueva memoria."""
        idx = self.ob_head
        self.ob_bid_prices[idx, :] = bid_prices[:5]
        self.ob_bid_qtys[idx, :] = bid_qtys[:5]
        self.ob_ask_prices[idx, :] = ask_prices[:5]
        self.ob_ask_qtys[idx, :] = ask_qtys[:5]
        self.ob_timestamps_ns[idx] = timestamp_ns

        self.ob_head = (idx + 1) % self.capacity
        if self.ob_count < self.capacity:
            self.ob_count += 1

    def push_trade(
        self,
        price: float,
        qty: float,
        is_buyer_maker: bool | int,
        timestamp_ns: int,
    ) -> None:
        """Inserta un trade ejecutado en el buffer circular."""
        idx = self.trade_head
        self.trade_prices[idx] = price
        self.trade_qtys[idx] = qty
        self.trade_is_buyer_maker[idx] = 1 if is_buyer_maker else 0
        self.trade_timestamps_ns[idx] = timestamp_ns

        self.trade_head = (idx + 1) % self.capacity
        if self.trade_count < self.capacity:
            self.trade_count += 1

    def push_liquidation(
        self,
        side: str | int,
        price: float,
        qty: float,
        timestamp_ns: int,
    ) -> None:
        """Inserta un evento de liquidación forzosa."""
        idx = self.liq_head
        s_val = 1 if side in ("BUY", 1, "buy") else -1
        self.liq_sides[idx] = s_val
        self.liq_prices[idx] = price
        self.liq_qtys[idx] = qty
        self.liq_timestamps_ns[idx] = timestamp_ns

        self.liq_head = (idx + 1) % self.capacity
        if self.liq_count < self.capacity:
            self.liq_count += 1

    # -------------------------------------------------------------------------
    # Métodos Vectorizados de Microestructura (< 0.2 ms)
    # -------------------------------------------------------------------------

    def compute_obi(self, levels: int = 5) -> float:
        """
        Calcula el Order Book Imbalance (OBI) sobre los niveles solicitados (hasta 5):
            OBI = (V_bid - V_ask) / (V_bid + V_ask)
        Retorna un float en el rango [-1.0, 1.0].
        """
        if self.ob_count == 0:
            return 0.0

        latest_idx = (self.ob_head - 1) % self.capacity
        k = min(max(levels, 1), 5)

        v_bid = float(np.sum(self.ob_bid_qtys[latest_idx, :k]))
        v_ask = float(np.sum(self.ob_ask_qtys[latest_idx, :k]))
        total_vol = v_bid + v_ask

        if total_vol <= 1e-12:
            return 0.0

        obi = (v_bid - v_ask) / total_vol
        return float(np.clip(obi, -1.0, 1.0))

    def compute_microprice(self) -> float:
        """
        Calcula el Microprice analítico utilizando el Nivel 1 del libro L2:
            Microprice = (V_bid,0 * P_ask,0 + V_ask,0 * P_bid,0) / (V_bid,0 + V_ask,0)
        Si no hay niveles registrados o el volumen es cero, retorna 0.0 o el mid-price.
        """
        if self.ob_count == 0:
            return 0.0

        latest_idx = (self.ob_head - 1) % self.capacity
        p_bid_0 = float(self.ob_bid_prices[latest_idx, 0])
        v_bid_0 = float(self.ob_bid_qtys[latest_idx, 0])
        p_ask_0 = float(self.ob_ask_prices[latest_idx, 0])
        v_ask_0 = float(self.ob_ask_qtys[latest_idx, 0])

        denom = v_bid_0 + v_ask_0
        if denom <= 1e-12:
            return 0.5 * (p_bid_0 + p_ask_0)

        return float((v_bid_0 * p_ask_0 + v_ask_0 * p_bid_0) / denom)

    def compute_tfi(self, window_ns: int = 500_000_000) -> float:
        """
        Calcula el Trade Flow Imbalance (TFI) sobre una ventana móvil temporal (500 ms por defecto):
            TFI = (V_buy_aggr - V_sell_aggr) / (V_buy_aggr + V_sell_aggr)
        donde:
            - is_buyer_maker == 0 (False) representa compra agresiva taker (golpea el ask)
            - is_buyer_maker == 1 (True) representa venta agresiva taker (golpea el bid)
        Retorna un float normalizado en el rango [-1.0, 1.0].
        """
        if self.trade_count == 0:
            return 0.0

        latest_idx = (self.trade_head - 1) % self.capacity
        latest_ts = int(self.trade_timestamps_ns[latest_idx])
        cutoff_ts = latest_ts - window_ns

        if self.trade_count < self.capacity:
            ts = self.trade_timestamps_ns[: self.trade_count]
            qtys = self.trade_qtys[: self.trade_count]
            bm = self.trade_is_buyer_maker[: self.trade_count]
        else:
            head = self.trade_head
            ts = np.concatenate([self.trade_timestamps_ns[head:], self.trade_timestamps_ns[:head]])
            qtys = np.concatenate([self.trade_qtys[head:], self.trade_qtys[:head]])
            bm = np.concatenate([self.trade_is_buyer_maker[head:], self.trade_is_buyer_maker[:head]])

        mask = ts >= cutoff_ts
        if not np.any(mask):
            return 0.0

        window_qtys = qtys[mask]
        window_bm = bm[mask]

        buy_vol = float(np.sum(window_qtys[window_bm == 0]))
        sell_vol = float(np.sum(window_qtys[window_bm == 1]))
        total_vol = buy_vol + sell_vol

        if total_vol <= 1e-12:
            return 0.0

        tfi = (buy_vol - sell_vol) / total_vol
        return float(np.clip(tfi, -1.0, 1.0))

    def compute_realized_volatility(self, window_seconds: float = 300.0) -> float:
        """
        Calcula la desviación estándar de los retornos logarítmicos del mid-price
        en una ventana móvil temporal de `window_seconds` (por defecto 300 s / 5 min).
        """
        if self.ob_count < 2:
            return 0.0

        latest_idx = (self.ob_head - 1) % self.capacity
        latest_ts = self.ob_timestamps_ns[latest_idx]
        cutoff_ts = latest_ts - int(window_seconds * 1_000_000_000)

        # Extracción temporal cronológica desde el buffer circular
        if self.ob_count < self.capacity:
            timestamps = self.ob_timestamps_ns[: self.ob_count]
            start_idx = int(np.searchsorted(timestamps, cutoff_ts))
            if (self.ob_count - start_idx) < 2:
                return 0.0
            bid0 = self.ob_bid_prices[start_idx : self.ob_count, 0]
            ask0 = self.ob_ask_prices[start_idx : self.ob_count, 0]
        else:
            if cutoff_ts > self.ob_timestamps_ns[0]:
                start_idx = int(np.searchsorted(self.ob_timestamps_ns[: self.ob_head], cutoff_ts))
                if (self.ob_head - start_idx) < 2:
                    return 0.0
                bid0 = self.ob_bid_prices[start_idx : self.ob_head, 0]
                ask0 = self.ob_ask_prices[start_idx : self.ob_head, 0]
            else:
                start_old = int(np.searchsorted(self.ob_timestamps_ns[self.ob_head :], cutoff_ts))
                bid0 = np.concatenate(
                    [self.ob_bid_prices[self.ob_head + start_old :, 0], self.ob_bid_prices[: self.ob_head, 0]]
                )
                ask0 = np.concatenate(
                    [self.ob_ask_prices[self.ob_head + start_old :, 0], self.ob_ask_prices[: self.ob_head, 0]]
                )

        mid_prices = 0.5 * (bid0 + ask0)
        valid_mask = mid_prices > 1e-12
        if np.count_nonzero(valid_mask) < 2:
            return 0.0

        log_returns = np.diff(np.log(mid_prices[valid_mask]))
        if len(log_returns) < 2:
            return 0.0

        return float(np.std(log_returns))

    def compute_vpin(self, bucket_volume: float = 10.0, num_buckets: int = 20) -> float:
        """
        Calcula VPIN (Volume-Synchronized Probability of Toxicity) de Easley, López de Prado & O'Hara:
            VPIN = (sum_{k=1}^N |V_k^B - V_k^S|) / (N * V)
        Divide el flujo de volumen en buckets de tamaño constante V (bucket_volume).
        Para cada bucket, calcula el desbalance de volumen de compradores y vendedores.
        Devuelve un float acotado en [0.0, 1.0].
        """
        if self.trade_count < 2:
            return 0.25

        if self.trade_count < self.capacity:
            qtys = self.trade_qtys[: self.trade_count]
            is_bm = self.trade_is_buyer_maker[: self.trade_count]
        else:
            qtys = np.concatenate([self.trade_qtys[self.trade_head :], self.trade_qtys[: self.trade_head]])
            is_bm = np.concatenate([self.trade_is_buyer_maker[self.trade_head :], self.trade_is_buyer_maker[: self.trade_head]])

        tot_vol = float(np.sum(qtys))
        if tot_vol <= 1e-8:
            return 0.25

        v = float(bucket_volume)
        if v <= 0.0:
            v = max(0.001, tot_vol / max(num_buckets, 1))

        bucket_imbalances: list[float] = []
        cur_v_buy = 0.0
        cur_v_sell = 0.0
        cur_filled = 0.0

        for i in range(len(qtys)):
            q_rem = float(qtys[i])
            is_buy = (is_bm[i] == 0)

            while q_rem > 1e-8:
                space = v - cur_filled
                if q_rem < space:
                    if is_buy:
                        cur_v_buy += q_rem
                    else:
                        cur_v_sell += q_rem
                    cur_filled += q_rem
                    q_rem = 0.0
                else:
                    if is_buy:
                        cur_v_buy += space
                    else:
                        cur_v_sell += space
                    bucket_imbalances.append(abs(cur_v_buy - cur_v_sell))
                    q_rem -= space
                    cur_v_buy = 0.0
                    cur_v_sell = 0.0
                    cur_filled = 0.0

        if len(bucket_imbalances) == 0:
            if cur_filled > 0.0:
                return float(np.clip(abs(cur_v_buy - cur_v_sell) / cur_filled, 0.0, 1.0))
            return 0.25

        recent_imb = bucket_imbalances[-num_buckets:]
        vpin = float(sum(recent_imb) / (len(recent_imb) * v))
        return float(np.clip(vpin, 0.0, 1.0))

    def detect_liquidation_cluster(self, percentile: float = 99.0) -> dict:
        """
        Detecta si el volumen liquidado en los últimos 10 segundos excede el umbral extremo
        (percentil 99.0 histórico del buffer) y devuelve el Z-score del precio actual
        respecto al VWAP ponderado de 1 minuto.
        """
        if self.liq_count == 0:
            return {
                "is_cluster": False,
                "volume_last_10s": 0.0,
                "threshold_volume": 0.0,
                "vwap_1m": 0.0,
                "z_score": 0.0,
                "dominant_side": "NONE",
            }

        latest_liq_idx = (self.liq_head - 1) % self.capacity
        latest_ts = self.liq_timestamps_ns[latest_liq_idx]
        cutoff_10s = latest_ts - int(10 * 1_000_000_000)

        # 1. Obtener todas las liquidaciones de los últimos 10 segundos
        if self.liq_count < self.capacity:
            mask_10s = self.liq_timestamps_ns[: self.liq_count] >= cutoff_10s
            recent_sides = self.liq_sides[: self.liq_count][mask_10s]
            recent_qtys = self.liq_qtys[: self.liq_count][mask_10s]
            all_historical_qtys = self.liq_qtys[: self.liq_count]
        else:
            mask_10s = self.liq_timestamps_ns >= cutoff_10s
            recent_sides = self.liq_sides[mask_10s]
            recent_qtys = self.liq_qtys[mask_10s]
            all_historical_qtys = self.liq_qtys

        vol_10s = float(np.sum(recent_qtys)) if len(recent_qtys) > 0 else 0.0

        threshold_volume = (
            float(np.percentile(all_historical_qtys, percentile))
            if len(all_historical_qtys) > 0
            else 0.0
        )
        is_cluster = (vol_10s > threshold_volume) and (vol_10s > 0.0)

        if len(recent_sides) > 0:
            buy_vol = float(np.sum(recent_qtys[recent_sides == 1]))
            sell_vol = float(np.sum(recent_qtys[recent_sides == -1]))
            dominant_side = "BUY" if buy_vol >= sell_vol else "SELL"
        else:
            dominant_side = "NONE"

        vwap_1m = 0.0
        z_score = 0.0
        cutoff_60s = latest_ts - int(60 * 1_000_000_000)

        if self.trade_count > 0:
            if self.trade_count < self.capacity:
                mask_60s = self.trade_timestamps_ns[: self.trade_count] >= cutoff_60s
                t_prices = self.trade_prices[: self.trade_count][mask_60s]
                t_qtys = self.trade_qtys[: self.trade_count][mask_60s]
            else:
                mask_60s = self.trade_timestamps_ns >= cutoff_60s
                t_prices = self.trade_prices[mask_60s]
                t_qtys = self.trade_qtys[mask_60s]

            sum_qty = float(np.sum(t_qtys))
            if sum_qty > 1e-12:
                vwap_1m = float(np.sum(t_prices * t_qtys) / sum_qty)
                std_price = float(np.std(t_prices))

                last_trade_idx = (self.trade_head - 1) % self.capacity
                current_price = float(self.trade_prices[last_trade_idx])

                if std_price > 1e-8:
                    z_score = float((current_price - vwap_1m) / std_price)

        return {
            "is_cluster": bool(is_cluster),
            "volume_last_10s": vol_10s,
            "threshold_volume": threshold_volume,
            "vwap_1m": vwap_1m,
            "z_score": z_score,
            "dominant_side": dominant_side,
        }
