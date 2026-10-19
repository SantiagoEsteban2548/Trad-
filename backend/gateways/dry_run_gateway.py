# backend/gateways/dry_run_gateway.py
from __future__ import annotations

import time
import logging
from typing import Any
from backend.ingestion.ring_buffer import MarketRingBuffer
from backend.mandate_gates.schemas import OrderRequest
from backend.validation.l2_simulator_and_dsr import L2QueueSimulator

logger = logging.getLogger("dry_run_gateway")

class DryRunWsGateway:
    """Adaptador de órdenes que enruta cotizaciones hacia L2QueueSimulator en modo DRY_RUN_L2."""

    def __init__(self, simulator: L2QueueSimulator, ring_buffer: MarketRingBuffer) -> None:
        self.simulator = simulator
        self.ring_buffer = ring_buffer

    async def cancel_order_ws(self, order_id: str, symbol: str) -> bool:
        return self.simulator.request_cancel(order_id, request_ts_ns=time.time_ns())

    async def place_order_ws(self, order: OrderRequest) -> str:
        sim_order = self.simulator.submit_gtx_order(order, self.ring_buffer)
        return sim_order.order_id

    async def prune_stale_orders(
        self,
        symbol: str,
        valid_order_ids: set[str] | list[str] | None = None,
        max_orders_per_side: int = 3,
    ) -> list[str]:
        cancelled: list[str] = []
        now_ns = time.time_ns()
        bids = [o for o in self.simulator.orders.values() if o.request.side == "BUY" and o.status in ("OPEN", "PARTIALLY_FILLED")]
        asks = [o for o in self.simulator.orders.values() if o.request.side == "SELL" and o.status in ("OPEN", "PARTIALLY_FILLED")]
        if len(bids) > max_orders_per_side:
            for o in bids[:-max_orders_per_side]:
                self.simulator.request_cancel(o.order_id, now_ns)
                cancelled.append(o.order_id)
        if len(asks) > max_orders_per_side:
            for o in asks[:-max_orders_per_side]:
                self.simulator.request_cancel(o.order_id, now_ns)
                cancelled.append(o.order_id)
        return cancelled




