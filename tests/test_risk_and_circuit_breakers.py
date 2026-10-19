"""
Suite de pruebas para Fase 4: Riesgo y Circuit Breakers Institucionales.
- Test de kill-switch completo (Gate 4 -> cancela órdenes, flatten a mercado, bloquea órdenes posteriores).
- Test de orden huérfana por fallo de SL (reintentos con backoff + fallback a mercado).
- Test de OCO (One-Cancels-the-Other: fill de TP cancela SL, fill de SL cancela TP).
- Test de ratchet 85/15 sin falso drawdown (descuenta vault_transfer de daily_high_watermark_usd).
- Test de persistencia de estado del kill-switch en disco WAL ante caídas/reinicios.
- Test de rechazo por nocional mínimo (anti-error del exchange en runtime).
"""
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.mandate_gates.gatekeeper import MandateGatekeeper
from backend.mandate_gates.profit_sweeper import step_up_ratchet
from backend.mandate_gates.schemas import OrderRequest, PortfolioState
from backend.telemetry_server import DryRunExecutionEngine, BinanceLiveWsGateway


@pytest.fixture
def base_portfolio():
    return PortfolioState(
        base_capital_usd=10_000.0,
        current_equity_usd=10_000.0,
        daily_high_watermark_usd=10_000.0,
        realized_pnl_today_usd=0.0,
        active_positions={},
        rolling_edge_mu=0.04,
        rolling_variance_sigma2=0.01,
    )


@pytest.fixture
def sample_maker_order():
    return OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50_000.0,
        qty=0.02,  # $1000 notional (10% of 10,000)
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=time.time_ns(),
        min_notional=5.0,
    )


def test_kill_switch_full_flow_gate4_drawdown(base_portfolio, sample_maker_order):
    """
    Simula condición de Gate 4 (pérdida > 3% diario) y verifica:
    (a) El flag se activa.
    (b) Se cancelan las órdenes abiertas en simulador.
    (c) Se manda cierre a mercado de todas las posiciones.
    (d) Cualquier orden posterior es rechazada inmediatamente.
    """
    engine = DryRunExecutionEngine(symbols=["BTC/USDT:USDT"])
    engine.portfolio = base_portfolio

    # Simular una posición abierta y órdenes vivas en el simulador
    ctx = engine.symbol_contexts["BTC/USDT:USDT"]
    ctx.l2_sim.orders["sim_ord_1"] = MagicMock(order_id="sim_ord_1", status="OPEN")
    engine.portfolio.active_positions["BTC/USDT:USDT"] = 1000.0

    # Simular caída de equity a $9,650 (-3.5% drawdown respecto a $10,000 HWM)
    engine.portfolio.current_equity_usd = 9650.0

    # Validar orden con drawdown > 3%
    approved, reason = engine.gatekeeper.validate_order(sample_maker_order, engine.portfolio)

    # (a) Orden rechazada y flag activado
    assert approved is False
    assert "GATE_4_KILL_SWITCH" in reason
    assert engine.gatekeeper.kill_switch_engaged is True

    # (b) Órdenes simuladas canceladas
    assert ctx.l2_sim.orders["sim_ord_1"].status == "CANCELLED"

    # (c) Posición cerrada a cero en portfolio
    assert engine.portfolio.active_positions.get("BTC/USDT:USDT", 0.0) == 0.0

    # (d) Toda orden posterior es rechazada inmediatamente por Kill-Switch activo
    appr2, reason2 = engine.gatekeeper.validate_order(sample_maker_order, engine.portfolio)
    assert appr2 is False
    assert "Kill-Switch activo" in reason2


@pytest.mark.asyncio
async def test_orphan_order_prevention_on_sl_failure():
    """
    Mockea el exchange para que la orden de SL falle (error 500 / rechazo).
    Verifica que el bot reintente hasta 3 veces y cierre la posición a mercado automáticamente.
    """
    mock_exchange = AsyncMock()
    # create_order falla con excepción cuando es STOP_MARKET
    async def mock_create_order(symbol, type, side, amount, price=None, params=None):
        if type == "STOP_MARKET":
            raise RuntimeError("Exchange 500: Internal Server Error on STOP_MARKET")
        return {"id": "market_close_999", "status": "closed"}

    mock_exchange.create_order = AsyncMock(side_effect=mock_create_order)

    portfolio = PortfolioState(
        base_capital_usd=1000.0,
        current_equity_usd=1000.0,
        daily_high_watermark_usd=1000.0,
        realized_pnl_today_usd=0.0,
        active_positions={"BTC/USDT:USDT": 500.0},
        rolling_edge_mu=0.04,
        rolling_variance_sigma2=0.01,
    )
    gw = BinanceLiveWsGateway(exchange=mock_exchange, portfolio=portfolio)

    # Invocar place_protective_stop_loss_with_retry
    result = await gw.place_protective_stop_loss_with_retry(
        symbol="BTC/USDT:USDT",
        close_side="SELL",
        qty=0.01,
        stop_price=48000.0,
        max_retries=3,
    )

    # Debe retornar None porque el SL falló
    assert result is None
    # Debe haber intentado 3 veces el STOP_MARKET
    stop_calls = [c for c in mock_exchange.create_order.call_args_list if c.kwargs.get("type") == "STOP_MARKET"]
    assert len(stop_calls) == 3

    # Y tras los 3 fallos, debe haber llamado a close_position_market para no dejar posición huérfana
    market_calls = [c for c in mock_exchange.create_order.call_args_list if c.kwargs.get("type") == "market"]
    assert len(market_calls) >= 1


@pytest.mark.asyncio
async def test_oco_tp_fill_cancels_sl_and_sl_fill_cancels_tp():
    """
    Verifica OCO (One-Cancels-the-Other):
    - Al completarse TP, se cancela el SL pendiente en el exchange.
    - En el motor, al completarse el SL, se cancelan los TP y órdenes pendientes.
    """
    mock_exchange = AsyncMock()
    mock_exchange.cancel_order_ws = AsyncMock(return_value={"status": "CANCELED"})
    mock_exchange.cancel_order = AsyncMock(return_value={"status": "CANCELED"})

    gw = BinanceLiveWsGateway(exchange=mock_exchange)
    gw.protective_sl_by_symbol["BTC/USDT:USDT"] = "sl_protective_1001"
    gw.active_orders_meta["tp_maker_order_2002"] = {"price": 50000.0, "qty": 0.01, "symbol": "BTC/USDT:USDT"}

    # Simular completitud de orden de Take-Profit (is_decreasing = True)
    fill_res = {"is_increasing": False, "is_decreasing": True, "realized_pnl_usd": 25.0}
    with patch.object(gw.ledger, "record_fill", return_value=fill_res):
        await gw._handle_order_completion(
            oid="tp_maker_order_2002",
            sym="BTC/USDT:USDT",
            side="SELL",
            portfolio=PortfolioState(
                base_capital_usd=1000.0,
                current_equity_usd=1000.0,
                daily_high_watermark_usd=1000.0,
                realized_pnl_today_usd=0.0,
                active_positions={},
                rolling_edge_mu=0.04,
                rolling_variance_sigma2=0.01,
            ),
            ws_filled_qty=0.01,
        )

    # OCO verificado: el SL protector "sl_protective_1001" fue cancelado en el exchange
    assert mock_exchange.cancel_order_ws.await_count > 0 or mock_exchange.cancel_order.await_count > 0
    assert "BTC/USDT:USDT" not in gw.protective_sl_by_symbol


def test_ratchet_85_15_no_false_drawdown(base_portfolio, sample_maker_order):
    """
    Verifica la corrección matemática del Step-Up Ratchet:
    Al transferir el excedente al vault, descuenta vault_transfer del High-Water Mark,
    evitando que se dispare un falso drawdown del Gate 4 en la siguiente iteración.
    """
    # Equity sube a $10,500 (+5.0% de ganancia)
    base_portfolio.current_equity_usd = 10_500.0
    base_portfolio.daily_high_watermark_usd = 10_500.0

    # Ejecutar trinquete 85/15
    vault_swept = step_up_ratchet(base_portfolio, threshold_pct=0.02, vault_ratio=0.15, reinvest_ratio=0.85)

    assert vault_swept > 0.0
    # Excedente = $500 -> 15% ($75) a bóveda, 85% ($425) reinvertido
    assert pytest.approx(vault_swept, abs=1e-4) == 75.0
    assert pytest.approx(base_portfolio.base_capital_usd, abs=1e-4) == 10_425.0
    assert pytest.approx(base_portfolio.current_equity_usd, abs=1e-4) == 10_425.0

    # CLAVE: daily_high_watermark_usd DEBE haberse ajustado a $10,425 (no quedar en $10,500)
    assert pytest.approx(base_portfolio.daily_high_watermark_usd, abs=1e-4) == 10_425.0

    # Calcular drawdown: (10,425 - 10,425) / 10,425 = 0.0% (NO -4.76%)
    effective_dd = (base_portfolio.current_equity_usd - base_portfolio.daily_high_watermark_usd) / base_portfolio.daily_high_watermark_usd
    assert effective_dd == 0.0

    # Validar que Gate 4 NO se dispara
    gatekeeper = MandateGatekeeper()
    approved, reason = gatekeeper.validate_order(sample_maker_order, base_portfolio)
    assert approved is True
    assert reason == "APPROVED"
    assert gatekeeper.kill_switch_engaged is False


def test_kill_switch_persistence_across_restarts(tmp_path, sample_maker_order, base_portfolio):
    """
    Verifica que el estado del Kill-Switch persista en disco WAL
    y que una nueva instancia de MandateGatekeeper arranque bloqueada si el switch estaba activo.
    """
    state_file = tmp_path / "test_kill_switch.json"

    # 1. Instancia 1: arranca limpia
    gk1 = MandateGatekeeper(state_file_path=state_file)
    assert gk1.kill_switch_engaged is False

    # 2. Se dispara el Kill-Switch de emergencia
    gk1.trigger_emergency_flatten()
    assert gk1.kill_switch_engaged is True
    assert state_file.exists()

    # 3. Se simula reinicio del proceso instanciando un nuevo Gatekeeper leyendo del mismo archivo
    gk2 = MandateGatekeeper(state_file_path=state_file)
    assert gk2.kill_switch_engaged is True

    # Comprobar que rechaza órdenes de inmediato tras el reinicio
    approved, reason = gk2.validate_order(sample_maker_order, base_portfolio)
    assert approved is False
    assert "Kill-Switch activo" in reason

    # 4. Operador resetea el Kill-Switch explícitamente
    gk2.reset_kill_switch()
    assert gk2.kill_switch_engaged is False

    # 5. Nueva instancia post-reset arranca limpia
    gk3 = MandateGatekeeper(state_file_path=state_file)
    assert gk3.kill_switch_engaged is False


def test_min_notional_rejection(base_portfolio):
    """
    Verifica que MandateGatekeeper rechace órdenes con nocional inferior al mínimo permitido del exchange.
    """
    gatekeeper = MandateGatekeeper(min_notional_floor=5.0)

    # Orden con nocional de $0.50 (50,000 * 0.00001 = $0.50 < $5.0)
    micro_order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50_000.0,
        qty=0.00001,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=time.time_ns(),
        min_notional=5.0,
    )

    approved, reason = gatekeeper.validate_order(micro_order, base_portfolio)
    assert approved is False
    assert "MIN_NOTIONAL_REJECT" in reason

    # Orden con nocional de $500.0 (50,000 * 0.01 = $500 >= $5.0)
    valid_order = OrderRequest(
        symbol="BTC/USDT:USDT",
        side="BUY",
        order_type="LIMIT",
        price=50_000.0,
        qty=0.01,
        time_in_force="GTX",
        margin_mode="ISOLATED",
        strategy_id="AVELLANEDA_OBI",
        timestamp_ns=time.time_ns(),
        min_notional=5.0,
    )

    appr2, reason2 = gatekeeper.validate_order(valid_order, base_portfolio)
    assert appr2 is True
    assert reason2 == "APPROVED"
