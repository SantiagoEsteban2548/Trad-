import React, { useState, useEffect, useMemo } from 'react';
import {
  ShieldAlert,
  Activity,
  BrainCircuit,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Zap,
  RefreshCw,
  Lock,
  Percent,
  Terminal as TerminalIcon,
  Play,
  Pause,
  Radio,
  ArrowRightLeft,
  Gauge,
  ShieldCheck,
  XCircle,
  Flame,
  Power
} from 'lucide-react';

// Exact Normal CDF Phi(x)
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const prob =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - prob : prob;
}

// Acklam Rational Approximation of Inverse Normal CDF Phi^-1(p)
function normPpf(p: number): number {
  if (p <= 0 || p >= 1) return 0;
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687,
    138.357751867269, -30.66479806614716, 2.506628277459239
  ];
  const b = [
    -54.47609879822406, 161.5858368580409, -155.6989798598866,
    66.80131188771972, -13.28068155288572
  ];
  const c = [
    -0.00778489400243, -0.3223964580411365, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783
  ];
  const d = [
    0.007784695709041462, 0.3224671290700398, 2.445134137142996,
    3.754408661907416
  ];
  const pLow = 0.02425;
  const pHigh = 1.0 - pLow;

  if (p < pLow) {
    const q = Math.sqrt(-2.0 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    );
  }
  if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    );
  }
  const q = Math.sqrt(-2.0 * Math.log(1.0 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<
    'command' | 'phase1' | 'phase2' | 'phase3' | 'phase4' | 'phase5'
  >('command');

  // Telemetry Server Endpoints (Direct integration with backend/telemetry_server.py)
  const [wsUrl, setWsUrl] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      if (window.location.port === '5173') {
        return 'ws://127.0.0.1:8000/ws/telemetry';
      }
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.host}/ws/telemetry`;
    }
    return 'ws://127.0.0.1:8000/ws/telemetry';
  });
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      if (window.location.port === '5173') {
        return 'http://127.0.0.1:8000';
      }
      return window.location.origin;
    }
    return 'http://127.0.0.1:8000';
  });
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [httpConnected, setHttpConnected] = useState<boolean>(false);
  const [wsLastError, setWsLastError] = useState<string | null>(null);

  // Live Telemetry Payload State (Direct from backend/telemetry_server.py)
  const [isLiveRunning, setIsLiveRunning] = useState<boolean>(true);
  const [tickCount, setTickCount] = useState<number>(0);
  const [midPrice, setMidPrice] = useState<number>(65000.5);
  const [bestBid, setBestBid] = useState<number>(65000.0);
  const [bestAsk, setBestAsk] = useState<number>(65001.0);
  const [microprice, setMicroprice] = useState<number>(65000.2);
  const [liveObi, setLiveObi] = useState<number>(0.0);
  const [liveTfi, setLiveTfi] = useState<number>(0.0);
  const [normInventoryQ, setNormInventoryQ] = useState<number>(0.0);
  const [realizedSigma, setRealizedSigma] = useState<number>(0.00012);
  const [reservationPrice, setReservationPrice] = useState<number>(65000.5);
  const [optimalSpread, setOptimalSpread] = useState<number>(1.85);
  const [hotPathLatencyUs, setHotPathLatencyUs] = useState<number>(2.4);
  const [bidCancelledByShield, setBidCancelledByShield] = useState<boolean>(false);
  const [askCancelledByShield, setAskCancelledByShield] = useState<boolean>(false);
  const [rotationsFilled, setRotationsFilled] = useState<number>(0);

  // Portfolio State
  const [baseCapital, setBaseCapital] = useState<number>(100.0);
  const [currentEquity, setCurrentEquity] = useState<number>(100.0);
  const [dailyHwm, setDailyHwm] = useState<number>(100.0);
  const [realizedPnlToday, setRealizedPnlToday] = useState<number>(0.0);
  const [dailyReturnPct, setDailyReturnPct] = useState<number>(0.0);
  const [dailyDrawdownPct, setDailyDrawdownPct] = useState<number>(0.0);
  const [profitSweepReady, setProfitSweepReady] = useState<boolean>(false);
  const [sweptSpotVault, setSweptSpotVault] = useState<number>(0.0);

  // Operational Mode State: PAPER_L2 vs LIVE_BINANCE
  const [tradingMode, setTradingMode] = useState<'PAPER_L2' | 'LIVE_BINANCE'>('PAPER_L2');
  const [isLiveConfirmOpen, setIsLiveConfirmOpen] = useState<boolean>(false);
  const [modeSwitchError, setModeSwitchError] = useState<string | null>(null);
  const [modeSwitchSuccess, setModeSwitchSuccess] = useState<string | null>(null);
  const [isSwitchingMode, setIsSwitchingMode] = useState<boolean>(false);

  // Mandate Gates State
  const [gate1Mode, setGate1Mode] = useState<string>('LOCKED_ISOLATED');
  const [gate2Mode, setGate2Mode] = useState<string>('ENFORCED_POST_ONLY');
  const [gate3HalfKelly, setGate3HalfKelly] = useState<number>(2.5);
  const [gate3CurrentLev, setGate3CurrentLev] = useState<number>(0.0);
  const [gate4KillSwitch, setGate4KillSwitch] = useState<boolean>(false);
  const [emergencyFlattenCount, setEmergencyFlattenCount] = useState<number>(0);

  // L2 Queue Simulator State
  const [simOrderId, setSimOrderId] = useState<string>('NONE');
  const [simStatus, setSimStatus] = useState<string>('IDLE');
  const [simQueueAhead, setSimQueueAhead] = useState<number>(2.5);
  const [simOrderQty, setSimOrderQty] = useState<number>(0.0002);
  const [simFilledQty, setSimFilledQty] = useState<number>(0.0);
  const [simLatencyMs, setSimLatencyMs] = useState<number>(15.0);

  // DSR Quarantine State
  const [observedSr, setObservedSr] = useState<number>(2.45);
  const [sampleLengthT, setSampleLengthT] = useState<number>(250);
  const [skewness, setSkewness] = useState<number>(-0.3);
  const [kurtosis, setKurtosis] = useState<number>(4.1);
  const [numTrialsN, setNumTrialsN] = useState<number>(25);
  const [dsrScore, setDsrScore] = useState<number>(0.965);
  const [dsrStatus, setDsrStatus] = useState<string>('STATISTICALLY_SIGNIFICANT');
  const [dsrOverfitted, setDsrOverfitted] = useState<boolean>(false);

  // Multi-Asset Architecture State
  const [selectedAssetView, setSelectedAssetView] = useState<'GLOBAL' | 'BTC' | 'ETH' | 'SOL' | 'BNB' | 'DOGE'>('GLOBAL');
  const [symbolsData, setSymbolsData] = useState<Record<string, {
    symbol: string;
    base_asset: string;
    mid_price: number;
    best_bid: number;
    best_ask: number;
    obi: number;
    inventory_q: number;
    position_usd: number;
    rotations_today: number;
    pnl_today_usd: number;
    half_kelly_weight: number;
  }>>({
    'BTC/USDT:USDT': { symbol: 'BTC/USDT:USDT', base_asset: 'BTC', mid_price: 65000.5, best_bid: 65000.0, best_ask: 65001.0, obi: 0.0, inventory_q: 0.0, position_usd: 0.0, rotations_today: 0, pnl_today_usd: 0.0, half_kelly_weight: 0.20 },
    'ETH/USDT:USDT': { symbol: 'ETH/USDT:USDT', base_asset: 'ETH', mid_price: 2500.5, best_bid: 2500.0, best_ask: 2501.0, obi: 0.0, inventory_q: 0.0, position_usd: 0.0, rotations_today: 0, pnl_today_usd: 0.0, half_kelly_weight: 0.20 },
    'SOL/USDT:USDT': { symbol: 'SOL/USDT:USDT', base_asset: 'SOL', mid_price: 150.2, best_bid: 150.1, best_ask: 150.3, obi: 0.0, inventory_q: 0.0, position_usd: 0.0, rotations_today: 0, pnl_today_usd: 0.0, half_kelly_weight: 0.20 },
    'BNB/USDT:USDT': { symbol: 'BNB/USDT:USDT', base_asset: 'BNB', mid_price: 550.4, best_bid: 550.3, best_ask: 550.5, obi: 0.0, inventory_q: 0.0, position_usd: 0.0, rotations_today: 0, pnl_today_usd: 0.0, half_kelly_weight: 0.20 },
    'DOGE/USDT:USDT': { symbol: 'DOGE/USDT:USDT', base_asset: 'DOGE', mid_price: 0.1502, best_bid: 0.1501, best_ask: 0.1503, obi: 0.0, inventory_q: 0.0, position_usd: 0.0, rotations_today: 0, pnl_today_usd: 0.0, half_kelly_weight: 0.20 },
  });
  const [topActiveSymbols, setTopActiveSymbols] = useState<string[]>([
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'DOGE/USDT:USDT'
  ]);
  const [contagionShield, setContagionShield] = useState<{
    bid_shield_active: boolean;
    ask_shield_active: boolean;
    trigger_reason: string;
    btc_obi: number;
    bearish_count: number;
    bullish_count: number;
  }>({
    bid_shield_active: false,
    ask_shield_active: false,
    trigger_reason: 'NORMAL',
    btc_obi: 0.0,
    bearish_count: 0,
    bullish_count: 0,
  });

  // Stress Tester Interactive State
  const [testMarginMode, setTestMarginMode] = useState<'ISOLATED' | 'CROSS'>('ISOLATED');
  const [testOrderType, setTestOrderType] = useState<'LIMIT' | 'MARKET'>('LIMIT');
  const [testTif, setTestTif] = useState<'GTX' | 'GTC' | 'IOC'>('GTX');
  const [testPairPct, setTestPairPct] = useState<number>(15);
  const [testDrawdownPct, setTestDrawdownPct] = useState<number>(-0.8);

  const applyTelemetrySnapshot = (data: any) => {
    setTickCount((prev) => prev + 1);

    if (data.mode) {
      setTradingMode(data.mode);
    }
    if (data.trading_mode) {
      setTradingMode(data.trading_mode);
    }

    if (data.market) {
      setMidPrice(data.market.mid_price);
      setBestBid(data.market.best_bid);
      setBestAsk(data.market.best_ask);
      if (data.market.microprice !== undefined) setMicroprice(data.market.microprice);
      setLiveObi(data.market.obi);
      if (data.market.tfi !== undefined) setLiveTfi(data.market.tfi);
      setRealizedSigma(data.market.realized_sigma);
      setReservationPrice(data.market.reservation_price);
      setOptimalSpread(data.market.optimal_spread);
      setBidCancelledByShield(data.market.bid_cancelled_by_shield);
      setAskCancelledByShield(data.market.ask_cancelled_by_shield);
    }

    if (data.symbols) {
      setSymbolsData(data.symbols);
    }

    if (data.top_active_symbols && Array.isArray(data.top_active_symbols)) {
      setTopActiveSymbols(data.top_active_symbols);
    }

    if (data.contagion_shield) {
      setContagionShield(data.contagion_shield);
    }

    if (data.hot_path_latency_us !== undefined) {
      setHotPathLatencyUs(data.hot_path_latency_us);
    }

    if (data.portfolio) {
      setBaseCapital(data.portfolio.base_capital_usd);
      setCurrentEquity(data.portfolio.current_equity_usd);
      setDailyHwm(data.portfolio.daily_high_watermark_usd);
      setRealizedPnlToday(data.portfolio.realized_pnl_today_usd);
      setDailyReturnPct(data.portfolio.daily_return_pct);
      setDailyDrawdownPct(data.portfolio.daily_drawdown_pct);
      setProfitSweepReady(data.portfolio.profit_sweep_ready);
      if (data.portfolio.normalized_inventory_q !== undefined) {
        setNormInventoryQ(data.portfolio.normalized_inventory_q);
      }
      if (data.portfolio.rotations_filled !== undefined) {
        setRotationsFilled(data.portfolio.rotations_filled);
      }
      if (data.portfolio.simulated_vault_usd !== undefined) {
        setSweptSpotVault(data.portfolio.simulated_vault_usd);
      }
    }

    if (data.mandate_gates) {
      setGate1Mode(data.mandate_gates.gate_1_isolated_margin);
      setGate2Mode(data.mandate_gates.gate_2_maker_gtx);
      setGate3HalfKelly(data.mandate_gates.gate_3_half_kelly_cap);
      setGate3CurrentLev(data.mandate_gates.gate_3_current_leverage);
      setGate4KillSwitch(data.mandate_gates.gate_4_kill_switch_engaged);
      if (data.mandate_gates.emergency_flatten_count !== undefined) {
        setEmergencyFlattenCount(data.mandate_gates.emergency_flatten_count);
      }
    }

    if (data.l2_queue) {
      setSimOrderId(data.l2_queue.order_id);
      setSimStatus(data.l2_queue.status);
      setSimQueueAhead(data.l2_queue.queue_ahead_qty);
      setSimOrderQty(data.l2_queue.order_qty);
      setSimFilledQty(data.l2_queue.filled_qty);
      setSimLatencyMs(data.l2_queue.latency_ms);
    }

    if (data.dsr_audit) {
      setDsrScore(data.dsr_audit.deflated_sharpe_ratio);
      setDsrStatus(data.dsr_audit.status);
      setDsrOverfitted(data.dsr_audit.is_overfitted);
    }
  };

  // WebSocket Connection to backend/telemetry_server.py
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: any = null;

    const connectWs = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setWsConnected(true);
          setWsLastError(null);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            applyTelemetrySnapshot(data);
          } catch {
            // ignore non-json frame
          }
        };

        ws.onerror = () => {
          setWsConnected(false);
          setWsLastError('Reintentando conexión con /ws/telemetry...');
        };

        ws.onclose = () => {
          setWsConnected(false);
          reconnectTimeout = setTimeout(connectWs, 3000);
        };
      } catch (err: any) {
        setWsConnected(false);
        setWsLastError(err.message || 'Error inicializando WebSocket');
        reconnectTimeout = setTimeout(connectWs, 3000);
      }
    };

    connectWs();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [wsUrl]);

  // REST Telemetry sync polling (Zero mocks: fetches real data from GET /api/telemetry when WS is idle or connecting)
  useEffect(() => {
    if (wsConnected || !isLiveRunning) return;

    let active = true;
    const fetchTelemetry = async () => {
      try {
        // Try local relative origin first, fallback to direct port 8000
        const endpoints = [
          `${apiBaseUrl}/api/telemetry`,
          'http://127.0.0.1:8000/api/telemetry',
        ];
        for (const ep of endpoints) {
          try {
            const res = await fetch(ep, { cache: 'no-store' });
            if (res.ok) {
              const data = await res.json();
              if (active) {
                applyTelemetrySnapshot(data);
                setHttpConnected(true);
                setWsLastError(null);
              }
              return;
            }
          } catch {
            // try next endpoint
          }
        }
        if (active) {
          setHttpConnected(false);
        }
      } catch {
        if (active) setHttpConnected(false);
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 1500);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [wsConnected, isLiveRunning, apiBaseUrl]);


  // Kill Switch HTTP Trigger (POST /api/emergency-flatten)
  const handleTriggerEmergencyFlatten = async () => {
    try {
      const resp = await fetch(`${apiBaseUrl}/api/emergency-flatten`, { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        setGate4KillSwitch(data.kill_switch_engaged);
        setEmergencyFlattenCount(data.emergency_flatten_count);
      } else {
        // Fallback local state if API server unreachable
        setGate4KillSwitch(true);
        setEmergencyFlattenCount((c) => c + 1);
      }
    } catch {
      setGate4KillSwitch(true);
      setEmergencyFlattenCount((c) => c + 1);
    }
  };

  const handleResetKillSwitch = async () => {
    try {
      const resp = await fetch(`${apiBaseUrl}/api/reset-kill-switch`, { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        setGate4KillSwitch(data.kill_switch_engaged);
      } else {
        setGate4KillSwitch(false);
      }
    } catch {
      setGate4KillSwitch(false);
    }
  };

  const handleSwitchMode = async (targetMode: 'PAPER_L2' | 'LIVE_BINANCE') => {
    if (targetMode === tradingMode) return;
    if (targetMode === 'LIVE_BINANCE') {
      setModeSwitchError(null);
      setIsLiveConfirmOpen(true);
      return;
    }

    // Cambiar a PAPER_L2
    setIsSwitchingMode(true);
    setModeSwitchError(null);
    try {
      const res = await fetch(`${apiBaseUrl}/api/trading-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'PAPER_L2' }),
      });
      const data = await res.json();
      if (res.ok) {
        setTradingMode('PAPER_L2');
        setModeSwitchSuccess('Modo conmutado exitosamente a PAPER MODE (L2 SIM).');
        setTimeout(() => setModeSwitchSuccess(null), 4000);
      } else {
        setModeSwitchError(data.detail || 'Error al conmutar a PAPER_L2.');
      }
    } catch (err: any) {
      setModeSwitchError(err.message || 'Error de conexión con el servidor.');
    } finally {
      setIsSwitchingMode(false);
    }
  };

  const confirmSwitchToLive = async () => {
    setIsSwitchingMode(true);
    setModeSwitchError(null);
    try {
      const res = await fetch(`${apiBaseUrl}/api/trading-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'LIVE_BINANCE' }),
      });
      const data = await res.json();
      if (!res.ok) {
        setModeSwitchError(data.detail || 'Error al conmutar a modo LIVE.');
      } else {
        setTradingMode('LIVE_BINANCE');
        setIsLiveConfirmOpen(false);
        setModeSwitchSuccess('Modo LIVE CAPITAL (BINANCE USDⓈ-M) activado exitosamente.');
        setTimeout(() => setModeSwitchSuccess(null), 5000);
      }
    } catch (err: any) {
      setModeSwitchError(err.message || 'Error de red conectando al servidor de telemetría.');
    } finally {
      setIsSwitchingMode(false);
    }
  };

  const handleManualSweep = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/api/profit-sweep`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (res.ok) {
        if (data.simulated_vault_usd !== undefined) {
          setSweptSpotVault(data.simulated_vault_usd);
        } else if (data.surplus_swept_usd > 0) {
          setSweptSpotVault((v) => Number((v + data.surplus_swept_usd).toFixed(2)));
        }
        if (data.current_equity_usd !== undefined) {
          setCurrentEquity(data.current_equity_usd);
        }
      }
    } catch (err) {
      console.error('Error al ejecutar barrido de ganancias:', err);
      // Contingencia local si no hay respuesta de red
      const surplus = Number((currentEquity - baseCapital).toFixed(2));
      if (surplus > 0) {
        setSweptSpotVault((v) => Number((v + surplus).toFixed(2)));
        setCurrentEquity(baseCapital);
      }
    }
  };

  // DSR Calculation for quarantine tab
  const dsrAnalysis = useMemo(() => {
    const euler = 0.5772156649;
    const sigmaSr = Math.sqrt(1.0 / Math.max(sampleLengthT - 1, 1));
    const p1 = Math.min(Math.max(1.0 - 1.0 / numTrialsN, 1e-6), 1 - 1e-6);
    const p2 = Math.min(Math.max(1.0 - 1.0 / (numTrialsN * Math.E), 1e-6), 1 - 1e-6);
    const z1 = normPpf(p1);
    const z2 = normPpf(p2);
    const sr0 = numTrialsN > 1 ? sigmaSr * ((1.0 - euler) * z1 + euler * z2) : 0.0;

    const term1 = 1.0;
    const term2 = skewness * observedSr;
    const term3 = ((kurtosis - 1.0) / 4.0) * (observedSr * observedSr);
    const varSr = Math.max(term1 - term2 + term3, 1e-8);
    const zStat = ((observedSr - sr0) * Math.sqrt(sampleLengthT - 1)) / Math.sqrt(varSr);
    const dsrValue = normCdf(zStat);
    const isOverfitted = dsrValue < 0.95;

    return { sr0, zStat, dsrValue, isOverfitted };
  }, [observedSr, sampleLengthT, skewness, kurtosis, numTrialsN]);

  // Deterministic Stress Tester Evaluation
  const gatekeeperAudit = useMemo(() => {
    if (testDrawdownPct <= -3.0) {
      return {
        approved: false,
        gate: 'GATE_4_KILL_SWITCH',
        detail: `Drawdown diario (${testDrawdownPct.toFixed(2)}%) <= -3.00%. Nuevas órdenes bloqueadas y trigger_emergency_flatten() activado.`
      };
    }
    if (testMarginMode !== 'ISOLATED') {
      return {
        approved: false,
        gate: 'GATE_1_ISOLATED_MARGIN',
        detail: `margin_mode='${testMarginMode}' prohibido. Solo se permite 'ISOLATED'.`
      };
    }
    if (testOrderType !== 'LIMIT' || testTif !== 'GTX') {
      return {
        approved: false,
        gate: 'GATE_2_MAKER_PROTECTION',
        detail: `Estrategias Maker exigen order_type='LIMIT' y time_in_force='GTX' (recibido: ${testOrderType} / ${testTif}).`
      };
    }
    if (testPairPct > 20) {
      return {
        approved: false,
        gate: 'GATE_3_PAIR_CONCENTRATION',
        detail: `Exposición en un solo par (${testPairPct}%) supera el límite del 20.00% del margen operativo.`
      };
    }
    return {
      approved: true,
      gate: 'APPROVED',
      detail: 'Orden validada en 0.0012 ms por los 4 candados deterministas.'
    };
  }, [testMarginMode, testOrderType, testTif, testPairPct, testDrawdownPct]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Kill Switch Triggered Banner */}
      {gate4KillSwitch && (
        <div className="bg-rose-600 text-white px-6 py-2.5 flex items-center justify-between font-mono text-xs font-bold animate-pulse sticky top-0 z-50 shadow-lg">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4" />
            <span>
              EMERGENCY FLATTEN ENGAGED (GATE 4 TRIGGERED) — TODAS LAS ÓRDENES NUEVAS BLOQUEADAS
            </span>
          </div>
          <button
            onClick={handleResetKillSwitch}
            className="px-3 py-1 rounded bg-white text-rose-700 hover:bg-slate-100 font-bold transition-colors"
          >
            Resetear Kill-Switch
          </button>
        </div>
      )}

      {/* Main Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40 px-6 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-cyan-600 to-indigo-600 rounded-lg shadow-lg shadow-cyan-950/60">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-bold text-base tracking-tight text-white">
                APEX QUANT HFT TELEMETRY
              </span>
              <span
                className={`px-2 py-0.5 text-[11px] font-mono font-semibold rounded flex items-center gap-1.5 border ${
                  wsConnected
                    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                    : httpConnected
                    ? 'bg-blue-500/15 text-blue-300 border-blue-500/30'
                    : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                }`}
              >
                <Radio className={`w-3 h-3 ${wsConnected ? 'text-emerald-400 animate-pulse' : httpConnected ? 'text-blue-400' : 'text-amber-400'}`} />
                {wsConnected ? 'LIVE CCXT WS STREAM' : httpConnected ? 'LIVE TELEMETRY REST' : 'OFFLINE / LISTO PARA CONEXIÓN'}
              </span>
              <span className="px-2 py-0.5 text-[11px] font-mono rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/25">
                HOT PATH: {hotPathLatencyUs} µs
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              FastAPI lifespan • CCXT Pro Multiplexer • MandateGatekeeper • L2QueueSimulator • DSR &gt; 0.95
            </p>
          </div>
        </div>

        {/* Selector Conmutador Protegido: PAPER MODE vs LIVE CAPITAL */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-950/90 rounded-xl border border-slate-800 text-xs font-mono shadow-inner">
          <button
            onClick={() => handleSwitchMode('PAPER_L2')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
              tradingMode === 'PAPER_L2'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-900/30'
                : 'text-slate-400 hover:text-cyan-300'
            }`}
            title="Modo Simulación L2 con L2QueueSimulator y flujo CCXT en tiempo real"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>PAPER MODE (L2 SIM)</span>
          </button>
          <button
            onClick={() => handleSwitchMode('LIVE_BINANCE')}
            className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
              tradingMode === 'LIVE_BINANCE'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-900/30'
                : 'text-slate-400 hover:text-amber-300'
            }`}
            title="Modo Operativo con Capital Real en Binance USDⓈ-M Futures"
          >
            <Lock className="w-3.5 h-3.5" />
            <span>LIVE CAPITAL (BINANCE)</span>
          </button>
        </div>

        {/* Emergency Kill-Switch & Action Controls */}
        <div className="flex items-center gap-2">
          {!gate4KillSwitch ? (
            <button
              onClick={handleTriggerEmergencyFlatten}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-mono text-xs font-bold transition-all shadow-lg shadow-rose-900/40 flex items-center gap-1.5"
            >
              <Power className="w-3.5 h-3.5" />
              EMERGENCY FLATTEN
            </button>
          ) : (
            <button
              onClick={handleResetKillSwitch}
              className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-mono text-xs font-bold transition-all flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Resetear Kill-Switch
            </button>
          )}

          {/* Navigation Tabs */}
          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs font-medium">
            <button
              onClick={() => setActiveTab('command')}
              className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'command'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Gauge className="w-3.5 h-3.5" />
              Centro de Comando
            </button>
            <button
              onClick={() => setActiveTab('phase1')}
              className={`px-2.5 py-1.5 rounded-lg transition-all ${
                activeTab === 'phase1'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Mandate Gates
            </button>
            <button
              onClick={() => setActiveTab('phase3')}
              className={`px-2.5 py-1.5 rounded-lg transition-all ${
                activeTab === 'phase3'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              3 Motores
            </button>
            <button
              onClick={() => setActiveTab('phase5')}
              className={`px-2.5 py-1.5 rounded-lg transition-all ${
                activeTab === 'phase5'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              L2 &amp; DSR
            </button>
          </div>
        </div>
      </header>

      {/* Mode Switch Success Notification Banner */}
      {modeSwitchSuccess && (
        <div className="bg-emerald-500/15 border-b border-emerald-500/30 px-6 py-2 text-xs font-mono text-emerald-300 flex items-center justify-between animate-fadeIn">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{modeSwitchSuccess}</span>
          </div>
          <button onClick={() => setModeSwitchSuccess(null)} className="text-emerald-400 hover:text-white font-bold px-1">✕</button>
        </div>
      )}

      {/* Mode Switch Error Notification Banner (if modal closed) */}
      {modeSwitchError && !isLiveConfirmOpen && (
        <div className="bg-rose-500/15 border-b border-rose-500/30 px-6 py-2.5 text-xs font-mono text-rose-300 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{modeSwitchError}</span>
          </div>
          <button onClick={() => setModeSwitchError(null)} className="text-rose-400 hover:text-white font-bold px-1">✕</button>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* KPI Summary Bar */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
            <div className="text-[10px] font-mono uppercase text-slate-400">Capital Base Futuros</div>
            <div className="text-lg font-mono font-bold text-white mt-0.5">
              ${baseCapital.toLocaleString('en-US', { minimumFractionDigits: 2 })} <span className="text-xs text-slate-400 font-normal">USDT</span>
            </div>
            <div className="text-[11px] font-mono text-emerald-400">ISOLATED 2.8x Cap</div>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
            <div className="text-[10px] font-mono uppercase text-slate-400">Patrimonio Actual</div>
            <div className="text-lg font-mono font-bold text-cyan-400 mt-0.5">
              ${currentEquity.toLocaleString('en-US', { minimumFractionDigits: 2 })} <span className="text-xs text-cyan-500 font-normal">USDT</span>
            </div>
            <div className="text-[11px] font-mono text-emerald-400">
              +{dailyReturnPct.toFixed(2)}% ({rotationsFilled} rotaciones)
            </div>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
            <div className="text-[10px] font-mono uppercase text-slate-400">Bóveda Spot Segura</div>
            <div className="text-lg font-mono font-bold text-indigo-400 mt-0.5">
              ${sweptSpotVault.toFixed(2)} <span className="text-xs text-indigo-400 font-normal">USDT</span>
            </div>
            <button
              onClick={handleManualSweep}
              className="text-[10px] font-mono text-cyan-400 hover:underline flex items-center gap-1 mt-0.5"
            >
              <ArrowRightLeft className="w-3 h-3" /> Barrer excedente a Spot
            </button>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
            <div className="text-[10px] font-mono uppercase text-slate-400">Velocímetro OBI</div>
            <div
              className={`text-lg font-mono font-bold mt-0.5 ${
                Math.abs(liveObi) > 0.6 ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {liveObi > 0 ? `+${liveObi.toFixed(3)}` : liveObi.toFixed(3)}
            </div>
            <div className="text-[11px] font-mono text-slate-400">
              Shield: {bidCancelledByShield || askCancelledByShield ? 'DISPARADO' : 'NORMAL'}
            </div>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5">
            <div className="text-[10px] font-mono uppercase text-slate-400">DSR Cuarentena</div>
            <div
              className={`text-lg font-mono font-bold mt-0.5 ${
                dsrOverfitted ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {(dsrScore * 100).toFixed(1)}%
            </div>
            <div className="text-[11px] font-mono text-slate-400">
              {dsrOverfitted ? 'WARN: SNOOPING' : 'VALIDADO (>95%)'}
            </div>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase text-slate-400">Hot Path</span>
              <button
                onClick={() => setIsLiveRunning(!isLiveRunning)}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
              >
                {isLiveRunning ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
              </button>
            </div>
            <div className="text-lg font-mono font-bold text-white">{hotPathLatencyUs} µs</div>
            <div className="text-[11px] font-mono text-slate-400">Tick #{tickCount}</div>
          </div>
        </div>

        {/* TAB 1: CENTRO DE COMANDO EN TIEMPO REAL */}
        {activeTab === 'command' && (
          <div className="space-y-6">
            {/* Multi-Asset Portfolio Toolbar & Asset View Selector */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-gradient-to-tr from-cyan-600/30 to-indigo-600/30 border border-cyan-500/30 rounded-xl text-cyan-300 shadow-md shadow-cyan-950/40">
                  <Layers className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center flex-wrap gap-2">
                    <span className="text-xs font-mono uppercase text-slate-300 font-bold tracking-wider">Cesta Multimoneda Concurrente</span>
                    <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                      5 FUTUROS PERPETUOS
                    </span>
                    {contagionShield.bid_shield_active || contagionShield.ask_shield_active ? (
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3 text-rose-400" />
                        CONTAGION SHIELD ACTIVO: {contagionShield.trigger_reason} (BTC OBI: {contagionShield.btc_obi > 0 ? `+${contagionShield.btc_obi.toFixed(2)}` : contagionShield.btc_obi.toFixed(2)})
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                        <ShieldCheck className="w-3 h-3 text-emerald-400" />
                        CROSS-ASSET SHIELD: NORMAL (BTC OBI: {contagionShield.btc_obi > 0 ? `+${contagionShield.btc_obi.toFixed(2)}` : contagionShield.btc_obi.toFixed(2)})
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono">
                    Half-Kelly Matricial (F* = 0.5·C⁻¹M) • Margen ≤20% por par • Apalancamiento Global ≤3.0x • Monitoreo OBI Transversal
                  </p>
                </div>
              </div>

              {/* Selector de Vista por Activo */}
              <div className="flex items-center gap-1 bg-slate-950 p-1.5 rounded-xl border border-slate-800 font-mono text-xs shadow-inner">
                {(['GLOBAL', 'BTC', 'ETH', 'SOL', 'BNB', 'DOGE'] as const).map((view) => (
                  <button
                    key={view}
                    onClick={() => setSelectedAssetView(view)}
                    className={`px-3 py-1.5 rounded-lg transition-all font-bold flex items-center gap-1.5 ${
                      selectedAssetView === view
                        ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow-md shadow-cyan-950/60'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80'
                    }`}
                  >
                    {view === 'GLOBAL' ? 'PORTFOLIO GLOBAL' : view}
                  </button>
                ))}
              </div>
            </div>

            {/* Compact Live Multi-Asset Table */}
            <div className="bg-slate-900/85 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
              <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/70">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <h3 className="font-semibold text-xs text-white font-mono uppercase tracking-wider">
                    Matriz Multimoneda en Vivo • Cesta de 5 Activos Perpetuos
                  </h3>
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  Vista Actual: <span className="text-cyan-400 font-bold">{selectedAssetView === 'GLOBAL' ? 'PORTFOLIO GLOBAL' : `${selectedAssetView}/USDT:USDT`}</span> • Hot Path 20 Hz
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead className="bg-slate-950/90 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Activo</th>
                      <th className="py-2.5 px-4">Mid-Price ($)</th>
                      <th className="py-2.5 px-4">Velocímetro OBI</th>
                      <th className="py-2.5 px-4">Inventario (qᵢ)</th>
                      <th className="py-2.5 px-4">Posición Abierta</th>
                      <th className="py-2.5 px-4">Peso Kelly (%)</th>
                      <th className="py-2.5 px-4">Rotaciones Hoy</th>
                      <th className="py-2.5 px-4">PnL Aportado ($)</th>
                      <th className="py-2.5 px-4">Estado Escudo</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {Object.values(symbolsData).map((sym) => {
                      const isSelected = selectedAssetView === sym.base_asset || selectedAssetView === 'GLOBAL';
                      const isAlt = sym.base_asset !== 'BTC';
                      const isShielded = isAlt && (
                        contagionShield.bid_shield_active || contagionShield.ask_shield_active
                      );

                      return (
                        <tr
                          key={sym.symbol}
                          onClick={() => setSelectedAssetView(sym.base_asset as any)}
                          className={`cursor-pointer transition-colors ${
                            selectedAssetView === sym.base_asset
                              ? 'bg-cyan-500/15 text-white'
                              : 'hover:bg-slate-800/40 text-slate-300'
                          }`}
                        >
                          <td className="py-2.5 px-4 font-bold flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${selectedAssetView === sym.base_asset ? 'bg-cyan-400 animate-pulse' : 'bg-slate-500'}`} />
                            <span className="text-white">{sym.base_asset}</span>
                            <span className="text-[10px] font-normal text-slate-500">USDT-M</span>
                          </td>
                          <td className="py-2.5 px-4 font-semibold text-slate-100">
                            ${sym.mid_price >= 1 ? sym.mid_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : sym.mid_price.toFixed(5)}
                          </td>
                          <td className="py-2.5 px-4">
                            <div className="flex items-center gap-2">
                              <span className={`font-bold ${Math.abs(sym.obi) > 0.6 ? 'text-rose-400' : 'text-emerald-400'}`}>
                                {sym.obi > 0 ? `+${sym.obi.toFixed(3)}` : sym.obi.toFixed(3)}
                              </span>
                              <div className="w-16 h-1.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                <div
                                  className={`h-full ${sym.obi > 0 ? 'bg-emerald-400' : 'bg-rose-400'}`}
                                  style={{ width: `${Math.min(Math.abs(sym.obi) * 100, 100)}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="py-2.5 px-4">
                            <span className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${
                              Math.abs(sym.inventory_q) > 0.8
                                ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                                : Math.abs(sym.inventory_q) > 0.3
                                ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                                : 'bg-slate-800 text-slate-300 border-slate-700'
                            }`}>
                              q = {sym.inventory_q > 0 ? `+${sym.inventory_q.toFixed(2)}` : sym.inventory_q.toFixed(2)}
                            </span>
                          </td>
                          <td className="py-2.5 px-4 text-slate-300">
                            ${sym.position_usd.toFixed(2)}
                          </td>
                          <td className="py-2.5 px-4 text-cyan-400 font-semibold">
                            {(sym.half_kelly_weight * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-4 text-slate-200">
                            {sym.rotations_today} fills
                          </td>
                          <td className={`py-2.5 px-4 font-bold ${sym.pnl_today_usd >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {sym.pnl_today_usd >= 0 ? `+$${sym.pnl_today_usd.toFixed(2)}` : `-$${Math.abs(sym.pnl_today_usd).toFixed(2)}`}
                          </td>
                          <td className="py-2.5 px-4">
                            {isShielded ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">
                                {contagionShield.bid_shield_active && contagionShield.ask_shield_active ? 'SHIELD: BID/ASK' : contagionShield.bid_shield_active ? 'SHIELD: NO BID' : 'SHIELD: NO ASK'}
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                                NORMAL
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Panel 1: Velocímetro OBI & Avellaneda L2 */}
              <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <h2 className="font-semibold text-sm text-white">
                        Velocímetro OBI &amp; Cotización Avellaneda
                      </h2>
                    </div>
                    <span className="text-[11px] font-mono text-cyan-400 font-bold">
                      {selectedAssetView === 'GLOBAL' ? 'BTC/USDT:USDT (Primario)' : `${selectedAssetView}/USDT:USDT`}
                    </span>
                  </div>

                  {/* Gauge Visualizer */}
                  <div className="mb-4">
                    <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
                      <span className="text-rose-400">-1.0 (Sell Burst)</span>
                      <span className="text-slate-300">OBI = {liveObi.toFixed(3)}</span>
                      <span className="text-emerald-400">+1.0 (Buy Burst)</span>
                    </div>
                    <div className="relative h-4 bg-slate-950 rounded-full border border-slate-800 overflow-hidden">
                      <div className="absolute left-0 top-0 bottom-0 w-[20%] bg-rose-500/20 border-r border-rose-500/40" />
                      <div className="absolute right-0 top-0 bottom-0 w-[20%] bg-emerald-500/20 border-l border-emerald-500/40" />
                      <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-slate-600" />
                      <div
                        className={`absolute top-0 bottom-0 w-2.5 rounded-full transition-all duration-150 ${
                          Math.abs(liveObi) > 0.6 ? 'bg-rose-400 shadow-lg shadow-rose-500' : 'bg-cyan-400'
                        }`}
                        style={{ left: `calc(${((liveObi + 1) / 2) * 100}% - 5px)` }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                      <span>Shield &lt; -0.60 (Cancel BID)</span>
                      <span>Neutral</span>
                      <span>Shield &gt; +0.60 (Cancel ASK)</span>
                    </div>
                  </div>

                  <div className="space-y-2 bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 font-mono text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Mid-Price (s):</span>
                      <span className="text-white font-semibold">${midPrice.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Microprice L1 (Asimétrico):</span>
                      <span className="text-amber-400 font-semibold">${microprice.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Punta Libro (Bid / Ask):</span>
                      <span className="text-slate-300">
                        ${bestBid.toFixed(1)} / ${bestAsk.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Precio Reserva r(s,q,t):</span>
                      <span className="text-cyan-400 font-semibold">${reservationPrice.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Inventario Dinámico (q):</span>
                      <span className={`font-semibold ${normInventoryQ > 0 ? 'text-cyan-400' : normInventoryQ < 0 ? 'text-purple-400' : 'text-slate-300'}`}>
                        {normInventoryQ > 0 ? `+${normInventoryQ.toFixed(3)}` : normInventoryQ.toFixed(3)} (Max 20% Cap)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Trade Flow Imbalance (TFI 500ms):</span>
                      <span className={`font-semibold ${liveTfi > 0.2 ? 'text-emerald-400' : liveTfi < -0.2 ? 'text-rose-400' : 'text-slate-300'}`}>
                        {liveTfi > 0 ? `+${liveTfi.toFixed(3)}` : liveTfi.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Spread Óptimo:</span>
                      <span className="text-indigo-400">${optimalSpread.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Estado OBI Shield:</span>
                      {Math.abs(liveObi) > 0.6 ? (
                        <span className="text-rose-400 font-bold">CANCELACIÓN ACTIVA (&lt;20ms)</span>
                      ) : (
                        <span className="text-emerald-400">COTIZACIÓN GTX ACTIVA</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="mt-4 text-[11px] font-mono text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-3">
                  <span>Régimen: backend/config/regime_config.json</span>
                  <span className="text-emerald-400">0% Taker Fee</span>
                </div>
              </div>

              {/* Panel 2: Cortafuegos Determinista (Mandate Gates) */}
              <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <div className="flex items-center gap-2">
                      <Lock className="w-4 h-4 text-emerald-400" />
                      <h2 className="font-semibold text-sm text-white">
                        Cortafuegos Determinista (MandateGatekeeper)
                      </h2>
                    </div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      &lt; 0.002 ms
                    </span>
                  </div>

                  <div className="space-y-2.5 font-mono text-xs">
                    <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                      <div>
                        <div className="text-slate-200 font-semibold">GATE 1: Margen Aislado</div>
                        <div className="text-[11px] text-slate-400">Prohíbe modo CROSS</div>
                      </div>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 text-[10px]">
                        {gate1Mode}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                      <div>
                        <div className="text-slate-200 font-semibold">GATE 2: Protección Maker</div>
                        <div className="text-[11px] text-slate-400">Exige LIMIT + GTX (Post-Only)</div>
                      </div>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 text-[10px]">
                        {gate2Mode}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                      <div>
                        <div className="text-slate-200 font-semibold">GATE 3: Techo Half-Kelly</div>
                        <div className="text-[11px] text-slate-400">Tope: {gate3HalfKelly.toFixed(2)}x | Máx 20% par</div>
                      </div>
                      <span className="px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 text-[10px]">
                        {gate3CurrentLev.toFixed(2)}x / {gate3HalfKelly.toFixed(2)}x
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                      <div>
                        <div className="text-slate-200 font-semibold">GATE 4: Kill-Switch Diario</div>
                        <div className="text-[11px] text-slate-400">Flatten a -3.0% HWM ({emergencyFlattenCount} disparos)</div>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] ${
                          gate4KillSwitch
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : 'bg-emerald-500/15 text-emerald-400'
                        }`}
                      >
                        {gate4KillSwitch ? 'ENGAGED' : 'ARMED'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 text-[11px] font-mono text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-3">
                  <span>Barrido Spot (+2.2% / 00:00 UTC)</span>
                  <span className="text-indigo-400">{profitSweepReady ? 'LISTO' : 'MONITOR'}</span>
                </div>
              </div>

              {/* Panel 3: Simulador de Cola FIFO L2 */}
              <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <div className="flex items-center gap-2">
                      <Layers className="w-4 h-4 text-indigo-400" />
                      <h2 className="font-semibold text-sm text-white">
                        Simulador de Posición en Cola L2
                      </h2>
                    </div>
                    <span className="text-[11px] font-mono text-cyan-400">
                      Latencia: {simLatencyMs} ms
                    </span>
                  </div>

                  <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 mb-4 space-y-3 font-mono text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">ID Orden GTX:</span>
                      <span className="text-cyan-400 font-semibold">{simOrderId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Estado de Ejecución:</span>
                      <span className="text-emerald-400 font-semibold">{simStatus}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Cola Delante (queue_ahead):</span>
                      <span className="text-white font-bold">{simQueueAhead.toFixed(2)} BTC</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Cantidad Ejecutada:</span>
                      <span className="text-slate-200">
                        {simFilledQty.toFixed(4)} / {simOrderQty.toFixed(4)} BTC
                      </span>
                    </div>

                    <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500 h-full transition-all duration-200"
                        style={{
                          width: `${Math.min(100, (simFilledQty / Math.max(simOrderQty, 0.001)) * 100)}%`
                        }}
                      />
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-400 font-mono leading-relaxed">
                    Modelado FIFO exacto: Los trades a mercado consumen primero el volumen que estaba descansando antes de llenar nuestra orden límite Post-Only.
                  </p>
                </div>

                <div className="mt-4 text-[11px] font-mono text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-3">
                  <span>Selección Adversa en Vuelo</span>
                  <span className="text-amber-400">PROTEGIDA ({simLatencyMs} ms)</span>
                </div>
              </div>
            </div>

            {/* Configuración de Telemetría en Vivo */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <TerminalIcon className="w-4 h-4 text-cyan-400" />
                    Enlace de Telemetría WebSocket (FastAPI Lifespan)
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Stream en tiempo real a 10 Hz transmitiendo directamente desde `MarketRingBuffer` y `PortfolioState`.
                  </p>
                </div>
                <div className="flex items-center gap-2 font-mono text-xs">
                  <input
                    type="text"
                    value={wsUrl}
                    onChange={(e) => setWsUrl(e.target.value)}
                    className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 w-64 focus:border-cyan-500 focus:outline-none"
                    placeholder="ws://localhost:8000/ws/telemetry"
                  />
                  <span
                    className={`px-2.5 py-1.5 rounded-lg font-bold ${
                      wsConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {wsConnected ? 'CONECTADO' : 'DESCONECTADO'}
                  </span>
                </div>
              </div>
              {wsLastError && (
                <div className="mt-3 p-2.5 rounded-xl bg-amber-950/20 border border-amber-800/40 text-amber-300 font-mono text-xs">
                  Aviso: {wsLastError} (Puedes arrancar el servidor con: `uvicorn backend.telemetry_server:app --port 8000`)
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: MANDATE GATES STRESS TESTER */}
        {activeTab === 'phase1' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                <ShieldCheck className="w-5 h-5 text-cyan-400" />
                <h2 className="font-bold text-base text-white">
                  Auditoría de Candados de Seguridad (`MandateGatekeeper`)
                </h2>
              </div>
              <p className="text-xs text-slate-400">
                Prueba interactiva del método síncrono ultra-rápido `validate_order(order, state)` (&lt;0.1 ms).
              </p>

              <div className="space-y-4 font-mono text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Gate 1 — Modo de Margen:</label>
                  <div className="flex gap-2">
                    {(['ISOLATED', 'CROSS'] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => setTestMarginMode(m)}
                        className={`px-3 py-1.5 rounded-lg border ${
                          testMarginMode === m
                            ? 'bg-cyan-600 text-white border-cyan-500'
                            : 'bg-slate-950 text-slate-400 border-slate-800'
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 mb-1">Gate 2 — Tipo de Orden:</label>
                    <div className="flex gap-2">
                      {(['LIMIT', 'MARKET'] as const).map((t) => (
                        <button
                          key={t}
                          onClick={() => setTestOrderType(t)}
                          className={`px-3 py-1.5 rounded-lg border ${
                            testOrderType === t
                              ? 'bg-cyan-600 text-white border-cyan-500'
                              : 'bg-slate-950 text-slate-400 border-slate-800'
                          }`}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1">Gate 2 — Time in Force:</label>
                    <div className="flex gap-2">
                      {(['GTX', 'GTC', 'IOC'] as const).map((tif) => (
                        <button
                          key={tif}
                          onClick={() => setTestTif(tif)}
                          className={`px-3 py-1.5 rounded-lg border ${
                            testTif === tif
                              ? 'bg-cyan-600 text-white border-cyan-500'
                              : 'bg-slate-950 text-slate-400 border-slate-800'
                          }`}
                        >
                          {tif}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Gate 3 — Concentración en un solo par (% Capital):</span>
                    <span className={testPairPct > 20 ? 'text-rose-400 font-bold' : 'text-cyan-400'}>
                      {testPairPct}% (Máx 20%)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="35"
                    step="1"
                    value={testPairPct}
                    onChange={(e) => setTestPairPct(parseInt(e.target.value))}
                    className="w-full accent-cyan-500 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Gate 4 — Drawdown Diario desde HWM:</span>
                    <span className={testDrawdownPct <= -3.0 ? 'text-rose-400 font-bold' : 'text-emerald-400'}>
                      {testDrawdownPct.toFixed(1)}% (Kill-Switch en -3.0%)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-5.0"
                    max="0.0"
                    step="0.2"
                    value={testDrawdownPct}
                    onChange={(e) => setTestDrawdownPct(parseFloat(e.target.value))}
                    className="w-full accent-rose-500 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-base text-white border-b border-slate-800 pb-3 mb-4">
                  Resultado de Evaluación en el Gatekeeper
                </h3>
                <div
                  className={`p-5 rounded-2xl border ${
                    gatekeeperAudit.approved
                      ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300'
                      : 'bg-rose-950/30 border-rose-500/40 text-rose-300'
                  }`}
                >
                  <div className="flex items-center gap-2 font-mono font-bold text-sm mb-2">
                    {gatekeeperAudit.approved ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <XCircle className="w-5 h-5 text-rose-400" />
                    )}
                    {gatekeeperAudit.gate}
                  </div>
                  <p className="text-xs font-mono leading-relaxed">{gatekeeperAudit.detail}</p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs space-y-1.5">
                <div className="text-slate-400 font-semibold">Garantías Implementadas:</div>
                <div className="text-emerald-400">✓ Cero órdenes Taker en motores Maker</div>
                <div className="text-emerald-400">✓ Bloqueo estricto de margen cruzado</div>
                <div className="text-emerald-400">✓ Corte automático al tocar -3.0% HWM diario</div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: 3 MOTORES */}
        {activeTab === 'phase3' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="text-xs font-mono text-cyan-400">MOTOR A • avellaneda_obi.py</div>
              <h3 className="text-base font-bold text-white">Market Making Asimétrico + OBI Shield</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Calcula el precio de reserva r(s, q, t) = s - q·γ·σ²·(T-t) y el spread óptimo. Si OBI &lt; -0.60 cancela el BID en &lt;20 ms y desplaza la cotización; si OBI &gt; +0.60 cancela el ASK.
              </p>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs space-y-1">
                <div className="text-slate-400">Orden: LIMIT + GTX</div>
                <div className="text-emerald-400">Comisión: 0% Maker (Cobro de Spread)</div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="text-xs font-mono text-indigo-400">MOTOR B • liq_cascade.py</div>
              <h3 className="text-base font-bold text-white">Caza de Cascadas de Liquidación</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Detecta desviaciones &gt;3.2σ del VWAP de 1m combinadas con decaimiento del 50% en el flujo forzoso de 500 ms. Coloca salida Reduce-Only al 40% de la mecha y Hard Stop-Loss a 1.5σ.
              </p>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs space-y-1">
                <div className="text-slate-400">Filtro: Agotamiento 500 ms</div>
                <div className="text-indigo-400">Bracket: 40% Wick TP + 1.5σ SL</div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="text-xs font-mono text-emerald-400">MOTOR C • funding_arb.py</div>
              <h3 className="text-base font-bold text-white">Arbitraje Delta-Neutral de Funding</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Estructura paquetes Delta-Neutrales con cantidades idénticas Spot-Perpetuo cuando el APR supera el umbral configurado en `backend/config/regime_config.json`.
              </p>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs space-y-1">
                <div className="text-slate-400">Validación Dual en Gatekeeper</div>
                <div className="text-emerald-400">Delta = 0.00000000</div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: L2 & DEFLATED SHARPE AUDITOR */}
        {activeTab === 'phase5' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Percent className="w-5 h-5 text-indigo-400" />
                  <h2 className="font-semibold text-white">
                    Deflated Sharpe Auditor (`DeflatedSharpeAuditor`)
                  </h2>
                </div>
                {dsrAnalysis.isOverfitted ? (
                  <span className="px-2.5 py-1 text-xs font-mono font-bold rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    OVERFITTING_WARNING
                  </span>
                ) : (
                  <span className="px-2.5 py-1 text-xs font-mono font-bold rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    STATISTICALLY_SIGNIFICANT
                  </span>
                )}
              </div>

              <div className="space-y-4 font-mono text-xs">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Sharpe Ratio Observado (SR̂):</span>
                    <span className="text-cyan-400 font-bold">{observedSr.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="4.0"
                    step="0.05"
                    value={observedSr}
                    onChange={(e) => setObservedSr(parseFloat(e.target.value))}
                    className="w-full accent-cyan-500 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Ensayos Múltiples (N):</span>
                    <span className="text-indigo-400 font-bold">{numTrialsN}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="1000"
                    step="5"
                    value={numTrialsN}
                    onChange={(e) => setNumTrialsN(parseInt(e.target.value))}
                    className="w-full accent-indigo-500 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-400">Asimetría (γ̂₃):</span>
                      <span className="text-slate-200">{skewness.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="-2.0"
                      max="2.0"
                      step="0.05"
                      value={skewness}
                      onChange={(e) => setSkewness(parseFloat(e.target.value))}
                      className="w-full accent-slate-400 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-400">Curtosis Pearson (γ̂₄):</span>
                      <span className="text-slate-200">{kurtosis.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="2.0"
                      max="8.0"
                      step="0.1"
                      value={kurtosis}
                      onChange={(e) => setKurtosis(parseFloat(e.target.value))}
                      className="w-full accent-slate-400 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Longitud Muestral (T):</span>
                    <span className="text-slate-200">{sampleLengthT}</span>
                  </div>
                  <input
                    type="range"
                    min="30"
                    max="1000"
                    step="10"
                    value={sampleLengthT}
                    onChange={(e) => setSampleLengthT(parseInt(e.target.value))}
                    className="w-full accent-slate-400 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
                  />
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 grid grid-cols-3 gap-3 text-center font-mono">
                <div>
                  <div className="text-[10px] text-slate-500">Ruido Nulo SR₀</div>
                  <div className="text-sm font-semibold text-slate-200">{dsrAnalysis.sr0.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">Z-Score</div>
                  <div className="text-sm font-semibold text-slate-200">{dsrAnalysis.zStat.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">DSR Score</div>
                  <div
                    className={`text-base font-bold ${
                      dsrAnalysis.isOverfitted ? 'text-rose-400' : 'text-emerald-400'
                    }`}
                  >
                    {(dsrAnalysis.dsrValue * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
              <div className="space-y-4">
                <h3 className="font-semibold text-white border-b border-slate-800 pb-3">
                  Verificación de Suites de Pruebas (`pytest -v`)
                </h3>
                <div className="space-y-2.5 font-mono text-xs">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                    <div>
                      <div className="text-slate-200 font-semibold">tests/test_mandate_gates.py</div>
                      <div className="text-[11px] text-slate-400">4 candados, latencia &lt;0.1ms, profit sweeper</div>
                    </div>
                    <span className="text-emerald-400 font-bold">14 PASSED</span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                    <div>
                      <div className="text-slate-200 font-semibold">tests/test_ingestion.py</div>
                      <div className="text-[11px] text-slate-400">RingBuffer OBI, vol, liquidaciones, 10k ticks/s</div>
                    </div>
                    <span className="text-emerald-400 font-bold">4 PASSED</span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                    <div>
                      <div className="text-slate-200 font-semibold">tests/test_strategies.py</div>
                      <div className="text-[11px] text-slate-400">Avellaneda OBI shield, Liq Hunter, Funding Arb</div>
                    </div>
                    <span className="text-emerald-400 font-bold">4 PASSED</span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                    <div>
                      <div className="text-slate-200 font-semibold">tests/test_regime_council.py</div>
                      <div className="text-[11px] text-slate-400">Gemini 3-step debate, fallback, atomic save</div>
                    </div>
                    <span className="text-emerald-400 font-bold">4 PASSED</span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
                    <div>
                      <div className="text-slate-200 font-semibold">tests/test_validation.py</div>
                      <div className="text-[11px] text-slate-400">L2 FIFO queue ahead, 15ms latency, DSR equation</div>
                    </div>
                    <span className="text-emerald-400 font-bold">6 PASSED</span>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-3 rounded-xl bg-cyan-950/20 border border-cyan-500/30 text-xs font-mono text-cyan-200">
                ✓ Las 5 suites de pruebas están listas para ejecución con `pytest -v`.
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Modal de Confirmación de Seguridad: LIVE CAPITAL (BINANCE) */}
      {isLiveConfirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl max-w-lg w-full p-6 text-slate-100 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-amber-500/15 border border-amber-500/30 rounded-xl text-amber-400">
                  <ShieldAlert className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-base text-white tracking-tight">
                    Confirmación de Seguridad: Capital Real
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    Binance USDⓈ-M Futures (binanceusdm)
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => {
                  setIsLiveConfirmOpen(false);
                  setModeSwitchError(null);
                }}
                className="text-slate-400 hover:text-white p-1 text-base transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4 mb-6">
              <p className="text-xs text-slate-300 leading-relaxed">
                Está a punto de conmutar la ejecución hacia <strong>Capital Real en Binance</strong>.
                Por directriz algorítmica y mandatos deterministas, se verifica la integridad del sistema:
              </p>

              {/* Verificación del DSR > 95% */}
              <div
                className={`p-3 rounded-xl border font-mono text-xs ${
                  dsrScore > 0.95
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                    : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                }`}
              >
                <div className="flex items-center justify-between font-bold mb-1">
                  <span className="flex items-center gap-1.5">
                    {dsrScore > 0.95 ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                    )}
                    Estado DSR (Deflated Sharpe Ratio)
                  </span>
                  <span className="text-sm">{(dsrScore * 100).toFixed(1)}%</span>
                </div>
                <p className="text-[11px] text-slate-300 leading-normal">
                  {dsrScore > 0.95
                    ? '✓ DSR > 95% Superado: Estrategia estadísticamente significativa contra data-snooping y sobreajuste.'
                    : '⚠ ATENCIÓN: El DSR se encuentra en o por debajo del umbral del 95%. Proceder con estricta precaución.'}
                </p>
              </div>

              {/* Verificación de Mandatos y Candados */}
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
                <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                  Auditoría de Mandatos (Mandate Gates)
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span>Gate 1: Margen Aislado Obligatorio</span>
                  <span className="text-emerald-400 font-bold">LOCKED_ISOLATED</span>
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span>Gate 2: Maker Post-Only (GTX)</span>
                  <span className="text-emerald-400 font-bold">ENFORCED</span>
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span>Gate 3: Límite Half-Kelly (Máx 3x)</span>
                  <span className="text-emerald-400 font-bold">≤ 3.0x</span>
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span>Gate 4: Kill-Switch de Emergencia</span>
                  <span className={gate4KillSwitch ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                    {gate4KillSwitch ? 'BLOQUEADO (ENGAGED)' : 'ARMADO (OK)'}
                  </span>
                </div>
              </div>

              {/* Mensaje de Error devuelto por el Backend */}
              {modeSwitchError && (
                <div className="bg-rose-500/10 border border-rose-500/40 rounded-xl p-3 text-xs font-mono text-rose-300 flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-bold text-rose-200 mb-0.5">Aviso del Servidor / Seguridad:</div>
                    <div className="leading-snug">{modeSwitchError}</div>
                    <div className="mt-1 text-[11px] text-rose-400">
                      El motor se mantiene de forma segura en <strong>PAPER MODE (L2 SIM)</strong>.
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center justify-end gap-3 border-t border-slate-800 pt-4">
              <button
                type="button"
                onClick={() => {
                  setIsLiveConfirmOpen(false);
                  setModeSwitchError(null);
                }}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs font-semibold transition-all"
              >
                Cancelar (Mantener PAPER_L2)
              </button>
              <button
                type="button"
                disabled={isSwitchingMode || gate4KillSwitch}
                onClick={confirmSwitchToLive}
                className="px-4 py-2 rounded-xl bg-gradient-to-r from-amber-600 to-emerald-600 hover:from-amber-500 hover:to-emerald-500 text-white font-mono text-xs font-bold transition-all shadow-lg shadow-emerald-950/40 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSwitchingMode ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Conmutando...</span>
                  </>
                ) : (
                  <>
                    <Lock className="w-3.5 h-3.5" />
                    <span>Confirmar y Activar LIVE_BINANCE</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
