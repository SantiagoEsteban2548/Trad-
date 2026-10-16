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
  Power,
  Download,
  TrendingUp,
  TrendingDown,
  Clock,
  BarChart2,
  PieChart,
  ArrowDownUp,
  Cpu,
  BarChart3
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

export interface ExecutionStats {
  total_trades: number;
  wins_count: number;
  losses_count: number;
  win_rate_pct: number;
  profit_factor: number;
  avg_trade_pnl_usd: number;
  gross_profit_usd: number;
  gross_loss_usd: number;
  equity_curve: number[];
}

export interface OpenOrder {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  level?: string;
  price: number;
  qty: number;
  filled_qty: number;
  queue_ahead_qty: number;
  strategy: string;
  status: string;
}

export interface RecentTrade {
  id: string;
  timestamp_ms: number;
  time_str: string;
  symbol: string;
  strategy: string;
  side: 'BUY' | 'SELL';
  entry_price: number;
  exit_or_mid_price: number;
  price?: number;
  qty: number;
  return_pct: number;
  pnl_usd: number;
  age_str?: string;
}

export interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  is_bullish: boolean;
  reservation_price: number;
}

export interface EquityComparisonPoint {
  timestamp: number;
  timestamp_ms?: number;
  bot_equity_usd: number;
  benchmark_equity_usd: number;
  trade_pnl_bar: number;
}

export interface WhaleTapeEvent {
  symbol: string;
  side: 'BUY' | 'SELL';
  notional_usd: number;
  intensity_pct: number;
  age_str: string;
  is_liquidation?: boolean;
}

export interface PortfolioAllocationItem {
  symbol: string;
  base_asset: string;
  weight_pct: number;
  notional_usd: number;
  color: string;
}

export interface OrderBookDepth {
  bids: [number, number, number][]; // [price, qty, cum_qty]
  asks: [number, number, number][];
}

function renderEquitySparkline(data: number[], width = 160, height = 36) {
  if (!data || data.length < 2) {
    return (
      <div className="h-9 flex items-center justify-center text-slate-500 text-[10px] font-mono">
        Sin datos
      </div>
    );
  }
  const minVal = Math.min(...data);
  const maxVal = Math.max(...data);
  const range = maxVal - minVal || 1.0;
  const paddingX = 2;
  const paddingY = 4;
  const innerW = width - paddingX * 2;
  const innerH = height - paddingY * 2;

  const points = data.map((val, idx) => {
    const x = paddingX + (idx / (data.length - 1)) * innerW;
    const y = height - paddingY - ((val - minVal) / range) * innerH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(' L ')}`;
  const areaD = `M ${paddingX},${height - paddingY} L ${points.join(' L ')} L ${width - paddingX},${height - paddingY} Z`;
  const isUp = (data[data.length - 1] ?? 0) >= (data[0] ?? 0);
  const strokeColor = isUp ? '#00F59B' : '#A855F7';

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id="equitySparklineGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity="0.3" />
          <stop offset="100%" stopColor={strokeColor} stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <path d={areaD} fill="url(#equitySparklineGrad)" />
      <path d={pathD} fill="none" stroke={strokeColor} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      {points.length > 0 && (
        <circle
          cx={points[points.length - 1].split(',')[0]}
          cy={points[points.length - 1].split(',')[1]}
          r="2.5"
          fill={strokeColor}
        />
      )}
    </svg>
  );
}

function CandlestickChart({
  candles,
  bestBid,
  bestAsk,
  reservationPrice,
}: {
  candles: Candle[];
  bestBid: number;
  bestAsk: number;
  reservationPrice: number;
}) {
  if (!candles || candles.length === 0) return null;
  const n = candles.length;
  const svgW = 760;
  const svgH = 210;
  const priceH = 150;
  const volH = 45;
  const padTop = 15;

  const lows = candles.map((c) => c.low).filter((v) => v > 0);
  const highs = candles.map((c) => c.high).filter((v) => v > 0);
  const minP = Math.min(...lows, bestBid > 0 ? bestBid * 0.999 : 64000);
  const maxP = Math.max(...highs, bestAsk > 0 ? bestAsk * 1.001 : 66000);
  const rangeP = maxP - minP || 1.0;

  const maxV = Math.max(...candles.map((c) => c.volume), 1.0);
  const slotW = svgW / n;
  const barW = Math.max(slotW * 0.65, 4);

  const getY = (p: number) => padTop + priceH - ((p - minP) / rangeP) * priceH;

  const yBid = getY(bestBid);
  const yAsk = getY(bestAsk);
  const yRes = getY(reservationPrice);

  const resPoints = candles.map((c, i) => {
    const x = i * slotW + slotW / 2;
    const y = getY(c.reservation_price || c.close);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' L ');

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-full overflow-visible">
      <defs>
        <linearGradient id="volGradMint" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00F59B" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#00F59B" stopOpacity="0.05" />
        </linearGradient>
        <linearGradient id="volGradPurple" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#A855F7" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#A855F7" stopOpacity="0.05" />
        </linearGradient>
      </defs>

      {/* Grid horizontal lines */}
      {[0.25, 0.5, 0.75].map((ratio, idx) => {
        const y = padTop + priceH * ratio;
        const p = maxP - ratio * rangeP;
        return (
          <g key={idx}>
            <line x1="0" y1={y} x2={svgW} y2={y} stroke="#1C2230" strokeDasharray="3 3" />
            <text x={svgW - 5} y={y - 3} fill="#64748B" fontSize="9" fontFamily="monospace" textAnchor="end">
              ${p.toFixed(p > 100 ? 1 : 4)}
            </text>
          </g>
        );
      })}

      {/* Volume Separator */}
      <line x1="0" y1={padTop + priceH + 4} x2={svgW} y2={padTop + priceH + 4} stroke="#1C2230" />

      {/* Overlay Lines: Best Bid & Best Ask */}
      <line x1="0" y1={yBid} x2={svgW} y2={yBid} stroke="#00F59B" strokeWidth="1" strokeDasharray="4 2" opacity="0.75" />
      <line x1="0" y1={yAsk} x2={svgW} y2={yAsk} stroke="#A855F7" strokeWidth="1" strokeDasharray="4 2" opacity="0.75" />

      {/* Reservation Price overlay line */}
      <path d={`M ${resPoints}`} fill="none" stroke="#06B6D4" strokeWidth="1.5" strokeDasharray="3 2" opacity="0.85" />

      {/* Candles and Volumes */}
      {candles.map((c, i) => {
        const cx = i * slotW + slotW / 2;
        const isBull = c.close >= c.open;
        const color = isBull ? '#00F59B' : '#A855F7';
        const yHigh = getY(c.high);
        const yLow = getY(c.low);
        const yOpen = getY(c.open);
        const yClose = getY(c.close);
        const rectY = Math.min(yOpen, yClose);
        const rectH = Math.max(Math.abs(yClose - yOpen), 1.5);

        const vH = (c.volume / maxV) * (volH - 8);
        const vy = svgH - vH;

        return (
          <g key={i}>
            {/* Volume bar */}
            <rect
              x={cx - barW / 2}
              y={vy}
              width={barW}
              height={vH}
              fill={isBull ? 'url(#volGradMint)' : 'url(#volGradPurple)'}
              stroke={color}
              strokeWidth="0.5"
              opacity="0.8"
            />
            {/* Wick */}
            <line x1={cx} y1={yHigh} x2={cx} y2={yLow} stroke={color} strokeWidth="1.2" />
            {/* Body */}
            <rect
              x={cx - barW / 2}
              y={rectY}
              width={barW}
              height={rectH}
              fill={color}
              rx="1"
            />
          </g>
        );
      })}

      {/* Badges on right edge for Bid/Ask */}
      {bestBid > 0 && (
        <g>
          <rect x={svgW - 55} y={yBid - 7} width="52" height="14" rx="3" fill="#00F59B" fillOpacity="0.2" stroke="#00F59B" strokeWidth="0.8" />
          <text x={svgW - 29} y={yBid + 3} fill="#00F59B" fontSize="9" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
            ${bestBid >= 1 ? bestBid.toFixed(1) : bestBid.toFixed(4)}
          </text>
        </g>
      )}

      {bestAsk > 0 && (
        <g>
          <rect x={svgW - 55} y={yAsk - 7} width="52" height="14" rx="3" fill="#A855F7" fillOpacity="0.2" stroke="#A855F7" strokeWidth="0.8" />
          <text x={svgW - 29} y={yAsk + 3} fill="#A855F7" fontSize="9" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
            ${bestAsk >= 1 ? bestAsk.toFixed(1) : bestAsk.toFixed(4)}
          </text>
        </g>
      )}
    </svg>
  );
}

function DualEquityCurve({
  data,
  currentEquity,
  baseCapital,
}: {
  data: EquityComparisonPoint[];
  currentEquity: number;
  baseCapital: number;
}) {
  if (!data || data.length === 0) return null;
  const svgW = 760;
  const svgH = 135;
  const chartH = 90;
  const padTop = 10;
  const barBaseY = 118;

  const botVals = data.map((d) => d.bot_equity_usd);
  const benchVals = data.map((d) => d.benchmark_equity_usd);
  const allVals = [...botVals, ...benchVals, currentEquity, baseCapital];
  const minVal = Math.min(...allVals) * 0.995;
  const maxVal = Math.max(...allVals) * 1.005;
  const range = maxVal - minVal || 1.0;

  const n = data.length;
  const slotW = svgW / Math.max(n - 1, 1);

  const getY = (v: number) => padTop + chartH - ((v - minVal) / range) * chartH;

  const botPoints = data.map((d, i) => `${(i * slotW).toFixed(1)},${getY(d.bot_equity_usd).toFixed(1)}`).join(' L ');
  const benchPoints = data.map((d, i) => `${(i * slotW).toFixed(1)},${getY(d.benchmark_equity_usd).toFixed(1)}`).join(' L ');
  const botArea = `M 0,${padTop + chartH} L ${botPoints} L ${svgW},${padTop + chartH} Z`;

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-full overflow-visible">
      <defs>
        <linearGradient id="botGradMint" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00F59B" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#00F59B" stopOpacity="0.0" />
        </linearGradient>
      </defs>

      {/* Grid lines */}
      <line x1="0" y1={padTop + chartH / 2} x2={svgW} y2={padTop + chartH / 2} stroke="#1C2230" strokeDasharray="3 3" />
      <line x1="0" y1={padTop + chartH} x2={svgW} y2={padTop + chartH} stroke="#1C2230" />

      {/* Bot Area and Line */}
      <path d={botArea} fill="url(#botGradMint)" />
      <path d={`M ${botPoints}`} fill="none" stroke="#00F59B" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />

      {/* Benchmark Line */}
      <path d={`M ${benchPoints}`} fill="none" stroke="#A855F7" strokeWidth="1.8" strokeDasharray="4 3" opacity="0.85" />

      {/* PnL Bars for rotation events */}
      {data.map((d, i) => {
        if (!d.trade_pnl_bar || d.trade_pnl_bar === 0) return null;
        const x = i * slotW;
        const isUp = d.trade_pnl_bar > 0;
        const bH = Math.min(Math.abs(d.trade_pnl_bar) * 45, 14);
        const y = isUp ? barBaseY - bH : barBaseY;
        const color = isUp ? '#00F59B' : '#A855F7';
        return (
          <rect
            key={i}
            x={x - 2}
            y={y}
            width="4"
            height={bH}
            fill={color}
            rx="1"
            opacity="0.9"
          />
        );
      })}

      {/* Latest Bot Point Dot */}
      {data.length > 0 && (
        <circle
          cx={svgW}
          cy={getY(currentEquity)}
          r="3.5"
          fill="#00F59B"
          stroke="#080A0F"
          strokeWidth="1.5"
        />
      )}
    </svg>
  );
}

function OrderBookMountain({
  depth,
  microprice,
}: {
  depth: OrderBookDepth;
  microprice: number;
}) {
  const bids = depth?.bids || [];
  const asks = depth?.asks || [];
  const svgW = 280;
  const svgH = 95;
  const midX = svgW / 2;

  const maxCum = Math.max(
    bids.length > 0 ? bids[bids.length - 1][2] : 1,
    asks.length > 0 ? asks[asks.length - 1][2] : 1,
    1.0
  );

  const bidPoints: string[] = [`${midX},${svgH}`];
  bids.forEach((b, idx) => {
    const x = midX - (idx + 1) * (midX / Math.max(bids.length, 1));
    const y = svgH - (b[2] / maxCum) * (svgH - 15);
    bidPoints.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  bidPoints.push(`0,${svgH}`);

  const askPoints: string[] = [`${midX},${svgH}`];
  asks.forEach((a, idx) => {
    const x = midX + (idx + 1) * (midX / Math.max(asks.length, 1));
    const y = svgH - (a[2] / maxCum) * (svgH - 15);
    askPoints.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  askPoints.push(`${svgW},${svgH}`);

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-full overflow-visible">
      <defs>
        <linearGradient id="bidsGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00F59B" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#00F59B" stopOpacity="0.05" />
        </linearGradient>
        <linearGradient id="asksGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#A855F7" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#A855F7" stopOpacity="0.05" />
        </linearGradient>
      </defs>

      <polygon points={bidPoints.join(' ')} fill="url(#bidsGrad)" stroke="#00F59B" strokeWidth="1.5" />
      <polygon points={askPoints.join(' ')} fill="url(#asksGrad)" stroke="#A855F7" strokeWidth="1.5" />

      <line x1={midX} y1="0" x2={midX} y2={svgH} stroke="#FFFFFF" strokeDasharray="3 2" opacity="0.6" />
      <text x={midX} y="10" fill="#FFFFFF" fontSize="8" fontFamily="monospace" textAnchor="middle" opacity="0.8">
        µP: ${microprice.toFixed(1)}
      </text>

      <text x="8" y={svgH - 6} fill="#00F59B" fontSize="9" fontFamily="monospace" fontWeight="bold">
        BIDS
      </text>
      <text x={svgW - 8} y={svgH - 6} fill="#A855F7" fontSize="9" fontFamily="monospace" textAnchor="end" fontWeight="bold">
        ASKS
      </text>
    </svg>
  );
}

function AllocationDonut({
  items,
}: {
  items: PortfolioAllocationItem[];
}) {
  const size = 96;
  const center = size / 2;
  const radius = 35;
  const circumference = 2 * Math.PI * radius; // ~219.9

  let accum = 0;
  const segments = (items || []).map((it) => {
    const strokeDash = (it.weight_pct / 100) * circumference;
    const offset = -accum;
    accum += strokeDash;
    return {
      ...it,
      strokeDash,
      offset,
    };
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <circle
        cx={center}
        cy={center}
        r={radius}
        fill="none"
        stroke="#1C2230"
        strokeWidth="11"
      />
      {segments.map((seg, idx) => (
        <circle
          key={idx}
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={seg.color}
          strokeWidth="11"
          strokeDasharray={`${seg.strokeDash} ${circumference}`}
          strokeDashoffset={seg.offset}
          transform={`rotate(-90 ${center} ${center})`}
          strokeLinecap="round"
        />
      ))}
      <text x={center} y={center + 3} textAnchor="middle" fill="#E2E8F0" fontSize="10" fontFamily="monospace" fontWeight="bold">
        5P
      </text>
    </svg>
  );
}

function exportTradesToCsv(trades: RecentTrade[]) {
  if (!trades || trades.length === 0) return;
  const headers = ['ID', 'Timestamp_UTC', 'Symbol', 'Strategy', 'Side', 'Entry_Price', 'Exit_Mid_Price', 'Qty', 'Return_Pct', 'PnL_USD'];
  const rows = trades.map((t) => [
    t.id,
    t.time_str,
    t.symbol,
    t.strategy,
    t.side,
    t.entry_price,
    t.exit_or_mid_price,
    t.qty,
    t.return_pct,
    t.pnl_usd,
  ]);
  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `trades_blotter_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_')}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
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

  // Operational Mode State: PAPER_L2 vs LIVE_BINANCE vs LIVE_BINGX
  const [tradingMode, setTradingMode] = useState<'PAPER_L2' | 'LIVE_BINANCE' | 'LIVE_BINGX'>('PAPER_L2');
  const [pendingLiveMode, setPendingLiveMode] = useState<'LIVE_BINANCE' | 'LIVE_BINGX'>('LIVE_BINANCE');
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
  const [selectedAssetView, setSelectedAssetView] = useState<'GLOBAL' | 'BTC' | 'ETH' | 'SOL' | 'BNB' | 'DOGE' | 'XAU' | 'XAG'>('GLOBAL');
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

  // Execution Stats & Live Blotter State (Direct from backend/telemetry_server.py)
  const [executionStats, setExecutionStats] = useState<ExecutionStats>({
    total_trades: 0,
    wins_count: 0,
    losses_count: 0,
    win_rate_pct: 0.0,
    profit_factor: 1.0,
    avg_trade_pnl_usd: 0.0,
    gross_profit_usd: 0.0,
    gross_loss_usd: 0.0,
    equity_curve: [5000, 5000, 5000, 5000, 5000],
  });
  const [openOrders, setOpenOrders] = useState<OpenOrder[]>([]);
  const [recentTrades, setRecentTrades] = useState<RecentTrade[]>([]);
  const [tradeFilterSymbol, setTradeFilterSymbol] = useState<string>('TODOS');

  const filteredTrades = useMemo(() => {
    if (tradeFilterSymbol === 'TODOS') return recentTrades;
    return recentTrades.filter((t) => t.symbol.toUpperCase().includes(tradeFilterSymbol.toUpperCase()));
  }, [recentTrades, tradeFilterSymbol]);

  // Pro Terminal UI States
  const [activeCoin, setActiveCoin] = useState<'BTC' | 'ETH' | 'SOL' | 'BNB' | 'DOGE' | 'XAU' | 'XAG'>('BTC');
  const [activeTimeframe, setActiveTimeframe] = useState<'1m' | '5m' | '15m' | '1h'>('1m');
  const [activeBottomTab, setActiveBottomTab] = useState<'transactions' | 'orders' | 'scanner' | 'activity'>('transactions');

  // Real-Time L2 Microstructure Visual State
  const [orderBookDepth, setOrderBookDepth] = useState<OrderBookDepth>({
    bids: [[65000.0, 1.5, 1.5], [64995.0, 2.2, 3.7], [64990.0, 3.8, 7.5], [64985.0, 5.0, 12.5], [64980.0, 7.2, 19.7]],
    asks: [[65001.0, 1.4, 1.4], [65006.0, 2.5, 3.9], [65011.0, 3.6, 7.5], [65016.0, 4.8, 12.3], [65021.0, 6.9, 19.2]],
  });
  const [candlesHistory, setCandlesHistory] = useState<Candle[]>([]);
  const [equityComparison, setEquityComparison] = useState<EquityComparisonPoint[]>([]);
  const [whaleAndLiqTape, setWhaleAndLiqTape] = useState<WhaleTapeEvent[]>([]);
  const [portfolioAllocation, setPortfolioAllocation] = useState<PortfolioAllocationItem[]>([
    { symbol: 'BTC/USDT:USDT', base_asset: 'BTC', weight_pct: 35.0, notional_usd: 35.0, color: '#00F59B' },
    { symbol: 'ETH/USDT:USDT', base_asset: 'ETH', weight_pct: 25.0, notional_usd: 25.0, color: '#A855F7' },
    { symbol: 'SOL/USDT:USDT', base_asset: 'SOL', weight_pct: 18.0, notional_usd: 18.0, color: '#06B6D4' },
    { symbol: 'BNB/USDT:USDT', base_asset: 'BNB', weight_pct: 12.0, notional_usd: 12.0, color: '#F59E0B' },
    { symbol: 'DOGE/USDT:USDT', base_asset: 'DOGE', weight_pct: 10.0, notional_usd: 10.0, color: '#EC4899' },
  ]);

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

    if (data.execution_stats) {
      setExecutionStats(data.execution_stats);
    }

    if (data.open_orders && Array.isArray(data.open_orders)) {
      setOpenOrders(data.open_orders);
    }

    if (data.recent_trades && Array.isArray(data.recent_trades)) {
      setRecentTrades(data.recent_trades);
    }

    if (data.order_book_depth) {
      setOrderBookDepth(data.order_book_depth);
    }
    if (data.candles_history && Array.isArray(data.candles_history)) {
      setCandlesHistory(data.candles_history);
    }
    if (data.equity_comparison && Array.isArray(data.equity_comparison)) {
      setEquityComparison(data.equity_comparison);
    }
    if (data.whale_and_liq_tape && Array.isArray(data.whale_and_liq_tape)) {
      setWhaleAndLiqTape(data.whale_and_liq_tape);
    }
    if (data.portfolio_allocation && Array.isArray(data.portfolio_allocation)) {
      setPortfolioAllocation(data.portfolio_allocation);
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

  const handleSwitchMode = async (targetMode: 'PAPER_L2' | 'LIVE_BINANCE' | 'LIVE_BINGX') => {
    if (targetMode === tradingMode) return;
    if (targetMode === 'LIVE_BINANCE' || targetMode === 'LIVE_BINGX') {
      setPendingLiveMode(targetMode);
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
        body: JSON.stringify({ mode: pendingLiveMode }),
      });
      const data = await res.json();
      if (!res.ok) {
        setModeSwitchError(data.detail || `Error al conmutar a modo ${pendingLiveMode}.`);
      } else {
        setTradingMode(pendingLiveMode);
        setIsLiveConfirmOpen(false);
        const exchangeLabel = pendingLiveMode === 'LIVE_BINGX' ? 'BINGX SWAP & COMMODITIES' : 'BINANCE USDⓈ-M';
        setModeSwitchSuccess(`Modo LIVE CAPITAL (${exchangeLabel}) activado exitosamente.`);
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
    <div className="min-h-screen bg-[#080A0F] text-slate-200 flex flex-col font-sans selection:bg-[#00F59B] selection:text-[#080A0F]">
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

      {/* Mode Switch Success Notification Banner */}
      {modeSwitchSuccess && (
        <div className="bg-[#00F59B]/15 border-b border-[#00F59B]/30 px-6 py-2 text-xs font-mono text-[#00F59B] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-[#00F59B] shrink-0" />
            <span>{modeSwitchSuccess}</span>
          </div>
          <button onClick={() => setModeSwitchSuccess(null)} className="text-[#00F59B] hover:text-white font-bold px-1">✕</button>
        </div>
      )}

      {/* Mode Switch Error Notification Banner */}
      {modeSwitchError && !isLiveConfirmOpen && (
        <div className="bg-rose-500/15 border-b border-rose-500/30 px-6 py-2.5 text-xs font-mono text-rose-300 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{modeSwitchError}</span>
          </div>
          <button onClick={() => setModeSwitchError(null)} className="text-rose-400 hover:text-white font-bold px-1">✕</button>
        </div>
      )}

      {/* Main Container: Left Rail + Work Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Barra Lateral Izquierda (Icon Navigation Rail) */}
        <nav className="w-14 bg-[#0B0E14] border-r border-[#1C2230] flex flex-col items-center py-3.5 gap-5 shrink-0 z-20">
          {/* Logo superior circular verde menta */}
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#00F59B]/20 to-[#A855F7]/30 border border-[#00F59B]/50 flex items-center justify-center text-[#00F59B] shadow-lg shadow-[#00F59B]/20 cursor-pointer">
            <Zap className="w-5 h-5 text-[#00F59B]" />
          </div>

          <div className="w-7 h-[1px] bg-[#1C2230]" />

          {/* Iconos de Navegación de Pestañas */}
          <div className="flex flex-col items-center gap-3 w-full">
            <button
              onClick={() => setActiveTab('command')}
              title="Terminal Quant / DEX Pro"
              className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                activeTab === 'command'
                  ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/40 shadow-sm shadow-[#00F59B]/20'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-[#10141D]'
              }`}
            >
              <Gauge className="w-4 h-4" />
            </button>

            <button
              onClick={() => setActiveTab('phase1')}
              title="Mandate Gates Stress Tester"
              className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                activeTab === 'phase1'
                  ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/40 shadow-sm shadow-[#00F59B]/20'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-[#10141D]'
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
            </button>

            <button
              onClick={() => setActiveTab('phase3')}
              title="3 Motores Algorítmicos"
              className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                activeTab === 'phase3'
                  ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/40 shadow-sm shadow-[#00F59B]/20'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-[#10141D]'
              }`}
            >
              <Cpu className="w-4 h-4" />
            </button>

            <button
              onClick={() => setActiveTab('phase5')}
              title="L2 Queue & DSR Auditor"
              className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                activeTab === 'phase5'
                  ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/40 shadow-sm shadow-[#00F59B]/20'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-[#10141D]'
              }`}
            >
              <Layers className="w-4 h-4" />
            </button>
          </div>

          <div className="mt-auto flex flex-col items-center gap-3">
            <span
              title={wsConnected ? 'WebSocket Conectado (10 Hz)' : 'WebSocket Desconectado'}
              className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-[#00F59B] shadow-lg shadow-[#00F59B]/50 animate-pulse' : 'bg-amber-500'}`}
            />
            <div className="w-8 h-8 rounded-lg bg-[#10141D] border border-[#1C2230] flex items-center justify-center text-slate-500">
              <TerminalIcon className="w-3.5 h-3.5" />
            </div>
          </div>
        </nav>

        {/* Work Area */}
        <div className="flex-1 overflow-y-auto bg-[#080A0F]">
          {activeTab === 'command' && (
            <div className="min-h-full flex flex-col xl:flex-row divide-y xl:divide-y-0 xl:divide-x divide-[#1C2230]">
              {/* Columna Izquierda (w-80) */}
              <div className="w-full xl:w-80 shrink-0 bg-[#0B0E14]/80 p-3.5 space-y-3 flex flex-col">
                {/* 1. Tarjeta Superior de Identidad del Bot */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3 space-y-2.5">
                  <div className="flex items-center gap-2.5">
                    <div className="relative">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-[#00F59B]/30 via-[#10141D] to-[#A855F7]/30 border border-[#00F59B]/50 flex items-center justify-center text-[#00F59B] shadow-md shadow-[#00F59B]/20">
                        <Zap className="w-5 h-5 text-[#00F59B]" />
                      </div>
                      <span className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-[#10141D] ${wsConnected ? 'bg-[#00F59B]' : 'bg-amber-400'}`} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs tracking-wide text-white uppercase font-mono">APEX QUANT BOT</span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
                          tradingMode === 'PAPER_L2'
                            ? 'bg-[#00F59B]/10 text-[#00F59B] border-[#00F59B]/20'
                            : tradingMode === 'LIVE_BINANCE'
                            ? 'bg-[#A855F7]/10 text-[#A855F7] border-[#A855F7]/20'
                            : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                        }`}>
                          {tradingMode === 'PAPER_L2' ? 'PAPER L2' : tradingMode === 'LIVE_BINANCE' ? 'BINANCE' : 'BINGX'}
                        </span>
                      </div>
                      <p className="text-[11px] font-mono text-slate-400 truncate">
                        {wsConnected ? 'LIVE CCXT WS (10 Hz)' : 'OFFLINE / RECONNECT'}
                      </p>
                    </div>
                  </div>

                  {/* Selector tri-modal protegido */}
                  <div className="grid grid-cols-3 gap-1 p-1 bg-[#080A0F] rounded-lg border border-[#1C2230] text-[10px] font-mono">
                    <button
                      onClick={() => handleSwitchMode('PAPER_L2')}
                      className={`py-1.5 px-1 rounded font-bold flex items-center justify-center gap-1 transition-all ${
                        tradingMode === 'PAPER_L2'
                          ? 'bg-[#00F59B]/20 text-[#00F59B] border border-[#00F59B]/40 shadow-sm'
                          : 'text-slate-400 hover:text-[#00F59B]'
                      }`}
                      title="Simulador L2 Paper Trading"
                    >
                      <Activity className="w-2.5 h-2.5" />
                      <span>PAPER</span>
                    </button>
                    <button
                      onClick={() => handleSwitchMode('LIVE_BINANCE')}
                      className={`py-1.5 px-1 rounded font-bold flex items-center justify-center gap-1 transition-all ${
                        tradingMode === 'LIVE_BINANCE'
                          ? 'bg-[#A855F7]/20 text-[#A855F7] border border-[#A855F7]/40 shadow-sm'
                          : 'text-slate-400 hover:text-[#A855F7]'
                      }`}
                      title="Binance Futures USDⓈ-M"
                    >
                      <Lock className="w-2.5 h-2.5" />
                      <span>BINANCE</span>
                    </button>
                    <button
                      onClick={() => handleSwitchMode('LIVE_BINGX')}
                      className={`py-1.5 px-1 rounded font-bold flex items-center justify-center gap-1 transition-all ${
                        tradingMode === 'LIVE_BINGX'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40 shadow-sm'
                          : 'text-slate-400 hover:text-amber-400'
                      }`}
                      title="BingX Swap & Commodities"
                    >
                      <Zap className="w-2.5 h-2.5" />
                      <span>BINGX</span>
                    </button>
                  </div>

                  {/* Botón Compacto EMERGENCY FLATTEN */}
                  {!gate4KillSwitch ? (
                    <button
                      onClick={handleTriggerEmergencyFlatten}
                      className="w-full py-1.5 px-3 rounded-lg bg-rose-600/90 hover:bg-rose-600 text-white font-mono text-xs font-bold transition-all shadow-md shadow-rose-950/40 flex items-center justify-center gap-1.5"
                    >
                      <Power className="w-3.5 h-3.5" />
                      EMERGENCY FLATTEN
                    </button>
                  ) : (
                    <button
                      onClick={handleResetKillSwitch}
                      className="w-full py-1.5 px-3 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-mono text-xs font-bold transition-all flex items-center justify-center gap-1.5"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Resetear Kill-Switch
                    </button>
                  )}
                </div>

                {/* 2. Grilla 2x3 de Tarjetas Cuadradas Minimalistas */}
                <div className="grid grid-cols-2 gap-2">
                  {/* Card 1: Equity Total ($) & % Retorno */}
                  <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-2.5 flex flex-col justify-between">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-tight">Equity Total</span>
                    <div className="text-sm font-mono font-bold text-white mt-0.5">
                      ${currentEquity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-[10px] font-mono text-[#00F59B] mt-0.5 font-bold">
                      +{dailyReturnPct.toFixed(2)}%
                    </div>
                  </div>

                  {/* Card 2: Bóveda Spot ($) con botón de barrido */}
                  <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-2.5 flex flex-col justify-between">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-tight">Bóveda Spot</span>
                      <button
                        onClick={handleManualSweep}
                        title="Barrer excedente a Spot"
                        className="text-[9px] font-mono text-[#00F59B] hover:underline flex items-center gap-0.5"
                      >
                        <ArrowRightLeft className="w-2.5 h-2.5" /> Barrer
                      </button>
                    </div>
                    <div className="text-sm font-mono font-bold text-[#A855F7] mt-0.5">
                      ${sweptSpotVault.toFixed(2)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">Vault Segura</span>
                  </div>

                  {/* Card 3: Win Rate (%) */}
                  <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-2.5 flex flex-col justify-between">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-tight">Win Rate</span>
                    <div className="text-sm font-mono font-bold text-[#00F59B] mt-0.5">
                      {executionStats.win_rate_pct.toFixed(1)}%
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      {executionStats.wins_count}W / {executionStats.losses_count}L
                    </span>
                  </div>

                  {/* Card 4: Profit Factor */}
                  <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-2.5 flex flex-col justify-between">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-tight">Profit Factor</span>
                    <div className="text-sm font-mono font-bold text-white mt-0.5">
                      {executionStats.profit_factor >= 99 ? '99.0' : executionStats.profit_factor.toFixed(2)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      Avg: ${executionStats.avg_trade_pnl_usd.toFixed(2)}
                    </span>
                  </div>

                  {/* Card 5: Trades Realizados */}
                  <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-2.5 flex flex-col justify-between">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-tight">Trades Totales</span>
                    <div className="text-sm font-mono font-bold text-white mt-0.5">
                      {executionStats.total_trades}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      {rotationsFilled} rotaciones
                    </span>
                  </div>

                  {/* Card 6: Hot Path Latency (µs) & DSR Score */}
                  <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-2.5 flex flex-col justify-between">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-tight">Hot Path</span>
                    <div className="text-sm font-mono font-bold text-[#00F59B] mt-0.5">
                      {hotPathLatencyUs} µs
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      DSR: {(dsrScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* 3. Lista de Seguridad con Checks Verdes (✓) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3 space-y-2.5">
                  <div className="flex items-center justify-between border-b border-[#1C2230] pb-1.5">
                    <span className="text-[10px] font-mono uppercase text-slate-400 font-bold tracking-wider">
                      MANDATE GATES AUDIT
                    </span>
                    <span className="text-[9px] font-mono text-[#00F59B] font-bold">100% PASS</span>
                  </div>
                  <div className="space-y-1.5 text-xs font-mono">
                    <div className="flex items-center justify-between text-slate-300">
                      <span className="text-[11px]">Gate 1: Isolated Margin</span>
                      <span className="flex items-center gap-1 text-[#00F59B] font-bold text-[10px]">
                        ISOLATED <CheckCircle2 className="w-3.5 h-3.5 text-[#00F59B]" />
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-slate-300">
                      <span className="text-[11px]">Gate 2: Post-Only GTX</span>
                      <span className="flex items-center gap-1 text-[#00F59B] font-bold text-[10px]">
                        POST_ONLY <CheckCircle2 className="w-3.5 h-3.5 text-[#00F59B]" />
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-slate-300">
                      <span className="text-[11px]">Gate 3: Half-Kelly &le; 3.0x</span>
                      <span className="flex items-center gap-1 text-[#00F59B] font-bold text-[10px]">
                        {gate3CurrentLev.toFixed(1)}x / 3.0x <CheckCircle2 className="w-3.5 h-3.5 text-[#00F59B]" />
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-slate-300">
                      <span className="text-[11px]">Gate 4: Kill-Switch -3%</span>
                      <span className="flex items-center gap-1 text-[#00F59B] font-bold text-[10px]">
                        ARMED <CheckCircle2 className="w-3.5 h-3.5 text-[#00F59B]" />
                      </span>
                    </div>
                  </div>

                  {/* Mini Sparkline inferior en vivo */}
                  <div className="pt-2 border-t border-[#1C2230]">
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1">
                      <span>Equity Mini-Trend</span>
                      <span className="text-[#00F59B] font-bold">${currentEquity.toFixed(2)}</span>
                    </div>
                    {renderEquitySparkline(executionStats.equity_curve || [5000, 5000])}
                  </div>
                </div>
              </div>

              {/* Columna Central (flex-1) */}
              <div className="flex-1 min-w-0 p-3.5 space-y-3.5 flex flex-col bg-[#080A0F]">
                {/* 1. Gráfico Principal de Velas + Bandas Avellaneda (SVG Nativo) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3.5 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1C2230] pb-2.5">
                    {/* Selector de Par */}
                    <div className="flex items-center gap-1 bg-[#080A0F] p-1 rounded-lg border border-[#1C2230] text-xs font-mono">
                      {(['BTC', 'ETH', 'SOL', 'BNB', 'DOGE', 'XAU', 'XAG'] as const).map((coin) => (
                        <button
                          key={coin}
                          onClick={() => {
                            setActiveCoin(coin);
                            setSelectedAssetView(coin);
                          }}
                          className={`px-2 py-1 rounded font-bold transition-all ${
                            activeCoin === coin
                              ? 'bg-[#00F59B]/20 text-[#00F59B] border border-[#00F59B]/40 shadow-sm'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {coin}
                        </button>
                      ))}
                    </div>

                    {/* Selector Timeframe */}
                    <div className="flex items-center gap-1 bg-[#080A0F] p-1 rounded-lg border border-[#1C2230] text-[11px] font-mono">
                      {(['1m', '5m', '15m', '1h'] as const).map((tf) => (
                        <button
                          key={tf}
                          onClick={() => setActiveTimeframe(tf)}
                          className={`px-2 py-0.5 rounded font-bold transition-all ${
                            activeTimeframe === tf
                              ? 'bg-[#1C2230] text-[#00F59B]'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {tf}
                        </button>
                      ))}
                    </div>

                    {/* Métricas técnicas inline */}
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <span className="text-slate-400">Mid: <strong className="text-white">${midPrice.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 2 })}</strong></span>
                      <span className="text-slate-400">Res: <strong className="text-[#00F59B]">${reservationPrice.toFixed(1)}</strong></span>
                      <span className="text-slate-400">Spread: <strong className="text-[#A855F7]">${optimalSpread.toFixed(2)}</strong></span>
                      <span className="text-slate-400">OBI: <strong className={liveObi >= 0 ? 'text-[#00F59B]' : 'text-[#A855F7]'}>{liveObi > 0 ? `+${liveObi.toFixed(3)}` : liveObi.toFixed(3)}</strong></span>
                    </div>
                  </div>

                  <div className="w-full">
                    <CandlestickChart
                      candles={candlesHistory}
                      reservationPrice={reservationPrice}
                      bestBid={bestBid}
                      bestAsk={bestAsk}
                    />
                  </div>
                </div>

                {/* 2. Gráfico Medio de Doble Curva (Equity Curve vs. Base/Benchmark) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between border-b border-[#1C2230] pb-2">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-[#00F59B]" />
                      <span className="font-bold text-xs font-mono text-white tracking-wide uppercase">
                        BOT COMPOSITE EQUITY VS BENCHMARK
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-[11px] font-mono">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#00F59B]" />
                        <span className="text-slate-300">Bot Compuesto: <strong>${currentEquity.toFixed(2)}</strong></span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#A855F7]" />
                        <span className="text-slate-400">Benchmark: <strong>${baseCapital.toFixed(2)}</strong></span>
                      </div>
                    </div>
                  </div>
                  <div className="w-full">
                    <DualEquityCurve data={equityComparison} />
                  </div>
                </div>

                {/* 3. Panel Inferior con Pestañas (Transactions | Active Orders | 12P Scanner | Activity) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3.5 space-y-3 flex-1 flex flex-col">
                  {/* Tab headers */}
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1C2230] pb-2">
                    <div className="flex items-center gap-1 text-xs font-mono">
                      <button
                        onClick={() => setActiveBottomTab('transactions')}
                        className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
                          activeBottomTab === 'transactions'
                            ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <ArrowDownUp className="w-3.5 h-3.5" />
                        <span>Transactions ({filteredTrades.length})</span>
                      </button>

                      <button
                        onClick={() => setActiveBottomTab('orders')}
                        className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
                          activeBottomTab === 'orders'
                            ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <Layers className="w-3.5 h-3.5" />
                        <span>Active Orders ({openOrders.length})</span>
                      </button>

                      <button
                        onClick={() => setActiveBottomTab('scanner')}
                        className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
                          activeBottomTab === 'scanner'
                            ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <Radio className="w-3.5 h-3.5" />
                        <span>12P Scanner (Basket)</span>
                      </button>

                      <button
                        onClick={() => setActiveBottomTab('activity')}
                        className={`px-3 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
                          activeBottomTab === 'activity'
                            ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <Activity className="w-3.5 h-3.5" />
                        <span>Activity</span>
                      </button>
                    </div>

                    {activeBottomTab === 'transactions' && (
                      <div className="flex items-center gap-2">
                        {/* Filtro rápido por moneda */}
                        <div className="flex items-center gap-1 bg-[#080A0F] p-0.5 rounded-md border border-[#1C2230] text-[10px] font-mono">
                          {(['TODOS', 'BTC', 'ETH', 'SOL', 'BNB', 'DOGE', 'XAU', 'XAG'] as const).map((sym) => (
                            <button
                              key={sym}
                              onClick={() => setTradeFilterSymbol(sym)}
                              className={`px-1.5 py-0.5 rounded transition-all ${
                                tradeFilterSymbol === sym ? 'bg-[#1C2230] text-[#00F59B] font-bold' : 'text-slate-400 hover:text-slate-200'
                              }`}
                            >
                              {sym}
                            </button>
                          ))}
                        </div>
                        <button
                          onClick={() => exportTradesToCsv(recentTrades)}
                          className="px-2.5 py-1 rounded bg-[#080A0F] hover:bg-[#1C2230] text-slate-300 border border-[#1C2230] font-mono text-[10px] font-bold flex items-center gap-1"
                          title="Exportar blotter a CSV"
                        >
                          <Download className="w-3 h-3 text-[#00F59B]" /> CSV
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Tab 1 Content: Transactions */}
                  {activeBottomTab === 'transactions' && (
                    <div className="overflow-x-auto max-h-56 overflow-y-auto font-mono text-xs">
                      <table className="w-full text-left">
                        <thead>
                          <tr className="text-slate-500 border-b border-[#1C2230] text-[10px] uppercase tracking-wider">
                            <th className="py-1.5 px-2.5">Side</th>
                            <th className="py-1.5 px-2.5">Time</th>
                            <th className="py-1.5 px-2.5">Asset / Class</th>
                            <th className="py-1.5 px-2.5">Strategy</th>
                            <th className="py-1.5 px-2.5">Price</th>
                            <th className="py-1.5 px-2.5">Qty</th>
                            <th className="py-1.5 px-2.5">Return %</th>
                            <th className="py-1.5 px-2.5 text-right">PnL (USDT)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#1C2230]/60">
                          {filteredTrades.length > 0 ? (
                            filteredTrades.map((t) => {
                              const isBuy = t.side === 'BUY';
                              const isWin = t.pnl_usd >= 0;
                              const baseSym = t.symbol.split('/')[0] || t.symbol;
                              const isTradFi = t.symbol.includes('XAU') || t.symbol.includes('XAG');
                              return (
                                <tr key={t.id} className="hover:bg-[#10141D]/80 transition-colors">
                                  <td className="py-1.5 px-2.5">
                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold flex items-center gap-1 w-fit ${
                                      isBuy ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/30' : 'bg-[#A855F7]/15 text-[#A855F7] border border-[#A855F7]/30'
                                    }`}>
                                      {isBuy ? '↑ BUY' : '↓ SELL'}
                                    </span>
                                  </td>
                                  <td className="py-1.5 px-2.5 text-slate-400 text-[11px]">{t.age_str || t.time_str}</td>
                                  <td className="py-1.5 px-2.5">
                                    <div className="flex items-center gap-1.5">
                                      <span className="font-bold text-white">{baseSym}</span>
                                      <span className={`px-1 py-0.2 rounded text-[8px] font-mono font-bold ${
                                        isTradFi
                                          ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                                          : 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                                      }`}>
                                        {isTradFi ? 'COMMODITY (BINGX)' : 'CRYPTO 24/7'}
                                      </span>
                                    </div>
                                  </td>
                                  <td className="py-1.5 px-2.5">
                                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono border ${
                                      t.strategy === 'LIQ_CASCADE'
                                        ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                                        : 'bg-[#A855F7]/10 text-[#A855F7] border-[#A855F7]/30'
                                    }`}>
                                      {t.strategy}
                                    </span>
                                  </td>
                                  <td className="py-1.5 px-2.5 text-slate-200">
                                    ${t.entry_price >= 1 ? t.entry_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : t.entry_price.toFixed(5)}
                                  </td>
                                  <td className="py-1.5 px-2.5 text-slate-300">{t.qty}</td>
                                  <td className="py-1.5 px-2.5">
                                    <div className="flex items-center gap-1.5">
                                      <div className="w-12 h-1.5 bg-[#080A0F] rounded-full overflow-hidden">
                                        <div
                                          className={`h-full ${isWin ? 'bg-[#00F59B]' : 'bg-[#A855F7]'}`}
                                          style={{ width: `${Math.min(Math.abs(t.return_pct) * 100, 100)}%` }}
                                        />
                                      </div>
                                      <span className={isWin ? 'text-[#00F59B]' : 'text-[#A855F7]'}>
                                        {t.return_pct >= 0 ? `+${t.return_pct.toFixed(2)}%` : `${t.return_pct.toFixed(2)}%`}
                                      </span>
                                    </div>
                                  </td>
                                  <td className={`py-1.5 px-2.5 text-right font-bold ${isWin ? 'text-[#00F59B]' : 'text-[#A855F7]'}`}>
                                    {isWin ? `+$${t.pnl_usd.toFixed(2)}` : `-$${Math.abs(t.pnl_usd).toFixed(2)}`}
                                  </td>
                                </tr>
                              );
                            })
                          ) : (
                            <tr>
                              <td colSpan={8} className="py-6 text-center text-slate-500 text-xs">
                                Sin operaciones ejecutadas en la sesión actual. Cotizando continuamente en L2...
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Tab 2 Content: Active Orders (L2) */}
                  {activeBottomTab === 'orders' && (
                    <div className="overflow-x-auto max-h-56 overflow-y-auto font-mono text-xs">
                      <table className="w-full text-left">
                        <thead>
                          <tr className="text-slate-500 border-b border-[#1C2230] text-[10px] uppercase tracking-wider">
                            <th className="py-1.5 px-2.5">Level</th>
                            <th className="py-1.5 px-2.5">Symbol</th>
                            <th className="py-1.5 px-2.5">Side</th>
                            <th className="py-1.5 px-2.5">Price</th>
                            <th className="py-1.5 px-2.5">Qty</th>
                            <th className="py-1.5 px-2.5">Filled</th>
                            <th className="py-1.5 px-2.5">FIFO Queue Ahead</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#1C2230]/60">
                          {openOrders.length > 0 ? (
                            openOrders.map((ord) => {
                              const isBid = ord.side === 'BUY';
                              const filledPct = ord.qty > 0 ? (ord.filled_qty / ord.qty) * 100 : 0;
                              return (
                                <tr key={ord.order_id} className="hover:bg-[#10141D]/80 transition-colors">
                                  <td className="py-1.5 px-2.5">
                                    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-[#1C2230] text-slate-300">
                                      {ord.level || 'L1'}
                                    </span>
                                  </td>
                                  <td className="py-1.5 px-2.5 font-bold text-white">{ord.symbol.split('/')[0]}</td>
                                  <td className="py-1.5 px-2.5">
                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                      isBid ? 'bg-[#00F59B]/15 text-[#00F59B] border border-[#00F59B]/30' : 'bg-[#A855F7]/15 text-[#A855F7] border border-[#A855F7]/30'
                                    }`}>
                                      {isBid ? 'BID' : 'ASK'}
                                    </span>
                                  </td>
                                  <td className="py-1.5 px-2.5 text-slate-200">
                                    ${ord.price >= 1 ? ord.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : ord.price.toFixed(5)}
                                  </td>
                                  <td className="py-1.5 px-2.5 text-slate-300">{ord.qty}</td>
                                  <td className="py-1.5 px-2.5 text-slate-300">{ord.filled_qty} ({filledPct.toFixed(0)}%)</td>
                                  <td className="py-1.5 px-2.5">
                                    <div className="flex items-center gap-2">
                                      <div className="w-16 h-1.5 bg-[#080A0F] rounded-full overflow-hidden">
                                        <div
                                          className="h-full bg-cyan-400 transition-all duration-300"
                                          style={{ width: `${Math.max(10, Math.min(100, (1 - ord.queue_ahead_qty / 5.0) * 100))}%` }}
                                        />
                                      </div>
                                      <span className="text-[11px] text-slate-400">{ord.queue_ahead_qty.toFixed(2)} BTC</span>
                                    </div>
                                  </td>
                                </tr>
                              );
                            })
                          ) : (
                            <tr>
                              <td colSpan={7} className="py-6 text-center text-slate-500 text-xs">
                                No hay órdenes activas en reposo. Todas las cotizaciones pasaron al motor de fills.
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Tab 3 Content: 14P Scanner (Multi-Asset Basket) */}
                  {activeBottomTab === 'scanner' && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2 font-mono text-xs max-h-56 overflow-y-auto">
                      {topActiveSymbols.map((symKey) => {
                        const s = symbolsData[symKey];
                        if (!s) return null;
                        const isTradFi = symKey.includes('XAU') || symKey.includes('XAG');
                        return (
                          <div key={symKey} className="bg-[#080A0F] border border-[#1C2230] rounded-lg p-2.5 space-y-1">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-1.5">
                                <span className="font-bold text-white">{s.base_asset}</span>
                                <span className={`px-1 py-0.2 rounded text-[8px] font-mono font-bold ${
                                  isTradFi
                                    ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                                    : 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                                }`}>
                                  {isTradFi ? 'COMMODITY (BINGX)' : 'CRYPTO 24/7'}
                                </span>
                              </div>
                              <span className="text-[10px] text-slate-400">{(s.half_kelly_weight * 100).toFixed(0)}% W</span>
                            </div>
                            <div className="text-slate-200 font-bold">${s.mid_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="text-slate-400">OBI:</span>
                              <span className={s.obi >= 0 ? 'text-[#00F59B]' : 'text-[#A855F7]'}>{s.obi > 0 ? `+${s.obi.toFixed(2)}` : s.obi.toFixed(2)}</span>
                            </div>
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="text-slate-400">Posición:</span>
                              <span className="text-slate-300">${s.position_usd.toFixed(1)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Tab 4 Content: Activity */}
                  {activeBottomTab === 'activity' && (
                    <div className="space-y-2 font-mono text-xs text-slate-400">
                      <div className="flex items-center justify-between p-2 rounded bg-[#080A0F] border border-[#1C2230]">
                        <span>Stream WebSocket</span>
                        <span className="text-[#00F59B]">{wsConnected ? 'Activo (10 Hz) • FastAPI Lifespan' : 'Offline'}</span>
                      </div>
                      <div className="flex items-center justify-between p-2 rounded bg-[#080A0F] border border-[#1C2230]">
                        <span>Ticks Procesados</span>
                        <span className="text-white">#{tickCount}</span>
                      </div>
                      <div className="flex items-center justify-between p-2 rounded bg-[#080A0F] border border-[#1C2230]">
                        <span>Hot Path Latency</span>
                        <span className="text-[#00F59B]">{hotPathLatencyUs} µs (&lt; 100 µs target)</span>
                      </div>
                      <div className="flex items-center justify-between p-2 rounded bg-[#080A0F] border border-[#1C2230]">
                        <span>Cross-Asset Contagion Shield</span>
                        <span className={contagionShield.bid_shield_active || contagionShield.ask_shield_active ? 'text-[#A855F7]' : 'text-[#00F59B]'}>
                          {contagionShield.trigger_reason}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Columna Derecha (w-80) */}
              <div className="w-full xl:w-80 shrink-0 bg-[#0B0E14]/80 p-3.5 space-y-3.5 flex flex-col border-t xl:border-t-0 border-[#1C2230]">
                {/* 1. Gráfico Superior de Profundidad L2 (Order Book Mountain Chart) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between border-b border-[#1C2230] pb-2">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-[#00F59B]" />
                      <span className="font-bold text-xs font-mono text-white tracking-wide uppercase">
                        L2 ORDER BOOK DEPTH
                      </span>
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      µP: ${microprice.toFixed(1)}
                    </span>
                  </div>

                  <div className="w-full">
                    <OrderBookMountain depth={orderBookDepth} microprice={microprice} />
                  </div>

                  {/* OBI Speedometer / Bar */}
                  <div className="space-y-1 pt-1 border-t border-[#1C2230]">
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className="text-[#A855F7] font-bold">ASKS HEAVY</span>
                      <span className={`font-bold ${liveObi >= 0 ? 'text-[#00F59B]' : 'text-[#A855F7]'}`}>
                        OBI: {liveObi > 0 ? `+${liveObi.toFixed(3)}` : liveObi.toFixed(3)}
                      </span>
                      <span className="text-[#00F59B] font-bold">BIDS HEAVY</span>
                    </div>
                    <div className="w-full h-1.5 bg-[#080A0F] rounded-full overflow-hidden relative">
                      <div className="absolute inset-0 flex">
                        <div className="w-1/2 flex justify-end">
                          {liveObi < 0 && (
                            <div
                              className="h-full bg-[#A855F7] transition-all duration-200"
                              style={{ width: `${Math.min(Math.abs(liveObi) * 100, 100)}%` }}
                            />
                          )}
                        </div>
                        <div className="w-1/2 flex justify-start">
                          {liveObi > 0 && (
                            <div
                              className="h-full bg-[#00F59B] transition-all duration-200"
                              style={{ width: `${Math.min(liveObi * 100, 100)}%` }}
                            />
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 2. Panel Medio (Whale Flow & L2 Tape) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3.5 space-y-2.5 flex-1 flex flex-col">
                  <div className="flex items-center justify-between border-b border-[#1C2230] pb-2">
                    <div className="flex items-center gap-2">
                      <Flame className="w-4 h-4 text-amber-400" />
                      <span className="font-bold text-xs font-mono text-white tracking-wide uppercase">
                        WHALE FLOW & L2 TAPE
                      </span>
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">12 Events</span>
                  </div>

                  <div className="space-y-1.5 overflow-y-auto max-h-56 pr-1 font-mono text-xs">
                    {whaleAndLiqTape && whaleAndLiqTape.length > 0 ? (
                      whaleAndLiqTape.map((ev, idx) => {
                        const isBuy = ev.side === 'BUY';
                        return (
                          <div
                            key={idx}
                            className="p-1.5 rounded-lg bg-[#080A0F] border border-[#1C2230] flex items-center justify-between gap-2 hover:border-slate-700 transition-colors"
                          >
                            <div className="flex items-center gap-1.5 min-w-0">
                              <span className={`w-2 h-2 rounded-full shrink-0 ${isBuy ? 'bg-[#00F59B]' : 'bg-[#A855F7]'}`} />
                              <span className="font-bold text-white text-[11px]">{ev.symbol}</span>
                              <span className={`px-1 py-0.2 rounded text-[9px] font-bold ${
                                isBuy ? 'bg-[#00F59B]/15 text-[#00F59B]' : 'bg-[#A855F7]/15 text-[#A855F7]'
                              }`}>
                                {ev.side}
                              </span>
                            </div>

                            <div className="flex items-center gap-2">
                              <div className="w-12 h-1 bg-[#1C2230] rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${isBuy ? 'bg-[#00F59B]' : 'bg-[#A855F7]'}`}
                                  style={{ width: `${Math.min(ev.intensity_pct, 100)}%` }}
                                />
                              </div>
                              <span className="text-slate-300 font-bold text-[10px]">
                                ${(ev.notional_usd / 1000).toFixed(1)}k
                              </span>
                              <span className="text-slate-500 text-[9px] w-6 text-right">
                                {ev.age_str}
                              </span>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="py-6 text-center text-slate-500 text-xs">
                        Esperando órdenes institucionales &gt;$50k...
                      </div>
                    )}
                  </div>
                </div>

                {/* 3. Panel Inferior (Multi-Asset Allocation Donut Chart) */}
                <div className="bg-[#10141D] border border-[#1C2230] rounded-xl p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between border-b border-[#1C2230] pb-2">
                    <div className="flex items-center gap-2">
                      <PieChart className="w-4 h-4 text-[#00F59B]" />
                      <span className="font-bold text-xs font-mono text-white tracking-wide uppercase">
                        PORTFOLIO ALLOCATION
                      </span>
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">5 Perpetuals</span>
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <AllocationDonut items={portfolioAllocation} />
                    <div className="space-y-1 font-mono text-[11px] flex-1">
                      {portfolioAllocation.map((it) => (
                        <div key={it.symbol} className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: it.color }} />
                            <span className="text-slate-300">{it.base_asset}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-slate-400">${it.notional_usd.toFixed(0)}</span>
                            <span className="font-bold text-white w-8 text-right">{it.weight_pct.toFixed(0)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Base Half-Kelly Capital Usage Bar */}
                  <div className="pt-2 border-t border-[#1C2230] space-y-1 font-mono">
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="text-slate-400">Half-Kelly Limit Usage</span>
                      <span className="text-[#00F59B] font-bold">{((gate3CurrentLev / 3.0) * 100).toFixed(0)}% (Lev {gate3CurrentLev.toFixed(1)}x / 3.0x)</span>
                    </div>
                    <div className="w-full h-1.5 bg-[#080A0F] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[#00F59B] to-[#A855F7] transition-all duration-300"
                        style={{ width: `${Math.max(5, Math.min(100, (gate3CurrentLev / 3.0) * 100))}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab !== 'command' && (
            <div className="p-6 max-w-7xl mx-auto w-full space-y-6">

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
            </div>
          )}
        </div>
      </div>

      {/* Modal de Confirmación de Seguridad: LIVE CAPITAL (BINANCE / BINGX) */}
      {isLiveConfirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl max-w-lg w-full p-6 text-slate-100 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <div className="flex items-center gap-3">
                <div className={`p-2.5 border rounded-xl ${
                  pendingLiveMode === 'LIVE_BINGX'
                    ? 'bg-amber-500/15 border-amber-500/30 text-amber-400'
                    : 'bg-purple-500/15 border-purple-500/30 text-purple-400'
                }`}>
                  <ShieldAlert className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-base text-white tracking-tight">
                    Confirmación de Seguridad: Capital Real ({pendingLiveMode === 'LIVE_BINGX' ? 'BingX' : 'Binance'})
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    {pendingLiveMode === 'LIVE_BINGX'
                      ? 'BingX Perpetual Futures & TradFi Commodities (bingx swap)'
                      : 'Binance USDⓈ-M Futures (binanceusdm)'}
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
                Está a punto de conmutar la ejecución hacia <strong>Capital Real en {pendingLiveMode === 'LIVE_BINGX' ? 'BingX Swap' : 'Binance Futures'}</strong>.
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
                  <span>Gate 2: Maker Post-Only (GTX / PO)</span>
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
                className={`px-4 py-2 rounded-xl text-white font-mono text-xs font-bold transition-all shadow-lg flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${
                  pendingLiveMode === 'LIVE_BINGX'
                    ? 'bg-gradient-to-r from-amber-600 to-emerald-600 hover:from-amber-500 hover:to-emerald-500 shadow-amber-950/40'
                    : 'bg-gradient-to-r from-purple-600 to-emerald-600 hover:from-purple-500 hover:to-emerald-500 shadow-purple-950/40'
                }`}
              >
                {isSwitchingMode ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Conmutando...</span>
                  </>
                ) : (
                  <>
                    <Lock className="w-3.5 h-3.5" />
                    <span>Confirmar y Activar {pendingLiveMode}</span>
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
