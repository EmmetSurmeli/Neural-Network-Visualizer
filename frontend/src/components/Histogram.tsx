import type {HistogramBin} from '../types';
export const fmt = (n: number) => Math.abs(n) >= 1000 ? n.toExponential(2) : n.toFixed(3);
export default function Histogram({bins, color = 'var(--accent)', label}: {bins: HistogramBin[]; color?: string; label: string}) {
  const max = Math.max(1, ...bins.map(b => b.count));
  return <div className="histogram"><svg viewBox="0 0 300 78" role="img" aria-label={label}>
    {[20, 45, 70].map(y => <line key={y} x1="0" x2="300" y1={y} y2={y} stroke="#292929" strokeDasharray="3 4"/>)}
    {bins.map((b, i) => <rect key={i} x={i * 300 / bins.length + 1} y={74 - b.count / max * 66} width={300 / bins.length - 3} height={Math.max(1, b.count / max * 66)} rx="2" fill={color} opacity={0.55 + b.count / max * 0.45}><title>{fmt(b.start)} to {fmt(b.end)}: {b.count}</title></rect>)}
  </svg><div className="axis"><span>{fmt(bins[0]?.start ?? 0)}</span><span>activation value</span><span>{fmt(bins.at(-1)?.end ?? 0)}</span></div></div>;
}
