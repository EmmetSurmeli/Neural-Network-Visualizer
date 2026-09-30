import {useEffect, useRef, type PointerEvent} from 'react';

export function normalizedPixels(canvas: HTMLCanvasElement): number[] {
  const ctx = canvas.getContext('2d')!;
  const {data} = ctx.getImageData(0, 0, 280, 280);
  let left = 280, right = 0, top = 280, bottom = 0;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
    if (data[(y * 280 + x) * 4] > 20) {left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);}
  }
  if (left > right) return Array(784).fill(0);
  const small = document.createElement('canvas'); small.width = small.height = 28;
  const sctx = small.getContext('2d')!;
  sctx.fillStyle = '#000'; sctx.fillRect(0, 0, 28, 28);
  const width = right - left + 1, height = bottom - top + 1, scale = 20 / Math.max(width, height);
  sctx.drawImage(canvas, left, top, width, height, (28 - width * scale) / 2, (28 - height * scale) / 2, width * scale, height * scale);
  const image = sctx.getImageData(0, 0, 28, 28).data;
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {const v = image[i * 4] / 255; mass += v; cx += (i % 28) * v; cy += Math.floor(i / 28) * v;}
  const dx = Math.round(13.5 - cx / mass), dy = Math.round(13.5 - cy / mass);
  const result = Array(784).fill(0);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const xx = x + dx, yy = y + dy;
    if (xx >= 0 && xx < 28 && yy >= 0 && yy < 28) result[yy * 28 + xx] = image[(y * 28 + x) * 4] / 255;
  }
  return result;
}

export default function DigitCanvas({source, onChange, label}: {label?: string; source: number[]; onChange: (pixels: number[]) => void}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const down = useRef(false);
  useEffect(() => {
    const canvas = ref.current!, ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280);
    source.forEach((value, i) => {ctx.fillStyle = `rgb(${value * 255},${value * 255},${value * 255})`; ctx.fillRect((i % 28) * 10, Math.floor(i / 28) * 10, 10, 10);});
  }, [source]);
  const point = (e: PointerEvent<HTMLCanvasElement>) => {const r = e.currentTarget.getBoundingClientRect(); return [(e.clientX - r.left) * 280 / r.width, (e.clientY - r.top) * 280 / r.height];};
  function start(e: PointerEvent<HTMLCanvasElement>) {
    down.current = true; e.currentTarget.setPointerCapture(e.pointerId);
    const ctx = e.currentTarget.getContext('2d')!, [x, y] = point(e);
    ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineWidth = 19; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.arc(x, y, 9.5, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.moveTo(x, y);
  }
  function move(e: PointerEvent<HTMLCanvasElement>) {if (!down.current) return; const ctx = e.currentTarget.getContext('2d')!; const [x, y] = point(e); ctx.lineTo(x, y); ctx.stroke();}
  function end(e: PointerEvent<HTMLCanvasElement>) {if (!down.current) return; down.current = false; onChange(normalizedPixels(e.currentTarget));}
  return <canvas ref={ref} width={280} height={280} onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={end} className="drawing-canvas" aria-label={label ?? "Draw a digit from zero to nine. You can also use the sample buttons below."} />;
}
