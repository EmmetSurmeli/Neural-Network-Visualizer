import {useCallback, useEffect, useRef, useState} from 'react';

export interface PlaybackPosition {step: number; progress: number}
export function advancePlayback(position: PlaybackPosition, delta: number, duration: number, lastStep: number) {
  let {step, progress} = position;
  progress += delta / duration;
  while (progress >= 1 && step < lastStep) {progress -= 1; step++;}
  return {step, progress: Math.min(progress, 1)};
}
// One clock drives the execution rail, signals and neuron brightness.
export default function usePlayback(layerCount: number) {
  const [position, setPosition] = useState<PlaybackPosition>({step: -1, progress: 0});
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const current = useRef(position);
  const lastStep = layerCount - 1;
  const seek = useCallback((step: number, progress = 1) => {
    const next = {step, progress}; current.current = next; setPosition(next);
  }, []);
  useEffect(() => {
    if (!playing || !layerCount) return;
    let frame: number, previous: number | null = null;
    const tick = (time: number) => {
      const delta = previous === null ? 0 : Math.min(time - previous, 100);
      previous = time;
      const next = advancePlayback(current.current, delta, 1200 / speed, lastStep);
      current.current = next; setPosition(next);
      if (next.step === lastStep && next.progress >= 1) setPlaying(false);
      else frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, layerCount, lastStep, speed]);
  return {step: position.step, progress: position.progress, playing, speed, setSpeed, setPlaying, seek,
    reset: () => {setPlaying(false); seek(-1, 0);},
    toggle: () => {if (!playing && current.current.step >= lastStep && current.current.progress >= 1) seek(-1, 0); setPlaying(v => !v);},
    previous: () => {setPlaying(false); seek(Math.max(-1, current.current.step - 1));},
    next: () => {setPlaying(false); seek(Math.min(lastStep, current.current.step + 1));},
  };
}
