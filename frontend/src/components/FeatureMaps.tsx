import {useEffect, useRef} from 'react';
import {featureReveal} from '../lib/diagram';
import type {FeatureMaps as Maps, Trace} from '../types';

export function MapTile({maps, channel, reveal = 1}: {maps: Maps; channel: number; reveal?: number}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const image = ctx.createImageData(maps.width, maps.height);
    maps.values[channel].forEach((v, i) => {
      const strength = Math.min(1, Math.abs(v) / maps.scale), value = Math.round(strength * 255);
      image.data[i * 4] = v < 0 ? value : Math.round(value * .65);
      image.data[i * 4 + 1] = v < 0 ? Math.round(value * .4) : value;
      image.data[i * 4 + 2] = v < 0 ? Math.round(value * .4) : value;
      image.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(image, 0, 0);
  }, [maps, channel]);
  return <canvas ref={ref} width={maps.width} height={maps.height} style={{opacity: reveal}} aria-label={`Feature map channel ${channel}, ${maps.width} by ${maps.height}`}/>;
}

export default function FeatureMaps({trace, step, progress, select}: {trace: Trace; step: number; progress: number; select: (i: number) => void}) {
  const stages = [{name:'Input image', type:'Input', order:-1, feature_maps:trace.input_maps}, ...trace.layers.filter(l => l.feature_maps)];
  return <div className="cnn-flow" aria-label="Convolution feature maps">{stages.map(layer => {
    const maps = layer.feature_maps!;
    const reached = step >= layer.order;
    const reveal = featureReveal(layer.order, step, progress);
    return <button key={layer.order} className={`feature-stage ${step === layer.order ? 'current' : ''}`} onClick={() => select(layer.order)} aria-label={`Inspect feature maps ${layer.name}`}><span>{layer.type}</span><small>{layer.name} · {maps.channels} × {maps.height} × {maps.width}</small><div className="feature-thumbnails">{maps.values.slice(0,4).map((_, c) => <MapTile key={c} maps={maps} channel={c} reveal={reveal}/>)}</div><small>{reached ? `${Math.min(4, maps.channels)} of ${maps.channels} previews · inspect all` : 'Not reached'}</small></button>;
  })}</div>;
}
export function FeatureMapInspector({maps}: {maps: Maps}) {
  return <div className="feature-inspector"><p className="tiny muted">{maps.channels} channels · cyan positive / red negative · one shared scale{maps.downsampled ? ' · spatially averaged preview' : ''}</p><div className="feature-grid">{maps.values.map((_, i) => <div key={i}><MapTile maps={maps} channel={i}/><span>Channel {i}</span></div>)}</div></div>;
}
