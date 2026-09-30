import {useEffect, useRef} from 'react';
import {Download, X} from 'lucide-react';

export default function ImportGuide({open, close}: {open: boolean; close: () => void}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open && !dialog.current?.open) dialog.current?.showModal();
    if (!open && dialog.current?.open) dialog.current.close();
  }, [open]);
  return <dialog ref={dialog} className="import-dialog" onClose={close} onCancel={close}>
    <div className="dialog-heading"><div><h2>Model import format</h2><p>Sequential PyTorch models, represented as JSON.</p></div><button aria-label="Close import guide" onClick={close}><X size={16}/></button></div>
    <div className="dialog-content">
      <div className="format-grid"><div><span>Supported layers</span><p>Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, Flatten</p></div><div><span>Input</span><p>One flat array of finite numbers</p></div><div><span>Limits</span><p>24 layers · 500,000 parameters · 4,096 inputs</p></div><div><span>Safety</span><p>Weights only. Python and pickle files are not executed.</p></div></div>
      <pre className="model-example">{`{
  "name": "My classifier",
  "input_features": 2,
  "layers": [
    {"type": "Linear", "weight": [[1, -1]], "bias": [0]},
    {"type": "Sigmoid"}
  ]
}`}</pre>
      <p className="dialog-note">For an existing <span className="mono">nn.Sequential</span> model, use <span className="mono">export_sequential</span> from <span className="mono">backend.services.export</span>.</p>
      <div className="example-links"><a href="/tiny-model.json" download><Download size={13}/> Tiny ReLU classifier</a><a href="/xor-model.json" download><Download size={13}/> XOR + Sigmoid example</a><a href="/tanh-classifier.json" download><Download size={13}/> Tanh + Dropout example</a></div>
    </div>
  </dialog>;
}
