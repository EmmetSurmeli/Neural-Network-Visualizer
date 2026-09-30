from collections import OrderedDict
from pathlib import Path
from threading import RLock
from uuid import uuid4
import json
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.models.demo import samples
from backend.models.catalog import load_catalog, IDS
from backend.models.vision import samples as vision_samples, VISION_CLASSES
from backend.instrumentation.trace import run_trace, neuron_details
from backend.instrumentation.visualization import focused_visualization
from backend.schemas.requests import ModelSpec, RunRequest
from backend.services.models import from_spec, summarize_spec
from backend.services.playground import PlaygroundConfig, Dataset, dataset_payload, train_model

torch.set_num_threads(2)
app = FastAPI(title='NeuralScope', version='1.0.0')
lock = RLock()
models = {}
traces = OrderedDict()

@app.middleware('http')
async def limit_body(request, call_next):
    if request.method in ('POST', 'PUT', 'PATCH'):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 8 * 1024 * 1024:
                return JSONResponse({'detail': 'Request exceeds the 8 MB limit.'}, status_code=413)
        request._body = bytes(body)
    return await call_next(request)

def demo():
    with lock:
        if 'mnist-mlp' not in models:
            models.update(load_catalog())
        return models['mnist-mlp']

@app.get('/health')
def health():
    demo()
    return {'status': 'ok', 'engine': 'PyTorch', 'version': torch.__version__}

@app.get('/demo-models')
def get_models():
    demo()
    return [models[key][1] for key in IDS]

@app.get('/samples')
def get_samples(model_id: str = 'mnist-mlp'):
    if model_id == 'mnist-mlp':
        return samples()
    if model_id in VISION_CLASSES:
        return vision_samples(model_id)
    raise HTTPException(404, 'No image samples for this model.')

def execute(request: RunRequest, demo_only=False):
    with lock:
        demo()
        model_id = 'mnist-mlp' if demo_only else request.model_id
        if model_id not in models:
            raise HTTPException(404, 'Model is no longer available. Train or import it again.')
        model, info = models[model_id]
        if len(request.input) != info['input_features']:
            raise HTTPException(422, f"Expected exactly {info['input_features']} input values.")
        if any(abs(v) > 1e6 for v in request.input):
            raise HTTPException(422, 'Input magnitude must be at most 1,000,000.')
        if info.get('input_kind') == 'image' and any(v < 0 or v > 1 for v in request.input):
            raise HTTPException(422, 'Image pixels must be normalized to [0, 1].')
        try:
            tensor = torch.tensor([request.input], dtype=torch.float32)
            if info.get('input_shape'):
                tensor = tensor.reshape(1, *info['input_shape'])
            trace, raw = run_trace(model, tensor, info['name'])
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        trace_id = str(uuid4())
        traces[trace_id] = raw
        while len(traces) > 16:
            traces.popitem(last=False)
        return dict(trace, trace_id=trace_id, model_id=model_id, classes=info.get('classes'))

@app.post('/run-demo')
def run_demo(request: RunRequest):
    return execute(request, demo_only=True)

@app.post('/run-model')
def run_model(request: RunRequest):
    return execute(request)

@app.get('/playground/dataset')
def playground_dataset(dataset: Dataset = 'moons', seed: int = Query(42, ge=0, le=2**31 - 1), noise: float = Query(0.1, ge=0, le=0.8, allow_inf_nan=False)):
    return dataset_payload(dataset, seed, noise)

@app.post('/playground/train')
def playground_train(config: PlaygroundConfig):
    with lock:
        try:
            model, result = train_model(config)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        for key in list(models):
            if key.startswith('playground-'):
                del models[key]
        models[result['model_id']] = (model, result['model'])
        return result

@app.post('/upload-model', status_code=201)
def upload_model(spec: ModelSpec):
    with lock:
        demo()
        if sum(key not in IDS and not key.startswith('playground-') for key in models) >= 8:
            raise HTTPException(409, 'Eight imported models are supported per session. Restart to clear them.')
        try:
            model = from_spec(spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        model_id = str(uuid4())
        info = {'id': model_id, 'name': spec.name, 'input_features': spec.input_features,
                'parameters': sum(p.numel() for p in model.parameters()), 'dataset': 'Imported JSON',
                'architecture': summarize_spec(spec)}
        models[model_id] = (model, info)
        return info

@app.get('/traces/{trace_id}/layers/{layer_index}/neurons/{neuron_index}')
def neuron(trace_id: str, layer_index: int, neuron_index: int, limit: int = Query(20, ge=1, le=50)):
    with lock:
        if trace_id not in traces:
            raise HTTPException(404, 'Trace expired. Run the input again to inspect neurons.')
        raw = traces[trace_id]
        if not 0 <= layer_index < len(raw):
            raise HTTPException(404, 'Layer not found.')
        try:
            return neuron_details(raw, layer_index, neuron_index, limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

@app.get('/traces/{trace_id}/focus')
def focus(trace_id: str, layer: int, neuron: int):
    with lock:
        if trace_id not in traces:
            raise HTTPException(404, 'Trace expired. Run the input again to explore connections.')
        try:
            return focused_visualization(traces[trace_id], layer, neuron)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')
