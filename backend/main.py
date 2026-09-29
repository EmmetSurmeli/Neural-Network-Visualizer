from collections import OrderedDict
from pathlib import Path
from threading import RLock
from uuid import uuid4
import json
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.models.demo import load_demo, metadata, samples
from backend.instrumentation.trace import run_trace, neuron_details
from backend.schemas.requests import ModelSpec, RunRequest
from backend.services.models import from_spec, summarize_spec

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
            models['mnist-mlp'] = (load_demo(), metadata())
        return models['mnist-mlp']

@app.get('/health')
def health():
    demo()
    return {'status': 'ok', 'engine': 'PyTorch', 'version': torch.__version__}

@app.get('/demo-models')
def get_models():
    return [demo()[1]]

@app.get('/samples')
def get_samples():
    return samples()

def execute(request: RunRequest, demo_only=False):
    with lock:
        demo()
        model_id = 'mnist-mlp' if demo_only else request.model_id
        if model_id not in models:
            raise HTTPException(404, 'Model not found. Reimport the model after restarting the server.')
        model, info = models[model_id]
        if len(request.input) != info['input_features']:
            raise HTTPException(422, f"Expected exactly {info['input_features']} input values.")
        if any(abs(v) > 1e6 for v in request.input):
            raise HTTPException(422, 'Input magnitude must be at most 1,000,000.')
        if model_id == 'mnist-mlp' and any(v < 0 or v > 1 for v in request.input):
            raise HTTPException(422, 'Digit pixels must be normalized to [0, 1].')
        try:
            trace, raw = run_trace(model, torch.tensor([request.input], dtype=torch.float32), info['name'])
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        trace_id = str(uuid4())
        traces[trace_id] = raw
        while len(traces) > 16:
            traces.popitem(last=False)
        return dict(trace, trace_id=trace_id, model_id=model_id)

@app.post('/run-demo')
def run_demo(request: RunRequest):
    return execute(request, demo_only=True)

@app.post('/run-model')
def run_model(request: RunRequest):
    return execute(request)

@app.post('/upload-model', status_code=201)
def upload_model(spec: ModelSpec):
    with lock:
        demo()
        if len(models) >= 9:
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

DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')
