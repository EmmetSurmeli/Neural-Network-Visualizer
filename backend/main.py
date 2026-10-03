from collections import OrderedDict
from pathlib import Path
from threading import RLock
from uuid import uuid4
import json
import asyncio
import logging
import os
from time import monotonic
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
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
from backend.runtime import sessions, owner, deadline, check_deadline, compute_slot, inspection_slots, limiter, ServiceError, PRODUCTION, SESSION_TTL
from backend.monitoring import initialize as initialize_monitoring, report

torch.set_num_threads(2)
@asynccontextmanager
async def lifespan(app):
    initialize_monitoring()
    await run_in_threadpool(demo)
    async def cleanup():
        while True:
            await asyncio.sleep(30)
            await run_in_threadpool(prune_sessions)
    task = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(title='NeuralScope', version='1.0.0', lifespan=lifespan)
lock = RLock()
REQUEST_TIMEOUT_SECONDS = 20
TRAIN_TIMEOUT_SECONDS = 90
models = {}
traces = OrderedDict()
sessions.bind(models, traces)
def prune_sessions():
    with lock, sessions.guard:
        sessions.prune()

logger = logging.getLogger('neuralscope')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def route_category(path):
    return {'/health': 'health', '/session': 'session', '/demo-models': 'models', '/samples': 'samples',
            '/run-demo': 'run', '/run-model': 'run', '/upload-model': 'import',
            '/playground/train': 'train', '/playground/dataset': 'dataset'}.get(path,
            'focus' if path.startswith('/traces/') and path.endswith('/focus') else 'neuron' if path.startswith('/traces/') else 'other')

def failure(request, code, message, status, retry_after=None):
    response = JSONResponse({'error': {'code': code, 'message': message, 'request_id': request.state.request_id},
                             'detail': message}, status_code=status)
    if retry_after:
        response.headers['Retry-After'] = str(retry_after)
    return response

@app.exception_handler(ServiceError)
async def service_error(request, exc):
    return failure(request, exc.code, exc.message, exc.status, exc.retry_after)

@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    if request.url.path == '/upload-model' and any('type' in error['loc'] for error in exc.errors()):
        return failure(request, 'UNSUPPORTED_MODEL', 'This model contains an unsupported layer. Open the format guide for supported Sequential JSON models.', 422)
    return failure(request, 'INVALID_INPUT', 'Invalid input or unsupported model structure. Check the number of values and the supported JSON format.', 422)

@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    code = {404: 'NOT_FOUND', 409: 'MODEL_LIMIT', 413: 'PAYLOAD_TOO_LARGE', 422: 'INVALID_INPUT'}.get(exc.status_code, 'REQUEST_FAILED')
    if exc.status_code == 404 and request.url.path.startswith('/traces/'):
        code = 'TRACE_EXPIRED'
    return failure(request, code, str(exc.detail), exc.status_code)

@app.middleware('http')
async def limit_body(request, call_next):
    request.state.request_id = uuid4().hex
    category = route_category(request.url.path)
    started = monotonic()
    acquired = False
    owner_token = deadline_token = None
    session_cookie = None
    admission = inspection_slots if category in ('focus', 'neuron') else compute_slot
    try:
        peer = request.client.host if request.client else 'unknown'
        if category != 'other' and category != 'health':
            limiter.check((peer, 'all'), 240)
        if category in ('run', 'import', 'train', 'session'):
            limiter.check((peer, category), {'run': 60, 'import': 12, 'train': 6, 'session': 20}[category])
        if category in ('run', 'import', 'train', 'focus', 'neuron'):
            token = request.headers.get('X-NeuralScope-Session') or request.cookies.get('ns_session')
            if not token and not PRODUCTION:
                token = sessions.create()
                session_cookie = token
            sessions.get(token)
            owner_token = owner.set(token)
        if category in ('run', 'import', 'train', 'focus', 'neuron'):
            acquired = admission.acquire(blocking=False)
            if not acquired:
                raise ServiceError('SERVICE_BUSY', 'The demo is processing another request. Please retry shortly.', retry_after=2)
        timeout = TRAIN_TIMEOUT_SECONDS if category == 'train' else REQUEST_TIMEOUT_SECONDS
        deadline_token = deadline.set(started + timeout)
        if request.method in ('POST', 'PUT', 'PATCH'):
            async def read_body():
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 8 * 1024 * 1024:
                        raise ServiceError('PAYLOAD_TOO_LARGE', 'Model files must be smaller than 8 MB.', 413)
                request._body = bytes(body)
            await asyncio.wait_for(read_body(), timeout=10)
        # Do not release admission while timed-out native code is still running.
        task = asyncio.create_task(call_next(request))
        try:
            response = await asyncio.wait_for(asyncio.shield(task), timeout=max(.01, deadline.get() - monotonic()))
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if acquired:
                def finish(completed):
                    admission.release()
                    if not completed.cancelled():
                        completed.exception()
                task.add_done_callback(finish)
                acquired = False
            raise
    except ServiceError as exc:
        response = failure(request, exc.code, exc.message, exc.status, exc.retry_after)
    except asyncio.TimeoutError:
        response = failure(request, 'REQUEST_TIMEOUT', 'This operation took too long. Please retry with a smaller model or fewer epochs.', 504)
    except Exception:
        report(request.state.request_id, category)
        response = failure(request, 'INTERNAL_ERROR', 'The service could not complete this request. Please retry.', 500)
    finally:
        if acquired:
            admission.release()
        if owner_token is not None:
            owner.reset(owner_token)
        if deadline_token is not None:
            deadline.reset(deadline_token)
    response.headers['X-Request-ID'] = request.state.request_id
    response.headers['Cache-Control'] = 'no-store' if category != 'other' else 'no-cache'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    if session_cookie:
        response.set_cookie('ns_session', session_cookie, httponly=True, samesite='lax', max_age=SESSION_TTL)
    if category != 'other':
        logger.info(json.dumps({'request_id': request.state.request_id, 'route': category, 'status': response.status_code,
                                'elapsed_ms': round((monotonic() - started) * 1000)}))
    return response

origins = [v.strip() for v in os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if v.strip()]
if '*' in origins:
    raise RuntimeError('ALLOWED_ORIGINS must contain explicit frontend origins, not a wildcard.')
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST', 'DELETE'],
                   allow_headers=['Content-Type', 'X-NeuralScope-Session'], expose_headers=['X-Request-ID', 'Retry-After'])

@app.post('/session')
def new_session():
    return {'token': sessions.create(), 'expires_in': SESSION_TTL}

@app.delete('/session')
def clear_session(request: Request):
    token = request.headers.get('X-NeuralScope-Session') or request.cookies.get('ns_session')
    if not compute_slot.acquire(blocking=False):
        raise ServiceError('SERVICE_BUSY', 'Wait for the current operation to finish, then clear the session.', retry_after=2)
    try:
        with sessions.guard:
            sessions.remove(token)
    finally:
        compute_slot.release()
    return {'status': 'cleared'}

def demo():
    with lock:
        if 'mnist-mlp' not in models:
            models.update(load_catalog())
        return models['mnist-mlp']

@app.get('/health')
def health():
    return {'status': 'ok'}

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
        if model_id not in models or (model_id not in IDS and model_id not in sessions.current().models):
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
        except ValueError as exc:
            raise HTTPException(422, 'The model could not produce finite activations. Check its structure and reduce input or weight magnitudes.') from exc
        trace_id = str(uuid4())
        check_deadline()
        with sessions.guard:
            session = sessions.current()
            traces[trace_id] = raw
            session.traces.intersection_update(traces)
            session.traces.add(trace_id)
            while len(session.traces) > 16:
                oldest = next(key for key in traces if key in session.traces)
                traces.pop(oldest)
                session.traces.remove(oldest)
            def cache_bytes():
                return sum(t.numel() * t.element_size() for entries in traces.values() for item in entries for t in (item['input'], item['output']))
            while len(traces) > 128 or cache_bytes() > 128 * 1024 * 1024:
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
        session = sessions.current()
        session.models.intersection_update(models)
        if len(models) - len(IDS) >= 64 and not any(k.startswith('playground-') for k in session.models):
            raise ServiceError('SERVICE_BUSY', 'Temporary model storage is full. Please retry later.', retry_after=30)
        try:
            model, result = train_model(config)
        except ValueError as exc:
            raise HTTPException(422, 'Training became unstable. Try a smaller learning rate.') from exc
        check_deadline()
        for key in list(session.models):
            if key.startswith('playground-'):
                models.pop(key, None)
                session.models.remove(key)
        models[result['model_id']] = (model, result['model'])
        session.models.add(result['model_id'])
        return result

@app.post('/upload-model', status_code=201)
def upload_model(spec: ModelSpec):
    with lock:
        demo()
        session = sessions.current()
        session.models.intersection_update(models)
        if sum(not key.startswith('playground-') for key in session.models) >= 8:
            raise HTTPException(409, 'Eight imported models are supported per session. Clear your temporary session to import more.')
        if len(models) - len(IDS) >= 64:
            raise ServiceError('SERVICE_BUSY', 'Temporary model storage is full. Please retry later.', retry_after=30)
        try:
            model = from_spec(spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        model_id = str(uuid4())
        info = {'id': model_id, 'name': spec.name, 'input_features': spec.input_features,
                'parameters': sum(p.numel() for p in model.parameters()), 'dataset': 'Imported JSON',
                'architecture': summarize_spec(spec)}
        check_deadline()
        models[model_id] = (model, info)
        session.models.add(model_id)
        return info

@app.get('/traces/{trace_id}/layers/{layer_index}/neurons/{neuron_index}')
def neuron(trace_id: str, layer_index: int, neuron_index: int, limit: int = Query(20, ge=1, le=50)):
    with lock:
        if trace_id not in traces or trace_id not in sessions.current().traces:
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
        if trace_id not in traces or trace_id not in sessions.current().traces:
            raise HTTPException(404, 'Trace expired. Run the input again to explore connections.')
        try:
            return focused_visualization(traces[trace_id], layer, neuron)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')
