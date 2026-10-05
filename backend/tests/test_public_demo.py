import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import backend.main as server
from backend.runtime import MAX_SESSIONS, Sessions, ServiceError, deadline, check_deadline, compute_slot, sessions
from backend.monitoring import scrub_event

SPEC = json.loads(Path('examples/tiny-model.json').read_text())

@pytest.fixture
def visitors():
    a, b = TestClient(server.app), TestClient(server.app)
    tokens = [c.post('/session').json()['token'] for c in (a, b)]
    for client, token in zip((a, b), tokens):
        client.headers['X-NeuralScope-Session'] = token
    yield a, b
    for token in tokens:
        sessions.remove(token)

def test_models_and_traces_are_private_to_session(visitors):
    a, b = visitors
    model = a.post('/upload-model', json=SPEC).json()
    assert b.post('/run-model', json={'model_id': model['id'], 'input': [1, .5, -1]}).status_code == 404
    result = a.post('/run-model', json={'model_id': model['id'], 'input': [1, .5, -1]})
    assert result.status_code == 200
    trace = result.json()['trace_id']
    assert a.get(f'/traces/{trace}/focus?layer=0&neuron=0').status_code == 200
    assert b.get(f'/traces/{trace}/focus?layer=0&neuron=0').json()['error']['code'] == 'TRACE_EXPIRED'
    assert a.delete('/session').status_code == 200
    assert model['id'] not in server.models and trace not in server.traces
    assert a.post('/run-model', json={'input': [1]}).json()['error']['code'] == 'SESSION_EXPIRED'

def test_health_request_ids_cors_and_safe_errors(visitors):
    a, _ = visitors
    response = a.get('/health', headers={'Origin': 'http://localhost:5173', 'X-Request-ID': 'user-secret'})
    assert response.json() == {'status': 'ok'}
    assert len(response.headers['X-Request-ID']) == 32
    assert response.headers['Access-Control-Allow-Origin'] == 'http://localhost:5173'
    assert response.headers['Cache-Control'] == 'no-store'
    assert 'Access-Control-Allow-Origin' not in a.get('/health', headers={'Origin': 'https://evil.example'}).headers
    response = a.post('/upload-model', json={'name': 'PRIVATE_FILENAME', 'layers': 'SECRET_WEIGHTS'})
    assert response.status_code == 422
    assert response.json()['error']['request_id'] == response.headers['X-Request-ID']
    assert 'PRIVATE_FILENAME' not in response.text and 'SECRET_WEIGHTS' not in response.text

@pytest.mark.parametrize('path,body,limit', [('/run-model', {'model_id': 'tiny-mlp', 'input': [1, .5, -1]}, 60), ('/upload-model', {}, 12), ('/playground/train', {}, 6)])
def test_rate_limits_include_retry_and_request_id(visitors, path, body, limit):
    a, _ = visitors
    for _ in range(limit):
        assert a.post(path, json=body).status_code != 429
    response = a.post(path, json=body)
    assert response.status_code == 429
    assert response.json()['error']['code'] == 'RATE_LIMITED'
    assert 1 <= int(response.headers['Retry-After']) <= 60
    assert response.json()['error']['request_id'] == response.headers['X-Request-ID']

def test_busy_requests_do_not_queue(visitors):
    a, _ = visitors
    assert compute_slot.acquire(False)
    try:
        response = a.post('/run-model', json={'model_id': 'tiny-mlp', 'input': [1, .5, -1]})
        assert response.status_code == 503
        assert response.json()['error']['code'] == 'SERVICE_BUSY'
    finally:
        compute_slot.release()
    assert a.post('/run-model', json={'model_id': 'tiny-mlp', 'input': [1, .5, -1]}).status_code == 200

def test_expiration_discards_only_owned_memory():
    store = Sessions()
    models, traces = {'built-in': object()}, {}
    store.bind(models, traces)
    token = store.create()
    store.items[token].models.add('secret')
    models['secret'] = object()
    store.items[token].traces.add('trace')
    traces['trace'] = object()
    store.items[token].touched = -100000
    with pytest.raises(ServiceError, match='expired'):
        store.get(token)
    assert list(models) == ['built-in'] and not traces

def test_session_cap_recycles_oldest_idle_session_and_its_data():
    store = Sessions()
    models, traces = {'built-in': object(), 'old-model': object()}, {'old-trace': object()}
    store.bind(models, traces)
    tokens = [store.create() for _ in range(MAX_SESSIONS)]
    recently_used = tokens[0]
    oldest = tokens[1]
    store.items[oldest].models.add('old-model')
    store.items[oldest].traces.add('old-trace')
    store.get(recently_used)  # Recently used sessions are moved to the back.
    next_token = store.create()
    assert next_token in store.items and recently_used in store.items
    assert oldest not in store.items
    assert list(models) == ['built-in'] and not traces
    with pytest.raises(ServiceError, match='expired'):
        store.get(oldest)

def test_active_sessions_cannot_be_recycled():
    store = Sessions()
    store.bind({}, {})
    tokens = [store.create() for _ in range(MAX_SESSIONS)]
    for token in tokens:
        store.enter(token)
    with pytest.raises(ServiceError) as exc:
        store.create()
    assert exc.value.code == 'SERVICE_BUSY'
    store.leave(tokens[0])
    store.create()
    assert tokens[0] not in store.items

def test_interactive_docs_are_local_only():
    assert (server.app.docs_url is None) == server.PRODUCTION
    assert (server.app.openapi_url is None) == server.PRODUCTION

def test_frontend_policy_blocks_inline_scripts_and_limits_api_destinations():
    deployment = json.loads(Path('vercel.json').read_text())
    headers = deployment['headers'][0]['headers']
    policy = next(header['value'] for header in headers if header['key'] == 'Content-Security-Policy')
    assert "script-src 'self'" in policy
    assert "script-src 'self' 'unsafe-inline'" not in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert 'https://neuralscope-api-24v5.onrender.com' in policy

def test_deadline_is_cooperative_and_returns_stable_error(visitors, monkeypatch):
    a, _ = visitors
    def timed_out(*args, **kwargs):
        value = deadline.set(time.monotonic() - 1)
        try:
            check_deadline()
        finally:
            deadline.reset(value)
    monkeypatch.setattr(server, 'run_trace', timed_out)
    response = a.post('/run-model', json={'model_id': 'tiny-mlp', 'input': [1, .5, -1]})
    assert response.status_code == 504
    assert response.json()['error']['code'] == 'REQUEST_TIMEOUT'

def test_unexpected_errors_never_return_raw_exception(visitors, monkeypatch, caplog):
    a, _ = visitors
    def broken(*args):
        raise RuntimeError('SECRET_INPUT_AND_WEIGHTS')
    monkeypatch.setattr(server, 'run_trace', broken)
    response = a.post('/run-model', json={'model_id': 'tiny-mlp', 'input': [1, .5, -1]})
    assert response.status_code == 500
    assert 'SECRET_INPUT_AND_WEIGHTS' not in response.text + caplog.text
    assert response.json()['error']['code'] == 'INTERNAL_ERROR'

def test_monitoring_rebuilds_payload():
    event = {'request': {'data': 'SECRET'}, 'user': {'ip_address': 'SECRET'}, 'extra': {'weights': 'SECRET'},
             'message': 'SECRET', 'breadcrumbs': [{'message': 'SECRET'}], 'tags': {'request_id': 'a' * 32, 'route': 'run', 'model_name': 'SECRET'},
             'exception': {'values': [{'value': 'SECRET', 'stacktrace': {'frames': [{'filename': '/uploads/SECRET', 'vars': {'x': 'SECRET'}}]}}]}}
    clean = scrub_event(event)
    assert 'SECRET' not in json.dumps(clean)
    assert clean['tags'] == {'request_id': 'a' * 32, 'route': 'run'}

def test_invalid_monitoring_configuration_does_not_stop_the_demo(monkeypatch, caplog):
    from backend.monitoring import initialize
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.setenv('SENTRY_DSN', 'PRIVATE_INVALID_VALUE')
    initialize()
    assert 'PRIVATE_INVALID_VALUE' not in caplog.text
    assert 'could not initialize' in caplog.text

def test_production_requires_explicit_session(monkeypatch):
    monkeypatch.setattr(server, 'PRODUCTION', True)
    response = TestClient(server.app).post('/run-model', json={'input': [1]})
    assert response.status_code == 401
    assert response.json()['error']['code'] == 'SESSION_EXPIRED'

def test_http_deadline_and_admission_survive_slow_native_work(visitors, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    a, b = visitors
    started = Event()
    real_trace = server.run_trace
    def slow(*args):
        started.set()
        time.sleep(.15)
        return real_trace(*args)
    monkeypatch.setattr(server, 'run_trace', slow)
    monkeypatch.setattr(server, 'REQUEST_TIMEOUT_SECONDS', .03)
    body = {'model_id': 'tiny-mlp', 'input': [1, .5, -1]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(a.post, '/run-model', json=body)
        assert started.wait(1)
        time.sleep(.06)
        assert b.post('/run-model', json=body).json()['error']['code'] == 'SERVICE_BUSY'
        response = pending.result(timeout=2)
        assert response.status_code == 504
        assert response.json()['error']['code'] == 'REQUEST_TIMEOUT'
    # Expired native work must not commit a trace after the response timed out.
    assert not sessions.items[a.headers['X-NeuralScope-Session']].traces

def test_retraining_one_visitor_does_not_delete_anothers_model(visitors):
    a, b = visitors
    config = {'epochs': 100, 'hidden_layers': [2]}
    first = a.post('/playground/train', json=config).json()['model_id']
    other = b.post('/playground/train', json=config).json()['model_id']
    replacement = a.post('/playground/train', json=config).json()['model_id']
    assert first not in server.models
    assert replacement in server.models and other in server.models
    assert b.post('/run-model', json={'model_id': other, 'input': [0, 1]}).status_code == 200

def test_upload_is_memory_only(visitors, monkeypatch):
    a, _ = visitors
    def forbidden(*args, **kwargs):
        raise AssertionError('Upload attempted to write to disk')
    monkeypatch.setattr(Path, 'write_text', forbidden)
    monkeypatch.setattr(Path, 'write_bytes', forbidden)
    monkeypatch.setattr(server.torch, 'save', forbidden)
    assert a.post('/upload-model', json=SPEC).status_code == 201
