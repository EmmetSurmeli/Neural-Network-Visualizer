"""Check a deployed API using only bundled public example data."""
import json
from http.client import RemoteDisconnected
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    base = sys.argv[1].rstrip('/')
    token = None

    def request(path, method='GET', body=None):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-NeuralScope-Session'] = token
        data = None if body is None else json.dumps(body).encode()
        with urlopen(Request(base + path, data=data, method=method, headers=headers), timeout=25) as response:
            assert response.headers.get('X-Request-ID'), 'Missing request ID'
            return json.load(response)

    for attempt in range(60):
        try:
            assert request('/health') == {'status': 'ok'}
            break
        except (URLError, TimeoutError, ConnectionError, RemoteDisconnected):
            if attempt == 59:
                raise
            time.sleep(1)

    token = request('/session', 'POST')['token']
    try:
        request('/docs')
    except HTTPError as exc:
        assert exc.code == 404, 'Production API documentation should not be public'
    else:
        raise AssertionError('Production API documentation should not be public')
    try:
        spec = json.loads((Path(__file__).resolve().parents[1] / 'examples/tiny-model.json').read_text())
        model = request('/upload-model', 'POST', spec)
        trace = request('/run-model', 'POST', {'model_id': model['id'], 'input': [1, .5, -1]})
        assert trace['predicted_class'] == 0
        assert math.isclose(trace['probabilities'][0], 1 / (1 + math.exp(-3.425)), abs_tol=1e-6)
        assert trace['layers'] and trace['trace_id']
    finally:
        request('/session', 'DELETE')
    print('Production API passed: readiness, request IDs, session, model import, prediction and cleanup.')


if __name__ == '__main__':
    main()
