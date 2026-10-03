"""Telemetry is opt-in by deployment configuration and contains no request payloads."""
import os
import re
import logging


def scrub_event(event, hint=None):
    # Rebuild from a small allowlist: no request, breadcrumbs, locals or exception messages.
    frames = []
    for exception in event.get('exception', {}).get('values', []):
        for frame in exception.get('stacktrace', {}).get('frames', []):
            filename = frame.get('filename', '')
            if isinstance(filename, str) and filename.startswith('backend/'):
                frames.append({'filename': filename, 'lineno': frame.get('lineno')})
    tags = event.get('tags', {})
    safe_tags = {}
    if re.fullmatch(r'[a-f0-9]{32}', str(tags.get('request_id', ''))):
        safe_tags['request_id'] = tags['request_id']
    if tags.get('route') in {'health', 'session', 'models', 'samples', 'run', 'import', 'train', 'dataset', 'focus', 'neuron', 'other'}:
        safe_tags['route'] = tags['route']
    return {'event_id': event.get('event_id'), 'timestamp': event.get('timestamp'), 'platform': 'python',
            'level': 'error', 'release': os.getenv('RELEASE', 'local'), 'environment': os.getenv('APP_ENV', 'development'),
            'tags': safe_tags, 'exception': {'values': [{'type': 'ServerError', 'value': 'Server operation failed',
                                                       'stacktrace': {'frames': frames}}]}}


def initialize():
    if os.getenv('APP_ENV') != 'production' or not os.getenv('SENTRY_DSN'):
        return
    import sentry_sdk
    try:
        sentry_sdk.init(dsn=os.environ['SENTRY_DSN'], environment='production', release=os.getenv('RELEASE', 'unknown'),
                        default_integrations=False, auto_enabling_integrations=False, send_default_pii=False,
                        include_local_variables=False, max_request_body_size='never', traces_sample_rate=0,
                        before_send=scrub_event)
    except Exception:
        # Monitoring configuration must not prevent serving the demo. Never log the DSN.
        logging.getLogger('neuralscope').warning('Error monitoring could not initialize; check deployment configuration.')


def report(request_id, route):
    if os.getenv('APP_ENV') == 'production' and os.getenv('SENTRY_DSN'):
        import sentry_sdk
        sentry_sdk.capture_event({'level': 'error', 'tags': {'request_id': request_id, 'route': route},
                                 'message': 'Server operation failed'})
