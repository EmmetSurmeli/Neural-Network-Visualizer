"""Single-process, bounded visitor sessions. Nothing here is persisted."""
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, BoundedSemaphore
from time import monotonic
import os
import secrets

PRODUCTION = os.getenv('APP_ENV', 'development') == 'production'
SESSION_TTL = int(os.getenv('SESSION_TTL_SECONDS', '1800'))
MAX_SESSIONS = 64
owner = ContextVar('owner', default='local')
deadline = ContextVar('deadline', default=float('inf'))
compute_slot = BoundedSemaphore(1)
inspection_slots = BoundedSemaphore(4)


class ServiceError(Exception):
    def __init__(self, code, message, status=503, retry_after=None):
        self.code, self.message, self.status, self.retry_after = code, message, status, retry_after


def check_deadline():
    if monotonic() > deadline.get():
        raise ServiceError('REQUEST_TIMEOUT', 'This operation took too long. Try a smaller model or fewer epochs.', 504)


@dataclass
class Session:
    touched: float
    models: set = field(default_factory=set)
    traces: set = field(default_factory=set)


class Sessions:
    def __init__(self):
        self.items = OrderedDict()
        self.guard = RLock()
        self.models = None
        self.traces = None

    def bind(self, models, traces):
        self.models, self.traces = models, traces

    def prune(self):
        with self.guard:
            now = monotonic()
            for token, session in list(self.items.items()):
                if now - session.touched > SESSION_TTL:
                    self.remove(token)

    def remove(self, token):
        with self.guard:
            session = self.items.pop(token, None)
            if session:
                for key in session.models:
                    self.models.pop(key, None)
                for key in session.traces:
                    self.traces.pop(key, None)

    def create(self):
        with self.guard:
            self.prune()
            if len(self.items) >= MAX_SESSIONS:
                raise ServiceError('SERVICE_BUSY', 'The demo is busy. Please try again shortly.', retry_after=30)
            token = secrets.token_urlsafe(32)
            self.items[token] = Session(monotonic())
            return token

    def get(self, token):
        with self.guard:
            self.prune()
            session = self.items.get(token)
            if not session:
                raise ServiceError('SESSION_EXPIRED', 'Your temporary session expired. Retry, then re-import or retrain your model if needed.', 401)
            session.touched = monotonic()
            return session

    def current(self):
        return self.get(owner.get())


sessions = Sessions()


class RateLimiter:
    """Bounded fixed-window counters; uses socket peer, never client-supplied forwarding headers."""
    def __init__(self):
        self.buckets = OrderedDict()
        self.guard = RLock()

    def check(self, key, limit, seconds=60):
        now = monotonic()
        with self.guard:
            for old in list(self.buckets):
                if not self.buckets[old] or self.buckets[old][-1] <= now - seconds:
                    del self.buckets[old]
            if key not in self.buckets and len(self.buckets) >= 4096:
                raise ServiceError('SERVICE_BUSY', 'The demo is busy. Please try again shortly.', retry_after=30)
            bucket = self.buckets.setdefault(key, deque())
            while bucket and bucket[0] <= now - seconds:
                bucket.popleft()
            if len(bucket) >= limit:
                raise ServiceError('RATE_LIMITED', 'Too many requests. Wait a moment, then try again.', 429, max(1, int(seconds - (now - bucket[0])) + 1))
            bucket.append(now)


limiter = RateLimiter()
