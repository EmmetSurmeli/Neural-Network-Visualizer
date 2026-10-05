# Operating the public demo

## Health and failure recovery

`GET /health` returns `{"status":"ok"}` after models load during service startup. It does not expose framework versions or internal paths. The frontend checks readiness on startup, reconnect and retry; it does not poll continuously. The service can be ready while its single computation slot is busy.

API responses carry `X-Request-ID`. Errors use `{ "error": { "code": "...", "message": "...", "request_id": "..." } }`; a legacy `detail` string is also present. Responses never include stack traces. The UI displays the request reference so it can be matched to structured server logs. It retains the last successful result and labels it stale when a newer run fails.

| Condition | Recovery |
|---|---|
| Service unavailable / cold start | Wait for Render health, then Retry; the shell and format guide remain visible |
| Browser offline | Reconnect; live controls are disabled while offline |
| 429 `RATE_LIMITED` | Respect `Retry-After`; the retry button counts down |
| 503 `SERVICE_BUSY` | Another operation or temporary storage cap; retry after the suggested delay |
| 504 `REQUEST_TIMEOUT` | Use fewer epochs or a smaller model; explicit retry only |
| 401 `SESSION_EXPIRED` | Client discards old token; retry starts a fresh session; re-import or retrain custom models |
| 404 `TRACE_EXPIRED` | Return to the network and run the input again |
| Invalid input / unsupported structure | Correct the JSON array or use the import guide; load the 3-input demo |
| UI rendering failure | Reload demo from the error boundary |

## Limits

- One process/worker/instance. One concurrent inference, import or training operation. Four concurrent inspection requests; registry access is locked. Requests do not form an unbounded computation queue.
- Request-body cap: 8 MB and 10 seconds to read. Server operation deadline: 20 seconds; training: 90 seconds. Browser deadlines: 25 seconds, health 10 seconds, training 100 seconds. Session creation has its own bounded request.
- PyTorch native work cannot be forcibly killed safely in a thread. A timed-out computation retains its admission slot until it ends. Training checks a deadline each epoch; inference checks before caching. These are response and cooperative deadlines, not an OS-enforced CPU kill switch. Restart the service if native work gets stuck.
- 64 visitor sessions; 30-minute idle expiration, with least recently used sessions reclaimed sooner at capacity when no request is in flight. 64 custom models globally; eight imports and latest Playground model per session. The built-ins are not charged to import limits.
- Up to 16 traces per session, 128 globally, plus a 128 MB input/output tensor-cache cap. Model weights retained by cached traces consume additional memory. Choose an instance with headroom; monitor RSS under realistic imports before increasing limits.
- Per socket peer per minute: 60 runs, 12 imports, 6 training requests, 20 session operations, 240 API requests overall. Maximum 4,096 active rate buckets. Reverse-proxy peers can cause visitors to share limits. Health checks are exempt.
- Import: safe Sequential JSON only, 24 layers, 500,000 parameters, 4,096 inputs, 1,024 outputs per Linear layer; finite values with magnitude ≤1,000,000. Images must be normalized to [0,1]. No pickle, raw checkpoints, ONNX, Python or arbitrary code uploads.

## Incident checklist

1. Check the frontend status and Render `/health`. Record a request reference and release, without copying user data.
2. Review matching structured logs for route, status and duration. Check Render memory/CPU and restart/deploy events. Avoid enabling body or access logging during an incident.
3. For widespread 429/503, reduce abusive traffic at the trusted edge or add capacity only after redesigning process-local sessions. Do not add workers blindly.
4. For repeated timeouts or exhausted memory, restart the single instance. This discards sessions; the frontend guides visitors to re-import/retrain.
5. If a release regressed, roll back frontend and backend to compatible versions. Recheck the digit and Playground handoff.
6. If telemetry contains unexpected content, disable provider keys, redeploy, request deletion from the provider, fix the allowlist and add a regression test before reenabling.

## Credential rotation

PostHog project tokens and Sentry DSNs are public ingestion identifiers, not administrative secrets. If abused, replace/disable them in the relevant provider project, update Vercel Production values and Render backend DSN, and redeploy. Verify old ingestion identifiers no longer accept events. Never paste administrative keys into frontend env variables.

Rotate provider API/deployment tokens in the provider's account settings, update only the CI/platform secret store that uses them, test the new credentials, then revoke the old ones. This repository uses Git integrations and does not require deployment secrets in GitHub Actions. Do not log token values. Analytics preferences and session tokens do not need a shared signing secret; server sessions use random opaque bearer tokens.

## Release gate

CI runs backend tests, frontend unit/diagram tests, TypeScript and production build, whitespace/JSON checks, and desktop/mobile browser smoke tests. Connect GitHub branch protection to the `Release checks` job before production auto-deploys. Vercel Preview deployments require the Git integration and environment configuration described in [DEPLOYMENT.md](DEPLOYMENT.md). A local passing suite does not confirm hosted CORS, provider delivery, cold starts, or production capacity; verify those on the actual deployment.
