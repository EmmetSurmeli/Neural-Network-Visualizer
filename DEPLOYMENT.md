# Deploy NeuralScope

The public demo is deployed at [neuralscope-smoky.vercel.app](https://neuralscope-smoky.vercel.app). Its API is [neuralscope-api-24v5.onrender.com](https://neuralscope-api-24v5.onrender.com/health). This guide also covers recreating the deployment or changing its settings.

## 1. Backend on Render

The existing Render Blueprint uses `render.yaml`. It defines one free-tier Docker web service. Review the current plan before upgrading. The CPU-only PyTorch image includes the five bundled models; it never downloads datasets at startup. No persistent disk is needed.

Set `ALLOWED_ORIGINS` to comma-separated **exact** frontend origins, including scheme, without trailing slashes. Initially use your expected Vercel project domain; update it once assigned. Never use `*`. Set `SENTRY_DSN` to a backend project DSN, or leave it empty to disable monitoring. Set `RELEASE` for each release (a Git commit or version).

Keep one instance and one worker. Sessions, imports, Playground models and traces live in this process. Deploys/restarts intentionally expire them. Render's health check is `GET /health`, which returns `{"status":"ok"}` only after startup finishes loading bundled models. Use HTTPS for public traffic.

The container disables access logs and proxy-header trust. Rate limiting uses the socket peer: behind Render's proxy, limits may be shared between visitors. Session capacity is bounded at 64, and the least recently used inactive session is reclaimed when full. A reclaimed visitor can retry with a fresh session, but their imported models and traces are lost. This is suitable for a small demo, not robust abuse prevention. Do not enable unrestricted forwarded-header trust; use a trusted edge rate limiter and revisit session storage before inviting substantial traffic.

## 2. Frontend on Vercel

Import the GitHub repository, with the repository root as Root Directory. `vercel.json` installs/builds `frontend` and publishes `frontend/dist`. Connect the main branch as Production. Vercel's Git integration creates Preview deployments for pull requests; no deployment token belongs in this repository.

Set the following in Vercel's **Production** environment:

| Variable | Value |
|---|---|
| `VITE_API_URL` | Render's HTTPS service URL, without trailing slash |
| `VITE_APP_ENV` | `production` (Vercel's own environment overrides this) |
| `VITE_APP_VERSION` | Release version or Git commit |
| `VITE_PRODUCTION_HOST` | Exact public hostname, without scheme; required for telemetry |
| `VITE_POSTHOG_KEY` | Optional PostHog project token |
| `VITE_POSTHOG_HOST` | Project ingestion host, usually `https://us.i.posthog.com` or EU equivalent |
| `VITE_SENTRY_DSN` | Optional frontend Sentry project DSN |

Only put public project ingestion identifiers in `VITE_` variables. They are visible in browser code. Never put a PostHog personal API key or Sentry auth token there. No source-map upload token is needed; we do not publish source maps.

Set Preview variables separately: `VITE_API_URL`, `VITE_APP_ENV=preview`, `VITE_APP_VERSION=preview`. Leave monitoring variables unset. Vercel's environment plus an exact hostname check prevents preview builds from initializing production telemetry even if variables were copied accidentally.

Add exact preview origins to Render's `ALLOWED_ORIGINS`. For frequent previews, use a stable Vercel branch alias and add it once, or maintain the allowlist for each generated preview URL. Preview frontend deployment is automatic after Git integration; CORS for newly generated origins requires this configuration. A separate staging backend is preferable if production availability matters. Do not grant CORS to every `vercel.app` tenant.

Redeploy after changing build-time frontend variables. Navigation uses `#/visualizer` and `#/playground`. Direct `/visualizer` and `/playground` URLs also have Vercel fallback rewrites.

The frontend Content Security Policy in `vercel.json` currently permits the live Render API and optional PostHog/Sentry ingestion hosts. If the backend URL or telemetry host changes, update `connect-src` and redeploy; otherwise the browser will block those requests. Interactive API docs are available in local development but disabled in production.

## 3. Verify the deployed release

1. Open the site and check **Service ready**. On a cold service, a bounded connection message should become ready after retry.
2. Run the default digit 7. Confirm predicted class 7, play/pause, and output-neuron focus.
3. Import `tiny-model.json`, enter `[1,0.5,-1]`, and run it.
4. Train a 100-epoch Playground run and use **Inspect in Visualizer**.
5. Use **Privacy & session** to check the notice and analytics choice.
6. Confirm PostHog receives only documented events after Allow, stops after opt-out, and receives nothing from Preview.
7. Confirm Sentry project scrubbing rules and retention settings; enable server-side IP removal. Browser checks can verify event contents, but ingestion-provider settings must be verified in those accounts.

The application works without either monitoring provider configured. [OPERATIONS.md](OPERATIONS.md) covers failures and limits; [PRIVACY.md](PRIVACY.md) describes storage and analytics.

## Local development and checks

`./start.sh` builds and serves at `http://127.0.0.1:8000`. For hot reload, run the Python server and `npm run dev --prefix frontend` separately as described in README. Node 24 and Python 3.12+ are recommended for the test toolchain.

Frontend example files are templates, not active environment files. Copy `.env.development.example` to `.env.development.local` if overrides are needed. For a deployment-like local build, copy `.env.production.example` to `.env.production.local` and replace placeholders. Actual env files are ignored by Git. Backend env variables are supplied by the hosting platform or shell; `backend/.env.example` is documentation and is not automatically loaded.

```sh
.venv/bin/python -m pytest backend/tests -q
npm test --prefix frontend
npm run build --prefix frontend
python3 scripts/check_format.py
git diff --check
```

The GitHub Actions workflow also installs Chromium and runs `npm run test:browser --prefix frontend` against a production-mode backend on port 8001 and a preview build on 4173. It covers the main flow on desktop and mobile, including network-failure simulation. Failed browser traces are CI artifacts and contain only bundled test data. Use disposable examples, never private uploaded models, in automated tests.

Provider references: [Vercel Vite deployments](https://vercel.com/docs/frameworks/frontend/vite), [Render Blueprint specification](https://render.com/docs/blueprint-spec), [PostHog browser configuration](https://posthog.com/docs/libraries/js/config), [Sentry filtering](https://docs.sentry.io/platforms/javascript/guides/react/configuration/filtering/).
