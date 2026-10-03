# NeuralScope launch checklist

The first release has one job: load a small network, run an input, and watch its actual recorded forward pass. Launch this before adding more model formats or product sections.

## Ready in the repository

- [x] Visualizer, built-in examples, safe JSON imports, neuron focus and Playground handoff.
- [x] Loading, empty, stale, offline, error and retry states; previous results survive failed requests.
- [x] Temporary server sessions with isolation, expiration, bounded memory and computation limits.
- [x] Optional anonymous analytics, privacy controls and scrubbed error monitoring.
- [x] Vercel/Render configuration, environment templates and GitHub Actions checks.
- [x] Local verification: 82 backend tests, 23 frontend tests, production build and whitespace checks pass.
- [x] Interactive browser checks: inference, replay, neuron focus, training handoff, expired-session recovery, service outage and mobile layout.

The desktop/mobile automated browser suite passed in [GitHub Actions](https://github.com/EmmetSurmeli/Neural-Network-Visualizer/actions/runs/37160850711), together with the backend tests, frontend tests and production build. CI also checks the production Docker image; the hosted integration still needs verification after connecting the hosting accounts.

## 1. Publish the code

- [x] Review and commit the current changes, then push to GitHub.
- [x] Confirm the **Release checks** workflow passes, including all four browser smoke scenarios.
- [ ] Require that check on the main branch before automatically deploying future changes.

## 2. Create the Python service

- [ ] In Render, create a Blueprint from this repository; it reads `render.yaml`.
- [ ] Review the selected 2 GB plan and its current price before creating resources.
- [ ] Keep one instance and one worker. Do not attach a persistent disk.
- [ ] Set `ALLOWED_ORIGINS` to your frontend's exact HTTPS origin once Vercel assigns it.
- [ ] Leave `SENTRY_DSN` empty initially, or add the backend project's DSN.
- [ ] Record the assigned backend URL and confirm `/health` returns `{"status":"ok"}`.

## 3. Create the frontend

- [ ] Import the same repository into Vercel using the repository root. The included `vercel.json` supplies the build commands and output directory.
- [ ] In Production environment variables, set:
  - `VITE_API_URL` = the Render HTTPS URL, without a trailing slash.
  - `VITE_APP_ENV` = `production`.
  - `VITE_APP_VERSION` = this release's version or Git commit.
  - `VITE_PRODUCTION_HOST` = the exact Vercel/custom hostname, without `https://`.
- [ ] Deploy, then put its exact origin in Render's `ALLOWED_ORIGINS` and redeploy the backend.
- [ ] Configure Preview variables separately with `VITE_APP_ENV=preview` and no production monitoring keys. Add exact preview origins to Render as needed; do not allow every `vercel.app` site.

## 4. Add monitoring

- [ ] Create a PostHog project; disable unnecessary collection in the provider settings. Add the public project token as `VITE_POSTHOG_KEY` and its ingestion host as `VITE_POSTHOG_HOST` in Vercel Production.
- [ ] Create separate frontend/backend Sentry projects. Add `VITE_SENTRY_DSN` in Vercel and `SENTRY_DSN` in Render; set matching release versions.
- [ ] Set retention and server-side IP scrubbing in the provider accounts. Never put administrative API tokens into `VITE_` variables.
- [ ] Redeploy after changing frontend environment variables.
- [ ] Verify a completed run appears only after allowing analytics; opt out and confirm sending stops. Check that Preview produces no production analytics and no model/input content appears in either provider.

## 5. Open the public trial

- [ ] From the deployed URL, run sample 7, pause/replay and focus an output neuron.
- [ ] Import the bundled tiny JSON model and test `[1, 0.5, -1]` (class 0, about 96.85%).
- [ ] Train 100 epochs in Playground and inspect a selected point.
- [ ] Check both routes on a phone. Temporarily block backend requests in a test browser and verify the previous trace, guide and retry path remain usable.
- [ ] Check behavior after a backend restart and during a cold start. Confirm sessions expire rather than exposing another visitor's model.
- [ ] Invite a few people to try their own compatible JSON models. Ask whether they can explain the prediction more clearly after using the visualization.

Use completed forward passes and successful imports as the first product signals. Broader model support should follow concrete failed use cases from those visitors.

Detailed settings: [Deployment](DEPLOYMENT.md). Storage and analytics: [Privacy](PRIVACY.md). Limits, incidents and credential rotation: [Operations](OPERATIONS.md).
