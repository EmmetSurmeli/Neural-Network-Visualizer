# Public demo release status

NeuralScope is live at [neuralscope-smoky.vercel.app](https://neuralscope-smoky.vercel.app). Its [Render API](https://neuralscope-api-24v5.onrender.com/health) serves the Visualizer and Playground. The site is a public, no-login demonstration; imported models and traces live only in temporary server sessions.

The repository includes automated backend, frontend, browser, build, and Docker checks in [Release checks](.github/workflows/checks.yml). The hosted Visualizer has completed a digit-7 forward pass, and the hosted Playground has trained a 100-epoch two-moons run. Both hash routes work directly. A free Render instance may need time to wake after inactivity; the frontend shows service status and retries automatically for a bounded period.

## Remaining launch tasks

- [ ] In GitHub repository settings, protect `main` and require **Release checks** before merging. This setting requires repository administration access.
- [ ] Review the [privacy notice](PRIVACY.md) against the jurisdictions where the site will be promoted. The repository describes the current data flow; it is not legal advice.
- [ ] Decide whether to enable optional PostHog and Sentry. They are not required for the demo. If enabled, set their production environment variables, retention limits, and IP-scrubbing settings as described in [DEPLOYMENT.md](DEPLOYMENT.md) and [OPERATIONS.md](OPERATIONS.md).
- [ ] Ask a few visitors to try a compatible JSON model and report where the visualization is confusing. Treat those concrete cases as the guide for future model support.

## Operating constraints

The current backend runs one worker on a free 512 MB instance, because the session registry lives in process memory. Sessions expire, uploads are not saved to disk, and service restarts clear temporary models and traces. Heavy public traffic or large imported models may require a larger instance or a different session architecture. The app intentionally limits imports, traces, model size, and concurrent computation; see [OPERATIONS.md](OPERATIONS.md). There are no accounts, saved histories, ONNX imports, or arbitrary PyTorch code execution.

For deployment settings and recovery procedures, use [DEPLOYMENT.md](DEPLOYMENT.md) and [OPERATIONS.md](OPERATIONS.md).
