# Contributing

Thanks for helping improve NeuralScope. The project stays focused on showing how small networks learn and how a real forward pass produces a prediction. Please open an issue before starting a large new model format or product area.

## Before a pull request

1. Use the local setup in [README.md](README.md#run-locally). The backend and frontend can run separately for hot reload; [DEPLOYMENT.md](DEPLOYMENT.md#local-development-and-checks) has the commands.
2. Keep sample inputs and model files synthetic or publicly distributable. Never commit credentials, private models, user inputs, or exported traces from real visitors.
3. Run `.venv/bin/python -m pytest backend/tests -q`, `npm test --prefix frontend`, and `npm run build --prefix frontend`. For UI changes, run `npm run test:browser --prefix frontend` with Playwright Chromium installed.
4. Explain what changed, why it helps users, and any limits of the visualization. If a drawing is illustrative rather than a measured tensor value, say so in the UI and the pull request.

For bugs, use the [issue tracker](https://github.com/EmmetSurmeli/Neural-Network-Visualizer/issues) with a minimal example and a request ID if one appeared. Do not post private model contents or sensitive inputs. Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public bug report.
