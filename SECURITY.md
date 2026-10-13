# Security policy

The current `main` branch and the public demo are supported. Older commits and personal forks are not maintained.

If you find a vulnerability, please use GitHub's **Report a vulnerability** option on this repository's Security tab if it is available. Otherwise, contact the maintainer through the [GitHub profile](https://github.com/EmmetSurmeli) to arrange a private channel. If no private channel is available, open an issue asking for a security contact **without including exploit details, credentials, private model files, or personal data**.

Please include a clear description, affected endpoint or component, reproducible steps using disposable data, and the likely impact. Do not run denial-of-service tests against the public service or access another visitor's data. We will acknowledge a report, investigate it, and coordinate a fix and disclosure.

The demo accepts only validated JSON model specifications; it does not execute uploaded Python, ONNX, or pickle files. Public imports and traces are temporary, session-scoped server data. See [PRIVACY.md](PRIVACY.md) for retention and [OPERATIONS.md](OPERATIONS.md) for service limits.
