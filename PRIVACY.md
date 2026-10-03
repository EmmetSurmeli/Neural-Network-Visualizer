# Privacy and temporary storage

NeuralScope has no accounts or saved history. Models, inputs and traces are processed by the Python service in **temporary server memory**. “Session-only” does not mean computation happens entirely in the browser. All public traffic should use HTTPS.

| Data | Location and lifetime |
|---|---|
| Uploaded JSON weights, generated Playground models | Backend memory; eight imports plus latest Playground model per visitor; expires after 30 minutes of session inactivity, clearing, or process restart |
| Inputs and captured activations | Browser state and backend trace memory; at most 16 traces per visitor, also bounded globally; older traces may expire sooner |
| Bearer session token | Browser tab sessionStorage, with memory fallback; expires at the server; never in analytics, URLs or application logs |
| Analytics choice | Browser localStorage until changed or browser storage is cleared |
| Anonymous analytics identifier | Provider-generated in-memory identifier for this page visit, only after Allow |
| Application logs | Route category, random request reference, status and elapsed time; no bodies, query strings, model names, raw inputs or weights |
| Exported/downloaded files | Saved only when the visitor uses download/export; outside server storage |

`Privacy & session → Clear temporary session & reload` removes server models and traces associated with this visitor. Closing a tab does not immediately delete server data; TTL cleanup runs every 30 seconds. Hosted platforms and ingestion providers see connection metadata such as IP addresses as part of delivering requests. Configure provider retention and IP scrubbing before launch. Application code does not promise to erase infrastructure logs.

The production frontend asks whether to allow anonymous usage counts. No analytics initialize until allowed; browser Do Not Track and Global Privacy Control override Allow. Visitors can opt out at any time. No automatic click collection, session replay, user identification, feature flags, URL/referrer collection, or person profiles are enabled. Local and Preview builds never initialize production telemetry.

## Product events

| Event | Trigger |
|---|---|
| `page_view` | Route view, or the current page when analytics first becomes enabled |
| `model_loaded` | User selects/imports a model |
| `forward_pass_started` | An explicit run or Playground inspection handoff starts |
| `forward_pass_completed` | A valid visualization trace is returned |
| `forward_pass_failed` | The requested forward pass fails |
| `playground_training_completed` | A validated complete training replay arrives |
| `playground_training_failed` | Training request fails |
| `visualizer_focus_opened` | Incoming-connections view successfully loads |
| `replay_started` | User starts network replay or restarts training replay |
| `model_import_started` | Visitor chooses a file to import |
| `model_import_failed` | File validation or server import fails |

Allowed properties: route (`visualizer`/`playground`), built-in model ID only, model category (`built-in`/`imported`/`playground`), outcome, input-size bucket, bounded layer count, elapsed-time bucket, small/large device class, and frontend release version. PostHog receives only those fields plus its anonymous distinct ID and flags disabling profiles/geolocation. Imported IDs, names, filenames, inputs, weights, trace payloads, feature maps and point data are excluded. Invalid JSON rejected before a server request produces only the coarse import-failed event; its contents are not forwarded.

The primary conversion is `forward_pass_completed / forward_pass_started`; a completed trace is not proof the visitor watched the entire animation. Initial sample loading produces no completion event. No unique-person or cross-device claim is made because identifiers are in memory and there are no accounts.

## Error monitoring

Frontend and backend Sentry events are rebuilt from a strict allowlist. They contain a generic error type/message, release, environment, route category, and optional request ID. Payloads exclude requests, arbitrary messages, inputs, drawings, filenames, locals, breadcrumbs, user data and attachments. SDK automatic integrations and performance/session recording are disabled. Runtime platform is recorded; full user-agent strings and arbitrary runtime contexts are not retained. Backend capture can retain safe repository source frame locations, but never variable values.

Production service errors are monitored independently of the optional product analytics choice. No input/model content is included. Configure Sentry retention and server-side IP removal to complement the application scrubber.
