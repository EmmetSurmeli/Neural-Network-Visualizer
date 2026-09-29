# NeuralScope

A working local React + FastAPI application for exploring **the actual forward pass of a PyTorch model on a particular input**. The bundled MNIST classifier is trained, not randomly initialized. Draw a digit or choose a test sample, run inference, then inspect the computation layer by layer.

## Run

Requires Python 3.11+ and Node.js 20.19+ (Node 22.12+ also works). Tested here with Python 3.13, Node 24, and PyTorch 2.14 on Apple Silicon.

```bash
./start.sh
```

Open **http://127.0.0.1:8000**. The script installs missing dependencies, builds React, and starts FastAPI serving both the UI and API. The trained checkpoint and samples are included, so no training or dataset download is needed for ordinary use. The first dependency installation needs internet. After installation, inference is entirely local. Stop with Ctrl+C.

## Inputs to test immediately

1. **Digit demo:** sample **7** is preloaded. Click **Run forward pass**. The supplied sample predicts **7**, with approximately **99.73%** probability. Click `fc1` and then **Neuron explorer** to inspect input contributions. Presets for all ten digits are included.
2. **Your drawing:** click **Clear**, draw one digit, then run. Drawings are cropped, scaled to a 20-pixel bounding box, centered by their center of mass in a 28×28 image, and normalized to `[0,1]`. Recognition may fail for unusual handwriting; try the presets to verify the pipeline.
3. **Small numeric example:** click **3-input example**. The input is prefilled with **`[1, 0.5, -1]`**. Run it: class **0** has about **96.85%** probability. Its first Linear layer outputs `[2.35, -0.5, -1.45, -0.5]`; ReLU produces `[2.35, 0, 0, 0]`. This is useful for checking the arithmetic by hand.
4. **API example:**

```bash
curl http://127.0.0.1:8000/run-demo \
  -H 'Content-Type: application/json' \
  --data-binary @examples/digit-7.json
```

## Features

- Trained `784 → 128 → ReLU → 64 → ReLU → 10 → Softmax` digit classifier.
- Canvas, ten sample inputs, and raw JSON numeric input.
- An execution graph built from actual forward-hook order and tensor shapes, linked to a neuron-and-connection diagram.
- Grayscale neuron brightness shows absolute activation, normalized per operation. Cyan/red edges show positive/negative input × weight contributions. Moving signals follow Linear operations; ReLU, Sigmoid, Tanh, Softmax and other pointwise operations update the same column.
- Play, pause, reset, previous-step, next-step, and 0.5×/1×/2× animation speed. A shared clock keeps the execution rail, signals and activation transitions synchronized. Animation replays a completed trace; it does not claim that the millisecond computation runs in slow motion.
- Layer statistics: min, max, mean, population standard deviation, and percent zero.
- Activation and weight histograms, strongest and weakest neurons by magnitude.
- Neuron inspection with actual `input × weight` contributions, bias, pre-activation, and the following activation's output. Positive and negative inputs are also available separately. Configurable top 10/20/50 contributions.
- Class probabilities, raw output, and downloadable trace JSON.
- Validated declarative JSON model imports; no uploaded Python or pickle execution.
- Responsive dark interface, keyboard-accessible controls, and stale-input/error feedback.

## Architecture

```text
backend/
  main.py                 FastAPI routes, local model registry and trace cache
  models/                 PyTorch demo, bundled checkpoint, metadata and samples
  instrumentation/        Reusable hooks, statistics, histograms and contributions
  schemas/                Pydantic request validation
  services/               Controlled JSON model import and local export helper
  tests/                  Pytest inference, instrumentation and API checks
frontend/
  src/App.tsx             Input and trace orchestration
  src/components/         Canvas, graph, inspector, histograms and output
  src/services/           Typed API client
  src/types/              Trace/model/tensor interfaces
  src/styles.css          Responsive visual design, Tailwind integration
examples/                 Digit-7 input and a tiny importable JSON model
scripts/train_demo.py     Reproducible MNIST training
start.sh                  Single-command local startup
```

React/TypeScript/Vite + Tailwind form the UI; FastAPI/Pydantic/PyTorch form the backend. The execution rail uses accessible HTML nodes; the neuron diagram uses interactive SVG. Large columns combine eight strong activations with uniform index coverage, showing at most 16 actual neurons. Connections are the strongest contributions between those displayed neurons, capped at 96 per transition and 600 overall. Omitted neurons and connections still participate in the full inference; the display is a projection, not a smaller replacement model. Numeric labels use recorded values; only brightness and signal positions interpolate. No database, account, cloud service, or authentication is required.

### Instrumentation

`run_trace(model, tensor, model_name)` registers forward hooks on supported leaf modules, switches to evaluation mode, and executes under `torch.inference_mode()`. Each hook clones detached CPU input/output tensors. Execution order is the order in which those hooks run. A `finally` block removes all hooks and restores each module's previous train/eval state, including after inference errors.

JSON summaries include at most 512 activation values per layer, full statistics/histograms, and the top/bottom 20 activations. Raw captures remain in a bounded in-memory cache for on-demand neuron queries; full weight matrices are not sent with every trace. The last 16 traces and up to eight imported models are retained until restart. A lock serializes inference and registry access so hooks from concurrent requests cannot mix captures.

For Linear neuron `j`, contributions are `x[i] * W[j,i]`; their sum plus `b[j]` matches the recorded Linear output to floating-point tolerance. If the immediately following layer is ReLU, Sigmoid, Tanh or Softmax, the inspector also shows its output for that neuron. Softmax is a vector operation, not an independent scalar activation.

### Model provenance

The bundled MLP was trained from scratch for 12 epochs using Adam, seed 42, on all 60,000 MNIST training images. The final checkpoint achieved **97.59% accuracy on the separate 10,000-image MNIST test set**. See `backend/models/metadata.json`. The ten presets are the first *correctly classified* held-out sample of each class, chosen to make initial testing predictable; the reported accuracy covers the entire test set. Drawing performance is not measured by this score, and Softmax probability is not a guarantee or calibrated confidence.

Dataset files are downloaded by the optional training script from the [PyTorch MNIST mirror](https://ossci-datasets.s3.amazonaws.com/mnist/). Forward capture uses [PyTorch module forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

To retrain (downloads about 12 MB on the first run):

```bash
.venv/bin/python scripts/train_demo.py
```

### Import your model

V1 imports a **JSON description of a Sequential model with numeric weights and biases**, such as `examples/tiny-model.json`. Click **Format** in the model toolbar to see the schema and download two more examples, then click **Import model** to choose the JSON file. Imported models display their reconstructed architecture and accept a flat numeric array of the indicated size. Outputs are labeled as probabilities only if the last layer is Softmax; otherwise the raw output vector is shown.

The checked examples deliberately cover different structures:

- `examples/tiny-model.json`: ReLU classifier, `3 → 4 → 2 → Softmax`.
- `examples/xor-model.json`: two hidden Sigmoid stages and a one-value raw output. Inputs `[0,1]` and `[1,0]` produce about `0.992356`; `[0,0]` and `[1,1]` produce about `0.007153`.
- `examples/tanh-classifier.json`: `4 → 5 → Tanh → Dropout → 3 → Softmax`. Dropout is intentionally disabled during inference.

Supported layers: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax` (last dimension), `Dropout` (disabled in evaluation mode), and default `Flatten`. Linear weights are shaped `[output_features, input_features]`; omitted/null bias means no bias. Layer dimensions are validated.

To export an already-constructed, trusted model in your own local Python code:

```python
from backend.services.export import export_sequential

# model is your existing torch.nn.Sequential instance.
export_sequential(model, input_features=3, path="my-model.json", name="My MLP")
```

Raw `.pt`/`.pth` uploads and arbitrary Python source are intentionally unsupported. The only `torch.load` reads this repository's own trusted checkpoint with `weights_only=True`.

Limits: 8 MB request bodies, one sample per run, 4,096 input features, 24 layers, 1,024 output features per Linear layer, and 500,000 imported parameters. Inputs and weights must be finite and at most 1,000,000 in magnitude; MNIST inputs must be in `[0,1]`. Layer activation previews, contribution bars, and large class lists are summarized rather than drawn in full.

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Runtime readiness and PyTorch version |
| `GET /demo-models` | Bundled model metadata |
| `GET /samples` | Ten normalized MNIST test samples |
| `POST /run-demo` | Run `{ "input": [784 values] }` |
| `POST /upload-model` | Import a controlled JSON model |
| `POST /run-model` | Run `{ "model_id": "…", "input": […] }` |
| `GET /traces/{id}/layers/{layer}/neurons/{neuron}?limit=20` | On-demand neuron contributions |

Interactive API documentation is at **http://127.0.0.1:8000/docs**. Layer and neuron indices are zero-based.

## Development and verification

For hot reload, use two terminals from the repository root:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm run dev
```

Open **http://127.0.0.1:5173**; Vite proxies `/api` to FastAPI. In production mode `start.sh` serves the compiled frontend and API from the same origin.

```bash
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m pytest backend/tests -q
cd frontend && npm run build
```

Tests cover all digit presets, hook order/cleanup/restoration, failed-inference cleanup, tensor summaries, population statistics, exact neuron arithmetic, activation types, inference-mode Dropout, health/forward APIs, JSON import validation, dimension/non-finite validation, expired traces, request-size limits, faithful/bounded neuron diagram data, XOR/Sigmoid behavior, Tanh/Dropout/Softmax inference, raw regression output, and exact export/import parity for a real PyTorch Sequential model. The frontend build includes strict TypeScript checking. The browser flow was checked with digit 7 and imported examples, neuron selection, moving signals, exact pause/resume behavior, and Linear-to-ReLU step transitions.

## Known limitations and next steps

V1 targets flat, single-sample feed-forward Sequential models. It does not reconstruct arbitrary control flow, branches, recurrent layers, CNN feature maps, transformers, or gradients. Traces and imported models are process-local; restarting or running multiple server workers will not preserve them. Run one worker. This is a localhost developer tool, not an authenticated multi-user deployment.

Useful next additions: compare two input traces, map pixel contributions onto the input image, add Conv2d feature-map inspection, then add loss/gradient visualization and transformer-specific views.
