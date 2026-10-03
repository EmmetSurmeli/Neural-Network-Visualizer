# NeuralScope

A React + FastAPI application for **seeing how small neural networks learn and make decisions**. Playground trains a small network on a 2D dataset; Visualizer inspects the actual forward pass for one input. The bundled MNIST classifier is trained, not randomly initialized. Draw a digit or choose a test sample, run inference, then inspect the computation layer by layer.

## Public demo deployment

Start with the [launch checklist](LAUNCH.md).

The repository includes Vercel and Render configuration, bounded session memory, request limits, recoverable loading/error states, optional anonymous analytics, and scrubbed error monitoring. See [DEPLOYMENT.md](DEPLOYMENT.md), [PRIVACY.md](PRIVACY.md), and [OPERATIONS.md](OPERATIONS.md). Provider accounts and environment variables must be configured before public launch.

## Run

Requires Python 3.11+ and Node.js 20.19+ (Node 22.12+ also works). Tested here with Python 3.13, Node 24, and PyTorch 2.14 on Apple Silicon.

```bash
./start.sh
```

Open **http://127.0.0.1:8000**. The script installs missing dependencies, builds React, and starts FastAPI serving both the UI and API. The trained checkpoint and samples are included, so no training or dataset download is needed for ordinary use. The first dependency installation needs internet. After installation, inference is entirely local. Stop with Ctrl+C.

## Inputs to test immediately

1. **Playground:** open Playground to start with Two moons with two 8-neuron Tanh layers. Click **Train network**, replay or scrub through the recorded training, then click a point and **Inspect in Visualizer**. The final trained network opens with that point prefilled and runs automatically. Try XOR, circles, or spiral from the dataset selector.
2. **Digit demo:** open **Visualizer**. sample **7** is preloaded. Click **Run forward pass**. The supplied sample predicts **7**, with approximately **99.73%** probability. Click `fc1` and then **Neuron explorer** to inspect input contributions. Presets for all ten digits are included.
3. **Your drawing:** click **Clear**, draw one digit, then run. Drawings are cropped, scaled to a 20-pixel bounding box, centered by their center of mass in a 28×28 image, and normalized to `[0,1]`. Recognition may fail for unusual handwriting; try the presets to verify the pipeline.
4. **Small numeric example:** click **3-input example**. The input is prefilled with **`[1, 0.5, -1]`**. Run it: class **0** has about **96.85%** probability. Its first Linear layer outputs `[2.35, -0.5, -1.45, -0.5]`; ReLU produces `[2.35, 0, 0, 0]`. This is useful for checking the arithmetic by hand.
5. **API example:**

```bash
curl http://127.0.0.1:8000/run-demo \
  -H 'Content-Type: application/json' \
  --data-binary @examples/digit-7.json
```

## Features

- Two persistent sections at `#/playground` and `#/visualizer`, with Visualizer as the default.
- Deterministic XOR, two moons, circles, and spiral datasets: 200 points, stratified 160/40 training/validation split, standardized using training statistics only.
- Adjustable feature noise (0–80% of each clean feature’s standard deviation), new samples, and three starting experiments: low noise, overlapping data, and limited network capacity. Noise changes preserve class labels and the train/validation split at a fixed seed.
- Highlight wrong predictions, jump between mistakes, compare training and validation loss, and retain the previous completed run’s validation score with its data settings.
- Guided MLP builder with 1–3 hidden layers, 2/4/8/16 neurons, ReLU/Tanh/Sigmoid, three learning rates, and 100/300/1,000 epochs. Uses Adam and cross-entropy on raw logits; the inspected final model includes Softmax.
- Training replay with snapshots every five epochs (including epoch zero), a 48×48 probability grid, loss curve, accuracy metrics, selectable points, and keyboard scrubbing. Training finishes before replay starts.
- Point selection works by click, arrow keys in the plot, or the point selector. Inspect always uses the **final trained model**, regardless of the displayed replay epoch.
- Switching sections preserves state until page reload. The backend keeps only the latest trained Playground model without consuming the eight JSON import slots; earlier model IDs expire when retraining succeeds.

- Five built-in models: MNIST MLP, shapes CNN, line-orientation CNN, XOR, and a three-input numeric example. Selecting a model loads a matching test input; press Run forward pass to visualize it.
- CNN playback displays actual convolution/ReLU/pooling feature maps, linked to each execution step. Select a stage to inspect all its channels with a shared signed color scale. Flattening then connects to the dense neuron diagram; convolution connectivity is not represented as fully connected edges.
- Hidden-neuron display switches between 16 and 32 without rerunning inference.
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
  src/App.tsx             Hash navigation and Playground/Visualizer handoff
  src/Visualizer.tsx      Input and trace orchestration
  src/components/         Canvas, graph, inspector, histograms and output
  src/services/           Typed API client
  src/types/              Trace/model/tensor interfaces
  src/styles.css          Responsive visual design, Tailwind integration
examples/                 Digit-7 input and a tiny importable JSON model
scripts/train_demo.py     Reproducible MNIST training
start.sh                  Single-command local startup
```

React/TypeScript/Vite + Tailwind form the UI; FastAPI/Pydantic/PyTorch form the backend. The execution rail uses accessible HTML nodes; the neuron diagram uses interactive SVG. Large columns combine eight strong activations with uniform index coverage, showing 16 or 32 actual hidden neurons (selected in the Network toolbar), plus up to 16 inputs. Connections are the strongest contributions between those displayed neurons, capped at 96 per transition and 600 overall. Omitted neurons and connections still participate in the full inference; the display is a projection, not a smaller replacement model. Numeric labels use recorded values; only brightness and signal positions interpolate. No database, account, cloud service, or authentication is required.

### Instrumentation

`run_trace(model, tensor, model_name)` registers forward hooks on supported leaf modules, switches to evaluation mode, and executes under `torch.inference_mode()`. Each hook clones detached CPU input/output tensors. Execution order is the order in which those hooks run. A `finally` block removes all hooks and restores each module's previous train/eval state, including after inference errors.

JSON summaries include at most 512 activation values per layer, full statistics/histograms, and the top/bottom 20 activations. Raw captures remain in a bounded in-memory cache for on-demand neuron queries; full weight matrices are not sent with every trace. Each visitor has a temporary isolated session with up to 16 traces, eight imported models, and the latest trained Playground model. Sessions expire after 30 idle minutes; global cache limits can expire older traces sooner. A lock serializes inference and registry access so hooks from concurrent requests cannot mix captures.

For Linear neuron `j`, contributions are `x[i] * W[j,i]`; their sum plus `b[j]` matches the recorded Linear output to floating-point tolerance. If the immediately following layer is ReLU, Sigmoid, Tanh or Softmax, the inspector also shows its output for that neuron. Softmax is a vector operation, not an independent scalar activation.

### Model provenance

The bundled MLP was trained from scratch for 12 epochs using Adam, seed 42, on all 60,000 MNIST training images. The final checkpoint achieved **97.59% accuracy on the separate 10,000-image MNIST test set**. See `backend/models/metadata.json`. The ten presets are the first *correctly classified* held-out sample of each class, chosen to make initial testing predictable; the reported accuracy covers the entire test set. Drawing performance is not measured by this score, and Softmax probability is not a guarantee or calibrated confidence.

Dataset files are downloaded by the optional training script from the [PyTorch MNIST mirror](https://ossci-datasets.s3.amazonaws.com/mnist/). Forward capture uses [PyTorch module forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

To retrain (downloads about 12 MB on the first run):

```bash
.venv/bin/python scripts/train_demo.py
```

### CNN examples

The shapes CNN classifies outline circles, squares, and triangles. The line-orientation CNN classifies horizontal, vertical, and downward-diagonal strokes. Each has two convolution blocks (8 then 16 channels), 2×2 max pooling, a 32-neuron dense layer, and three outputs. Both checkpoints are bundled and loaded with `weights_only=True`.

Each model was trained for 15 epochs on 1,800 generated images (seed 42) and scored 100% on 600 separately generated evaluation images (seed 4042). These are deliberately small synthetic tasks; that score does not measure hand-drawn or real-world images. Samples are correctly classified held-out examples. Metadata and generator/training code are included. Reproduce with `.venv/bin/python -m scripts.train_vision`; no download is required.

The input API still accepts 784 flattened normalized pixels. Model metadata supplies `[1,28,28]` so the backend reshapes them before tracing. CNN traces include named classes and bounded feature maps. Linear neuron contributions continue to use exact flattened input values.

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
| `GET /playground/dataset?dataset=moons&seed=42` | Preview deterministic points and plotting bounds |
| `POST /playground/train` | Train, register the final model, and return metrics and replay snapshots |
| `GET /health` | Readiness after built-in models load |
| `POST /session` | Create an anonymous temporary session; send its token in `X-NeuralScope-Session` |
| `DELETE /session` | Delete this session’s in-memory models and traces |
| `GET /demo-models` | Bundled model metadata |
| `GET /samples?model_id=…` | Model-specific image samples (defaults to MNIST) |
| `POST /run-demo` | Run `{ "input": [784 values] }` |
| `POST /upload-model` | Import a controlled JSON model |
| `POST /run-model` | Run `{ "model_id": "…", "input": […] }` |
| `GET /traces/{id}/layers/{layer}/neurons/{neuron}?limit=20` | On-demand neuron contributions |
| `GET /traces/{id}/focus?layer=0&neuron=0` | Two-column view of the strongest 16 incoming contributions across the full layer |

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

The fidelity audit (`backend/tests/test_fidelity.py`) independently evaluates all five built-in models using NumPy, including convolution and pooling. Across 28 preset/random inputs it checks every layer tensor, displayed neuron identity/value, feature map, displayed edge contribution, selected neuron sums and prediction. The largest absolute activation difference from PyTorch was about 0.000011 (float32 rounding). Run `npm test --prefix frontend` with Node 22.6+ to check neuron sampling, activation rendering, replay visibility and circle sizing.

The diagram is a sampled view: omitted neurons and connections still participate in inference. Brightness represents absolute activation relative to the operation's maximum; edge color represents the sign of input × weight. Replay follows operation order, but particles and fades are illustrative timing. CNN tiles show actual feature maps, not individual convolution receptive-field connections. The in-app “How to read this view” disclosure explains these limits.

Select a dense neuron to focus on its incoming connections; select a source neuron to continue backward, or use **Whole network** to return. Focus ranks every input by absolute input × weight contribution, independently of the overview sample. The coverage percentage excludes bias and describes local contribution magnitude, not the fraction of a prediction explained. The full linear score includes all inputs and bias. Layer activity strips cover every activation in up to 48 consecutive averaged blocks; exact zero percentages also cover the whole layer. Zoom buttons enlarge the scrollable canvas, and the replay slider pauses and scrubs the same clock used by the graph and feature maps. No 3D view or additional import format is introduced.

## Known limitations and next steps

V1 targets flat, single-sample feed-forward Sequential models. It does not reconstruct arbitrary control flow, branches, recurrent layers, transformers, or gradients. Built-in CNNs support Conv2d, ReLU, MaxPool2d, Flatten, and dense layers; JSON imports remain limited to flat Sequential models. Traces and imported models are process-local; restarting or running multiple server workers will not preserve them. Run one worker. Public deployment uses anonymous, temporary bearer sessions; no accounts or database are required. Keep a single instance until the in-memory session architecture is replaced.

Playground stays focused on small binary 2D classification problems. It does not train MNIST or imported models. Training requests run synchronously under the inference lock. The default seed is 42; the API accepts a nonnegative 32-bit seed. No datasets or packages are downloaded during training.

Example training request: `{"dataset":"moons","hidden_layers":[8,8],"activation":"Tanh","learning_rate":0.01,"epochs":300,"seed":42,"noise":0.1}`. The response includes `model_id`, `model`, `config`, `points` (ID, input, label, split), `grid` bounds/size, final `metrics`, and `snapshots` (epoch, loss, train/validation accuracy, grid and point probabilities for class 1). Grid values are ordered by increasing y, then increasing x. The frontend reverses y for screen display.
