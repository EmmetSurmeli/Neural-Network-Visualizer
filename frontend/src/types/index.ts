export interface HistogramBin {start: number; end: number; count: number}
export interface Stats {min: number; max: number; mean: number; std: number; percent_zero: number}
export interface Ranked {index: number; value: number}
export interface Layer {
  id: string; name: string; type: string; order: number; input_shape: number[]; output_shape: number[];
  activations: number[]; activation_count: number; truncated: boolean; stats: Stats; histogram: HistogramBin[];
  strongest: Ranked[]; weakest: Ranked[];
  parameters: {count: number; weight_shape?: number[]; bias_shape?: number[] | null; weight_histogram?: HistogramBin[]};
}
export interface VisualState {step: number; name: string; type: string; values: number[]; scale: number}
export interface VisualColumn {size: number; indices: number[]; states: VisualState[]; linear_layer: number | null}
export interface VisualConnection {step: number; source_column: number; target_column: number; edges: {source: number; target: number; weight: number; input: number; contribution: number}[]}
export interface Trace {trace_id: string; model_id: string; model_name: string; layers: Layer[]; output: number[]; probabilities: number[] | null; predicted_class: number | null; elapsed_ms: number; parameter_count: number; input_shape: number[]; visualization: {columns: VisualColumn[]; connections: VisualConnection[]; node_limit: number; edge_limit: number}}
export interface NeuronSelection {layer: number; index: number; key: number}
export interface Model {id: string; name: string; input_features: number; parameters: number; test_accuracy?: number; dataset: string}
export interface Sample {label: number; test_index: number; pixels: number[]}
export interface Contribution {index: number; input: number; weight: number; contribution: number}
export interface Neuron {index: number; pre_activation: number; post_activation: number; activation: string; bias: number; contribution_sum: number; top_contributions: Contribution[]; positive: Contribution[]; negative: Contribution[]; input_count: number}
