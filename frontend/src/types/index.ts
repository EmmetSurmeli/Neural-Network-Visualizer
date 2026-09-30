export interface FeatureMaps {channels: number; height: number; width: number; scale: number; downsampled: boolean; values: number[][]}
export interface HistogramBin {start: number; end: number; count: number}
export interface Stats {min: number; max: number; mean: number; std: number; percent_zero: number}
export interface Ranked {index: number; value: number}
export interface Layer {
  id: string; name: string; type: string; order: number; input_shape: number[]; output_shape: number[];
  feature_maps?: FeatureMaps | null; activations: number[]; activation_count: number; truncated: boolean; stats: Stats; histogram: HistogramBin[];
  strongest: Ranked[]; weakest: Ranked[];
  parameters: {count: number; weight_shape?: number[]; bias_shape?: number[] | null; weight_histogram?: HistogramBin[]};
}
export interface VisualState {step: number; name: string; type: string; values: number[]; scale: number; activity?: number[]; zero_fraction?: number}
export interface VisualColumn {label?: string; size: number; indices: number[]; states: VisualState[]; linear_layer: number | null}
export interface VisualConnection {step: number; source_column: number; target_column: number; edges: {source: number; target: number; weight: number; input: number; contribution: number}[]}
export interface Trace {classes?: string[] | null; input_maps?: FeatureMaps | null; trace_id: string; model_id: string; model_name: string; layers: Layer[]; output: number[]; probabilities: number[] | null; predicted_class: number | null; elapsed_ms: number; parameter_count: number; input_shape: number[]; visualization: {columns: VisualColumn[]; connections: VisualConnection[]; node_limit: number; edge_limit: number; input_label?: string}}
export interface NeuronSelection {layer: number; index: number; key: number}
export interface FocusView {diagram: Trace['visualization']; shown: number; total: number; magnitude_coverage: number; bias: number; pre_activation: number}
export interface Model {input_kind?: string; input_shape?: number[]; classes?: string[]; description?: string; default_input?: number[]; id: string; name: string; input_features: number; parameters: number; test_accuracy?: number; dataset: string; architecture?: string}
export interface Sample {label: number; test_index: number; pixels: number[]}
export interface Contribution {index: number; input: number; weight: number; contribution: number}
export interface Neuron {index: number; pre_activation: number; post_activation: number; activation: string; bias: number; contribution_sum: number; top_contributions: Contribution[]; positive: Contribution[]; negative: Contribution[]; input_count: number}
export type DatasetName = 'xor' | 'moons' | 'circles' | 'spiral';
export interface PlaygroundConfig {dataset: DatasetName; hidden_layers: number[]; activation: 'ReLU' | 'Tanh' | 'Sigmoid'; learning_rate: number; epochs: number; seed: number; noise: number}
export interface DataPoint {id: number; input: [number, number]; label: number; split: 'train' | 'validation'}
export interface PlaygroundDataset {dataset: DatasetName; points: DataPoint[]; grid: {size: number; min: number; max: number}}
export interface TrainingSnapshot {epoch: number; loss: number; validation_loss: number; train_accuracy: number; validation_accuracy: number; grid_probabilities: number[]; point_probabilities: number[]}
export interface PlaygroundRun extends PlaygroundDataset {config: PlaygroundConfig; model_id: string; model: Model; snapshots: TrainingSnapshot[]; metrics: {loss: number; train_accuracy: number; validation_accuracy: number}}
export interface PlaygroundHandoff {key: number; model: Model; dataset: string; point: DataPoint}
