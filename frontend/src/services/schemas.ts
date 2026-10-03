import {z} from 'zod';
const n = z.number().finite(), numbers = z.array(n), index = n.int().nonnegative();
const maps = z.object({channels: index, height: index, width: index, scale: n, downsampled: z.boolean(), values: z.array(numbers)});
const histogram = z.array(z.object({start: n, end: n, count: index}));
const ranked = z.array(z.object({index, value: n}));
const state = z.object({step: n.int(), name: z.string(), type: z.string(), values: numbers, scale: n, activity: numbers.optional(), zero_fraction: n.optional()});
const diagram = z.object({columns: z.array(z.object({label: z.string().optional(), size: index, indices: z.array(index), states: z.array(state).min(1), linear_layer: index.nullable()})).min(1), connections: z.array(z.object({step: index, source_column: index, target_column: index, edges: z.array(z.object({source: index, target: index, weight: n, input: n, contribution: n}))})), node_limit: index, edge_limit: index, input_label: z.string().optional()}).superRefine((v, ctx) => {
  for (const col of v.columns) if (col.states.some(s => s.values.length !== col.indices.length)) ctx.addIssue({code: 'custom', message: 'Incomplete activations'});
  for (const edge of v.connections) if (!v.columns[edge.source_column] || !v.columns[edge.target_column]) ctx.addIssue({code: 'custom', message: 'Missing connection column'});
});
export const model = z.object({id: z.string(), name: z.string(), input_features: index.positive(), parameters: index, dataset: z.string(), input_kind: z.string().optional(), input_shape: numbers.optional(), classes: z.array(z.string()).optional(), description: z.string().optional(), default_input: numbers.optional(), test_accuracy: n.optional(), architecture: z.string().optional()});
export const models = z.array(model).min(1);
export const samples = z.array(z.object({label: index, test_index: index, pixels: z.array(n.min(0).max(1)).length(784)})).min(1);
export const trace = z.object({trace_id: z.string(), model_id: z.string(), model_name: z.string(), classes: z.array(z.string()).nullish(), input_maps: maps.nullish(), input_shape: numbers, output: numbers.min(1), probabilities: numbers.nullable(), predicted_class: index.nullable(), elapsed_ms: n, parameter_count: index,
  visualization: diagram, layers: z.array(z.object({id: z.string(), name: z.string(), type: z.string(), order: index, input_shape: numbers, output_shape: numbers, feature_maps: maps.nullish(), activations: numbers, activation_count: index, truncated: z.boolean(), stats: z.object({min: n, max: n, mean: n, std: n, percent_zero: n}), histogram, strongest: ranked, weakest: ranked, parameters: z.object({count: index, weight_shape: numbers.optional(), bias_shape: numbers.nullish(), weight_histogram: histogram.optional()})})).min(1)});
export const dataset = z.object({dataset: z.enum(['xor', 'moons', 'circles', 'spiral']), points: z.array(z.object({id: index, input: z.tuple([n, n]), label: n.int().min(0).max(1), split: z.enum(['train', 'validation'])})).length(200), grid: z.object({size: z.literal(48), min: n, max: n})});
export const run = dataset.extend({model_id: z.string(), model, config: z.object({dataset: z.enum(['xor', 'moons', 'circles', 'spiral']), hidden_layers: numbers, activation: z.enum(['ReLU', 'Tanh', 'Sigmoid']), learning_rate: n, epochs: index, seed: index, noise: n}), metrics: z.object({loss: n, train_accuracy: n, validation_accuracy: n}), snapshots: z.array(z.object({epoch: index, loss: n, validation_loss: n, train_accuracy: n, validation_accuracy: n, grid_probabilities: z.array(n.min(0).max(1)).length(2304), point_probabilities: z.array(n.min(0).max(1)).length(200)})).min(2)});
const contribution = z.object({index, input: n, weight: n, contribution: n});
export const neuron = z.object({index, pre_activation: n, post_activation: n, activation: z.string(), bias: n, contribution_sum: n, top_contributions: z.array(contribution), positive: z.array(contribution), negative: z.array(contribution), input_count: index});
export const focus = z.object({diagram, shown: index, total: index, magnitude_coverage: n, bias: n, pre_activation: n});
export const session = z.object({token: z.string().min(32), expires_in: index});
export const health = z.object({status: z.literal('ok')});
