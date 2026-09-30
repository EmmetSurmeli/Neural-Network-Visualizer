import test from 'node:test';
import assert from 'node:assert/strict';
import {projectDiagram, displayed, featureReveal, neuronRadius, replayPosition} from '../src/lib/diagram.ts';

// Deliberately non-contiguous model indices and signed values detect positional/index mixups.
const indices = Array.from({length:32}, (_, i) => i * 3);
const hidden = {size:100, indices, linear_layer:0, states:[
  {step:0, name:'linear', type:'Linear', values:indices.map(i => i - 45), scale:48},
  {step:1, name:'relu', type:'ReLU', values:indices.map(i => Math.max(0, i - 45)), scale:48},
]};
const trace = {visualization:{columns:[
  {size:2, indices:[0,1], linear_layer:null, states:[{step:-1,name:'Input',type:'Input',values:[-2,1],scale:2}]},
  hidden,
  {size:1,indices:[0],linear_layer:2,states:[{step:2,name:'out',type:'Linear',values:[3],scale:3}]},
],connections:[{step:0,source_column:0,target_column:1,edges:indices.map(i=>({source:1,target:i,weight:i-45,input:1,contribution:i-45}))}]}};

for (const limit of [16,32]) {
  test(`${limit}-neuron view keeps actual neuron identities and corresponding values`, () => {
    const graph = projectDiagram(trace as any, limit);
    assert.equal(graph.columns[1].indices.length, limit);
    for (const [position, index] of graph.columns[1].indices.entries()) {
      assert.equal(graph.columns[1].states[0].values[position], index-45);
      assert.equal(graph.columns[1].states[1].values[position], Math.max(0,index-45));
    }
    for (const edge of graph.connections[0].edges) {
      assert.ok(graph.columns[1].indices.includes(edge.target));
      assert.equal(edge.contribution, edge.target-45);
    }
    assert.deepEqual(graph.columns[0].indices,[0,1]);
  });
}
test('future layers stay dark and completed activations match their recorded stage', () => {
  assert.equal(displayed(hidden,0,-1,1).ready,false);
  assert.equal(displayed(hidden,0,-1,1).brightness,0);
  const negative = displayed(hidden,0,0,1);
  assert.equal(negative.value,-45);
  assert.ok(negative.brightness>0); // absolute magnitude, not a binary firing flag
  assert.equal(displayed(hidden,0,1,1).value,0);
  assert.equal(displayed(hidden,0,1,1).brightness,0); // ReLU kills this negative value
  assert.equal(displayed(hidden,31,1,1).value,48);
  assert.equal(displayed(hidden,31,1,1).brightness,232);
});
test('animation changes brightness without fabricating numerical activations', () => {
  for (const progress of [0,.2,.55,.8,1]) {
    assert.equal(displayed(hidden,31,0,progress).value,48);
  }
  assert.equal(displayed(hidden,31,0,0).brightness,0);
  assert.equal(displayed(hidden,31,0,1).brightness,232);
});
test('CNN output maps reveal only at their execution step and stay visible afterward', () => {
  assert.equal(featureReveal(3,2,1),0);
  assert.equal(featureReveal(3,3,0),0);
  assert.equal(featureReveal(3,3,.5),.5);
  assert.equal(featureReveal(3,3,1),1);
  assert.equal(featureReveal(3,4,0),1);
  assert.equal(featureReveal(-1,-1,0),1);
});
test('connection endpoints use the same radius as 16/32-neuron circles', () => {
  assert.equal(neuronRadius(16),10);
  assert.equal(neuronRadius(32),5);
});

test('scrubbing includes the input and the complete final operation', () => {
  assert.deepEqual(replayPosition(0, 4), {step:-1, progress:0});
  assert.deepEqual(replayPosition(1.5, 4), {step:0, progress:.5});
  assert.deepEqual(replayPosition(4.5, 4), {step:3, progress:.5});
  assert.deepEqual(replayPosition(5, 4), {step:3, progress:1});
  assert.deepEqual(replayPosition(100, 4), {step:3, progress:1});
});
