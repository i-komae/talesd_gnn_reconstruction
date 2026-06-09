Heterogeneous model explained
=============================

This page explains the current TALE-SD heterogeneous model from input arrays to
output predictions.
It follows the teaching style of the official
`PyTorch Dataset/DataLoader tutorial <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_,
the official
`PyTorch training tutorial <https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html>`_,
and the official
`PyG heterogeneous graph guide <https://pytorch-geometric.readthedocs.io/en/stable/notes/heterogeneous.html>`_:
first define the data object, then define the model, then show how training and
inference use the same inputs.

The code paths described here are the actual code paths used by the repository:

.. list-table::
   :header-rows: 1

   * - Step
     - Implementation
   * - HDF5 sample read/write
     - ``talesd_gnn_reconstruction.hetero_graph_io``
   * - Tensor / PyG conversion
     - ``talesd_gnn_reconstruction.hetero_data``
   * - Model
     - ``talesd_gnn_reconstruction.hetero_model.MinimalHeteroTaleSdGNN``
   * - Training
     - ``talesd_gnn_reconstruction.hetero_training.train_hetero_model``
   * - Direct DST inference
     - ``talesd_gnn_reconstruction.hetero_predict.reconstruct_dst``

Forward-pass overview
---------------------

.. figure:: ../fig/hetero_model_forward.svg
   :alt: Forward pass of the TALE-SD heterogeneous model from GraphEvent to output heads.
   :width: 100%

   The model keeps a full event graph. Detector waveforms are stored once on
   detector nodes. Pulse nodes keep ``pulse_detector_index`` and
   ``pulse_bounds``. Relation attention updates detector and pulse states, and
   type-wise readout produces one event vector.

One event as a heterogeneous data object
----------------------------------------

PyG represents heterogeneous graphs with separate node stores and edge stores.
The TALE-SD graph uses the same idea. One event has two node types and seven
v3 relation types:

.. code-block:: text

   node types:
     detector
     pulse

   edge types:
     ("pulse", "same_detector_next", "pulse")
     ("pulse", "same_detector_prev", "pulse")
     ("pulse", "near_space", "pulse")
     ("pulse", "time_causal", "pulse")
     ("detector", "near", "detector")
     ("detector", "observes", "pulse")
     ("pulse", "observed_by", "detector")

The production training path uses PyG ``HeteroData`` for batching. The HDF5
dataset returns one ``HeteroData`` object per event, and
``torch_geometric.loader.DataLoader`` batches those objects. Inside
``MinimalHeteroTaleSdGNN.forward``, the batched ``HeteroData`` object is
converted back to the repository's explicit tensor dictionary by
``hetero_data_to_tensors``. Direct single-event paths can also feed the explicit
tensor dictionary directly. This is the actual structure built by
``hetero_data.sample_to_hetero_data`` before batching:

.. code-block:: python

   data["detector"].x = tensors["detector"]["x"]
   data["detector"].context = tensors["detector"]["context"]
   data["detector"].pos = tensors["detector"]["pos"]
   data["detector"].lid = tensors["detector"]["lid"]
   data["detector"].waveform = tensors["detector"]["waveform"]

   data["pulse"].x = tensors["pulse"]["x"]
   data["pulse"].pos = tensors["pulse"]["pos"]
   data["pulse"].lid = tensors["pulse"]["lid"]
   data["pulse"].detector_index = tensors["pulse"]["detector_index"]
   data["pulse"].pulse_bounds = tensors["pulse"]["pulse_bounds"]

   for relation, edge_type in EDGE_TYPE_BY_RELATION.items():
       data[edge_type].edge_index = tensors["edge_index_by_type"][relation]
       data[edge_type].edge_attr = tensors["edge_features_by_type"][relation]

This is not a single homogeneous feature matrix. Detector nodes and pulse nodes
carry different fields because they represent different physical objects.

Input fields
------------

``detector_features`` are detector-level signal, timing, and local geometry
features. ``detector_context_features`` are readout and calibration context.
Keeping them separate makes it possible to include, remove, or ablate context
without silently mixing it with shower features.

``detector_waveforms`` are full detector-level calibrated VEM waveforms. They
are not duplicated on pulse nodes. A pulse points back to its detector with
``pulse_detector_index`` and records the relevant time window through
``pulse_bounds``. In schema v3, no-signal live detectors have
``detector_waveform_valid = 0``. Their waveform arrays are zero-filled by
``dstio`` and the model masks the waveform embedding to zero before it is
concatenated with detector scalar/context embeddings.

Schema v3 detector features also include
``detector_has_ising_kept_pulse``, ``detector_ising_kept_pulse_count``, and
``detector_ising_removed_pulse_count``. These values are detector-level
summaries of the retained pulse nodes. They prevent an Ising-rejected-only
detector from being treated the same as a detector with at least one
Ising-kept pulse.

``pulse_features`` contain pulse timing, charge, core-relative coordinates when
the Ising reference core exists, and Ising annotations. Ising-rejected pulse
candidates are kept as input because delayed pulses, waveform tails, and
multi-pulse structure are physically relevant for mass and energy.

``edge_features_by_type`` stores continuous physical edge attributes. For
example, pulse-pulse edges include timing differences, spatial separation, and
Ising weights. These are not just labels; they are numerical inputs to the
attention calculation.

The v3 pulse-pulse relations are deliberately separated:

``pulse__same_detector_next__pulse`` / ``pulse__same_detector_prev__pulse``
   Consecutive pulses in the same detector, with explicit forward and reverse
   time-order relations.

``pulse__near_space__pulse``
   Pulses on different detectors with detector distance ``<= 1.5 km``. This is
   bidirectional and has no time cut.

``pulse__time_causal__pulse``
   A stricter near-space subset: ``abs(dt) <= distance / c + 2 FADC bins`` and
   Ising ``raw_weight >= 0.2``. This is still bidirectional because it marks a
   compatible pair, not a directed shower-front ordering.

These relations are graph schema decisions. The GNN model consumes the edge
sets written by ``dstio`` and does not cut or add edges during training.

Conversion and scaling
----------------------

``hetero_sample_to_tensors`` converts NumPy arrays from HDF5 or direct DST
graphs into tensors. If scalers are supplied, detector, context, pulse, edge,
and target arrays are standardized using training-split statistics.

.. code-block:: python

   detector_features = _scale_tensor(
       detector_features,
       _scaler_for(scalers, "detector", "detector_features"),
   )
   detector_context = _scale_tensor(
       detector_context,
       _scaler_for(scalers, "detector_context", "detector_context_features"),
   )
   pulse_features = _scale_tensor(
       pulse_features,
       _scaler_for(scalers, "pulse", "pulse_features"),
   )
   edge_features_by_type[relation] = _scale_tensor(
       edge_features,
       _scaler_for(scalers, f"edge:{relation}", relation),
   )
   target_tensor = _scale_tensor(target_tensor, _scaler_for(scalers, "target"))

This mirrors the PyTorch Dataset/DataLoader separation: the dataset reads one
sample, the conversion layer turns it into a typed graph object, the PyG
DataLoader batches typed graph objects, and the model receives a unified tensor
dictionary internally.

Tensor shapes and matrix operations
-----------------------------------

After PyG batching, nodes from several events are concatenated. The model keeps
batch vectors so readout can later separate the events again. In the notation
below, ``N_d`` is the number of detector nodes in the batch, ``N_p`` is the
number of pulse nodes, ``E_r`` is the number of edges for relation ``r``,
``H`` is the hidden dimension, and ``B`` is the number of events in the batch.

.. list-table::
   :header-rows: 1

   * - Tensor
     - Shape
     - Meaning
   * - ``detector.x``
     - ``[N_d, F_detector]``
     - detector signal, timing, and local geometry features
   * - ``detector.context``
     - ``[N_d, F_context]``
     - readout/calibration context features
   * - ``detector.waveform``
     - ``[N_d, C_waveform, T]``
     - detector-level waveform channels over time
   * - ``pulse.x``
     - ``[N_p, F_pulse]``
     - pulse features, including Ising annotations
   * - ``edge_index_by_type[r]``
     - ``[2, E_r]``
     - source and destination node indices for relation ``r``
   * - ``edge_features_by_type[r]``
     - ``[E_r, F_edge_r]``
     - continuous physical edge attributes
   * - ``detector.batch`` / ``pulse.batch``
     - ``[N_d]`` / ``[N_p]``
     - event index for each node after batching

The model does not form a dense adjacency matrix. ``edge_index_by_type[r][0]``
selects source rows and ``edge_index_by_type[r][1]`` selects destination rows.
All graph propagation is then done with row selection, grouped softmax, and
``index_add_`` accumulation.

The first encoding step is a set of ordinary learned matrix multiplications
inside MLPs:

.. code-block:: text

   detector_feature_embedding = MLP_detector(detector.x)        -> [N_d, H]
   detector_context_embedding = MLP_context(detector.context)   -> [N_d, H]
   pulse_hidden_0             = MLP_pulse(pulse.x)              -> [N_p, H]

For detector waveforms, the tensor is first processed as a 1D time series per
detector:

.. code-block:: text

   detector.waveform [N_d, C_waveform, T]
     -> Conv1d layers
     -> encoded waveform time sequence [N_d, T, H_waveform]

If ``WAVEFORM_ENCODER=transformer``, the Transformer operates only on this
per-detector waveform time sequence. It computes self-attention between time
bins of the same detector waveform, not between detector and pulse graph nodes.
The waveform sequence is then pooled:

.. code-block:: text

   waveform_embedding =
     Linear(concat(mean_time(encoded), max_time(encoded))) -> [N_d, H_waveform]

Finally, the detector branch concatenates the three detector embeddings and
projects them into the graph hidden space:

.. code-block:: text

   detector_hidden_0 =
     Linear(concat(detector_feature_embedding,
                   detector_context_embedding,
                   waveform_embedding)) -> [N_d, H]

Detector and pulse encoders
---------------------------

The model first embeds each detector and each pulse into a common hidden
dimension.

Detector nodes have three input branches:

.. code-block:: text

   detector_features        -> detector feature MLP
   detector_context_features -> detector context MLP
   detector_waveforms       -> waveform encoder
                         concat -> detector_node_encoder

Pulse nodes have one scalar branch:

.. code-block:: text

   pulse_features -> pulse_node_encoder

The waveform encoder is applied once per detector. For the first transformer
waveform sweep, the submitter sets ``WAVEFORM_ENCODER=transformer``. The graph
attention architecture remains ``hetero_attention``.
If ``detector_waveform_valid`` is zero, the waveform embedding is multiplied by
zero. The detector can still contribute through live-status, geometry, detector
scalar features, and detector-detector edges, but not through a fake waveform
signal.

Where attention appears
-----------------------

The word ``attention`` appears in more than one place in this repository, so it
is important to separate the meanings.

.. list-table::
   :header-rows: 1

   * - Component
     - Code path
     - What attends to what
   * - Waveform transformer
     - ``WaveformEncoder(mode="transformer")`` in ``model.py``
     - time bins inside one detector waveform
   * - Graph relation attention
     - ``HeteroAttentionMessageLayer`` in ``hetero_model.py``
     - source nodes connected to a destination node by one typed edge relation
   * - Event readout attention
     - ``HeteroAttentiveReadout`` in ``hetero_model.py``
     - detector nodes or pulse nodes inside one event graph

Only the first row is a PyTorch ``TransformerEncoder`` block. The graph
relation attention is not ``torch.nn.MultiheadAttention`` and is not a full
Transformer block over all event objects. It uses the same query/key/value
scaled dot-product idea, but the attention candidates are restricted by the
physics graph:

.. code-block:: text

   Transformer-style sequence attention:
     every token can usually attend to every other token

   TALE heterogeneous graph attention:
     a node attends only over incoming edge_index entries
     relation type chooses a separate Q/K/V projection
     edge_attr is part of key/value

For a destination node ``j`` and an incoming source node ``i``:

.. code-block:: text

   src_input_ij = concat(hidden_i, edge_attr_ij)
   q_j = W_Q[relation](hidden_j)
   k_ij = W_K[relation](src_input_ij)
   v_ij = W_V[relation](src_input_ij)

   score_ij = dot(q_j, k_ij) / sqrt(head_dim)
   alpha_ij = softmax(score_ij over incoming edges to j)
   message_j = sum_i alpha_ij * v_ij

This is why the current model is closer to relation-specific graph attention
than to a plain sequence Transformer. Making the graph fully connected would
only change the candidate edges; it would not by itself make the model a
standard Transformer, because the model still keeps detector/pulse node types,
relation-specific projections, edge attributes, detector-level waveforms, and
type-wise readout.

Relation attention
------------------

The core message-passing layer is ``HeteroAttentionMessageLayer``. For each
relation type, it builds separate query, key, and value projections:

.. code-block:: python

   src_input = torch.cat([src_state[src_index], edge_attr], dim=-1)
   query = self.query[relation](dst_state[dst_index]).view(-1, self.heads, self.head_dim)
   key = self.key[relation](src_input).view(-1, self.heads, self.head_dim)
   value = self.value[relation](src_input).view(-1, self.heads, self.head_dim)
   scores = (query * key).sum(dim=-1) * scale
   weights = _scatter_softmax(scores, dst_index, dst_state.shape[0])
   messages = (value * weights[:, :, None]).reshape(-1, self.hidden_dim)

The important TALE-specific point is that ``key`` and ``value`` include
``edge_attr``. This lets the model decide which neighboring pulse or detector
matters while seeing physical quantities such as ``dt_usec``, ``distance_km``,
``dt_per_km``, and ``ising_weight``.

After messages are accumulated, detector and pulse states are updated with a
residual connection, layer normalization, and a feed-forward block:

.. code-block:: text

   new_state = LayerNorm(old_state + update(old_state, aggregated_message))
   new_state = LayerNorm(new_state + FFN(new_state))

This is inspired by HGT, but it is not an exact HGT implementation. The current
model does not use PyG ``HGTConv`` and does not use HGSampling. Each TALE event
is one graph and is read as a whole.

Direction output normalization
------------------------------

The reconstruction head emits six raw target values:

.. code-block:: text

   log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z

Before physics loss, NLL loss, quality-target construction, or predicted-error
target construction, the scaled output is converted back to physical target
units and the three direction components are normalized:

.. code-block:: text

   u = (dir_x, dir_y, dir_z)
   n_hat = u / (||u|| + eps)

The direction loss is computed from the opening angle between ``n_hat`` and the
true unit direction. This removes the unused direction-vector norm from the
reconstruction objective while keeping the three-component output format.

Relation to GAT, HGT, Transformer, and Laplacian GCNs
-----------------------------------------------------

The current implementation should be named precisely:

.. list-table::
   :header-rows: 1

   * - Reference family
     - Relation to the current model
   * - GAT
     - similar because edge messages are weighted by learned attention
   * - Transformer
     - similar because the score uses query/key/value scaled dot-product attention
   * - HGT
     - similar because node and edge types use relation-specific projections
   * - PyG ``HGTConv`` / ``HeteroConv``
     - not used directly; the repository implements its own layer
   * - Laplacian / spectral GCN
     - not used; there is no explicit ``L = D - A`` or normalized Laplacian propagation

The graph is represented by sparse ``edge_index`` tensors, not by a dense
adjacency matrix. Message aggregation is implemented with ``_scatter_softmax``
and ``index_add_``. Some graph construction code contains degree-based
normalization for Ising edge features, but that is an input feature
normalization, not Laplacian graph convolution.

Readout
-------

Training and inference need one prediction per event, not one prediction per
node. ``HeteroAttentiveReadout`` therefore pools detector nodes and pulse nodes
separately:

.. code-block:: text

   detector states -> mean, max, attention-weighted sums
   pulse states    -> mean, max, attention-weighted sums
   concat          -> event vector

This makes the readout type-aware. A detector and a pulse are not forced into
one shared pool before the model has summarized them.

Output heads
------------

The event vector is passed to task-specific heads:

.. list-table::
   :header-rows: 1

   * - Head
     - Output
     - Meaning
   * - Reconstruction
     - 6 values
     - ``log10_energy_eV``, ``core_x_km``, ``core_y_km``, ``dir_x``, ``dir_y``, ``dir_z``
   * - Mass
     - 1 logit
     - ``sigmoid(logit)`` gives iron probability
   * - Quality
     - 1 logit
     - auxiliary quality score
   * - Predicted error
     - 3 values
     - predicted energy, angular, and core uncertainties

The current comparison plan is not to enable every auxiliary head at once.
For the first transformer waveform sweep, run six jobs: three dataset sizes
times quality-only or predicted-error-only reco+mass.

Training
--------

``train_hetero_model`` follows the standard PyTorch training pattern:

.. code-block:: text

   H5HeteroGraphDataset
     -> split train / validation / test
     -> fit scalers on train split
     -> build model from first sample
     -> PyG DataLoader batches HeteroData samples
     -> model converts batched HeteroData to explicit tensor dict
     -> forward
     -> compute loss against MC target
     -> backward
     -> optimizer step
     -> save checkpoint and scalers

The checkpoint stores both model weights and the scalers. That is why direct
DST reconstruction can use the same feature normalization as training.

Direct DST reconstruction
-------------------------

Direct inference does not require HDF5. It uses the same graph schema and the
same model conversion:

.. code-block:: text

   DST
     -> dstio.tale.graph.iter_graphs
     -> graph_event_to_sample
     -> hetero_sample_to_tensors or sample_to_hetero_data
     -> hetero_attention checkpoint
     -> reconstruction CSV

This is the final reconstruction path. HDF5 remains a training cache, not a
required intermediate file for production DST reconstruction.

Server command for the first transformer sweep
----------------------------------------------

After the balanced HDF5 files are ready, submit the first six transformer jobs
with:

.. code-block:: bash

   cd /dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction

   RUN_ID=hetero_balance_20260606_143020 \
   SUBMIT_EXPORTS=0 \
   SUBMIT_TRAINING=1 \
   MODEL_ARCHITECTURE=hetero_attention \
   WAVEFORM_ENCODER=transformer \
   PARTITION=v100-al9_long \
   scripts/submit_server_hetero_dataset_size_sweep.sh

This submits quality-only and predicted-error-only reco+mass training for the
50000, 20000, and 10000 events/bin datasets. The ``cnn-gru`` waveform encoder
should be compared later under the selected condition, not run as a simultaneous
six-job sweep at this stage.
