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
For the attention notation, this page follows the Query/Key/Value language used
by PyTorch's
`MultiheadAttention <https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html>`_.
For heterogeneous graph notation, it follows PyG's
`HeteroData <https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.HeteroData.html>`_
and compares the repository code with PyG
`HGTConv <https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.HGTConv.html>`_.

The code paths described here are the actual code paths used by the repository:

.. list-table::
   :header-rows: 1

   * - Step
     - Implementation
   * - HDF5 sample write / read
     - ``dstio.tale.graph.write_balanced_graph_h5`` or ``write_graph_h5`` for writing; ``talesd_gnn_reconstruction.hetero_graph_io`` for reading.
   * - Tensor / PyG conversion
     - ``talesd_gnn_reconstruction.hetero_data``
   * - Model
     - ``talesd_gnn_reconstruction.hetero_model.MinimalHeteroTaleSdGNN``
   * - Training
     - ``talesd_gnn_reconstruction.hetero_training.train_hetero_model``
   * - Direct DST inference
     - ``talesd_gnn_reconstruction.hetero_predict.reconstruct_dst``

In one sentence, the current model is an
``edge-attribute-conditioned, relation-specific heterogeneous graph attention``
model for TALE-SD events.  It represents one event with detector nodes and
pulse nodes, updates those nodes with relation-specific Q/K/V message passing,
and reads one event-level vector for energy, core, direction, mass, quality,
and predicted-error heads.  It is inspired by HGT, but it is not PyG
``HGTConv`` or ``HeteroConv``.  It also does not use HGSampling: one TALE event
is small enough to keep as one complete event graph, so the model can keep
waveform tails, delayed pulses, multi-pulse structure, and Ising-rejected pulse
candidates in the input.

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

Within schema v3, ``pulse_arrival_usec_rel`` is built from the pulse onset, the
earlier of the upper/lower rise times, not from the averaged coincidence time.
Its zero point is the first accepted graph pulse candidate in the event. The
zero point is not shifted to the first Ising-kept pulse, because Ising-rejected
candidate pulses remain part of the ML input.

``detector_trigger_usec_rel`` is a compatibility name. The value is used as the
detector node's representative arrival time: with ``cleaning="ising"``, it is
the first ``ising_keep = 1`` pulse onset attached to that detector, expressed on
the same ``pulse_arrival_usec_rel`` axis. It is not used as the detector
waveform start time. Detectors without an Ising-kept pulse have
``detector_arrival_time_valid = 0`` and a zero value in this column, so the
validity flag must be checked when interpreting detector timing.

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
For one detector waveform, each time bin is a token.  For time bin ``t`` and
another time bin ``u`` in the same detector waveform, the Transformer forms

.. code-block:: text

   q_t = W_Q x_t
   k_u = W_K x_u
   v_u = W_V x_u

   score_tu = dot(q_t, k_u) / sqrt(d_head)
   alpha_tu = softmax(score_tu over waveform time bins u)
   y_t = sum_u alpha_tu * v_u

Physically, this lets one waveform time bin learn which other time bins in the
same detector waveform matter.  It can therefore represent rise shape, peak,
tail, double peaks, delayed components, and multi-pulse waveform structure more
flexibly than a fixed local convolution alone.  The waveform sequence is then
pooled:

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

For the graph relation attention, the physical interpretation is:

.. list-table::
   :header-rows: 1

   * - Name
     - Intuition
     - In this model
   * - Query, ``Q``
     - "What information do I need now?"
     - made from the destination node state
   * - Key, ``K``
     - "What kind of source information is this?"
     - made from source node state plus edge attributes
   * - Value, ``V``
     - "What information should be passed?"
     - made from source node state plus edge attributes
   * - score
     - compatibility between ``Q`` and ``K``
     - softmax logit for one edge and one head
   * - attention weight
     - normalized edge weight
     - relative weight among edges entering the same destination for the same relation
   * - message
     - transported information
     - ``attention weight * V``

For a destination node ``j`` and an incoming source node ``i`` in relation
``r``:

.. code-block:: text

   z_ij = concat(hidden_i, edge_attr_ij)

   q_ij = W_Q[r](hidden_j)
   k_ij = W_K[r](z_ij)
   v_ij = W_V[r](z_ij)

   score_ij = dot(q_j, k_ij) / sqrt(head_dim)
   alpha_ij = softmax(score_ij over incoming edges to j in relation r)
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

Why the query comes from the destination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The destination node is the node being updated.  It supplies the query because
the useful source information depends on the current destination state.  The
same source pulse can be compatible with one destination pulse and not useful
for another.  In physical language, the destination pulse says "I need
information that helps decide whether I lie on a consistent shower front"; a
source pulse plus its edge says "I have this time, charge, distance, and
time-gradient relation."  The Q/K dot product tests whether those two statements
match.

Why key and value include edge attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a sequence Transformer, key and value usually come only from the source
token.  In this model they come from ``source state + edge_attr``.  That is
important for TALE-SD because the same source pulse has different meaning for
different destinations.  A close source with a shower-front-consistent
``dt_per_km`` should not be treated the same as a close source with a delayed or
inconsistent time.  Putting edge attributes in the key lets the model decide
whether this edge is a good information source; putting edge attributes in the
value lets the transported message carry the relative time, distance, and Ising
context.

Multi-head shape
~~~~~~~~~~~~~~~~

If the hidden dimension is ``H`` and the number of heads is ``A``, one head has
``d_head = H / A`` dimensions.  For relation ``r`` with ``E_r`` edges:

.. code-block:: text

   src_input = concat(src_state[src_index], edge_attr) -> [E_r, H + F_edge_r]
   query -> [E_r, A, d_head]
   key   -> [E_r, A, d_head]
   value -> [E_r, A, d_head]
   scores  = sum(query * key, dim=-1) / sqrt(d_head) -> [E_r, A]
   weights = scatter_softmax(scores, dst_index)      -> [E_r, A]
   messages = value * weights[:, :, None]            -> [E_r, A, d_head]

The softmax is not over the whole event.  It is taken separately for each
relation and destination node, using ``dst_index``.  A relation with only one
incoming edge to a destination will have weight one for that edge in that
relation; the model still combines the relation output with other relation
messages and the old node state.

Small numerical example
~~~~~~~~~~~~~~~~~~~~~~~

Consider one destination pulse ``P0`` with two incoming source pulses.  Use one
head and two-dimensional vectors only for illustration:

.. code-block:: text

   query(P0) = q = (1, 0)

   key(P1 -> P0) = k1 = (2.0, 0)
   key(P2 -> P0) = k2 = (0.5, 0)

   score1 = dot(q, k1) = 2.0
   score2 = dot(q, k2) = 0.5

   softmax(2.0, 0.5) ~= (0.82, 0.18)

   value(P1 -> P0) = v1 = (10, 1)
   value(P2 -> P0) = v2 = (2, 8)

   message(P0) = 0.82 * v1 + 0.18 * v2
               = (8.56, 2.26)

The keys decide which edge is selected more strongly; the values decide what is
actually transported.  Keeping key and value separate lets the model learn that
the criteria for trusting an edge can differ from the content that should pass
through that edge.

Relation-specific information flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pulse__same_detector_next__pulse`` / ``pulse__same_detector_prev__pulse``
   Pass information between consecutive pulses in the same detector.  This is
   where multi-pulse timing order can enter message passing explicitly.

``pulse__near_space__pulse``
   Connects pulses on nearby detectors without a time cut.  This carries local
   spatial footprint information and keeps delayed or complex pulse structure
   available instead of hard-dropping it.

``pulse__time_causal__pulse``
   Uses the stricter near-space, time-compatible, Ising-weighted subset.  It is
   the relation most directly connected to shower-front timing and direction.

``detector__near__detector``
   Passes local detector geometry, signal/no-signal context, and neighborhood
   density information between nearby detector nodes.

``detector__observes__pulse``
   Sends detector scalar/context/waveform information to the pulse observed by
   that detector.  This is how a pulse receives the detector waveform embedding
   without duplicating the waveform on every pulse node.

``pulse__observed_by__detector``
   Sends pulse information back to the detector.  This lets the detector state
   summarize its pulse candidates, including Ising-rejected candidates, before
   detector-detector propagation and event readout.

After messages are accumulated, detector and pulse states are updated with a
residual connection, layer normalization, and a feed-forward block:

.. code-block:: text

   new_state = LayerNorm(old_state + update(old_state, aggregated_message))
   new_state = LayerNorm(new_state + FFN(new_state))

This is inspired by HGT, but it is not an exact HGT implementation. The current
model does not use PyG ``HGTConv`` and does not use HGSampling. Each TALE event
is one graph and is read as a whole.

Attention weights are useful diagnostics, but they are not proof of physical
causality by themselves.  A high attention weight means that this trained model,
in this layer/relation/head, transported value strongly through that edge.  A
physical interpretation still needs event displays, ablations, and comparison
with conventional reconstruction.

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
separately.  After the final message-passing layer, the model has

.. code-block:: text

   detector states H_d -> [N_d, H]
   pulse states    H_p -> [N_p, H]

For one event ``b``, let ``D_b`` be its detector nodes and ``P_b`` its pulse
nodes.  The readout computes three summaries for each node type.

Mean pooling:

.. code-block:: text

   mean_detector_b = mean_{i in D_b} H_d[i]
   mean_pulse_b    = mean_{i in P_b} H_p[i]

Max pooling:

.. code-block:: text

   max_detector_b = max_{i in D_b} H_d[i]
   max_pulse_b    = max_{i in P_b} H_p[i]

Attention readout:

.. code-block:: text

   score_i = W_read H[i] + b_read
   beta_i  = softmax(score_i over nodes in the same event and node type)
   attn_b  = sum_i beta_i * H[i]

The implementation has ``readout_heads`` independent readout-attention heads.
With hidden dimension ``H`` and ``R`` readout heads, one node type produces
``H * (2 + R)`` features: one mean vector, one max vector, and ``R`` attention
weighted sums.  Detector and pulse summaries are then concatenated:

.. code-block:: text

   detector_readout -> [B, H * (2 + R)]
   pulse_readout    -> [B, H * (2 + R)]
   event vector     -> [B, 2 * H * (2 + R)]

This makes the readout type-aware. A detector and a pulse are not forced into
one shared pool before the model has summarized them.  Detector readout can
therefore focus on shower footprint, detector geometry, signal/no-signal
context, and waveform summaries, while pulse readout can focus on timing,
charge, Ising annotations, and delayed or multi-pulse structure.

During a forward pass, hidden states and attention weights change from event to
event.  They are temporary values.  During training, the learnable parameters
that change are the detector/context/pulse MLP weights, waveform encoder
weights, relation-specific ``W_Q``, ``W_K``, ``W_V`` and output projections,
LayerNorm parameters, readout attention weights, and output-head weights.  The
feature scalers are fit on the training split and then stored with the
checkpoint; they are not neural-network weights updated by backpropagation.

Output heads
------------

The event vector is passed to task-specific heads.  Each head is a small MLP
applied to the same event vector; the enabled heads are concatenated in this
order: reconstruction, mass, quality, predicted error.

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

Reconstruction head
~~~~~~~~~~~~~~~~~~~

The reconstruction head emits six scaled values:

.. code-block:: text

   y_reco_hat = Head_reco(event_vector)
              = (log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z)

These six values are inverse-transformed with the target scaler before physical
losses and metrics are computed.  The direction part is then normalized:

.. code-block:: text

   u_hat = (dir_x, dir_y, dir_z)
   n_hat = u_hat / (||u_hat|| + eps)

This keeps the output as three components, but removes the unphysical freedom
for the model to change the direction-vector norm.  The final evaluation uses
opening-angle metrics, not the raw vector norm.

Mass head
~~~~~~~~~

The mass head emits one logit ``z_mass``.  This is not already a probability.
The iron probability used for diagnostics is

.. code-block:: text

   P(iron) = sigmoid(z_mass) = 1 / (1 + exp(-z_mass))

Thus ``z_mass = 0`` means 50% iron probability, ``z_mass = 2`` means about 88%,
and ``z_mass = -2`` means about 12%.

Quality head
~~~~~~~~~~~~

The quality head emits one logit.  The target is derived from the current
reconstruction error during training.  The code computes energy, core, and
angular error scores in physical units, averages them, and uses
``exp(-score)`` as a target in ``binary_cross_entropy_with_logits``.  In words,
the head learns an event-wise score for whether the reconstruction is expected
to be reliable.

Predicted-error head
~~~~~~~~~~~~~~~~~~~~

The predicted-error head emits three raw values.  They are converted with
``softplus`` and physical scales:

.. code-block:: text

   predicted_errors = softplus(error_raw) * (energy_scale,
                                             angular_scale_deg,
                                             core_scale_km)

The training target is the event's current reconstruction error:
relative energy error, opening-angle error, and xy core displacement, each
scaled by the corresponding configured scale.  The loss compares
``log1p(predicted_error)`` and ``log1p(target_error)`` with SmoothL1.

The current comparison plan is not to enable every auxiliary head at once.
For the first transformer waveform sweep, run six jobs: three dataset sizes
times quality-only or predicted-error-only reco+mass.

Whole forward pass as equations
-------------------------------

For one batch, the model can be summarized as the following sequence.

Input tensors:

.. code-block:: text

   X_d = detector.x       -> [N_d, F_d]
   C_d = detector.context -> [N_d, F_c]
   W_d = detector.waveform -> [N_d, C_waveform, T]
   X_p = pulse.x          -> [N_p, F_p]
   E_r = {(src, dst, edge_attr)} for each relation r

Scaling:

.. code-block:: text

   X_d, C_d, X_p, edge_attr, target
     -> standardized with train-split scalers

Initial encoders:

.. code-block:: text

   a_d = MLP_detector(X_d)
   c_d = MLP_context(C_d)
   w_d = WaveformEncoder(W_d)

   h_d^0 = Linear(concat(a_d, c_d, w_d))
   h_p^0 = MLP_pulse(X_p)

Message passing layer ``l`` for relation ``r``:

.. code-block:: text

   z_ij^(r,l) = concat(h_src_i^l, edge_attr_ij^r)

   q_ij^(r,l) = W_Q^r h_dst_j^l + b_Q^r
   k_ij^(r,l) = W_K^r z_ij^(r,l) + b_K^r
   v_ij^(r,l) = W_V^r z_ij^(r,l) + b_V^r

   score_ij = dot(q_ij, k_ij) / sqrt(d_head)
   alpha_ij = softmax(score_ij over incoming edges to j in relation r)
   m_j^r = sum_i alpha_ij * v_ij

Node update:

.. code-block:: text

   h_j <- LayerNorm(h_j + Update(h_j, m_j))
   h_j <- LayerNorm(h_j + FFN(h_j))

Readout and heads:

.. code-block:: text

   event_vector = Readout({h_d^L}, {h_p^L})

   y_reco_hat = Head_reco(event_vector)
   z_mass     = Head_mass(event_vector)
   z_quality  = Head_quality(event_vector)
   error_raw  = Head_error(event_vector)

What changes during training
----------------------------

It is useful to separate temporary forward-pass values from learned parameters.

Temporary values:

- detector and pulse hidden states at each layer;
- graph-relation attention weights for each event, layer, relation, head, and edge;
- readout attention weights for detector nodes and pulse nodes;
- output predictions for the current batch.

These values are recomputed for every event.  The same trained checkpoint can
give different attention maps for different events.

Learned parameters:

- detector feature MLP weights;
- detector context MLP weights;
- pulse MLP weights;
- waveform Conv1d and optional Transformer weights;
- relation-specific ``W_Q``, ``W_K``, ``W_V`` and relation output projections;
- node update blocks, FFNs, and LayerNorm scale/bias;
- readout attention weights;
- reconstruction, mass, quality, and predicted-error head weights.

The training loop computes

.. code-block:: text

   L = L_reco
       + lambda_mass    * L_mass
       + lambda_quality * L_quality
       + lambda_error   * L_error
       + optional bias penalties

and updates learnable parameters by backpropagation.  Scalers are different:
they are fit once on the training split and stored in the checkpoint, but they
are not neural-network weights.

Physical interpretation by output
---------------------------------

Energy
~~~~~~

Energy is mainly constrained by shower size: detector signal sums, pulse charge,
waveform integral/shape, hit or pulse multiplicity, and the relation between
signal size and core position.  In this model, those enter through
``detector.x``, ``detector.waveform``, ``pulse.x``, pulse-pulse relations,
detector-detector relations, and the final readout.  The waveform encoder gives
each detector a compact representation of the full calibrated waveform, while
pulse nodes keep local pulse candidates and Ising annotations.

Core position
~~~~~~~~~~~~~

Core position is constrained by the spatial signal footprint, local detector
geometry, arrival-time pattern, and signal/no-signal context.  The most direct
relations are ``detector__near__detector``, ``pulse__near_space__pulse``, and
``pulse__time_causal__pulse``.  Keeping both near-space and time-causal pulse
relations lets the model distinguish a geometrically nearby but delayed pulse
from a pulse that is both nearby and shower-front compatible.

Direction
~~~~~~~~~

Arrival direction is tied most directly to pulse arrival time and detector
position.  Pulse-pulse edge attributes include ``dt_usec``, ``distance_km`` and
``dt_per_km``.  Since edge attributes enter both key and value, the attention
calculation can use time-gradient information when deciding which pulse pairs
to trust and what timing message to send.

Mass
~~~~

Mass can depend on waveform tails, delayed components, multi-pulse structure,
and muon-richness-related information.  The model therefore does not hard-drop
Ising-rejected pulse candidates from the ML graph.  Rejected candidates remain
as pulse nodes with Ising annotations, and detector waveforms remain stored once
per detector.  This gives the mass head a path to use delayed pulses and waveform
tail structure instead of losing them at graph construction time.

Quality and predicted error
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quality and predicted-error heads are event-wise consistency outputs.  Events
with inconsistent pulse timing, geometry/signal mismatch, many rejected
candidates, complex waveforms, or diffuse attention can learn lower quality or
larger predicted error.  These heads are auxiliary outputs; the next comparison
tests quality-only and predicted-error-only separately under the same reco+mass
task.

Short explanation
-----------------

For papers or talks, the concise description is:

   The model represents each TALE-SD event as a heterogeneous graph with detector
   nodes and pulse nodes. Detector waveforms are stored once on detector nodes,
   while pulse nodes refer to their detector and waveform window. The graph uses
   separate physical relations for same-detector pulse order, nearby pulse
   pairs, time-causal pulse pairs, detector proximity, and detector-pulse
   observation. Each relation performs multi-head attention with query from the
   destination node and key/value from the source node plus edge attributes.
   Detector and pulse states are read out separately and then combined to predict
   energy, core position, arrival direction, mass, quality, and predicted error.

The shortest accurate name is:

   An edge-attribute-conditioned heterogeneous graph attention model for
   TALE-SD detector, waveform, and pulse reconstruction.

Key points
----------

- The model is not a plain Transformer over all detector and pulse objects.
- It is not PyG ``HGTConv`` or ``HeteroConv`` directly.
- It does not use HGSampling; one TALE event is kept as one full graph.
- Waveform Transformer attention looks within each detector waveform over time bins.
- Graph relation attention looks only along ``dstio``-defined graph edges.
- In graph relation attention, query comes from the destination node, while key
  and value come from source node plus edge attributes.
- Readout attention decides which detector or pulse nodes matter for the event
  summary.
- Attention maps are diagnostics, not physical proof by themselves; ablation,
  event displays, and conventional-reconstruction comparisons are still needed.

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

After the balanced HDF5 files are ready and their split/input distributions
have been checked, submit the first six transformer jobs with:

.. code-block:: bash

   cd /dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction

   RUN_ID=<same RUN_ID used for export> \
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
