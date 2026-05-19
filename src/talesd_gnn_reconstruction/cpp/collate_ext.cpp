#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct ScalerView {
    bool enabled = false;
    py::array_t<float, py::array::c_style | py::array::forcecast> mean;
    py::array_t<float, py::array::c_style | py::array::forcecast> std;
    const float* mean_ptr = nullptr;
    const float* std_ptr = nullptr;
    py::ssize_t size = 0;
};

struct GraphView {
    py::array node_owner;
    py::array edge_owner;
    py::array edge_index_owner;
    py::array pulse_owner;
    py::array waveform_owner;
    py::array detector_lids_owner;
    py::array target_owner;

    const float* node_ptr = nullptr;
    const float* edge_ptr = nullptr;
    const std::int64_t* edge_index_ptr = nullptr;
    const float* pulse_ptr = nullptr;
    const float* waveform_ptr = nullptr;
    const std::int64_t* detector_lids_ptr = nullptr;
    const float* target_ptr = nullptr;

    py::ssize_t n_nodes = 0;
    py::ssize_t n_edges = 0;
    py::ssize_t n_pulses = 0;
    py::ssize_t waveform_channels = 0;
    py::ssize_t waveform_bins = 0;
    py::ssize_t n_detector_lids = 0;
    py::ssize_t pulse_cols = 0;
    py::ssize_t target_dim = 0;
    py::ssize_t valid_pulses = 0;

    py::ssize_t node_offset = 0;
    py::ssize_t edge_offset = 0;
    py::ssize_t pulse_offset = 0;
    py::ssize_t target_offset = 0;
    bool has_target = false;
    bool has_waveform = false;
    bool has_detector_lids = false;
    bool has_particle_label = false;
    float particle_label = std::numeric_limits<float>::quiet_NaN();
};

ScalerView make_scaler(py::object mean_obj, py::object std_obj, const char* name) {
    ScalerView scaler;
    if (mean_obj.is_none() || std_obj.is_none()) {
        return scaler;
    }
    scaler.mean = py::array_t<float, py::array::c_style | py::array::forcecast>(mean_obj);
    scaler.std = py::array_t<float, py::array::c_style | py::array::forcecast>(std_obj);
    if (scaler.mean.ndim() != 1 || scaler.std.ndim() != 1 || scaler.mean.shape(0) != scaler.std.shape(0)) {
        throw std::runtime_error(std::string("invalid scaler shape: ") + name);
    }
    scaler.enabled = true;
    scaler.size = scaler.mean.shape(0);
    scaler.mean_ptr = static_cast<const float*>(scaler.mean.data());
    scaler.std_ptr = static_cast<const float*>(scaler.std.data());
    return scaler;
}

void require_2d(const py::buffer_info& info, const char* name) {
    if (info.ndim != 2) {
        throw std::runtime_error(std::string(name) + " must be 2D");
    }
}

void require_scaler_dim(const ScalerView& scaler, py::ssize_t dim, const char* name) {
    if (scaler.enabled && scaler.size != dim) {
        throw std::runtime_error(std::string("scaler dimension mismatch: ") + name);
    }
}

float scale_value(float value, const ScalerView& scaler, py::ssize_t column) {
    if (!scaler.enabled) {
        return value;
    }
    return (value - scaler.mean_ptr[column]) / scaler.std_ptr[column];
}

int parse_positive_int(const char* text) {
    if (text == nullptr || text[0] == '\0') {
        return 0;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || value <= 0) {
        return 0;
    }
    return static_cast<int>(std::min<long>(value, 1024));
}

int resolve_thread_count(int requested_threads, py::ssize_t n_graphs, py::ssize_t total_work) {
    if (requested_threads > 0) {
        return std::max(1, std::min<int>(requested_threads, static_cast<int>(n_graphs)));
    }
    const int env_threads = parse_positive_int(std::getenv("TALESD_GNN_COLLATE_THREADS"));
    if (env_threads > 0) {
        return std::max(1, std::min<int>(env_threads, static_cast<int>(n_graphs)));
    }
    if (n_graphs < 128 || total_work < 200000) {
        return 1;
    }
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int auto_threads = hardware_threads == 0 ? 4 : static_cast<int>(hardware_threads);
    return std::max(1, std::min<int>({auto_threads, 8, static_cast<int>(n_graphs)}));
}

template <typename Func>
void parallel_for_graphs(py::ssize_t n_graphs, int num_threads, Func&& fn) {
    if (num_threads <= 1 || n_graphs <= 1) {
        for (py::ssize_t graph_index = 0; graph_index < n_graphs; ++graph_index) {
            fn(graph_index);
        }
        return;
    }

    std::atomic<py::ssize_t> next_graph{0};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
        workers.emplace_back([&]() {
            while (true) {
                const py::ssize_t graph_index = next_graph.fetch_add(1, std::memory_order_relaxed);
                if (graph_index >= n_graphs) {
                    break;
                }
                fn(graph_index);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

}  // namespace

py::dict collate_numeric(
    py::list samples,
    py::object node_mean,
    py::object node_std,
    py::object edge_mean,
    py::object edge_std,
    py::object pulse_mean,
    py::object pulse_std,
    py::object target_mean,
    py::object target_std,
    bool require_target,
    int num_threads
) {
    const py::ssize_t n_graphs = py::len(samples);
    if (n_graphs == 0) {
        throw std::runtime_error("empty batch");
    }

    ScalerView node_scaler = make_scaler(node_mean, node_std, "node");
    ScalerView edge_scaler = make_scaler(edge_mean, edge_std, "edge");
    ScalerView pulse_scaler = make_scaler(pulse_mean, pulse_std, "pulse");
    ScalerView target_scaler = make_scaler(target_mean, target_std, "target");

    std::vector<GraphView> graphs;
    graphs.reserve(static_cast<std::size_t>(n_graphs));

    py::ssize_t total_nodes = 0;
    py::ssize_t total_edges = 0;
    py::ssize_t total_pulses = 0;
    py::ssize_t target_count = 0;
    bool has_any_particle_label = false;
    bool has_any_detector_lids = false;
    py::ssize_t node_dim = -1;
    py::ssize_t edge_dim = -1;
    py::ssize_t pulse_dim = -1;
    py::ssize_t waveform_channels = -1;
    py::ssize_t waveform_bins = -1;
    py::ssize_t target_dim = -1;

    for (py::ssize_t graph_index = 0; graph_index < n_graphs; ++graph_index) {
        py::dict sample = samples[graph_index].cast<py::dict>();

        auto node = py::array_t<float, py::array::c_style | py::array::forcecast>(sample["node_features"]);
        auto edge = py::array_t<float, py::array::c_style | py::array::forcecast>(sample["edge_features"]);
        auto edge_index =
            py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(sample["edge_index"]);
        auto pulses = py::array_t<float, py::array::c_style | py::array::forcecast>(sample["pulse_features"]);
        auto waveform = py::array_t<float, py::array::c_style | py::array::forcecast>(sample["waveform_features"]);

        auto node_info = node.request();
        auto edge_info = edge.request();
        auto edge_index_info = edge_index.request();
        auto pulse_info = pulses.request();
        auto waveform_info = waveform.request();
        require_2d(node_info, "node_features");
        require_2d(edge_info, "edge_features");
        require_2d(edge_index_info, "edge_index");
        require_2d(pulse_info, "pulse_features");
        if (waveform_info.ndim != 3) {
            throw std::runtime_error("waveform_features must be 3D");
        }
        if (waveform_info.shape[0] != node_info.shape[0]) {
            throw std::runtime_error("waveform_features first dimension must match node_features");
        }
        if (edge_index_info.shape[0] != 2 || edge_index_info.shape[1] != edge_info.shape[0]) {
            throw std::runtime_error("edge_index shape is inconsistent with edge_features");
        }
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> detector_lids;
        py::buffer_info detector_lids_info;
        bool graph_has_detector_lids = false;
        if (sample.contains("detector_lids")) {
            detector_lids =
                py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(sample["detector_lids"]);
            detector_lids_info = detector_lids.request();
            if (detector_lids_info.ndim != 1) {
                throw std::runtime_error("detector_lids must be 1D");
            }
            graph_has_detector_lids = detector_lids_info.shape[0] == node_info.shape[0];
            has_any_detector_lids = has_any_detector_lids || graph_has_detector_lids;
        }

        const py::ssize_t this_pulse_dim = pulse_info.shape[1] > 1 ? pulse_info.shape[1] - 1 : 0;
        const py::ssize_t this_waveform_channels = waveform_info.shape[1];
        const py::ssize_t this_waveform_bins = waveform_info.shape[2];
        if (node_dim < 0) {
            node_dim = node_info.shape[1];
            edge_dim = edge_info.shape[1];
            pulse_dim = this_pulse_dim;
            waveform_channels = this_waveform_channels;
            waveform_bins = this_waveform_bins;
        } else if (node_info.shape[1] != node_dim || edge_info.shape[1] != edge_dim || this_pulse_dim != pulse_dim) {
            throw std::runtime_error("inconsistent graph feature dimensions");
        } else if (this_waveform_channels != waveform_channels || this_waveform_bins != waveform_bins) {
            throw std::runtime_error("inconsistent waveform feature dimensions");
        }
        GraphView view;
        view.node_owner = node;
        view.edge_owner = edge;
        view.edge_index_owner = edge_index;
        view.pulse_owner = pulses;
        view.waveform_owner = waveform;
        view.detector_lids_owner = detector_lids;
        view.node_ptr = static_cast<const float*>(node_info.ptr);
        view.edge_ptr = static_cast<const float*>(edge_info.ptr);
        view.edge_index_ptr = static_cast<const std::int64_t*>(edge_index_info.ptr);
        view.pulse_ptr = static_cast<const float*>(pulse_info.ptr);
        view.waveform_ptr = static_cast<const float*>(waveform_info.ptr);
        view.detector_lids_ptr =
            graph_has_detector_lids ? static_cast<const std::int64_t*>(detector_lids_info.ptr) : nullptr;
        view.n_nodes = node_info.shape[0];
        view.n_edges = edge_info.shape[0];
        view.n_pulses = pulse_info.shape[0];
        view.waveform_channels = this_waveform_channels;
        view.waveform_bins = this_waveform_bins;
        view.n_detector_lids = graph_has_detector_lids ? detector_lids_info.shape[0] : 0;
        view.pulse_cols = pulse_info.shape[1];
        view.node_offset = total_nodes;
        view.edge_offset = total_edges;
        view.pulse_offset = total_pulses;
        view.has_waveform = this_waveform_channels > 0 && this_waveform_bins > 0;
        view.has_detector_lids = graph_has_detector_lids;

        if (pulse_dim > 0 && view.pulse_cols > 1) {
            for (py::ssize_t row = 0; row < view.n_pulses; ++row) {
                const auto local_node = static_cast<std::int64_t>(view.pulse_ptr[row * view.pulse_cols]);
                if (local_node >= 0 && local_node < view.n_nodes) {
                    view.valid_pulses += 1;
                }
            }
        }

        py::object target_obj = sample["target"];
        if (target_obj.is_none()) {
            if (require_target) {
                throw std::runtime_error("sample has no target");
            }
        } else {
            auto target = py::array_t<float, py::array::c_style | py::array::forcecast>(target_obj);
            auto target_info = target.request();
            if (target_info.ndim != 1) {
                throw std::runtime_error("target must be 1D");
            }
            if (target_dim < 0) {
                target_dim = target_info.shape[0];
            } else if (target_info.shape[0] != target_dim) {
                throw std::runtime_error("inconsistent target dimension");
            }
            view.target_owner = target;
            view.target_ptr = static_cast<const float*>(target_info.ptr);
            view.target_dim = target_info.shape[0];
            view.target_offset = target_count;
            view.has_target = true;
            target_count += 1;
        }
        if (sample.contains("particle_label")) {
            py::object label_obj = sample["particle_label"];
            if (!label_obj.is_none()) {
                view.particle_label = py::cast<float>(label_obj);
                view.has_particle_label = true;
                has_any_particle_label = true;
            }
        }

        total_nodes += view.n_nodes;
        total_edges += view.n_edges;
        total_pulses += view.valid_pulses;
        graphs.push_back(std::move(view));
    }

    if (node_dim < 0 || edge_dim < 0 || pulse_dim < 0 || waveform_channels < 0 || waveform_bins < 0) {
        throw std::runtime_error("invalid batch dimensions");
    }
    require_scaler_dim(node_scaler, node_dim, "node");
    require_scaler_dim(edge_scaler, edge_dim, "edge");
    require_scaler_dim(pulse_scaler, pulse_dim, "pulse");
    if (target_count > 0) {
        require_scaler_dim(target_scaler, target_dim, "target");
    }

    py::array_t<float> out_x(std::vector<py::ssize_t>{total_nodes, node_dim});
    py::array_t<std::int64_t> out_edge_index(std::vector<py::ssize_t>{2, total_edges});
    py::array_t<float> out_edge_attr(std::vector<py::ssize_t>{total_edges, edge_dim});
    py::array_t<float> out_pulse_x(std::vector<py::ssize_t>{total_pulses, pulse_dim});
    py::array_t<std::int64_t> out_pulse_node_index(total_pulses);
    py::array_t<float> out_waveform_x(std::vector<py::ssize_t>{total_nodes, waveform_channels, waveform_bins});
    py::array_t<std::int64_t> out_detector_lids;
    if (has_any_detector_lids) {
        out_detector_lids = py::array_t<std::int64_t>(total_nodes);
    }
    py::array_t<std::int64_t> out_batch(total_nodes);
    py::array_t<float> out_y;
    if (target_count > 0) {
        out_y = py::array_t<float>(std::vector<py::ssize_t>{target_count, target_dim});
    }
    py::array_t<float> out_mass_label;
    if (has_any_particle_label) {
        out_mass_label = py::array_t<float>(n_graphs);
    }

    float* x_ptr = static_cast<float*>(out_x.mutable_data());
    std::int64_t* edge_index_ptr = static_cast<std::int64_t*>(out_edge_index.mutable_data());
    float* edge_attr_ptr = static_cast<float*>(out_edge_attr.mutable_data());
    float* pulse_x_ptr = static_cast<float*>(out_pulse_x.mutable_data());
    std::int64_t* pulse_node_index_ptr = static_cast<std::int64_t*>(out_pulse_node_index.mutable_data());
    float* waveform_x_ptr = static_cast<float*>(out_waveform_x.mutable_data());
    std::int64_t* detector_lids_ptr =
        has_any_detector_lids ? static_cast<std::int64_t*>(out_detector_lids.mutable_data()) : nullptr;
    std::int64_t* batch_ptr = static_cast<std::int64_t*>(out_batch.mutable_data());
    float* y_ptr = target_count > 0 ? static_cast<float*>(out_y.mutable_data()) : nullptr;
    float* mass_label_ptr =
        has_any_particle_label ? static_cast<float*>(out_mass_label.mutable_data()) : nullptr;
    const int resolved_threads = resolve_thread_count(num_threads, n_graphs, total_nodes + total_edges + total_pulses);

    {
        py::gil_scoped_release release;
        parallel_for_graphs(n_graphs, resolved_threads, [&](py::ssize_t graph_index) {
            const GraphView& graph = graphs[static_cast<std::size_t>(graph_index)];

            for (py::ssize_t row = 0; row < graph.n_nodes; ++row) {
                batch_ptr[graph.node_offset + row] = static_cast<std::int64_t>(graph_index);
                if (has_any_detector_lids) {
                    detector_lids_ptr[graph.node_offset + row] =
                        graph.has_detector_lids ? graph.detector_lids_ptr[row] : static_cast<std::int64_t>(-1);
                }
                for (py::ssize_t col = 0; col < node_dim; ++col) {
                    const float value = graph.node_ptr[row * node_dim + col];
                    x_ptr[(graph.node_offset + row) * node_dim + col] = scale_value(value, node_scaler, col);
                }
                for (py::ssize_t channel = 0; channel < waveform_channels; ++channel) {
                    for (py::ssize_t bin = 0; bin < waveform_bins; ++bin) {
                        const auto src_index = (row * waveform_channels + channel) * waveform_bins + bin;
                        const auto dst_index =
                            ((graph.node_offset + row) * waveform_channels + channel) * waveform_bins + bin;
                        waveform_x_ptr[dst_index] = graph.has_waveform ? graph.waveform_ptr[src_index] : 0.0f;
                    }
                }
            }

            for (py::ssize_t row = 0; row < graph.n_edges; ++row) {
                const auto src_node = graph.edge_index_ptr[row] + graph.node_offset;
                const auto dst_node = graph.edge_index_ptr[graph.n_edges + row] + graph.node_offset;
                edge_index_ptr[graph.edge_offset + row] = src_node;
                edge_index_ptr[total_edges + graph.edge_offset + row] = dst_node;
                for (py::ssize_t col = 0; col < edge_dim; ++col) {
                    const float value = graph.edge_ptr[row * edge_dim + col];
                    edge_attr_ptr[(graph.edge_offset + row) * edge_dim + col] = scale_value(value, edge_scaler, col);
                }
            }

            if (pulse_dim > 0 && graph.pulse_cols > 1) {
                py::ssize_t out_row = graph.pulse_offset;
                for (py::ssize_t row = 0; row < graph.n_pulses; ++row) {
                    const auto local_node = static_cast<std::int64_t>(graph.pulse_ptr[row * graph.pulse_cols]);
                    if (local_node < 0 || local_node >= graph.n_nodes) {
                        continue;
                    }
                    pulse_node_index_ptr[out_row] = local_node + graph.node_offset;
                    for (py::ssize_t col = 0; col < pulse_dim; ++col) {
                        const float value = graph.pulse_ptr[row * graph.pulse_cols + col + 1];
                        pulse_x_ptr[out_row * pulse_dim + col] = scale_value(value, pulse_scaler, col);
                    }
                    out_row += 1;
                }
            }

            if (graph.has_target) {
                for (py::ssize_t col = 0; col < target_dim; ++col) {
                    y_ptr[graph.target_offset * target_dim + col] =
                        scale_value(graph.target_ptr[col], target_scaler, col);
                }
            }
            if (has_any_particle_label) {
                mass_label_ptr[graph_index] = graph.has_particle_label
                    ? graph.particle_label
                    : std::numeric_limits<float>::quiet_NaN();
            }
        });
    }

    py::dict result;
    result["x"] = out_x;
    result["edge_index"] = out_edge_index;
    result["edge_attr"] = out_edge_attr;
    result["pulse_x"] = out_pulse_x;
    result["pulse_node_index"] = out_pulse_node_index;
    result["waveform_x"] = out_waveform_x;
    if (has_any_detector_lids) {
        result["detector_lids"] = out_detector_lids;
    }
    result["batch"] = out_batch;
    result["collate_threads"] = resolved_threads;
    if (target_count > 0) {
        result["y"] = out_y;
    }
    if (has_any_particle_label) {
        result["mass_label"] = out_mass_label;
    }
    return result;
}

#ifndef TALESD_GNN_COLLATE_MODULE
#define TALESD_GNN_COLLATE_MODULE talesd_gnn_collate_ext
#endif

PYBIND11_MODULE(TALESD_GNN_COLLATE_MODULE, m) {
    m.def(
        "collate_numeric",
        &collate_numeric,
        py::arg("samples"),
        py::arg("node_mean"),
        py::arg("node_std"),
        py::arg("edge_mean"),
        py::arg("edge_std"),
        py::arg("pulse_mean"),
        py::arg("pulse_std"),
        py::arg("target_mean"),
        py::arg("target_std"),
        py::arg("require_target"),
        py::arg("num_threads") = 0,
        "Collate TALE-SD graph arrays with optional C++ thread parallelism"
    );
}
