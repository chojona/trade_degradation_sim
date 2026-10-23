#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "degradation.hpp"
#include "batch.hpp"

namespace py = pybind11;

PYBIND11_MODULE(trade_degradation_cpp, m) {
    m.def("evaluate_batch", [](const py::dict& spec, const py::dict& regime,
                                const py::buffer& delta_buffer, const py::buffer& slip_buffer,
                                bool include_samples) {
        // Convert/snapshot inputs while holding the GIL. No borrowed mutable data
        // or Python API calls are used while the numerical kernel releases it.
        const std::array<const char*, 10> keys = {"side", "ideal_entry", "stop", "target", "tick_size",
            "max_entry_distance_ticks", "max_late_ticks_gate", "min_rr", "allow_stop_expansion", "max_stop_ticks"};
        if (spec.size() != keys.size()) throw std::invalid_argument("unexpected spec fields");
        for (const auto key : keys) if (!spec.contains(key)) throw std::invalid_argument("missing spec field");
        if (regime.size() != 5) throw std::invalid_argument("unexpected regime fields");
        for (const auto key : {"sigma_per_sqrt_s_ticks", "slip_mu_ticks", "slip_sigma_ticks", "stop_k_regime", "stop_k_late"}) {
            if (!regime.contains(key)) throw std::invalid_argument("missing regime field");
            tds::nonnegative(py::cast<double>(regime[key]));
        }
        tds::KernelSpec p{py::cast<std::string>(spec["side"]), py::cast<double>(spec["ideal_entry"]),
            py::cast<double>(spec["stop"]), py::cast<double>(spec["target"]), py::cast<double>(spec["tick_size"]),
            py::cast<double>(spec["max_entry_distance_ticks"]), py::cast<double>(spec["max_late_ticks_gate"]),
            py::cast<double>(spec["min_rr"]), py::cast<double>(spec["max_stop_ticks"]),
            py::cast<double>(regime["stop_k_regime"]), py::cast<double>(regime["stop_k_late"]),
            py::cast<bool>(spec["allow_stop_expansion"])};
        const auto snapshot = [](const py::buffer& buffer) {
            const auto info = buffer.request();
            if (info.ndim != 1 || info.itemsize != sizeof(double) ||
                info.format != py::format_descriptor<double>::format() || info.strides[0] != sizeof(double) ||
                info.shape[0] < 1 || info.shape[0] > 1000000)
                throw std::invalid_argument("draws require contiguous native float64 buffers, length 1..1000000");
            const auto data = static_cast<const double*>(info.ptr);
            return std::vector<double>(data, data + info.shape[0]);
        };
        const auto deltas = snapshot(delta_buffer), slips = snapshot(slip_buffer);
        tds::BatchResult result;
        {
            py::gil_scoped_release release;
            result = tds::evaluate_batch(p, deltas, slips, include_samples);
        }
        py::dict stats;
        stats["n"] = result.n;
        stats["p_clean"] = static_cast<double>(result.counts[0]) / result.n;
        stats["p_compromised"] = static_cast<double>(result.counts[1]) / result.n;
        stats["p_invalid"] = static_cast<double>(result.counts[2]) / result.n;
        stats["edge_remaining_mean"] = result.mean;
        stats["edge_remaining_p10"] = result.p10;
        stats["edge_remaining_p50"] = result.p50;
        stats["edge_remaining_p90"] = result.p90;
        stats["degradation_mean"] = result.degradation_mean;
        py::dict output;
        output["stats"] = stats;
        output["samples"] = py::cast(result.samples);
        return output;
    }, py::arg("spec"), py::arg("regime"), py::arg("delay_draws"), py::arg("slip_draws"),
       py::arg("include_samples") = false,
       "Experimental explicit-draw batch kernel; snapshots inputs before releasing the GIL.");

    m.doc() = "C++ core math for Trade Degradation Simulator";

    m.def("clamp", &tds::clamp, "Clamp x into [lo, hi]");
    m.def("ticks_to_price", &tds::ticks_to_price, "Convert ticks to price");
    m.def("price_to_ticks", &tds::price_to_ticks, "Convert price delta to ticks");

    m.def("rr", &tds::rr, "Compute RR (reward/risk) for long/short");
    m.def("actual_entry_from_ticks", &tds::actual_entry_from_ticks,
          py::arg("side"), py::arg("ideal_entry"), py::arg("tick_size"),
          py::arg("delta_delay_ticks"), py::arg("slip_ticks"));

    py::class_<tds::EdgeMetrics>(m, "EdgeMetrics")
        .def_readonly("rr_ideal", &tds::EdgeMetrics::rr_ideal)
        .def_readonly("rr_actual", &tds::EdgeMetrics::rr_actual)
        .def_readonly("edge_ratio", &tds::EdgeMetrics::edge_ratio)
        .def_readonly("edge_remaining_clamped_pct", &tds::EdgeMetrics::edge_remaining_clamped_pct)
        .def_readonly("edge_remaining_unclamped_pct", &tds::EdgeMetrics::edge_remaining_unclamped_pct);

    m.def("edge_metrics_with_stops", &tds::edge_metrics_with_stops,
          py::arg("side"), py::arg("ideal_entry"), py::arg("actual_entry"),
          py::arg("original_stop"), py::arg("used_stop"), py::arg("target"));

    m.def("edge_metrics", &tds::edge_metrics,
          py::arg("side"), py::arg("ideal_entry"), py::arg("actual_entry"),
          py::arg("stop"), py::arg("target"));
}
