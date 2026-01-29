#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "degradation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(trade_degradation_cpp, m) {
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

    m.def("edge_metrics", &tds::edge_metrics,
          py::arg("side"), py::arg("ideal_entry"), py::arg("actual_entry"),
          py::arg("stop"), py::arg("target"));
}
