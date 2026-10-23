#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tds {
inline double finite(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("value must be finite");
    return x;
}
inline double clamp(double x, double lo, double hi) {
    finite(x); finite(lo); finite(hi);
    if (lo > hi) throw std::invalid_argument("invalid clamp bounds");
    return std::max(lo, std::min(x, hi));
}
inline int side_sign(std::string side) {
    const auto first = side.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) throw std::invalid_argument("invalid side");
    side = side.substr(first, side.find_last_not_of(" \t\r\n") - first + 1);
    for (auto& ch : side) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (side == "long" || side == "buy") return 1;
    if (side == "short" || side == "sell") return -1;
    throw std::invalid_argument("invalid side");
}
inline double valid_tick(double tick) {
    finite(tick);
    if (tick <= 0) throw std::invalid_argument("tick_size must be positive");
    return tick;
}
inline double ticks_to_price(double ticks, double tick) { return finite(finite(ticks) * valid_tick(tick)); }
inline double price_to_ticks(double price, double tick) { return finite(finite(price) / valid_tick(tick)); }
inline double round_grid(double price, double tick, int direction) {
    const double units = price_to_ticks(price, tick);
    if (std::abs(units) >= std::pow(2., 50)) throw std::invalid_argument("price/tick outside supported precision range");
    const double nearest = std::round(units);
    return (std::abs(units - nearest) <= 1e-8 ? nearest : direction > 0 ? std::ceil(units) : std::floor(units)) * tick;
}
inline double rr(const std::string& side, double entry, double stop, double target) {
    finite(entry); finite(stop); finite(target);
    const int sgn = side_sign(side);
    const double risk = finite((entry - stop) * sgn), reward = finite((target - entry) * sgn);
    return risk > 0 ? finite(reward / risk) : 0.;
}
struct EdgeMetrics {
    double rr_ideal, rr_actual, edge_ratio, edge_remaining_clamped_pct, edge_remaining_unclamped_pct;
};
inline EdgeMetrics edge_metrics_with_stops(const std::string& side, double ideal, double actual,
                                         double original_stop, double used_stop, double target) {
    const double baseline = rr(side, ideal, original_stop, target);
    const double observed = rr(side, actual, used_stop, target);
    const double ratio = baseline > 0 ? finite(observed / baseline) : 0.;
    return {baseline, observed, ratio, clamp(ratio, 0., 1.) * 100, clamp(ratio, 0., 10.) * 100};
}
inline EdgeMetrics edge_metrics(const std::string& side, double ideal, double actual, double stop, double target) {
    return edge_metrics_with_stops(side, ideal, actual, stop, stop, target);
}
inline double actual_entry_from_ticks(const std::string& side, double ideal, double tick, double delta, double slip) {
    finite(ideal); finite(delta); finite(slip); valid_tick(tick);
    if (slip < 0) throw std::invalid_argument("slippage must be nonnegative");
    const int sgn = side_sign(side);
    return round_grid(ideal + ticks_to_price(delta + sgn * slip, tick), tick, sgn);
}
} // namespace tds
