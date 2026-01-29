#pragma once
#include <string>
#include <stdexcept>
#include <algorithm>

namespace tds {

inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(x, hi));
}

inline double ticks_to_price(double ticks, double tick_size) {
    return ticks * tick_size;
}

inline double price_to_ticks(double price_delta, double tick_size) {
    if (tick_size <= 0.0) throw std::invalid_argument("tick_size must be > 0");
    return price_delta / tick_size;
}

// RR (reward/risk) for long/short.
// Returns negative if trade is "inverted" (e.g., entry beyond target, or stop on wrong side).
inline double rr(const std::string& side, double entry, double stop, double target) {
    const bool is_long = (side == "long");
    const bool is_short = (side == "short");
    if (!is_long && !is_short) throw std::invalid_argument("side must be 'long' or 'short'");

    if (is_long) {
        const double risk = entry - stop;
        const double reward = target - entry;
        if (risk == 0.0) return 0.0;
        return reward / risk;
    } else {
        const double risk = stop - entry;
        const double reward = entry - target;
        if (risk == 0.0) return 0.0;
        return reward / risk;
    }
}

struct EdgeMetrics {
    double rr_ideal;
    double rr_actual;
    double edge_ratio;                    // rr_actual / rr_ideal
    double edge_remaining_clamped_pct;    // clamp(edge_ratio, 0..1)*100
    double edge_remaining_unclamped_pct;  // clamp(edge_ratio, 0..10)*100
};

// Computes RR metrics given ideal entry (for rr_ideal) and actual entry (for rr_actual).
inline EdgeMetrics edge_metrics(
    const std::string& side,
    double ideal_entry,
    double actual_entry,
    double stop,
    double target
) {
    const double rr_ideal = rr(side, ideal_entry, stop, target);
    const double rr_actual = rr(side, actual_entry, stop, target);

    const double edge_ratio = (rr_ideal != 0.0) ? (rr_actual / rr_ideal) : 0.0;
    const double edge_remaining_clamped_pct = clamp(edge_ratio, 0.0, 1.0) * 100.0;
    const double edge_remaining_unclamped_pct = clamp(edge_ratio, 0.0, 10.0) * 100.0;

    return EdgeMetrics{
        rr_ideal,
        rr_actual,
        edge_ratio,
        edge_remaining_clamped_pct,
        edge_remaining_unclamped_pct
    };
}

// Converts delay+slip (in ticks) into an actual entry price.
// Convention:
//  - long: worse fills => higher entry (positive ticks)
 // - short: worse fills => lower entry (negative ticks)
inline double actual_entry_from_ticks(
    const std::string& side,
    double ideal_entry,
    double tick_size,
    double delta_delay_ticks,   // signed (can be +/-)
    double slip_ticks           // non-negative
) {
    const bool is_long = (side == "long");
    const bool is_short = (side == "short");
    if (!is_long && !is_short) throw std::invalid_argument("side must be 'long' or 'short'");

    const double sgn = is_long ? 1.0 : -1.0;
    const double total_ticks = delta_delay_ticks + sgn * slip_ticks;
    return ideal_entry + ticks_to_price(total_ticks, tick_size);
}

} // namespace tds
