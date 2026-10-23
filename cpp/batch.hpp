#pragma once
#include "degradation.hpp"
#include <array>
#include <vector>

namespace tds {
struct KernelSpec {
    std::string side;
    double ideal, stop, target, tick, max_entry, max_late, min_rr, max_stop, k_regime, k_late;
    bool expand;
};
inline void nonnegative(double x) {
    if (finite(x) < 0) throw std::invalid_argument("parameter must be nonnegative");
}
inline void validate_spec(const KernelSpec& p) {
    const int sign = side_sign(p.side);
    valid_tick(p.tick);
    for (double price : {p.ideal, p.stop, p.target}) {
        const double units = price_to_ticks(price, p.tick);
        if (std::abs(units) >= std::pow(2., 50) || std::abs(units - std::round(units)) > 1e-8)
            throw std::invalid_argument("plan price must lie on supported tick grid");
    }
    if ((p.ideal - p.stop) * sign <= 0 || (p.target - p.ideal) * sign <= 0)
        throw std::invalid_argument("stop and target must bracket ideal entry");
    for (double x : {p.max_entry, p.max_late, p.min_rr, p.k_regime, p.k_late}) nonnegative(x);
    if (finite(p.max_stop) <= 0) throw std::invalid_argument("max stop must be positive");
}
using Sample = std::array<double, 8>;
struct BatchResult {
    std::size_t n = 0;
    std::array<std::size_t, 3> counts{};
    double mean = 0, p10 = 0, p50 = 0, p90 = 0, degradation_mean = 0;
    std::vector<Sample> samples;
};
// Neumaier compensated summation; deterministic order, no parallel reduction.
struct Sum {
    double sum = 0, correction = 0;
    void add(double x) {
        double next = sum + x;
        correction += std::abs(sum) >= std::abs(x) ? (sum - next) + x : (x - next) + sum;
        sum = next;
    }
    double value() const { return sum + correction; }
};
inline std::size_t percentile_index(std::size_t n, int percent) {
    // Exact rational nearest index with ties-to-even, matching Python round.
    const std::size_t numerator = static_cast<std::size_t>(percent) * (n - 1);
    std::size_t q = numerator / 100, rem = numerator % 100;
    return q + (rem > 50 || (rem == 50 && q % 2 != 0));
}
inline BatchResult evaluate_batch(const KernelSpec& p, const std::vector<double>& delays,
                                const std::vector<double>& slips, bool include_samples) {
    validate_spec(p);
    if (delays.empty() || delays.size() != slips.size() || delays.size() > 1000000)
        throw std::invalid_argument("require equal nonempty arrays of at most 1000000 draws");
    const int sign = side_sign(p.side);
    const double baseline = rr(p.side, p.ideal, p.stop, p.target);
    const double ideal_risk = (p.ideal - p.stop) * sign;
    const double base_ticks = std::abs(price_to_ticks(p.ideal - p.stop, p.tick));
    BatchResult result;
    result.n = delays.size();
    if (include_samples) result.samples.reserve(result.n);
    std::vector<double> retained;
    retained.reserve(result.n);
    Sum rr_sum, score_sum;
    for (std::size_t i = 0; i < result.n; ++i) {
        finite(delays[i]); nonnegative(slips[i]);
        const double entry = actual_entry_from_ticks(p.side, p.ideal, p.tick, delays[i], slips[i]);
        const double lateness = std::abs(price_to_ticks(entry - p.ideal, p.tick));
        double stop = p.stop;
        if (p.expand) {
            const double factor = clamp(lateness / std::max(p.max_entry, 1e-9), 0., 2.);
            const double desired = finite(base_ticks * (1 + p.k_regime + p.k_late * factor));
            const double distance = std::max(base_ticks, std::min(std::ceil(desired - 1e-8), std::floor(p.max_stop)));
            stop = round_grid(p.ideal - sign * distance * p.tick, p.tick, -sign);
        }
        const double actual_rr = rr(p.side, entry, stop, p.target);
        const double ratio = finite(actual_rr / baseline);
        const double retention = clamp(ratio, 0., 1.) * 100;
        const double stop_ticks = (entry - stop) * sign / p.tick;
        const bool invalid = lateness > p.max_entry + 1e-8 || lateness > p.max_late + 1e-8 ||
                             stop_ticks > p.max_stop + 1e-8 || stop_ticks <= 0 ||
                             actual_rr < p.min_rr || (p.target - entry) * sign <= 0;
        const double penalty = invalid ? .5 : 0.;
        const double risk_increase = clamp(((entry - stop) * sign / ideal_risk) - 1, 0., 1.);
        const double score = 100 * clamp(.7 * clamp(1 - ratio, 0., 1.) + .2 * risk_increase + .1 * penalty, 0., 1.);
        const std::size_t classification = invalid || retention < 50 ? 2 : retention < 80 ? 1 : 0;
        ++result.counts[classification];
        retained.push_back(retention);
        rr_sum.add(retention); score_sum.add(score);
        if (include_samples) result.samples.push_back({entry, stop, baseline, actual_rr, retention, score, stop_ticks,
                                                       static_cast<double>(classification)});
    }
    std::sort(retained.begin(), retained.end());
    result.mean = rr_sum.value() / result.n;
    result.degradation_mean = score_sum.value() / result.n;
    result.p10 = retained[percentile_index(result.n, 10)];
    result.p50 = retained[percentile_index(result.n, 50)];
    result.p90 = retained[percentile_index(result.n, 90)];
    return result;
}
} // namespace tds
