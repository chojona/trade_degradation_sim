# python/app.py
# Streamlit UI for Trade Degradation Simulator
# Run from repo root:
#   streamlit run python/app.py

import streamlit as st
from monte_carlo import run_monte_carlo


from simulator import (
    TradeSpec,
    SimInput,
    simulate_once,
    prob_bad,
    max_slippage_allowed_prob,
    max_delay_allowed_prob,
    REGIMES,
)


st.set_page_config(page_title="Trade Degradation Simulator", layout="wide")
st.title("Trade Degradation Simulator")
st.caption("Quantify how much edge is gone by the time you enter.")

# -----------------------------
# Sidebar: trade + model inputs
# -----------------------------
with st.sidebar:
    st.header("Trade")
    side = st.selectbox("Side", ["long", "short"], index=0)

    colA, colB = st.columns(2)
    with colA:
        E = st.number_input("Ideal entry (E)", value=20000.0, step=0.25, format="%.2f")
        S = st.number_input("Stop (S)", value=19995.0, step=0.25, format="%.2f")
    with colB:
        T = st.number_input("Target (T)", value=20020.0, step=0.25, format="%.2f")
        tick = st.number_input("Tick size", value=0.25, step=0.01, format="%.2f")

    st.divider()
    st.header("Execution / Regime")
    regime = st.selectbox("Volatility regime", list(REGIMES.keys()), index=list(REGIMES.keys()).index("High") if "High" in REGIMES else 0)
    delay = st.slider("Entry delay (seconds)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
    slip = st.slider("Slippage (ticks)", min_value=0, max_value=20, value=2, step=1)

    st.divider()
    st.header("Rules")
    max_entry_ticks = st.slider("Max entry distance (ticks)", 1.0, 30.0, 12.0, 0.5)
    max_late_gate = st.slider("Hard lateness gate (ticks)", 1.0, 15.0, 4.0, 0.5)
    min_rr = st.slider("Minimum RR", 0.25, 4.0, 1.0, 0.05)

    allow_stop_expansion = st.checkbox("Allow stop expansion", value=False)
    max_stop_ticks = st.slider("Max stop size (ticks)", 5.0, 200.0, 60.0, 1.0)

    st.divider()
    st.header("Probabilistic boundary")
    edge_floor = st.slider("Edge floor (%)", 0.0, 100.0, 70.0, 1.0)
    p_bad = st.slider("p_bad threshold", 0.05, 0.95, 0.40, 0.01)
    trials = st.slider("Trials per step", 50, 2000, 300, 50)
    seed = st.number_input("Seed", value=7, step=1)


# -----------------------------
# Build spec + input
# -----------------------------
spec = TradeSpec(
    side=side,
    ideal_entry=E,
    stop=S,
    target=T,
    tick_size=tick,
    max_entry_distance_ticks=max_entry_ticks,
    max_late_ticks_gate=max_late_gate,
    min_rr=min_rr,
    allow_stop_expansion=allow_stop_expansion,
    max_stop_ticks=max_stop_ticks,
)

inp = SimInput(
    delay_seconds=delay,
    regime=regime,
    forced_slip_ticks=float(slip),
)

# -----------------------------
# Compute single simulation
# -----------------------------
r = simulate_once(spec, inp)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1.0], gap="large")

with col1:
    st.subheader("Single Simulation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classification", r.classification)
    c2.metric("Edge remaining (clamped)", f"{r.edge_remaining_pct:.2f}%")
    c3.metric("Degradation score", f"{r.degradation_score:.2f}/100")
    c4.metric("RR (ideal → actual)", f"{r.edge_ideal_R:.2f}R → {r.edge_actual_R:.2f}R")

    st.write("**Execution details**")
    d = r.details
    st.code(
        "\n".join(
            [
                f"regime: {d['regime']}",
                f"delay_seconds: {d['delay_seconds']}",
                f"delta_delay_ticks: {d['delta_delay_ticks']:.3f}",
                f"slip_ticks: {d['slip_ticks']}",
                f"lateness_ticks: {d['lateness_ticks']:.3f}",
                f"late_gate_ticks: {d['late_gate_ticks']}",
                f"edge_ratio: {d['edge_ratio']:.4f}",
                f"edge_remaining_unclamped_pct: {d['edge_remaining_unclamped_pct']:.2f}",
                f"rule_penalty: {d['rule_penalty']}",
                f"actual_entry: {r.actual_entry:.2f}",
                f"used_stop: {r.used_stop:.2f}",
            ]
        ),
        language="text",
    )

with col2:
    st.subheader("Tolerances (probabilistic)")

    max_slip = max_slippage_allowed_prob(
        spec,
        delay_seconds=float(delay),
        regime=regime,
        edge_floor_pct=float(edge_floor),
        p_bad=float(p_bad),
        trials=int(trials),
        max_search_ticks=60,
        seed=int(seed),
    )

    max_delay = max_delay_allowed_prob(
        spec,
        forced_slip_ticks=0.0,  # by design: delay tolerance assumes 0 slip
        regime=regime,
        edge_floor_pct=float(edge_floor),
        p_bad=float(p_bad),
        trials=int(trials),
        max_search_seconds=60.0,
        seed=int(seed),
    )

    p_at_max_slip = prob_bad(
        spec,
        SimInput(delay_seconds=float(delay), regime=regime, forced_slip_ticks=float(max_slip)),
        edge_floor_pct=float(edge_floor),
        trials=int(trials),
        seed=int(seed),
    )

    p_at_max_delay = prob_bad(
        spec,
        SimInput(delay_seconds=float(max_delay), regime=regime, forced_slip_ticks=0.0),
        edge_floor_pct=float(edge_floor),
        trials=int(trials),
        seed=int(seed),
    )

    st.metric("Edge floor", f"{edge_floor:.0f}%")
    st.metric("Current lateness", f"{d['lateness_ticks']:.2f} ticks (gate {d['late_gate_ticks']})")

    st.divider()
    st.metric("Max slippage allowed", f"{max_slip} ticks")
    st.caption(f"P(bad) @ max_slip: {p_at_max_slip:.3f}")

    st.divider()
    st.metric("Max delay allowed", f"{max_delay:.3f} s (assuming 0 slip)")
    st.caption(f"P(bad) @ max_delay: {p_at_max_delay:.3f}")

    st.divider()
    st.caption("Tip: Tighten the filter by increasing Edge floor or p_bad, or lowering late_gate.")

st.divider()
st.subheader("Quick Presets")

st.divider()
st.subheader("Monte Carlo Distribution")

if st.button("Run Monte Carlo Simulation"):
    with st.spinner("Running Monte Carlo..."):
        stats, edges = run_monte_carlo(
            spec,
            inp,
            n=2000,
            seed=int(seed),
        )

    st.write("**Summary statistics**")
    st.json({
        "p_clean": round(stats.p_clean, 3),
        "p_compromised": round(stats.p_compromised, 3),
        "p_invalid": round(stats.p_invalid, 3),
        "edge_remaining_mean": round(stats.edge_remaining_mean, 2),
        "edge_remaining_p10": round(stats.edge_remaining_p10, 2),
        "edge_remaining_p50": round(stats.edge_remaining_p50, 2),
        "edge_remaining_p90": round(stats.edge_remaining_p90, 2),
    })

    # Simple histogram via bar_chart (Streamlit-native)
    st.write("**Edge Remaining Distribution (%)**")
    st.bar_chart(edges)


preset_col1, preset_col2, preset_col3 = st.columns(3)
with preset_col1:
    st.write("**Strict filter**")
    st.code("edge_floor=80, p_bad=0.35, late_gate=3", language="text")
with preset_col2:
    st.write("**Balanced**")
    st.code("edge_floor=70, p_bad=0.40, late_gate=4", language="text")
with preset_col3:
    st.write("**Loose**")
    st.code("edge_floor=60, p_bad=0.50, late_gate=6", language="text")
