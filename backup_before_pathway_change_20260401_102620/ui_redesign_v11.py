import math
import pandas as pd
import streamlit as st


NODE_LABELS = {
    "N1": "Symptom Onset",
    "N2": "Risk Screening",
    "N3": "Specialist Referral",
    "N4": "Diagnosis Confirmation",
    "N5": "Treatment Initiation",
    "N6": "Clinical Inertia",
    "N7": "Hospital Transition",
    "N8": "Adherence & Monitoring",
    "N9": "HHF Event & Readmission",
}

STAGE_MAP = {
    "Stage 1 — Pre-Diagnosis": ["N1", "N2"],
    "Stage 2 — Presentation & Diagnosis": ["N3", "N4"],
    "Stage 3 — Treatment Initiation": ["N5", "N6"],
    "Stage 4 — Ongoing Management": ["N7", "N8"],
    "Stage 5 — Advanced Disease & Acute Events": ["N9"],
}


def _fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"


def _fmt_money(x, currency="$"):
    try:
        return f"{currency}{float(x):,.0f}"
    except Exception:
        return f"{currency}0"


def _fmt_days(x):
    try:
        return f"{float(x):.0f}d"
    except Exception:
        return "-"


def _score_node(row):
    delay = float(row.get("cumulative_delay_days", row.get("avg_delay_days", 0)) or 0)
    leakage = 0.0
    inflow = float(row.get("inflow_patients", 0) or 0)
    lost = float(row.get("leakage_patients", 0) or 0)
    if inflow > 0:
        leakage = lost / inflow
    cost = float(row.get("total_cost", 0) or 0)
    excess_hhf = float(row.get("excess_events_from_delay", 0) or 0)

    score = (
        min(delay / 90.0, 1.0) * 0.30 +
        min(leakage / 0.50, 1.0) * 0.25 +
        min(cost / 50000000.0, 1.0) * 0.25 +
        min(excess_hhf / 10000.0, 1.0) * 0.20
    )
    return score


def _node_color(score):
    if score >= 0.66:
        return "#FDE2E1"
    if score >= 0.33:
        return "#FFF4D6"
    return "#E6F4EA"


def prepare_client_view(v11_aggregate_node_results_df):
    df = v11_aggregate_node_results_df.copy()

    if "node_id" not in df.columns:
        raise ValueError("Expected column 'node_id' in Version 11 aggregate results.")

    df["node_name"] = df["node_id"].map(NODE_LABELS).fillna(df["node_id"])
    df["node_score"] = df.apply(_score_node, axis=1)

    if "leakage_patients" in df.columns and "inflow_patients" in df.columns:
        df["leakage_pct"] = df.apply(
            lambda r: (float(r["leakage_patients"]) / float(r["inflow_patients"]) * 100.0)
            if float(r["inflow_patients"]) > 0 else 0.0,
            axis=1
        )
    else:
        df["leakage_pct"] = 0.0

    return df.sort_values("node_id")


def render_top_kpis(client_df, currency="$"):
    total_cost = float(client_df["total_cost"].sum()) if "total_cost" in client_df.columns else 0.0
    total_remaining = float(client_df["surviving_patients"].sum()) if "surviving_patients" in client_df.columns else 0.0
    total_excess_hhf = float(client_df["excess_events_from_delay"].sum()) if "excess_events_from_delay" in client_df.columns else 0.0

    if client_df.empty:
        bottleneck_label = "-"
    else:
        top = client_df.sort_values("node_score", ascending=False).iloc[0]
        bottleneck_label = f'{top["node_id"]} — {top["node_name"]}'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total system cost", _fmt_money(total_cost, currency))
    c2.metric("Patients remaining in pathway", _fmt_int(total_remaining))
    c3.metric("Biggest bottleneck", bottleneck_label)
    c4.metric("Avoidable hospitalizations", _fmt_int(total_excess_hhf))


def render_story_summary(client_df, show_vica):
    if client_df.empty:
        return

    top = client_df.sort_values("node_score", ascending=False).iloc[0]
    top_hhf = float(top.get("excess_events_from_delay", 0) or 0)
    mode = "with VicaEmpa" if show_vica else "in the current pathway"

    st.info(
        f"The biggest current barrier {mode} is {top['node_id']} — {top['node_name']}. "
        f"It is associated with about {_fmt_int(top_hhf)} avoidable hospitalizations from accumulated delay."
    )


def render_pathway_cards(client_df, currency="$"):
    st.subheader("HFpEF Care Pathway")

    for stage_name, stage_nodes in STAGE_MAP.items():
        st.markdown(f"### {stage_name}")
        cols = st.columns(len(stage_nodes))

        for idx, node_id in enumerate(stage_nodes):
            col = cols[idx]
            row_df = client_df[client_df["node_id"] == node_id]

            if row_df.empty:
                with col:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:12px; padding:14px; min-height:190px;">
                            <div style="font-size:14px; color:#666;">{node_id}</div>
                            <div style="font-size:18px; font-weight:700;">{NODE_LABELS.get(node_id, node_id)}</div>
                            <div style="margin-top:12px; color:#888;">No data</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                continue

            row = row_df.iloc[0]
            bg = _node_color(float(row["node_score"]))

            with col:
                st.markdown(
                    f"""
                    <div style="background:{bg}; border:1px solid #ddd; border-radius:16px; padding:14px; min-height:210px;">
                        <div style="font-size:13px; color:#555;">{row['node_id']}</div>
                        <div style="font-size:18px; font-weight:700; line-height:1.2; margin-bottom:10px;">
                            {row['node_name']}
                        </div>
                        <div style="font-size:14px; margin-bottom:6px;"><b>Patients:</b> {_fmt_int(row.get('inflow_patients', 0))}</div>
                        <div style="font-size:14px; margin-bottom:6px;"><b>Delay:</b> {_fmt_days(row.get('cumulative_delay_days', row.get('avg_delay_days', 0)))}</div>
                        <div style="font-size:14px; margin-bottom:6px;"><b>Lost:</b> {float(row.get('leakage_pct', 0)):.1f}%</div>
                        <div style="font-size:14px; margin-bottom:6px;"><b>Cost:</b> {_fmt_money(row.get('total_cost', 0), currency)}</div>
                        <div style="font-size:14px;"><b>Avoidable HHF:</b> {_fmt_int(row.get('excess_events_from_delay', 0))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")


def render_node_detail(client_df, v11_transition_results_df, currency="$"):
    st.subheader("Node detail")
    node_id = st.selectbox("Select a node", client_df["node_id"].tolist(), key="node_detail_select")
    row = client_df[client_df["node_id"] == node_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients entering", _fmt_int(row.get("inflow_patients", 0)))
    c2.metric("Cumulative delay", _fmt_days(row.get("cumulative_delay_days", row.get("avg_delay_days", 0))))
    c3.metric("Cost at node", _fmt_money(row.get("total_cost", 0), currency))

    c4, c5, c6 = st.columns(3)
    c4.metric("Patients lost", _fmt_int(row.get("leakage_patients", 0)))
    c5.metric("Leakage", f'{float(row.get("leakage_pct", 0)):.1f}%')
    c6.metric("Avoidable HHF", _fmt_int(row.get("excess_events_from_delay", 0)))

    node_trans = v11_transition_results_df[v11_transition_results_df["from_node"] == node_id].copy()
    if not node_trans.empty:
        st.markdown("#### Where patients go next")
        view = node_trans[["to_node", "probability", "patients_moved", "cumulative_delay_sent"]].copy()
        view["to_node"] = view["to_node"].map(NODE_LABELS).fillna(view["to_node"])
        view["probability"] = (view["probability"] * 100).round(1).astype(str) + "%"
        view["patients_moved"] = view["patients_moved"].round(0).astype(int)
        view["cumulative_delay_sent"] = view["cumulative_delay_sent"].round(1)
        st.dataframe(view, use_container_width=True, hide_index=True)


def render_hidden_technical_section(v11_aggregate_node_results_df, v11_transition_results_df, v11_segment_summary_df):
    with st.expander("Technical details"):
        st.markdown("#### Aggregate node results")
        st.dataframe(v11_aggregate_node_results_df, use_container_width=True)

        st.markdown("#### Transition flows")
        st.dataframe(v11_transition_results_df, use_container_width=True)

        st.markdown("#### Segment summary")
        st.dataframe(v11_segment_summary_df, use_container_width=True)
