import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ui_redesign_v11 import (
    render_top_kpis,
    render_story_summary,
    render_pathway_cards,
    render_node_detail,
)

from segment_explanations import get_segment_explanation
from dynamic_segments_v12 import (
    summarize_segment_evolution,
    build_segment_transition_matrix,
    build_segment_evolution_chart_data,
)

def _make_bar_chart(df, x_col, y_col, title, y_label):
    plot_df = df.copy()
    if plot_df.empty or x_col not in plot_df.columns or y_col not in plot_df.columns:
        st.info(f"No data available for: {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col].astype(float), color="#00E47C", edgecolor="#08312A")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def _make_comparison_chart(current_value, vica_value, title, y_label):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Current", "VicaEmpa"], [float(current_value), float(vica_value)], color=["#08312A", "#00E47C"], edgecolor="#08312A")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    st.pyplot(fig)


def render_executive_summary(
    client_view_df,
    v11_transition_results_df,
    currency,
    show_vica,
    v11_current_results,
    v11_vica_results,
):
    st.title("Executive Summary")
    st.caption(
        "A simple view of where patients are delayed, where the system loses value, "
        "and how VicaEmpa changes the pathway."
    )

    render_top_kpis(client_view_df, currency=currency)
    render_story_summary(client_view_df, show_vica=show_vica)
    render_pathway_cards(client_view_df, currency=currency)

    st.subheader("Key visuals")
    c1, c2 = st.columns(2)

    with c1:
        _make_bar_chart(
            client_view_df.sort_values("total_cost", ascending=False),
            x_col="node_id",
            y_col="total_cost",
            title="Cost by node",
            y_label=f"Cost ({currency})",
        )

    with c2:
        _make_bar_chart(
            client_view_df.sort_values("excess_events_from_delay", ascending=False),
            x_col="node_id",
            y_col="excess_events_from_delay",
            title="Avoidable hospitalizations by node",
            y_label="Avoidable HHF",
        )

    current_df = v11_current_results["aggregate_node_results"]
    vica_df = v11_vica_results["aggregate_node_results"]

    current_cost = float(current_df["total_cost"].sum()) if not current_df.empty else 0.0
    vica_cost = float(vica_df["total_cost"].sum()) if not vica_df.empty else 0.0

    current_hhf = float(current_df["excess_events_from_delay"].sum()) if not current_df.empty else 0.0
    vica_hhf = float(vica_df["excess_events_from_delay"].sum()) if not vica_df.empty else 0.0

    c3, c4 = st.columns(2)

    if show_vica:
        with c3:
            _make_comparison_chart(
                current_cost,
                vica_cost,
                title="Current vs VicaEmpa: total system cost",
                y_label=f"Cost ({currency})",
            )

        with c4:
            _make_comparison_chart(
                current_hhf,
                vica_hhf,
                title="Current vs VicaEmpa: avoidable hospitalizations",
                y_label="Avoidable HHF",
            )
    else:
        with c3:
            _make_bar_chart(
                current_df.sort_values("total_cost", ascending=False),
                x_col="node_id",
                y_col="total_cost",
                title="Current scenario: cost by node",
                y_label=f"Cost ({currency})",
            )

        with c4:
            _make_bar_chart(
                current_df.sort_values("excess_events_from_delay", ascending=False),
                x_col="node_id",
                y_col="excess_events_from_delay",
                title="Current scenario: avoidable hospitalizations by node",
                y_label="Avoidable HHF",
            )

    render_node_detail(client_view_df, v11_transition_results_df, currency=currency)


def render_detailed_analysis(
    v11_aggregate_node_results_df,
    v11_transition_results_df,
    v11_segment_summary_df,
    client_view_df,
    currency,
    show_vica,
    v11_current_results,
    v11_vica_results,
):
    st.title("Detailed Analysis")
    st.caption("Structured outputs for deeper inspection.")

    st.subheader("Charts")
    c1, c2 = st.columns(2)

    with c1:
        _make_bar_chart(
            v11_aggregate_node_results_df.sort_values("total_cost", ascending=False),
            x_col="node_id",
            y_col="total_cost",
            title="Node cost ranking",
            y_label=f"Cost ({currency})",
        )

    with c2:
        _make_bar_chart(
            v11_aggregate_node_results_df.sort_values("cumulative_delay_days", ascending=False),
            x_col="node_id",
            y_col="cumulative_delay_days",
            title="Cumulative delay by node",
            y_label="Days",
        )

    if not v11_transition_results_df.empty:
        trans_plot = (
            v11_transition_results_df.groupby(["from_node", "to_node"], as_index=False)["patients_moved"]
            .sum()
        )
        trans_plot["transition"] = trans_plot["from_node"] + " → " + trans_plot["to_node"]

        st.subheader("Transition flows chart")
        _make_bar_chart(
            trans_plot.sort_values("patients_moved", ascending=False),
            x_col="transition",
            y_col="patients_moved",
            title="Patients moved by transition",
            y_label="Patients moved",
        )

    if not v11_segment_summary_df.empty:
        seg_plot = v11_segment_summary_df.copy()
        if "segment" in seg_plot.columns and "total_cost" in seg_plot.columns:
            st.subheader("Segment comparison chart")
            _make_bar_chart(
                seg_plot.sort_values("total_cost", ascending=False),
                x_col="segment",
                y_col="total_cost",
                title="Total cost by segment",
                y_label=f"Cost ({currency})",
            )

    if show_vica:
        current_df = v11_current_results["aggregate_node_results"]
        vica_df = v11_vica_results["aggregate_node_results"]

        current_cost = float(current_df["total_cost"].sum()) if not current_df.empty else 0.0
        vica_cost = float(vica_df["total_cost"].sum()) if not vica_df.empty else 0.0

        current_hhf = float(current_df["excess_events_from_delay"].sum()) if not current_df.empty else 0.0
        vica_hhf = float(vica_df["excess_events_from_delay"].sum()) if not vica_df.empty else 0.0

        st.subheader("Scenario comparison")
        c3, c4 = st.columns(2)

        with c3:
            _make_comparison_chart(
                current_cost,
                vica_cost,
                title="Current vs VicaEmpa: total system cost",
                y_label=f"Cost ({currency})",
            )

        with c4:
            _make_comparison_chart(
                current_hhf,
                vica_hhf,
                title="Current vs VicaEmpa: avoidable hospitalizations",
                y_label="Avoidable HHF",
            )

    st.subheader("1. Aggregate node results")
    st.dataframe(v11_aggregate_node_results_df, use_container_width=True)

    st.subheader("2. Transition flows")
    st.dataframe(v11_transition_results_df, use_container_width=True)

    if not v11_transition_results_df.empty:
        transition_matrix = v11_transition_results_df.pivot_table(
            index="from_node",
            columns="to_node",
            values="probability",
            aggfunc="mean",
            fill_value=0
        )
        st.subheader("3. Transition matrix")
        st.dataframe(transition_matrix, use_container_width=True)

    st.subheader("4. Segment summary")
    st.dataframe(v11_segment_summary_df, use_container_width=True)

    st.subheader("Segment evolution over time")
    st.caption("Version 12 view: patients may move between pathway segments over time as their care journey changes.")

    v12_segment_evolution_df = summarize_segment_evolution()
    v12_segment_matrix_df = build_segment_transition_matrix()
    v12_segment_chart_df = build_segment_evolution_chart_data()

    if not v12_segment_chart_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(
            v12_segment_chart_df["transition_label"],
            v12_segment_chart_df["annual_transition_rate"] * 100,
            color="#00E47C",
            edgecolor="#08312A"
        )
        ax.set_ylabel("Annual transition rate (%)")
        ax.set_title("Illustrative segment evolution rates")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    if not v12_segment_evolution_df.empty:
        st.markdown("**Illustrative segment evolution summary**")
        st.dataframe(v12_segment_evolution_df, use_container_width=True, hide_index=True)

    if not v12_segment_matrix_df.empty:
        st.markdown("**Segment transition matrix**")
        st.dataframe(v12_segment_matrix_df, use_container_width=True)

    st.subheader("Segment explanations")
    if not v11_segment_summary_df.empty and "segment" in v11_segment_summary_df.columns:
        for segment_name in v11_segment_summary_df["segment"].dropna().unique():
            expl = get_segment_explanation(segment_name)
            with st.expander(f"{segment_name}", expanded=False):
                st.write(f"**Plain English:** {expl['plain_english']}")
                st.write(f"**Main Nodes:** {expl['main_nodes']}")
                st.write(f"**Clinical Logic:** {expl['clinical_logic']}")

    with st.expander("Technical details", expanded=False):
        st.write("Raw node-level results")
        st.dataframe(client_view_df, use_container_width=True)
