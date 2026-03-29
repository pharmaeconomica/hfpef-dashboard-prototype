import streamlit as st
from ui_redesign_v11 import (
    render_top_kpis,
    render_story_summary,
    render_pathway_cards,
    render_node_detail,
    render_hidden_technical_section,
)


def render_executive_summary(
    client_view_df,
    v11_transition_results_df,
    currency,
    show_vica,
):
    st.title("HFpEF Care Pathway Dashboard")
    st.caption(
        "A simple view of where patients are delayed, where the system loses value, "
        "and how VicaEmpa changes the pathway."
    )

    render_top_kpis(client_view_df, currency=currency)
    render_story_summary(client_view_df, show_vica=show_vica)
    render_pathway_cards(client_view_df, currency=currency)
    render_node_detail(client_view_df, v11_transition_results_df, currency=currency)


def render_detailed_analysis(
    v11_aggregate_node_results_df,
    v11_transition_results_df,
    v11_segment_summary_df,
    client_view_df,
    currency,
):
    st.title("Detailed Analysis")
    st.caption("Technical outputs, flow tables, and detailed node-level information.")

    st.subheader("Aggregate node results")
    st.dataframe(v11_aggregate_node_results_df, use_container_width=True)

    st.subheader("Transition flows")
    st.dataframe(v11_transition_results_df, use_container_width=True)

    if not v11_transition_results_df.empty:
        transition_matrix = v11_transition_results_df.pivot_table(
            index="from_node",
            columns="to_node",
            values="probability",
            aggfunc="mean",
            fill_value=0
        )
        st.subheader("Transition matrix")
        st.dataframe(transition_matrix, use_container_width=True)

    st.subheader("Segment summary")
    st.dataframe(v11_segment_summary_df, use_container_width=True)

    st.subheader("Node-level drilldown")
    from ui_redesign_v11 import render_node_detail
    render_node_detail(client_view_df, v11_transition_results_df, currency=currency)

    render_hidden_technical_section(
        v11_aggregate_node_results_df,
        v11_transition_results_df,
        v11_segment_summary_df
    )
