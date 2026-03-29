import pandas as pd
import numpy as np


def _find_column(df, aliases, required=True):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias.lower().strip() in lower_map:
            return lower_map[alias.lower().strip()]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {aliases}")
    return None


def _scenario_is_vicaempa(scenario_name):
    if scenario_name is None:
        return False
    return str(scenario_name).strip().lower() == "vicaempa"


def _safe_get_value(row, col_name, default=0.0):
    if col_name is None:
        return default
    value = row.get(col_name, default)
    if pd.isna(value):
        return default
    return value


def validate_transition_table(node_params_df, transitions_df, selected_market):
    market_col_nodes = _find_column(node_params_df, ["market"], required=False)
    node_id_col = _find_column(node_params_df, ["node_id", "node", "node_name"])
    order_col = _find_column(node_params_df, ["node_order", "order", "sequence"], required=False)

    market_col_trans = _find_column(transitions_df, ["market"], required=False)
    from_col = _find_column(transitions_df, ["from_node", "source_node"])
    to_col = _find_column(transitions_df, ["to_node", "target_node"])
    prob_col = _find_column(transitions_df, ["probability", "transition_probability", "prob"])

    if market_col_nodes:
        valid_nodes = set(
            node_params_df[node_params_df[market_col_nodes] == selected_market][node_id_col].astype(str)
        )
    else:
        valid_nodes = set(node_params_df[node_id_col].astype(str))

    if market_col_trans:
        trans_market_df = transitions_df[transitions_df[market_col_trans] == selected_market].copy()
    else:
        trans_market_df = transitions_df.copy()

    if trans_market_df.empty:
        raise ValueError(f"No transitions found for market '{selected_market}'.")

    invalid_from = set(trans_market_df[from_col].astype(str)) - valid_nodes
    invalid_to = set(trans_market_df[to_col].astype(str)) - valid_nodes

    if invalid_from:
        raise ValueError(f"Invalid from_node values in transition table: {sorted(list(invalid_from))}")

    if invalid_to:
        raise ValueError(f"Invalid to_node values in transition table: {sorted(list(invalid_to))}")

    if ((trans_market_df[prob_col] < 0) | (trans_market_df[prob_col] > 1)).any():
        raise ValueError("All transition probabilities must be between 0 and 1.")

    grouped = trans_market_df.groupby(from_col, dropna=False)[prob_col].sum().reset_index()
    bad_rows = grouped[np.abs(grouped[prob_col] - 1.0) > 0.000001]

    if not bad_rows.empty:
        bad_text = ", ".join(
            [f"{row[from_col]} = {row[prob_col]:.4f}" for _, row in bad_rows.iterrows()]
        )
        raise ValueError(
            f"Outgoing transition probabilities must sum to 1. Problem nodes: {bad_text}"
        )

    warning_messages = []
    if order_col:
        if market_col_nodes:
            market_nodes = node_params_df[node_params_df[market_col_nodes] == selected_market].copy()
        else:
            market_nodes = node_params_df.copy()

        order_map = dict(zip(market_nodes[node_id_col].astype(str), market_nodes[order_col]))
        for _, row in trans_market_df.iterrows():
            from_node = str(row[from_col])
            to_node = str(row[to_col])
            if from_node in order_map and to_node in order_map:
                if order_map[to_node] < order_map[from_node]:
                    warning_messages.append(
                        f"Transition goes backwards in node order: {from_node} -> {to_node}"
                    )

    return {
        "is_valid": True,
        "warnings": warning_messages
    }


def run_transition_model_v10(
    node_params_df,
    segment_params_df,
    transitions_df,
    total_population,
    selected_market,
    scenario_name="Current"
):
    node_id_col = _find_column(node_params_df, ["node_id", "node", "node_name"])
    market_col_nodes = _find_column(node_params_df, ["market"], required=False)
    order_col = _find_column(node_params_df, ["node_order", "order", "sequence"], required=False)

    delay_col = _find_column(node_params_df, ["delay", "delay_days", "avg_delay_days"], required=False)
    leakage_col = _find_column(node_params_df, ["leakage", "leakage_rate", "dropout_rate"], required=False)
    cost_col = _find_column(node_params_df, ["cost", "node_cost", "cost_per_patient"], required=False)

    vica_delay_mult_col = _find_column(
        node_params_df,
        ["vicaempa_delay_multiplier", "vica_delay_multiplier", "scenario_delay_multiplier"],
        required=False
    )
    vica_leakage_mult_col = _find_column(
        node_params_df,
        ["vicaempa_leakage_multiplier", "vica_leakage_multiplier", "scenario_leakage_multiplier"],
        required=False
    )
    vica_cost_mult_col = _find_column(
        node_params_df,
        ["vicaempa_cost_multiplier", "vica_cost_multiplier", "scenario_cost_multiplier"],
        required=False
    )

    segment_col = _find_column(segment_params_df, ["segment", "segment_name", "segment_id"])
    pop_share_col = _find_column(segment_params_df, ["population_share", "share", "segment_share"])

    seg_delay_mult_col = _find_column(segment_params_df, ["delay_multiplier"], required=False)
    seg_leakage_mult_col = _find_column(segment_params_df, ["leakage_multiplier"], required=False)
    seg_cost_mult_col = _find_column(segment_params_df, ["cost_multiplier"], required=False)

    market_col_segments = _find_column(segment_params_df, ["market"], required=False)

    market_col_trans = _find_column(transitions_df, ["market"], required=False)
    from_col = _find_column(transitions_df, ["from_node", "source_node"])
    to_col = _find_column(transitions_df, ["to_node", "target_node"])
    prob_col = _find_column(transitions_df, ["probability", "transition_probability", "prob"])

    if market_col_nodes:
        nodes_df = node_params_df[node_params_df[market_col_nodes] == selected_market].copy()
    else:
        nodes_df = node_params_df.copy()

    if market_col_segments:
        segments_df = segment_params_df[segment_params_df[market_col_segments] == selected_market].copy()
    else:
        segments_df = segment_params_df.copy()

    if market_col_trans:
        trans_df = transitions_df[transitions_df[market_col_trans] == selected_market].copy()
    else:
        trans_df = transitions_df.copy()

    if nodes_df.empty:
        raise ValueError(f"No node parameters found for market '{selected_market}'.")
    if segments_df.empty:
        raise ValueError(f"No segment parameters found for market '{selected_market}'.")
    if trans_df.empty:
        raise ValueError(f"No transition rows found for market '{selected_market}'.")

    if order_col:
        nodes_df = nodes_df.sort_values(order_col).copy()
    else:
        nodes_df = nodes_df.copy()

    node_ids = nodes_df[node_id_col].astype(str).tolist()
    node_map = {str(row[node_id_col]): row for _, row in nodes_df.iterrows()}
    start_node = node_ids[0]

    transition_map = {}
    for _, row in trans_df.iterrows():
        from_node = str(row[from_col])
        to_node = str(row[to_col])
        prob = float(row[prob_col])

        if from_node not in transition_map:
            transition_map[from_node] = []
        transition_map[from_node].append((to_node, prob))

    use_vicaempa = _scenario_is_vicaempa(scenario_name)

    segment_node_rows = []
    segment_transition_rows = []

    for _, seg_row in segments_df.iterrows():
        segment_name = str(seg_row[segment_col])
        population_share = float(seg_row[pop_share_col])
        segment_population = total_population * population_share

        seg_delay_multiplier = float(_safe_get_value(seg_row, seg_delay_mult_col, 1.0))
        seg_leakage_multiplier = float(_safe_get_value(seg_row, seg_leakage_mult_col, 1.0))
        seg_cost_multiplier = float(_safe_get_value(seg_row, seg_cost_mult_col, 1.0))

        inflow_map = {node_id: 0.0 for node_id in node_ids}
        inflow_map[start_node] = segment_population

        for node_id in node_ids:
            row = node_map[node_id]
            inflow = float(inflow_map.get(node_id, 0.0))

            if inflow <= 0:
                continue

            base_delay = float(_safe_get_value(row, delay_col, 0.0))
            base_leakage = float(_safe_get_value(row, leakage_col, 0.0))
            base_cost = float(_safe_get_value(row, cost_col, 0.0))

            if use_vicaempa:
                delay_multiplier = float(_safe_get_value(row, vica_delay_mult_col, 1.0))
                leakage_multiplier = float(_safe_get_value(row, vica_leakage_mult_col, 1.0))
                cost_multiplier = float(_safe_get_value(row, vica_cost_mult_col, 1.0))
            else:
                delay_multiplier = 1.0
                leakage_multiplier = 1.0
                cost_multiplier = 1.0

            final_delay = base_delay * seg_delay_multiplier * delay_multiplier
            final_leakage = base_leakage * seg_leakage_multiplier * leakage_multiplier
            final_cost = base_cost * seg_cost_multiplier * cost_multiplier

            final_leakage = max(0.0, min(1.0, final_leakage))

            leakage_patients = inflow * final_leakage
            surviving_patients = inflow - leakage_patients
            node_total_cost = inflow * final_cost
            node_total_delay = inflow * final_delay

            segment_node_rows.append({
                "market": selected_market,
                "scenario": scenario_name,
                "segment": segment_name,
                "node_id": node_id,
                "inflow_patients": inflow,
                "leakage_patients": leakage_patients,
                "surviving_patients": surviving_patients,
                "avg_delay_days": final_delay,
                "total_delay_days": node_total_delay,
                "cost_per_patient": final_cost,
                "total_cost": node_total_cost
            })

            if node_id in transition_map:
                for to_node, probability in transition_map[node_id]:
                    moved_patients = surviving_patients * probability
                    inflow_map[to_node] = inflow_map.get(to_node, 0.0) + moved_patients

                    segment_transition_rows.append({
                        "market": selected_market,
                        "scenario": scenario_name,
                        "segment": segment_name,
                        "from_node": node_id,
                        "to_node": to_node,
                        "probability": probability,
                        "patients_moved": moved_patients
                    })

    node_results_df = pd.DataFrame(segment_node_rows)
    transition_results_df = pd.DataFrame(segment_transition_rows)

    if node_results_df.empty:
        raise ValueError("Model produced no node results.")

    aggregate_node_results_df = (
        node_results_df.groupby(["market", "scenario", "node_id"], as_index=False)
        .agg({
            "inflow_patients": "sum",
            "leakage_patients": "sum",
            "surviving_patients": "sum",
            "total_delay_days": "sum",
            "total_cost": "sum"
        })
    )

    aggregate_node_results_df["avg_delay_days"] = np.where(
        aggregate_node_results_df["inflow_patients"] > 0,
        aggregate_node_results_df["total_delay_days"] / aggregate_node_results_df["inflow_patients"],
        0.0
    )

    aggregate_transition_results_df = (
        transition_results_df.groupby(["market", "scenario", "from_node", "to_node"], as_index=False)
        .agg({
            "patients_moved": "sum",
            "probability": "mean"
        })
    )

    segment_summary_df = (
        node_results_df.groupby(["market", "scenario", "segment"], as_index=False)
        .agg({
            "inflow_patients": "sum",
            "leakage_patients": "sum",
            "surviving_patients": "sum",
            "total_delay_days": "sum",
            "total_cost": "sum"
        })
    )

    return {
        "node_results": node_results_df,
        "transition_results": transition_results_df,
        "aggregate_node_results": aggregate_node_results_df,
        "aggregate_transition_results": aggregate_transition_results_df,
        "segment_summary": segment_summary_df
    }
