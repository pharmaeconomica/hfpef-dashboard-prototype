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


def _safe_get_value(row, col_name, default=0.0):
    if col_name is None:
        return default
    value = row.get(col_name, default)
    if pd.isna(value):
        return default
    return value


def _scenario_is_vicaempa(scenario_name):
    if scenario_name is None:
        return False
    return str(scenario_name).strip().lower() == "vicaempa"


def validate_time_risk_table(time_risk_df, selected_market):
    market_col = _find_column(time_risk_df, ["market"], required=False)
    target_node_col = _find_column(time_risk_df, ["target_node"])
    base_risk_col = _find_column(time_risk_df, ["base_risk_per_patient"])
    threshold_col = _find_column(time_risk_df, ["delay_threshold_days"])
    risk_increase_col = _find_column(time_risk_df, ["risk_increase_per_30d"])
    cost_increase_col = _find_column(time_risk_df, ["cost_increase_per_30d"])

    if market_col:
        df = time_risk_df[time_risk_df[market_col] == selected_market].copy()
    else:
        df = time_risk_df.copy()

    if df.empty:
        raise ValueError(f"No time-risk rows found for market '{selected_market}'.")

    for col in [base_risk_col, threshold_col, risk_increase_col, cost_increase_col]:
        if (pd.to_numeric(df[col], errors="coerce").isna()).any():
            raise ValueError(f"Column '{col}' contains invalid numeric values.")

    if ((df[base_risk_col] < 0) | (df[base_risk_col] > 1)).any():
        raise ValueError("base_risk_per_patient must be between 0 and 1.")

    if (df[threshold_col] < 0).any():
        raise ValueError("delay_threshold_days must be 0 or more.")

    if (df[risk_increase_col] < 0).any():
        raise ValueError("risk_increase_per_30d must be 0 or more.")

    if (df[cost_increase_col] < 0).any():
        raise ValueError("cost_increase_per_30d must be 0 or more.")

    return {"is_valid": True}


def run_transition_time_model_v11(
    node_params_df,
    segment_params_df,
    transitions_df,
    time_risk_df,
    total_population,
    selected_market,
    scenario_name="Current",
    delay_factor=100,
    leakage_factor=100,
    cost_factor=100
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

    market_col_risk = _find_column(time_risk_df, ["market"], required=False)
    target_node_col = _find_column(time_risk_df, ["target_node"])
    base_risk_col = _find_column(time_risk_df, ["base_risk_per_patient"])
    threshold_col = _find_column(time_risk_df, ["delay_threshold_days"])
    risk_increase_col = _find_column(time_risk_df, ["risk_increase_per_30d"])
    cost_increase_col = _find_column(time_risk_df, ["cost_increase_per_30d"])

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

    if market_col_risk:
        risk_df = time_risk_df[time_risk_df[market_col_risk] == selected_market].copy()
    else:
        risk_df = time_risk_df.copy()

    if nodes_df.empty:
        raise ValueError(f"No node parameters found for market '{selected_market}'.")
    if segments_df.empty:
        raise ValueError(f"No segment parameters found for market '{selected_market}'.")
    if trans_df.empty:
        raise ValueError(f"No transition rows found for market '{selected_market}'.")
    if risk_df.empty:
        raise ValueError(f"No time-risk parameters found for market '{selected_market}'.")

    if order_col:
        nodes_df = nodes_df.sort_values(order_col).copy()

    node_ids = nodes_df[node_id_col].astype(str).tolist()
    node_map = {str(row[node_id_col]): row for _, row in nodes_df.iterrows()}
    start_node = node_ids[0]

    transition_map = {}
    for _, row in trans_df.iterrows():
        source = str(row[from_col])
        target = str(row[to_col])
        prob = float(row[prob_col])
        transition_map.setdefault(source, []).append((target, prob))

    risk_map = {}
    for _, row in risk_df.iterrows():
        target_node = str(row[target_node_col])
        risk_map[target_node] = {
            "base_risk_per_patient": float(row[base_risk_col]),
            "delay_threshold_days": float(row[threshold_col]),
            "risk_increase_per_30d": float(row[risk_increase_col]),
            "cost_increase_per_30d": float(row[cost_increase_col]),
        }

    use_vica = _scenario_is_vicaempa(scenario_name)

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
        cumulative_delay_map = {node_id: 0.0 for node_id in node_ids}
        inflow_map[start_node] = segment_population
        cumulative_delay_map[start_node] = 0.0

        for node_id in node_ids:
            row = node_map[node_id]
            inflow = float(inflow_map.get(node_id, 0.0))

            if inflow <= 0:
                continue

            incoming_cumulative_delay = float(cumulative_delay_map.get(node_id, 0.0))

            base_delay = float(_safe_get_value(row, delay_col, 0.0))
            base_leakage = float(_safe_get_value(row, leakage_col, 0.0))
            base_cost = float(_safe_get_value(row, cost_col, 0.0))

            if use_vica:
                delay_multiplier = float(_safe_get_value(row, vica_delay_mult_col, 1.0))
                leakage_multiplier = float(_safe_get_value(row, vica_leakage_mult_col, 1.0))
                cost_multiplier = float(_safe_get_value(row, vica_cost_mult_col, 1.0))
            else:
                delay_multiplier = 1.0
                leakage_multiplier = 1.0
                cost_multiplier = 1.0

            final_delay = base_delay * (delay_factor / 100.0) * seg_delay_multiplier * delay_multiplier
            final_leakage = base_leakage * (leakage_factor / 100.0) * seg_leakage_multiplier * leakage_multiplier
            final_cost = base_cost * (cost_factor / 100.0) * seg_cost_multiplier * cost_multiplier

            final_leakage = max(0.0, min(1.0, final_leakage))

            cumulative_delay_after_node = incoming_cumulative_delay + final_delay

            adjusted_event_risk = np.nan
            expected_events = 0.0
            excess_events_from_delay = 0.0
            delay_cost_multiplier = 1.0

            if node_id in risk_map:
                base_risk = risk_map[node_id]["base_risk_per_patient"]
                delay_threshold = risk_map[node_id]["delay_threshold_days"]
                risk_increase_per_30d = risk_map[node_id]["risk_increase_per_30d"]
                cost_increase_per_30d = risk_map[node_id]["cost_increase_per_30d"]

                excess_delay = max(0.0, cumulative_delay_after_node - delay_threshold)
                delay_units = excess_delay / 30.0

                adjusted_event_risk = base_risk + (delay_units * risk_increase_per_30d)
                adjusted_event_risk = max(0.0, min(1.0, adjusted_event_risk))

                expected_events = inflow * adjusted_event_risk
                excess_events_from_delay = inflow * max(0.0, adjusted_event_risk - base_risk)

                delay_cost_multiplier = 1.0 + (delay_units * cost_increase_per_30d)

            node_total_cost = inflow * final_cost * delay_cost_multiplier
            leakage_patients = inflow * final_leakage
            surviving_patients = inflow - leakage_patients
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
                "cumulative_delay_days": cumulative_delay_after_node,
                "total_delay_days": node_total_delay,
                "cost_per_patient": final_cost,
                "delay_cost_multiplier": delay_cost_multiplier,
                "total_cost": node_total_cost,
                "adjusted_event_risk": adjusted_event_risk,
                "expected_events": expected_events,
                "excess_events_from_delay": excess_events_from_delay
            })

            if node_id in transition_map:
                for to_node, probability in transition_map[node_id]:
                    moved_patients = surviving_patients * probability

                    inflow_map[to_node] = inflow_map.get(to_node, 0.0) + moved_patients

                    existing_delay = cumulative_delay_map.get(to_node, 0.0)
                    if inflow_map[to_node] > 0:
                        # weighted average cumulative delay when branches merge
                        previous_patients = inflow_map[to_node] - moved_patients
                        total_patients = previous_patients + moved_patients

                        if total_patients > 0:
                            weighted_delay = (
                                (existing_delay * previous_patients) +
                                (cumulative_delay_after_node * moved_patients)
                            ) / total_patients
                        else:
                            weighted_delay = cumulative_delay_after_node
                    else:
                        weighted_delay = cumulative_delay_after_node

                    cumulative_delay_map[to_node] = weighted_delay

                    segment_transition_rows.append({
                        "market": selected_market,
                        "scenario": scenario_name,
                        "segment": segment_name,
                        "from_node": node_id,
                        "to_node": to_node,
                        "probability": probability,
                        "patients_moved": moved_patients,
                        "cumulative_delay_sent": cumulative_delay_after_node
                    })

    node_results_df = pd.DataFrame(segment_node_rows)
    transition_results_df = pd.DataFrame(segment_transition_rows)

    aggregate_node_results_df = (
        node_results_df.groupby(["market", "scenario", "node_id"], as_index=False)
        .agg({
            "inflow_patients": "sum",
            "leakage_patients": "sum",
            "surviving_patients": "sum",
            "total_delay_days": "sum",
            "total_cost": "sum",
            "expected_events": "sum",
            "excess_events_from_delay": "sum"
        })
    )

    weighted_delay_df = (
        node_results_df.groupby(["market", "scenario", "node_id"], as_index=False)
        .apply(
            lambda x: pd.Series({
                "avg_delay_days": (x["avg_delay_days"] * x["inflow_patients"]).sum() / x["inflow_patients"].sum() if x["inflow_patients"].sum() > 0 else 0.0,
                "cumulative_delay_days": (x["cumulative_delay_days"] * x["inflow_patients"]).sum() / x["inflow_patients"].sum() if x["inflow_patients"].sum() > 0 else 0.0,
                "adjusted_event_risk": (x["adjusted_event_risk"].fillna(0) * x["inflow_patients"]).sum() / x["inflow_patients"].sum() if x["inflow_patients"].sum() > 0 else 0.0
            })
        )
        .reset_index()
    )

    aggregate_node_results_df = aggregate_node_results_df.merge(
        weighted_delay_df,
        on=["market", "scenario", "node_id"],
        how="left"
    )

    aggregate_transition_results_df = (
        transition_results_df.groupby(["market", "scenario", "from_node", "to_node"], as_index=False)
        .agg({
            "patients_moved": "sum",
            "probability": "mean",
            "cumulative_delay_sent": "mean"
        })
    )

    segment_summary_df = (
        node_results_df.groupby(["market", "scenario", "segment"], as_index=False)
        .agg({
            "inflow_patients": "sum",
            "leakage_patients": "sum",
            "surviving_patients": "sum",
            "total_delay_days": "sum",
            "total_cost": "sum",
            "expected_events": "sum",
            "excess_events_from_delay": "sum"
        })
    )

    return {
        "node_results": node_results_df,
        "transition_results": transition_results_df,
        "aggregate_node_results": aggregate_node_results_df,
        "aggregate_transition_results": aggregate_transition_results_df,
        "segment_summary": segment_summary_df
    }
