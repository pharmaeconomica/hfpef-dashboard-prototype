import pandas as pd


DEFAULT_SEGMENTS = [
    "Delayed Diagnosis Pathway",
    "High Inertia Pathway",
    "Lower Friction Pathway",
    "Post-Hospital High Risk",
]


def load_segment_transition_parameters(csv_path="segment_transition_parameters.csv"):
    expected_cols = [
        "from_segment",
        "to_segment",
        "annual_transition_rate",
        "clinical_rationale",
        "enabled",
    ]

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=expected_cols)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols].copy()
    df["annual_transition_rate"] = pd.to_numeric(df["annual_transition_rate"], errors="coerce").fillna(0.0)
    df["enabled"] = pd.to_numeric(df["enabled"], errors="coerce").fillna(0).astype(int)

    return df


def get_active_segment_transitions(csv_path="segment_transition_parameters.csv"):
    df = load_segment_transition_parameters(csv_path)
    if df.empty:
        return df
    return df[df["enabled"] == 1].copy()


def build_segment_transition_matrix(csv_path="segment_transition_parameters.csv", segment_names=None):
    if segment_names is None:
        segment_names = DEFAULT_SEGMENTS

    active_df = get_active_segment_transitions(csv_path)
    matrix = pd.DataFrame(0.0, index=segment_names, columns=segment_names)

    if active_df.empty:
        return matrix

    for _, row in active_df.iterrows():
        from_seg = row["from_segment"]
        to_seg = row["to_segment"]
        rate = float(row["annual_transition_rate"])

        if from_seg not in matrix.index:
            matrix.loc[from_seg] = 0.0
        if to_seg not in matrix.columns:
            matrix[to_seg] = 0.0

        matrix.loc[from_seg, to_seg] = rate

    return matrix.fillna(0.0)


def summarize_segment_evolution(csv_path="segment_transition_parameters.csv"):
    active_df = get_active_segment_transitions(csv_path)

    if active_df.empty:
        return pd.DataFrame(columns=[
            "From Segment",
            "To Segment",
            "Annual Transition Rate",
            "Clinical Interpretation",
        ])

    summary = active_df[[
        "from_segment",
        "to_segment",
        "annual_transition_rate",
        "clinical_rationale",
    ]].rename(columns={
        "from_segment": "From Segment",
        "to_segment": "To Segment",
        "annual_transition_rate": "Annual Transition Rate",
        "clinical_rationale": "Clinical Interpretation",
    }).copy()

    summary["Annual Transition Rate"] = (summary["Annual Transition Rate"] * 100).round(1).astype(str) + "%"

    return summary.reset_index(drop=True)


def build_segment_evolution_chart_data(csv_path="segment_transition_parameters.csv"):
    active_df = get_active_segment_transitions(csv_path)

    if active_df.empty:
        return pd.DataFrame(columns=["transition_label", "annual_transition_rate"])

    chart_df = active_df.copy()
    chart_df["transition_label"] = chart_df["from_segment"] + " → " + chart_df["to_segment"]

    return chart_df[["transition_label", "annual_transition_rate"]].sort_values(
        by="annual_transition_rate",
        ascending=False
    ).reset_index(drop=True)
