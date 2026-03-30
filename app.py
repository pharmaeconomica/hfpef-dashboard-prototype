import streamlit as st
import pandas as pd
from transition_model_v10 import validate_transition_table, run_transition_model_v10
from transition_time_model_v11 import validate_time_risk_table, run_transition_time_model_v11
from ui_redesign_v11 import (
    prepare_client_view,
    render_top_kpis,
    render_story_summary,
    render_pathway_cards,
    render_node_detail,
    render_hidden_technical_section,
)
from page_layout_v11 import (
    render_executive_summary,
    render_detailed_analysis,
)





# --- Logo ---
import base64

def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("assets/logo.png")

st.markdown(
    f"""
    <div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px;">
        <img src="data:image/png;base64,{logo_base64}" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="HFpEF Dashboard Prototype", layout="wide")



# -----------------------------
# Required column definitions
# -----------------------------
REQUIRED_NODE_COLUMNS = [
    "market", "node_id", "node_name", "delay_days", "leakage_rate",
    "cost_per_patient", "vica_delay_multiplier", "vica_leakage_multiplier",
    "vica_cost_multiplier", "unit", "min_value", "max_value", "editable",
    "source_type", "evidence_strength", "parameter_note"
]

REQUIRED_GLOBAL_COLUMNS = [
    "market", "default_population", "currency", "market_note"
]

REQUIRED_CONTROL_COLUMNS = [
    "parameter_name", "market", "label", "control_type", "default_value",
    "min_value", "max_value", "step_value", "editable", "unit",
    "source_type", "evidence_strength", "parameter_note"
]

REQUIRED_SEGMENT_COLUMNS = [
    "market", "segment_id", "segment_name", "population_share",
    "delay_multiplier", "leakage_multiplier", "cost_multiplier",
    "source_type", "evidence_strength", "parameter_note"
]


# -----------------------------
# Built-in file loaders
# -----------------------------
@st.cache_data
def load_node_parameters():
    return pd.read_csv("data/node_parameters.csv")


@st.cache_data
def load_global_parameters():
    return pd.read_csv("data/global_parameters.csv")


@st.cache_data
def load_parameter_controls():
    return pd.read_csv("data/parameter_controls.csv")


@st.cache_data
def load_segment_parameters():
    return pd.read_csv("data/segment_parameters.csv")
transitions_df = pd.read_csv("node_transitions.csv")
time_risk_df = pd.read_csv("time_risk_parameters.csv")


# -----------------------------
# Validation helpers
# -----------------------------
def validate_columns(df, required_columns, file_label):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return [f"{file_label}: missing columns -> {', '.join(missing)}"]
    return []


def validate_node_file(df):
    errors = validate_columns(df, REQUIRED_NODE_COLUMNS, "node_parameters.csv")
    if errors:
        return errors

    if df[["market", "node_id"]].duplicated().any():
        errors.append("node_parameters.csv: duplicate combination of market + node_id found")

    numeric_cols = [
        "delay_days", "leakage_rate", "cost_per_patient",
        "vica_delay_multiplier", "vica_leakage_multiplier",
        "vica_cost_multiplier", "min_value", "max_value"
    ]
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"node_parameters.csv: column '{col}' contains invalid numeric values")

    return errors


def validate_global_file(df):
    errors = validate_columns(df, REQUIRED_GLOBAL_COLUMNS, "global_parameters.csv")
    if errors:
        return errors

    if df["market"].duplicated().any():
        errors.append("global_parameters.csv: duplicate market values found")

    converted = pd.to_numeric(df["default_population"], errors="coerce")
    if converted.isna().any():
        errors.append("global_parameters.csv: default_population contains invalid numeric values")

    return errors


def validate_control_file(df):
    errors = validate_columns(df, REQUIRED_CONTROL_COLUMNS, "parameter_controls.csv")
    if errors:
        return errors

    if df[["market", "parameter_name"]].duplicated().any():
        errors.append("parameter_controls.csv: duplicate combination of market + parameter_name found")

    numeric_cols = ["default_value", "min_value", "max_value", "step_value"]
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"parameter_controls.csv: column '{col}' contains invalid numeric values")

    return errors


def validate_segment_file(df):
    errors = validate_columns(df, REQUIRED_SEGMENT_COLUMNS, "segment_parameters.csv")
    if errors:
        return errors

    if df[["market", "segment_id"]].duplicated().any():
        errors.append("segment_parameters.csv: duplicate combination of market + segment_id found")

    numeric_cols = ["population_share", "delay_multiplier", "leakage_multiplier", "cost_multiplier"]
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"segment_parameters.csv: column '{col}' contains invalid numeric values")

    share_check = df.groupby("market")["population_share"].sum().round(6)
    invalid = share_check[share_check != 1.0]
    if not invalid.empty:
        for market_name, total_share in invalid.items():
            errors.append(f"segment_parameters.csv: population shares for {market_name} sum to {total_share}, not 1.0")

    return errors


def check_market_alignment(node_df, global_df, control_df, segment_df):
    errors = []
    node_markets = set(node_df["market"].unique())
    global_markets = set(global_df["market"].unique())
    control_markets = set(control_df["market"].unique())
    segment_markets = set(segment_df["market"].unique())

    if not node_markets.issubset(global_markets):
        missing = node_markets - global_markets
        errors.append(f"Markets in node_parameters.csv missing from global_parameters.csv -> {', '.join(sorted(missing))}")

    if not control_markets.issubset(global_markets):
        missing = control_markets - global_markets
        errors.append(f"Markets in parameter_controls.csv missing from global_parameters.csv -> {', '.join(sorted(missing))}")

    if not segment_markets.issubset(global_markets):
        missing = segment_markets - global_markets
        errors.append(f"Markets in segment_parameters.csv missing from global_parameters.csv -> {', '.join(sorted(missing))}")

    return errors


# -----------------------------
# Access helpers
# -----------------------------
def get_market_globals(global_df, market):
    row = global_df[global_df["market"] == market]
    if row.empty:
        raise ValueError(f"No global parameters found for market: {market}")
    return row.iloc[0]


def get_market_nodes(node_df, market):
    df = node_df[node_df["market"] == market].copy()
    if df.empty:
        raise ValueError(f"No node parameters found for market: {market}")
    return df


def get_market_controls(control_df, market):
    df = control_df[control_df["market"] == market].copy()
    if df.empty:
        raise ValueError(f"No controls found for market: {market}")
    return df


def get_market_segments(segment_df, market):
    df = segment_df[segment_df["market"] == market].copy()
    if df.empty:
        raise ValueError(f"No segment parameters found for market: {market}")
    return df


def get_control_value(control_row):
    label = control_row.get("control_label", control_row.get("parameter_name", "Parameter"))
    default_value = control_row["default_value"]
    editable = bool(control_row["editable"])
    control_type = control_row["control_type"]
    min_value = control_row["min_value"]
    max_value = control_row["max_value"]
    step_value = control_row["step_value"]

    if not editable:
        return default_value

    if control_type == "slider":
        return st.sidebar.slider(
            label,
            int(min_value),
            int(max_value),
            int(default_value),
            step=int(step_value)
        )

    return default_value


def run_segment_scenario(nodes_df, segment_row, population, delay_factor, leakage_factor, cost_factor, use_vica=False):
    results = []
    segment_population = int(population * float(segment_row["population_share"]))
    patients = segment_population
    total_cost = 0
    total_lost = 0
    cumulative_cost = 0

    seg_delay_mult = float(segment_row["delay_multiplier"])
    seg_leakage_mult = float(segment_row["leakage_multiplier"])
    seg_cost_mult = float(segment_row["cost_multiplier"])

    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        name = row["node_name"]

        delay = row["delay_days"] * (delay_factor / 100) * seg_delay_mult
        leakage = row["leakage_rate"] * (leakage_factor / 100) * seg_leakage_mult
        cost = row["cost_per_patient"] * (cost_factor / 100) * seg_cost_mult

        leakage = min(leakage, 0.95)

        if use_vica:
            delay *= row["vica_delay_multiplier"]
            leakage *= row["vica_leakage_multiplier"]
            cost *= row["vica_cost_multiplier"]

        lost = int(patients * leakage)
        exiting = patients - lost
        node_cost = int(patients * cost)
        cumulative_cost += node_cost
        total_cost += node_cost
        total_lost += lost

        severity_score = leakage * 100 + delay / 5
        if severity_score >= 35:
            severity = "High"
        elif severity_score >= 20:
            severity = "Moderate"
        else:
            severity = "Low"

        results.append({
            "Segment ID": segment_row["segment_id"],
            "Segment": segment_row["segment_name"],
            "Node ID": node_id,
            "Node": name,
            "Patients Entering": patients,
            "Delay (days)": int(delay),
            "Leakage %": round(leakage * 100, 1),
            "Patients Lost": lost,
            "Patients Exiting": exiting,
            "Node Cost": node_cost,
            "Cumulative Cost": cumulative_cost,
            "Severity": severity,
            "Severity Score": round(severity_score, 2),
            "Source Type": row["source_type"],
            "Evidence Strength": row["evidence_strength"],
            "Editable": row["editable"],
            "Parameter Note": row["parameter_note"],
        })

        patients = exiting

    return pd.DataFrame(results), total_cost, total_lost, patients, segment_population


def aggregate_segments(segment_outputs):
    combined_df = pd.concat([x["df"] for x in segment_outputs], ignore_index=True)

    agg_df = (
        combined_df.groupby(["Node ID", "Node"], as_index=False)
        .agg({
            "Patients Entering": "sum",
            "Delay (days)": "mean",
            "Leakage %": "mean",
            "Patients Lost": "sum",
            "Patients Exiting": "sum",
            "Node Cost": "sum",
            "Cumulative Cost": "sum",
            "Severity Score": "mean"
        })
    )

    def classify(score):
        if score >= 35:
            return "High"
        elif score >= 20:
            return "Moderate"
        return "Low"

    agg_df["Severity"] = agg_df["Severity Score"].apply(classify)
    agg_df["Delay (days)"] = agg_df["Delay (days)"].round(0).astype(int)
    agg_df["Leakage %"] = agg_df["Leakage %"].round(1)
    agg_df["Severity Score"] = agg_df["Severity Score"].round(2)

    total_cost = sum(x["total_cost"] for x in segment_outputs)
    total_lost = sum(x["total_lost"] for x in segment_outputs)
    remaining = sum(x["remaining"] for x in segment_outputs)

    segment_summary = pd.DataFrame([
        {
            "Segment ID": x["segment_id"],
            "Segment": x["segment_name"],
            "Population": x["population"],
            "Total Cost": x["total_cost"],
            "Patients Lost": x["total_lost"],
            "Patients Remaining": x["remaining"]
        }
        for x in segment_outputs
    ])

    return agg_df, total_cost, total_lost, remaining, segment_summary, combined_df


def run_market_scenario(nodes_df, segments_df, population, delay_factor, leakage_factor, cost_factor, use_vica=False):
    outputs = []

    for _, seg in segments_df.iterrows():
        seg_df, seg_cost, seg_lost, seg_remaining, seg_pop = run_segment_scenario(
            nodes_df, seg, population, delay_factor, leakage_factor, cost_factor, use_vica=use_vica
        )
        outputs.append({
            "segment_id": seg["segment_id"],
            "segment_name": seg["segment_name"],
            "population": seg_pop,
            "total_cost": seg_cost,
            "total_lost": seg_lost,
            "remaining": seg_remaining,
            "df": seg_df
        })

    return aggregate_segments(outputs)


def get_node_style(severity, node_id):
    if node_id == "N6":
        return "#fff4cc", "2px solid #d97706"
    if severity == "High":
        return "#fee2e2", "1px solid #ef4444"
    if severity == "Moderate":
        return "#fef3c7", "1px solid #f59e0b"
    return "#ecfdf5", "1px solid #10b981"


# -----------------------------
# App header
# -----------------------------
st.markdown("""
<div style="padding-top: 0.25rem; padding-bottom: 0.75rem;">
    <h1 style="margin-bottom: 0.2rem; font-size: 2.2rem;">
        HFpEF Care Pathway Simulation Dashboard
    </h1>
    <p style="margin-top: 0; color: #6b7280; font-size: 1.05rem;">
        Interactive scenario explorer for pathway bottlenecks, delays, leakage, cost burden, and VicaEmpa impact.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Load default files
# -----------------------------
default_node_params = load_node_parameters()
default_global_params = load_global_parameters()
default_control_params = load_parameter_controls()
default_segment_params = load_segment_parameters()

# -----------------------------
# Sidebar: data source selection
# -----------------------------
st.sidebar.header("Model Configuration")

# --- Mode selection ---
mode = st.sidebar.radio(
    "Mode",
    ["Demo Mode", "Advanced Mode"],
    index=0
)

if mode == "Demo Mode":
    st.sidebar.info(
        "Demo Mode uses default assumptions. Switch to Advanced Mode to customize inputs."
    )

st.sidebar.markdown("---")
("Data Source")
use_uploaded_files = st.sidebar.toggle("Use uploaded country template", value=False)

node_params = default_node_params
global_params = default_global_params
control_params = default_control_params
segment_params = default_segment_params

upload_errors = []

if use_uploaded_files:
    st.sidebar.markdown("### Upload template files")

    
    
    
    

    # --- File upload (Advanced Mode only) ---
    if mode == "Advanced Mode":
        uploaded_node_file = st.sidebar.file_uploader("Upload pathway structure (optional)", type=["csv"])
        uploaded_global_file = st.sidebar.file_uploader("Upload population & system context (optional)", type=["csv"])
        uploaded_control_file = st.sidebar.file_uploader("Upload scenario assumptions (optional)", type=["csv"])
        uploaded_segment_file = st.sidebar.file_uploader("Upload patient segmentation (optional)", type=["csv"])
    else:
        uploaded_node_file = None
        uploaded_global_file = None
        uploaded_control_file = None
        uploaded_segment_file = None

    if uploaded_node_file is not None and uploaded_global_file is not None:
        uploaded_node_df = pd.read_csv(uploaded_node_file)
        uploaded_global_df = pd.read_csv(uploaded_global_file)

        upload_errors.extend(validate_node_file(uploaded_node_df))
        upload_errors.extend(validate_global_file(uploaded_global_df))

        if uploaded_control_file is not None:
            uploaded_control_df = pd.read_csv(uploaded_control_file)
            upload_errors.extend(validate_control_file(uploaded_control_df))
        else:
            uploaded_control_df = default_control_params.copy()

        if uploaded_segment_file is not None:
            uploaded_segment_df = pd.read_csv(uploaded_segment_file)
            upload_errors.extend(validate_segment_file(uploaded_segment_df))
        else:
            uploaded_segment_df = default_segment_params.copy()

        if not upload_errors:
            upload_errors.extend(check_market_alignment(uploaded_node_df, uploaded_global_df, uploaded_control_df, uploaded_segment_df))

        if not upload_errors:
            node_params = uploaded_node_df
            global_params = uploaded_global_df
            control_params = uploaded_control_df
            segment_params = uploaded_segment_df
            st.sidebar.success("Uploaded country template validated and loaded.")
        else:
            st.sidebar.error("Uploaded files failed validation. Using built-in defaults.")
    else:
        st.sidebar.info("Upload at least node_parameters.csv and global_parameters.csv to activate country template mode.")

# -----------------------------
# Sidebar: scenario controls
# -----------------------------
st.sidebar.header("Scenario Controls")

available_markets = global_params["market"].tolist()
market = st.sidebar.selectbox("Market", available_markets)

market_globals = get_market_globals(global_params, market)
market_nodes = get_market_nodes(node_params, market)
market_controls = get_market_controls(control_params, market)
market_segments = get_market_segments(segment_params, market)

control_lookup = {}
for _, row in market_controls.iterrows():
    control_lookup[row["parameter_name"]] = get_control_value(row)

population = int(control_lookup["population_size"])
delay_factor = int(control_lookup["delay_factor"])
leakage_factor = int(control_lookup["leakage_factor"])
cost_factor = int(control_lookup["cost_factor"])

show_vica = st.sidebar.toggle("Show VicaEmpa scenario", value=True)

page_view = st.sidebar.radio(
    "View",
    ["Executive Summary", "Detailed Analysis"],
    index=0
)


currency = market_globals["currency"]
market_note = market_globals["market_note"]

# -----------------------------
# Run scenarios
# -----------------------------
base_df, base_cost, base_lost, base_remaining, base_segment_summary, base_detailed_segments = run_market_scenario(
    market_nodes, market_segments, population, delay_factor, leakage_factor, cost_factor, use_vica=False
)

vica_df, vica_cost, vica_lost, vica_remaining, vica_segment_summary, vica_detailed_segments = run_market_scenario(
    market_nodes, market_segments, population, delay_factor, leakage_factor, cost_factor, use_vica=True
)


# -----------------------------
# Version 10 parallel run
# -----------------------------
v10_current_results = run_transition_model_v10(
    node_params_df=node_params,
    segment_params_df=segment_params,
    transitions_df=transitions_df,
    total_population=population,
    selected_market=market,
    scenario_name="Current"
)

v10_vica_results = run_transition_model_v10(
    node_params_df=node_params,
    segment_params_df=segment_params,
    transitions_df=transitions_df,
    total_population=population,
    selected_market=market,
    scenario_name="VicaEmpa"
)

v10_active_results = v10_vica_results if show_vica else v10_current_results
v10_node_results_df = v10_active_results["node_results"]
v10_transition_results_df = v10_active_results["transition_results"]
v10_aggregate_node_results_df = v10_active_results["aggregate_node_results"]
v10_aggregate_transition_results_df = v10_active_results["aggregate_transition_results"]
v10_segment_summary_df = v10_active_results["segment_summary"]

# -----------------------------
# Version 11 parallel run
# -----------------------------
time_risk_validation = validate_time_risk_table(
    time_risk_df=time_risk_df,
    selected_market=market
)

v11_current_results = run_transition_time_model_v11(
    node_params_df=node_params,
    segment_params_df=segment_params,
    transitions_df=transitions_df,
    time_risk_df=time_risk_df,
    total_population=population,
    selected_market=market,
    scenario_name="Current"
)

v11_vica_results = run_transition_time_model_v11(
    node_params_df=node_params,
    segment_params_df=segment_params,
    transitions_df=transitions_df,
    time_risk_df=time_risk_df,
    total_population=population,
    selected_market=market,
    scenario_name="VicaEmpa"
)

v11_active_results = v11_vica_results if show_vica else v11_current_results
v11_node_results_df = v11_active_results["node_results"]
v11_transition_results_df = v11_active_results["transition_results"]
v11_aggregate_node_results_df = v11_active_results["aggregate_node_results"]
v11_aggregate_transition_results_df = v11_active_results["aggregate_transition_results"]
v11_segment_summary_df = v11_active_results["segment_summary"]

# -----------------------------
# Client-ready UI view
# -----------------------------
client_view_df = prepare_client_view(v11_aggregate_node_results_df)



active_df = vica_df if show_vica else base_df
active_cost = vica_cost if show_vica else base_cost
active_lost = vica_lost if show_vica else base_lost
active_remaining = vica_remaining if show_vica else base_remaining
active_segment_summary = vica_segment_summary if show_vica else base_segment_summary

savings = base_cost - vica_cost
avoided_loss = base_lost - vica_lost

top_bottleneck_row = active_df.sort_values("Severity Score", ascending=False).iloc[0]
top_bottleneck_name = f"{top_bottleneck_row['Node ID']} - {top_bottleneck_row['Node']}"
top_bottleneck_delay = int(top_bottleneck_row["Delay (days)"])
top_bottleneck_leakage = float(top_bottleneck_row["Leakage %"])

delta_df = base_df[["Node ID", "Node", "Node Cost", "Patients Lost", "Delay (days)", "Leakage %"]].merge(
    vica_df[["Node ID", "Node Cost", "Patients Lost", "Delay (days)", "Leakage %"]],
    on="Node ID",
    suffixes=("_Base", "_Vica")
)

delta_df["Cost Delta"] = delta_df["Node Cost_Base"] - delta_df["Node Cost_Vica"]
delta_df["Patients Lost Delta"] = delta_df["Patients Lost_Base"] - delta_df["Patients Lost_Vica"]
delta_df["Delay Delta"] = delta_df["Delay (days)_Base"] - delta_df["Delay (days)_Vica"]
delta_df["Leakage Delta"] = delta_df["Leakage %_Base"] - delta_df["Leakage %_Vica"]

value_df = delta_df[[
    "Node ID",
    "Node",
    "Cost Delta",
    "Patients Lost Delta",
    "Delay Delta",
    "Leakage Delta"
]].copy().sort_values("Cost Delta", ascending=False)

# -----------------------------
# Top metrics
# -----------------------------
with st.expander("Legacy prototype sections", expanded=False):
    col1, col2, col3, col4 = st.columns(4)



# =============================
# PAGE-BASED CLIENT UI
# =============================
if page_view == "Executive Summary":
    render_executive_summary(
        client_view_df=client_view_df,
        v11_transition_results_df=v11_transition_results_df,
        currency=currency,
        show_vica=show_vica,
        v11_current_results=v11_current_results,
        v11_vica_results=v11_vica_results,
    )
else:
    render_detailed_analysis(
        v11_aggregate_node_results_df=v11_aggregate_node_results_df,
        v11_transition_results_df=v11_transition_results_df,
        v11_segment_summary_df=v11_segment_summary_df,
        client_view_df=client_view_df,
        currency=currency,
        show_vica=show_vica,
        v11_current_results=v11_current_results,
        v11_vica_results=v11_vica_results,
    )

st.markdown("---")
st.stop()


col1.metric("Current View Cost", f"{currency}{active_cost:,.0f}")
col2.metric("Patients Remaining", f"{active_remaining:,}")
col3.metric("Potential Savings", f"{currency}{savings:,.0f}")
col4.metric("Avoided Patient Loss", f"{avoided_loss:,}")






with st.expander("Removed empty block"):
    st.write("Removed during cleanup")
st.markdown("---")

# -----------------------------
# Template status
# -----------------------------
st.subheader("Country Template Status")

s1, s2 = st.columns([1, 2])

with s1:
    st.metric("Template Mode", "Uploaded" if use_uploaded_files and not upload_errors else "Built-in")
    st.metric("Active Market", market)

with s2:
    if upload_errors:
        st.error("Validation issues detected:")
        for err in upload_errors:
            st.write(f"- {err}")
    else:
        st.success("No validation issues detected for active data source.")

st.markdown("---")

# -----------------------------
# Market summary
# -----------------------------
st.subheader("Market Summary")
m1, m2 = st.columns([2, 1])

with m1:
    st.markdown(
        f"""
        <div style="
            background:#f8fafc;
            border:1px solid #cbd5e1;
            border-radius:16px;
            padding:18px;
            min-height:120px;
        ">
            <div style="font-size:13px;color:#475569;">Active market</div>
            <div style="font-size:22px;font-weight:700;margin-top:6px;">{market}</div>
            <div style="margin-top:8px;">{market_note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.metric("Scenario Mode", "VicaEmpa" if show_vica else "Current State")
    st.metric("Population Input", f"{population:,}")

st.markdown("---")

# -----------------------------
# Segment summary
# -----------------------------
st.subheader("Segmented Population Summary")

seg1, seg2 = st.columns(2)

with seg1:
    st.markdown("### Segment Definitions")
    st.dataframe(
        market_segments[[
            "segment_id", "segment_name", "population_share",
            "delay_multiplier", "leakage_multiplier", "cost_multiplier",
            "source_type", "evidence_strength"
        ]],
        use_container_width=True,
        hide_index=True
    )

with seg2:
    st.markdown("### Active Scenario by Segment")
    st.dataframe(active_segment_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# Bottleneck summary
# -----------------------------
st.subheader("Top Bottleneck Summary")
b1, b2, b3 = st.columns([2, 1, 1])

with b1:
    st.markdown(
        f"""
        <div style="
            background:#fff7ed;
            border:1px solid #fdba74;
            border-radius:16px;
            padding:18px;
            min-height:120px;
        ">
            <div style="font-size:13px;color:#9a3412;">Highest friction node</div>
            <div style="font-size:22px;font-weight:700;margin-top:6px;">{top_bottleneck_name}</div>
            <div style="margin-top:8px;">
                This node currently appears to create the greatest combined friction in the pathway based on delay and leakage across segments.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with b2:
    st.metric("Delay at Bottleneck", f"{top_bottleneck_delay} days")

with b3:
    st.metric("Leakage at Bottleneck", f"{top_bottleneck_leakage:.1f}%")

st.markdown("---")

# -----------------------------
# Pathway view
# -----------------------------
st.subheader("Aggregated Pathway View")
node_cols = st.columns(len(active_df))

for i, row in active_df.iterrows():
    bg, border = get_node_style(row["Severity"], row["Node ID"])

    with node_cols[i]:
        st.markdown(
            f"""
            <div style="
                background:{bg};
                border:{border};
                border-radius:14px;
                padding:12px;
                min-height:260px;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);
            ">
                <div style="font-size:12px;color:#666;">{row['Node ID']}</div>
                <div style="font-size:16px;font-weight:700;margin-bottom:10px;">{row['Node']}</div>
                <div><b>Entering:</b> {int(row['Patients Entering']):,}</div>
                <div><b>Delay:</b> {row['Delay (days)']} days</div>
                <div><b>Leakage:</b> {row['Leakage %']}%</div>
                <div><b>Lost:</b> {int(row['Patients Lost']):,}</div>
                <div><b>Node Cost:</b> {currency}{int(row['Node Cost']):,}</div>
                <div><b>Cumulative:</b> {currency}{int(row['Cumulative Cost']):,}</div>
                <div><b>Severity:</b> {row['Severity']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption("Green = lower friction, amber = moderate friction, red = high friction, highlighted amber = flagship node for demo focus.")

st.markdown("---")

# -----------------------------
# Scenario comparison
# -----------------------------
st.subheader("Scenario Comparison")
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Current State")
    st.metric("Total Cost", f"{currency}{base_cost:,.0f}")
    st.metric("Patients Lost", f"{base_lost:,}")
    st.metric("Patients Remaining", f"{base_remaining:,}")

with c2:
    st.markdown("### With VicaEmpa")
    st.metric("Total Cost", f"{currency}{vica_cost:,.0f}")
    st.metric("Patients Lost", f"{vica_lost:,}")
    st.metric("Patients Remaining", f"{vica_remaining:,}")

st.markdown("---")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Node Analytics")
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("### Node Cost by Step")
    st.bar_chart(active_df[["Node ID", "Node Cost"]].set_index("Node ID"))

with chart2:
    st.markdown("### Leakage by Step")
    st.bar_chart(active_df[["Node ID", "Leakage %"]].set_index("Node ID"))

st.markdown("---")

# -----------------------------
# Segment analytics
# -----------------------------
st.subheader("Segment Analytics")

sa1, sa2 = st.columns(2)

with sa1:
    st.markdown("### Cost by Segment")
    seg_cost_chart = active_segment_summary[["Segment", "Total Cost"]].set_index("Segment")
    st.bar_chart(seg_cost_chart)

with sa2:
    st.markdown("### Patients Remaining by Segment")
    seg_remaining_chart = active_segment_summary[["Segment", "Patients Remaining"]].set_index("Segment")
    st.bar_chart(seg_remaining_chart)

st.markdown("---")

# -----------------------------
# Value view
# -----------------------------
st.subheader("Where VicaEmpa Creates Value")
v1, v2 = st.columns(2)

with v1:
    st.markdown("### Node Delta View")
    st.dataframe(value_df, use_container_width=True, hide_index=True)

with v2:
    st.markdown("### Cost Delta by Node")
    st.bar_chart(value_df[["Node ID", "Cost Delta"]].set_index("Node ID"))

st.markdown("---")

# -----------------------------
# Template preview
# -----------------------------
st.subheader("Country Template Preview")

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown("### Node Parameters")
    st.dataframe(market_nodes.head(10), use_container_width=True, hide_index=True)

with p2:
    st.markdown("### Global Parameters")
    st.dataframe(global_params, use_container_width=True, hide_index=True)

with p3:
    st.markdown("### Control Parameters")
    st.dataframe(market_controls, use_container_width=True, hide_index=True)

with p4:
    st.markdown("### Segment Parameters")
    st.dataframe(market_segments, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# Download example templates
# -----------------------------
st.subheader("Download Example Templates")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        label="Download example node_parameters.csv",
        data=default_node_params.to_csv(index=False).encode("utf-8"),
        file_name="example_node_parameters.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        label="Download example global_parameters.csv",
        data=default_global_params.to_csv(index=False).encode("utf-8"),
        file_name="example_global_parameters.csv",
        mime="text/csv"
    )

with d3:
    st.download_button(
        label="Download example parameter_controls.csv",
        data=default_control_params.to_csv(index=False).encode("utf-8"),
        file_name="example_parameter_controls.csv",
        mime="text/csv"
    )

with d4:
    st.download_button(
        label="Download example segment_parameters.csv",
        data=default_segment_params.to_csv(index=False).encode("utf-8"),
        file_name="example_segment_parameters.csv",
        mime="text/csv"
    )

st.markdown("---")

# -----------------------------
# Detailed active tables
# -----------------------------
st.subheader("Detailed Active Scenario Table")
display_df = active_df.drop(columns=["Severity Score"])
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    label="Download active aggregated scenario as CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name=f"hfpef_dashboard_{market.lower()}_{'vica' if show_vica else 'base'}_aggregated.csv",
    mime="text/csv"
)

st.subheader("Detailed Segment-Level Scenario Table")
st.dataframe(active_segment_summary, use_container_width=True, hide_index=True)

st.download_button(
    label="Download active segment summary as CSV",
    data=active_segment_summary.to_csv(index=False).encode("utf-8"),
    file_name=f"hfpef_dashboard_{market.lower()}_{'vica' if show_vica else 'base'}_segments.csv",
    mime="text/csv"
)

st.markdown("---")

with st.expander("Presenter Notes"):
    st.write("1. Explain that the model now reflects heterogeneity by splitting the population into pathway segments.")
    st.write("2. Show that each segment has different delay, leakage, and cost behavior.")
    st.write("3. Emphasize that this is a bridge between a simple cohort model and a future DES engine.")
    st.write("4. Use the segment summary to show how burden differs by patient archetype.")
    st.write("5. Position this as a meaningful realism upgrade aligned with the pathway intent of the RFP.")