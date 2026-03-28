import streamlit as st
import pandas as pd

st.set_page_config(page_title="HFpEF Dashboard Prototype", layout="wide")

st.title("HFpEF Care Pathway Simulation Dashboard")
st.caption("Rapid prototype for client demonstration")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Scenario Controls")

market = st.sidebar.selectbox("Market", ["US", "Germany"])
population = st.sidebar.slider("Population size", 100000, 5000000, 1000000, step=100000)
delay_factor = st.sidebar.slider("Delay intensity (%)", 50, 150, 100, step=5)
leakage_factor = st.sidebar.slider("Leakage intensity (%)", 50, 150, 100, step=5)
cost_factor = st.sidebar.slider("Cost intensity (%)", 50, 150, 100, step=5)
show_vica = st.sidebar.toggle("Show VicaEmpa scenario", value=True)

# -----------------------------
# Base node data
# -----------------------------
if market == "US":
    nodes = [
        ("N1", "Symptom Onset", 90, 0.12, 220),
        ("N2", "Risk Screening", 21, 0.18, 140),
        ("N3", "Specialist Referral", 28, 0.14, 260),
        ("N4", "Diagnosis Confirmation", 20, 0.10, 340),
        ("N5", "Treatment Initiation", 18, 0.12, 420),
        ("N6", "Clinical Inertia", 120, 0.28, 680),
        ("N7", "Hospital Transition", 14, 0.16, 710),
        ("N8", "Adherence & Monitoring", 30, 0.20, 390),
        ("N9", "HHF Event & Readmission", 10, 0.08, 14000),
    ]
    currency = "$"
    market_note = "US base case prototype. Designed to represent a high-cost, higher-volume pathway with substantial downstream event burden."
else:
    nodes = [
        ("N1", "Symptom Onset", 85, 0.10, 180),
        ("N2", "Risk Screening", 25, 0.20, 120),
        ("N3", "Specialist Referral", 30, 0.15, 210),
        ("N4", "Diagnosis Confirmation", 20, 0.11, 300),
        ("N5", "Treatment Initiation", 11, 0.10, 350),
        ("N6", "Clinical Inertia", 95, 0.25, 590),
        ("N7", "Hospital Transition", 18, 0.18, 620),
        ("N8", "Adherence & Monitoring", 25, 0.08, 270),
        ("N9", "HHF Event & Readmission", 9, 0.07, 9000),
    ]
    currency = "€"
    market_note = "Germany adaptation prototype. Designed to show how the same interaction layer can run on a different country parameter set."

# -----------------------------
# Scenario engine
# -----------------------------
def run_scenario(use_vica=False):
    results = []
    patients = population
    total_cost = 0
    total_lost = 0
    cumulative_cost = 0

    for node in nodes:
        node_id, name, delay, leakage, cost = node

        delay = delay * (delay_factor / 100)
        leakage = leakage * (leakage_factor / 100)
        cost = cost * (cost_factor / 100)

        if use_vica and node_id in ["N5", "N6", "N7", "N8", "N9"]:
            leakage *= 0.80
            cost *= 0.85
            delay *= 0.85

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
        })

        patients = exiting

    return pd.DataFrame(results), total_cost, total_lost, patients

base_df, base_cost, base_lost, base_remaining = run_scenario(use_vica=False)
vica_df, vica_cost, vica_lost, vica_remaining = run_scenario(use_vica=True)

active_df = vica_df if show_vica else base_df
active_cost = vica_cost if show_vica else base_cost
active_lost = vica_lost if show_vica else base_lost
active_remaining = vica_remaining if show_vica else base_remaining

savings = base_cost - vica_cost
avoided_loss = base_lost - vica_lost

# -----------------------------
# Top bottleneck
# -----------------------------
top_bottleneck_row = active_df.sort_values("Severity Score", ascending=False).iloc[0]
top_bottleneck_name = f"{top_bottleneck_row['Node ID']} - {top_bottleneck_row['Node']}"
top_bottleneck_delay = int(top_bottleneck_row["Delay (days)"])
top_bottleneck_leakage = float(top_bottleneck_row["Leakage %"])

# -----------------------------
# Delta tables
# -----------------------------
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
]].copy()

value_df = value_df.sort_values("Cost Delta", ascending=False)

# -----------------------------
# Summary row
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current View Cost", f"{currency}{active_cost:,.0f}")
col2.metric("Patients Remaining", f"{active_remaining:,}")
col3.metric("Potential Savings", f"{currency}{savings:,.0f}")
col4.metric("Avoided Patient Loss", f"{avoided_loss:,}")

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
                This node currently appears to create the greatest combined friction in the pathway based on delay and leakage.
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
st.subheader("Pathway View")

node_cols = st.columns(len(active_df))

def get_node_style(severity, node_id):
    if node_id == "N6":
        return "#fff4cc", "2px solid #d97706"
    if severity == "High":
        return "#fee2e2", "1px solid #ef4444"
    if severity == "Moderate":
        return "#fef3c7", "1px solid #f59e0b"
    return "#ecfdf5", "1px solid #10b981"

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
                min-height:235px;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);
            ">
                <div style="font-size:12px;color:#666;">{row['Node ID']}</div>
                <div style="font-size:16px;font-weight:700;margin-bottom:10px;">{row['Node']}</div>
                <div><b>Entering:</b> {row['Patients Entering']:,}</div>
                <div><b>Delay:</b> {row['Delay (days)']} days</div>
                <div><b>Leakage:</b> {row['Leakage %']}%</div>
                <div><b>Lost:</b> {row['Patients Lost']:,}</div>
                <div><b>Node Cost:</b> {currency}{row['Node Cost']:,}</div>
                <div><b>Cumulative:</b> {currency}{row['Cumulative Cost']:,}</div>
                <div><b>Severity:</b> {row['Severity']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption("Green = lower friction, amber = moderate friction, red = high friction, highlighted amber = flagship node for demo focus.")

st.markdown("---")

# -----------------------------
# Comparison section
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
    cost_chart_df = active_df[["Node ID", "Node Cost"]].copy().set_index("Node ID")
    st.bar_chart(cost_chart_df)

with chart2:
    st.markdown("### Leakage by Step")
    leakage_chart_df = active_df[["Node ID", "Leakage %"]].copy().set_index("Node ID")
    st.bar_chart(leakage_chart_df)

st.markdown("---")

# -----------------------------
# Value creation view
# -----------------------------
st.subheader("Where VicaEmpa Creates Value")

v1, v2 = st.columns(2)

with v1:
    st.markdown("### Node Delta View")
    st.dataframe(value_df, use_container_width=True, hide_index=True)

with v2:
    st.markdown("### Cost Delta by Node")
    delta_chart_df = value_df[["Node ID", "Cost Delta"]].copy().set_index("Node ID")
    st.bar_chart(delta_chart_df)

st.markdown("---")

# -----------------------------
# Side-by-side node comparison
# -----------------------------
st.subheader("Side-by-Side Node Comparison")

compare_nodes_df = base_df[["Node ID", "Node", "Node Cost", "Patients Lost", "Delay (days)", "Leakage %"]].merge(
    vica_df[["Node ID", "Node Cost", "Patients Lost", "Delay (days)", "Leakage %"]],
    on="Node ID",
    suffixes=(" - Current", " - VicaEmpa")
)

st.dataframe(compare_nodes_df, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------
# Detailed active table
# -----------------------------
st.subheader("Detailed Active Scenario Table")

display_df = active_df.drop(columns=["Severity Score"])
st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download active scenario as CSV",
    data=csv_data,
    file_name=f"hfpef_dashboard_{market.lower()}_{'vica' if show_vica else 'base'}.csv",
    mime="text/csv"
)

st.markdown("---")

# -----------------------------
# Presenter notes
# -----------------------------
with st.expander("Presenter Notes"):
    st.write("1. Start with the market summary to anchor the audience.")
    st.write("2. Move to the Top Bottleneck Summary and explain where the greatest friction sits.")
    st.write("3. Walk across the full pathway view.")
    st.write("4. Compare current state versus VicaEmpa.")
    st.write("5. Use the delta view to explain where value is created node by node.")
    st.write("6. Use the side-by-side comparison table to show operational differences across the pathway.")
    st.write("7. Position this as a rapid prototype of the interaction layer, not the final DES model.")
