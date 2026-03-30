SEGMENT_EXPLANATIONS = {
    "Delayed Diagnosis Pathway": {
        "plain_english": "Patients whose symptoms are not recognized early, so they spend too long before diagnosis.",
        "main_nodes": "Main bottleneck nodes: N3–N4",
        "clinical_logic": "The system fails during referral and diagnosis, which increases future hospitalization risk."
    },
    "High Inertia Pathway": {
        "plain_english": "Patients who are diagnosed, but treatment is delayed or not escalated appropriately.",
        "main_nodes": "Main bottleneck nodes: N5–N6",
        "clinical_logic": "The system knows the patient has HFpEF, but action is too slow or too weak."
    },
    "Lower Friction Pathway": {
        "plain_english": "Patients who move relatively smoothly through diagnosis, treatment, and follow-up.",
        "main_nodes": "Main bottleneck nodes: none dominant",
        "clinical_logic": "This is the closest thing to a best-case or benchmark pathway."
    },
    "Post-Hospital High Risk": {
        "plain_english": "Patients who already had a hospitalization and remain unstable after discharge.",
        "main_nodes": "Main bottleneck nodes: N7–N9",
        "clinical_logic": "These patients are expensive and vulnerable, with high risk of repeat events."
    },
}


def get_segment_explanation(segment_name):
    return SEGMENT_EXPLANATIONS.get(
        segment_name,
        {
            "plain_english": "No explanation available.",
            "main_nodes": "Not defined",
            "clinical_logic": "Not defined"
        }
    )
