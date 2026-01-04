"""
Privacy-Safe Cross-Organization Analytics Prototype
--------------------------------------------------
• Single-file Flask app (frontend + backend)
• Snowflake Clean Room–style governed analytics
• Aggregate-only, pre-approved queries
• AI-style explainability
• Real Snowflake integration (optional, env-based)
"""

import os
from flask import Flask, render_template_string, jsonify
from typing import List, Dict

# ============================================================
# Optional Snowflake Integration
# ============================================================

USE_SNOWFLAKE = False
SNOWFLAKE_ERROR = None

try:
    import snowflake.connector
    USE_SNOWFLAKE = all([
        os.getenv("SNOWFLAKE_ACCOUNT"),
        os.getenv("SNOWFLAKE_USER"),
        os.getenv("SNOWFLAKE_PASSWORD"),
        os.getenv("SNOWFLAKE_WAREHOUSE"),
        os.getenv("SNOWFLAKE_DATABASE"),
        os.getenv("SNOWFLAKE_SCHEMA")
    ])
except Exception as e:
    SNOWFLAKE_ERROR = str(e)

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

# ============================================================
# Mock De-Identified Data (Fallback / Demo Mode)
# ============================================================

BANK_DATA = {
    "18-25": {"customers": 200, "defaults": 12},
    "26-35": {"customers": 300, "defaults": 18},
    "36-50": {"customers": 280, "defaults": 25},
    "50+": {"customers": 150, "defaults": 10}
}

INSURANCE_DATA = {
    "18-25": {"claims": 20},
    "26-35": {"claims": 35},
    "36-50": {"claims": 40},
    "50+": {"claims": 22}
}

# ============================================================
# Governance Layer – Pre-Approved Queries Only
# ============================================================

APPROVED_QUERIES = {
    "combined_risk_by_age": {
        "description": "Combined default + claim risk by age group",
        "type": "AGGREGATE_ONLY"
    }
}

# ============================================================
# Snowflake Clean Room–Style Execution
# ============================================================

def run_clean_room_query() -> List[Dict]:
    """
    Executes governed, aggregate-only analytics.
    If Snowflake credentials exist → real execution.
    Else → deterministic mock execution.
    """

    if USE_SNOWFLAKE:
        return run_real_snowflake_query()
    else:
        return run_mock_query()


def run_real_snowflake_query() -> List[Dict]:
    """
    Real Snowflake execution (aggregate-only).
    Assumes partner tables already exist in clean room.
    """

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = """
    SELECT
        age_group,
        ROUND((SUM(defaults) + SUM(claims)) / SUM(customers), 3) AS combined_risk_score
    FROM clean_room_aggregate_view
    GROUP BY age_group
    ORDER BY combined_risk_score DESC;
    """

    cursor = conn.cursor()
    cursor.execute(query)

    results = [
        {"age_group": row[0], "combined_risk_score": row[1]}
        for row in cursor.fetchall()
    ]

    cursor.close()
    conn.close()

    return results


def run_mock_query() -> List[Dict]:
    """
    Deterministic, privacy-safe mock analytics.
    """

    output = []
    for age in BANK_DATA:
        customers = BANK_DATA[age]["customers"]
        defaults = BANK_DATA[age]["defaults"]
        claims = INSURANCE_DATA[age]["claims"]

        score = round((defaults + claims) / customers, 3)
        output.append({
            "age_group": age,
            "combined_risk_score": score
        })

    return sorted(output, key=lambda x: x["combined_risk_score"], reverse=True)

# ============================================================
# AI Explanation Layer (Simulated Cortex Analyst)
# ============================================================

def generate_explanation(results: List[Dict]) -> str:
    highest = results[0]

    return (
        f"The analysis identifies the {highest['age_group']} age group as having "
        f"the highest combined risk score ({highest['combined_risk_score']}). "
        f"This reflects a stronger overlap between credit defaults and insurance claims. "
        f"Such insights can guide targeted fraud monitoring, inclusive product design, "
        f"or policy interventions—without exposing any individual-level data."
    )

# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Privacy-Safe Cross-Organization Analytics</title>
    <style>
        body { font-family: Arial; background: #f4f6f8; }
        .container { max-width: 1100px; margin: auto; padding: 30px; }
        h1 { color: #1f3c88; }
        .card { background: white; padding: 25px; border-radius: 6px; }
        button { background: #1f3c88; color: white; padding: 12px 20px; border: none; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
        th { background: #eef1f6; }
        .meta { font-size: 13px; color: #555; margin-top: 10px; }
    </style>
</head>
<body>

<div class="container">
    <h1>Privacy-Safe Cross-Organization Analytics</h1>

    <div class="card">
        <p>
            This system demonstrates a <b>Snowflake Data Clean Room–aligned architecture</b>
            enabling multiple organizations to collaborate without sharing raw data.
        </p>

        <button onclick="runAnalysis()">Run Approved Analysis</button>

        <div id="output"></div>
    </div>
</div>

<script>
function runAnalysis() {
    fetch("/analyze")
        .then(res => res.json())
        .then(data => {
            let html = `
                <h3>Anonymous Aggregate Output</h3>
                <table>
                    <tr><th>Age Group</th><th>Combined Risk Score</th></tr>
            `;

            data.results.forEach(r => {
                html += `<tr><td>${r.age_group}</td><td>${r.combined_risk_score}</td></tr>`;
            });

            html += `
                </table>
                <h3>AI-Generated Explanation</h3>
                <p>${data.explanation}</p>
                <p class="meta">
                    ✔ Pre-approved query only<br>
                    ✔ Aggregate-level output<br>
                    ✔ No raw data access<br>
                    ✔ Governance-first execution
                </p>
            `;

            document.getElementById("output").innerHTML = html;
        });
}
</script>

</body>
</html>
    """)

@app.route("/analyze")
def analyze():
    results = run_clean_room_query()
    explanation = generate_explanation(results)

    return jsonify({
        "results": results,
        "explanation": explanation,
        "execution_mode": "Snowflake" if USE_SNOWFLAKE else "Mock"
    })

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
