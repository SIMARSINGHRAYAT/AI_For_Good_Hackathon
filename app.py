from flask import Flask, render_template_string, request, jsonify, session
import json
import random
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'hackathon-secret-key-2024'

# -----------------------------
# Enhanced Mock Datasets with Multiple Organizations
# -----------------------------
datasets = {
    "bank": {
        "name": "First National Bank",
        "description": "Consumer banking data (de-identified)",
        "data": {
            "18-25": {"defaults": 12, "customers": 200, "avg_balance": 2500},
            "26-35": {"defaults": 18, "customers": 300, "avg_balance": 7500},
            "36-50": {"defaults": 25, "customers": 280, "avg_balance": 15000},
            "50+": {"defaults": 10, "customers": 150, "avg_balance": 18000}
        }
    },
    "insurer": {
        "name": "SafeGuard Insurance",
        "description": "Auto insurance claims (aggregated)",
        "data": {
            "18-25": {"claims": 20, "premium": 2200, "payout": 45000},
            "26-35": {"claims": 35, "premium": 1800, "payout": 52000},
            "36-50": {"claims": 40, "premium": 1500, "payout": 48000},
            "50+": {"claims": 22, "premium": 1300, "payout": 28000}
        }
    },
    "retailer": {
        "name": "Urban Retail Co",
        "description": "Consumer spending patterns",
        "data": {
            "18-25": {"avg_spend": 350, "electronics_ratio": 0.45},
            "26-35": {"avg_spend": 620, "electronics_ratio": 0.32},
            "36-50": {"avg_spend": 780, "electronics_ratio": 0.25},
            "50+": {"avg_spend": 520, "electronics_ratio": 0.18}
        }
    }
}

# -----------------------------
# Pre-approved Query Templates
# -----------------------------
query_templates = {
    "risk_analysis": {
        "name": "Cross-Industry Risk Analysis",
        "description": "Combines banking defaults with insurance claims to identify high-risk demographics",
        "organizations": ["bank", "insurer"],
        "function": "calculate_combined_risk"
    },
    "subsidy_impact": {
        "name": "Subsidy Impact Simulation",
        "description": "Simulates how targeted subsidies could reduce financial distress",
        "organizations": ["bank", "insurer"],
        "function": "calculate_subsidy_impact"
    },
    "spending_correlation": {
        "name": "Spending & Financial Health",
        "description": "Correlates retail spending with banking metrics",
        "organizations": ["bank", "retailer"],
        "function": "calculate_spending_correlation"
    }
}

# -----------------------------
# Enhanced Analytics Functions
# -----------------------------
def calculate_combined_risk(bank_data, insurer_data):
    results = []
    for age in bank_data:
        defaults = bank_data[age]["defaults"]
        customers = bank_data[age]["customers"]
        claims = insurer_data[age]["claims"]
        
        risk_score = round((defaults + claims) / customers, 3)
        financial_stress = round((defaults / customers) * (claims / 100), 3)
        
        results.append({
            "age_group": age,
            "risk_score": risk_score,
            "financial_stress": financial_stress,
            "bank_customers": customers,
            "insurance_claims": claims
        })
    return results

def calculate_subsidy_impact(bank_data, insurer_data):
    results = []
    for age in bank_data:
        defaults = bank_data[age]["defaults"]
        customers = bank_data[age]["customers"]
        avg_balance = bank_data[age]["avg_balance"]
        claims = insurer_data[age]["claims"]
        
        # Simulate subsidy impact
        potential_reduction = round(defaults * 0.3, 2)  # 30% reduction with subsidies
        subsidy_cost = round(customers * 500, 2)  # $500 per customer
        roi = round((potential_reduction * 10000) / subsidy_cost, 2) if subsidy_cost > 0 else 0
        
        results.append({
            "age_group": age,
            "current_defaults": defaults,
            "predicted_reduction": potential_reduction,
            "subsidy_cost": f"${subsidy_cost:,}",
            "estimated_roi": f"{roi}x",
            "recommended": roi > 2
        })
    return results

def calculate_spending_correlation(bank_data, retailer_data):
    results = []
    for age in bank_data:
        defaults_rate = bank_data[age]["defaults"] / bank_data[age]["customers"]
        avg_spend = retailer_data[age]["avg_spend"]
        correlation = round(1 - (defaults_rate * 10), 3)  # Simplified correlation
        
        results.append({
            "age_group": age,
            "default_rate": f"{defaults_rate*100:.1f}%",
            "avg_retail_spend": f"${avg_spend}",
            "health_score": max(0, min(10, correlation * 10)),
            "insight": "High spend, low defaults" if correlation > 0.7 else "Moderate correlation"
        })
    return results

def generate_ai_explanation(results, query_type):
    if query_type == "risk_analysis":
        highest = max(results, key=lambda x: x["risk_score"])
        return (
            f"🚨 **Risk Alert**: Age group {highest['age_group']} shows highest combined risk (score: {highest['risk_score']}). "
            f"This group has {highest['bank_customers']} banking customers with {highest['insurance_claims']} insurance claims. "
            f"**Recommendation**: Targeted financial literacy programs and customized insurance products."
        )
    elif query_type == "subsidy_impact":
        best_roi = max(results, key=lambda x: float(x["estimated_roi"].replace('x', '')) if x["estimated_roi"] != 'N/A' else 0)
        return (
            f"💰 **Subsidy Opportunity**: Age group {best_roi['age_group']} shows highest ROI ({best_roi['estimated_roi']}) "
            f"for subsidy intervention. Current defaults: {best_roi['current_defaults']}, "
            f"Predicted reduction: {best_roi['predicted_reduction']} cases. "
            f"**Recommendation**: Prioritize subsidies for this demographic."
        )
    else:
        best_health = max(results, key=lambda x: x["health_score"])
        return (
            f"📊 **Consumer Insight**: Age group {best_health['age_group']} has strongest financial health "
            f"(score: {best_health['health_score']}/10) with {best_health['avg_retail_spend']} average spend. "
            f"**Recommendation**: Partner with retailers for bundled financial products."
        )

# -----------------------------
# Simulated NLP Query Processing
# -----------------------------
def process_natural_language(query):
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['risk', 'danger', 'problem']):
        return "risk_analysis"
    elif any(word in query_lower for word in ['subsidy', 'fund', 'aid', 'help']):
        return "subsidy_impact"
    elif any(word in query_lower for word in ['spend', 'retail', 'shopping']):
        return "spending_correlation"
    else:
        return "risk_analysis"  # Default

# -----------------------------
# Enhanced Routes
# -----------------------------
@app.route("/")
def index():
    session['role'] = session.get('role', 'analyst')
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🔐 Privacy-Safe Analytics Platform</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #1f3c88;
            --secondary: #2d46b9;
            --accent: #6c63ff;
            --success: #00b894;
            --warning: #fdcb6e;
            --danger: #e17055;
            --light: #f8f9fa;
            --dark: #2d3436;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.12);
        }
        h1 {
            color: var(--primary);
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: var(--dark);
            opacity: 0.8;
            font-size: 1.1em;
            margin-bottom: 30px;
        }
        .pill {
            display: inline-block;
            padding: 6px 15px;
            background: var(--light);
            border-radius: 50px;
            margin-right: 10px;
            margin-bottom: 10px;
            font-size: 0.9em;
            color: var(--dark);
            border: 1px solid #dee2e6;
        }
        .query-card {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        .btn-primary:hover {
            background: var(--secondary);
            transform: translateY(-2px);
        }
        .btn-outline {
            background: transparent;
            color: var(--primary);
            border: 2px solid var(--primary);
        }
        .btn-outline:hover {
            background: var(--primary);
            color: white;
        }
        .query-input {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #dee2e6;
            font-size: 16px;
            margin-bottom: 15px;
            transition: border-color 0.3s ease;
        }
        .query-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .results-section {
            display: none;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            background: var(--light);
            border-radius: 20px;
            font-size: 0.85em;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .role-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .role-btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: 2px solid transparent;
            background: var(--light);
            cursor: pointer;
        }
        .role-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        .insight-box {
            background: linear-gradient(135deg, #00b89420 0%, #00b89440 100%);
            border-left: 4px solid var(--success);
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin: 20px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary);
        }
        .stat-label {
            color: var(--dark);
            opacity: 0.7;
            font-size: 0.9em;
        }
        .privacy-badge {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Privacy-Safe Analytics Platform</h1>
            <p class="subtitle">Cross-organization insights without sharing raw data • Snowflake Clean Room Technology</p>
            
            <div class="role-selector">
                <button class="role-btn active" onclick="setRole('analyst')">📊 Data Analyst</button>
                <button class="role-btn" onclick="setRole('executive')">👔 Executive</button>
                <button class="role-btn" onclick="setRole('compliance')">⚖️ Compliance Officer</button>
            </div>
            
            <div id="orgChips">
                <span class="chip">🏦 First National Bank</span>
                <span class="chip">🛡️ SafeGuard Insurance</span>
                <span class="chip">🛒 Urban Retail Co</span>
                <span class="chip">🔒 Data Clean Room v2.1</span>
            </div>
        </div>

        <div class="card query-card">
            <h2 style="color: white;">Ask a Natural Language Question</h2>
            <input type="text" id="nlQuery" class="query-input" placeholder="e.g., Which age group has the highest financial risk and could benefit from subsidies?">
            <button class="btn btn-primary" onclick="processNaturalLanguage()">
                🔍 Analyze Across Organizations
            </button>
            <p style="color: rgba(255,255,255,0.8); margin-top: 15px; font-size: 0.9em;">
                <strong>Try:</strong> "Show me spending patterns that correlate with low defaults" or "Where should we target subsidies?"
            </p>
        </div>

        <div class="card">
            <h2>Pre-Approved Analytics Templates</h2>
            <div class="stats-grid">
                <div class="stat-card" onclick="runQuery('risk_analysis')" style="cursor: pointer;">
                    <div class="stat-value">⚠️</div>
                    <div class="stat-label">Risk Analysis</div>
                    <small>Bank + Insurance data</small>
                </div>
                <div class="stat-card" onclick="runQuery('subsidy_impact')" style="cursor: pointer;">
                    <div class="stat-value">💰</div>
                    <div class="stat-label">Subsidy Impact</div>
                    <small>ROI simulation</small>
                </div>
                <div class="stat-card" onclick="runQuery('spending_correlation')" style="cursor: pointer;">
                    <div class="stat-value">📊</div>
                    <div class="stat-label">Spending Correlation</div>
                    <small>Retail + Banking</small>
                </div>
            </div>
        </div>

        <div id="resultsSection" class="results-section">
            <div class="card">
                <h2 id="queryTitle">Analysis Results</h2>
                <div id="resultsTable"></div>
                <div class="chart-container">
                    <canvas id="resultsChart"></canvas>
                </div>
                <div class="insight-box" id="aiInsight"></div>
                
                <h3>🔒 Privacy & Compliance Details</h3>
                <div id="privacyDetails">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">🎯</div>
                            <div class="stat-label">Query Type</div>
                            <small id="queryType">Pre-approved</small>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">📈</div>
                            <div class="stat-label">Data Processed</div>
                            <small id="dataVolume">~900 records</small>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">🔐</div>
                            <div class="stat-label">Privacy Level</div>
                            <small>Aggregate-only output</small>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">⏱️</div>
                            <div class="stat-label">Processing Time</div>
                            <small id="processingTime">~120ms</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="privacy-badge">
        <div style="color: green; font-size: 1.5em;">✓</div>
        <div>
            <strong>Privacy Safe</strong><br>
            <small>No PII exposed • Audit logged</small>
        </div>
    </div>

    <script>
        let currentChart = null;
        
        function setRole(role) {
            document.querySelectorAll('.role-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            fetch('/set_role/' + role);
            updateUIForRole(role);
        }
        
        function updateUIForRole(role) {
            const chips = document.getElementById('orgChips');
            if (role === 'compliance') {
                chips.innerHTML += '<span class="chip">📋 Audit Log Enabled</span>';
            }
        }
        
        function runQuery(queryType) {
            showLoading();
            fetch('/run_query/' + queryType)
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                });
        }
        
        function processNaturalLanguage() {
            const query = document.getElementById('nlQuery').value;
            if (!query.trim()) return;
            
            showLoading();
            fetch('/process_query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            });
        }
        
        function showLoading() {
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('resultsTable').innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <div style="font-size: 3em; margin-bottom: 20px;">⏳</div>
                    <p>Processing query in secure Data Clean Room...</p>
                    <p><small>Aggregating de-identified data from multiple organizations</small></p>
                </div>
            `;
        }
        
        function displayResults(data) {
            document.getElementById('queryTitle').textContent = data.query_name;
            document.getElementById('queryType').textContent = data.query_type;
            document.getElementById('processingTime').textContent = data.processing_time;
            
            // Create table
            let html = `<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <thead><tr style="background: #f8f9fa;">`;
            
            // Headers
            Object.keys(data.results[0]).forEach(col => {
                html += `<th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">${col.replace('_', ' ').toUpperCase()}</th>`;
            });
            html += `</tr></thead><tbody>`;
            
            // Rows
            data.results.forEach(row => {
                html += `<tr>`;
                Object.values(row).forEach(val => {
                    const isRecommended = val === true || val === false;
                    const cellStyle = isRecommended ? 
                        (val ? 'background: #d4edda; color: #155724;' : 'background: #f8d7da; color: #721c24;') : '';
                    html += `<td style="padding: 12px; border-bottom: 1px solid #dee2e6; ${cellStyle}">${val}</td>`;
                });
                html += `</tr>`;
            });
            html += `</tbody></table>`;
            document.getElementById('resultsTable').innerHTML = html;
            
            // AI Insight
            document.getElementById('aiInsight').innerHTML = `
                <h3>🤖 AI-Generated Insight</h3>
                <p>${data.explanation}</p>
                <small>Generated using privacy-safe aggregate data only</small>
            `;
            
            // Create chart
            createChart(data.results, data.query_type);
        }
        
        function createChart(results, queryType) {
            const ctx = document.getElementById('resultsChart').getContext('2d');
            
            if (currentChart) {
                currentChart.destroy();
            }
            
            const labels = results.map(r => r.age_group);
            let datasets = [];
            
            if (queryType === 'risk_analysis') {
                datasets = [
                    {
                        label: 'Risk Score',
                        data: results.map(r => r.risk_score),
                        borderColor: '#e17055',
                        backgroundColor: '#e1705533',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Financial Stress',
                        data: results.map(r => r.financial_stress),
                        borderColor: '#00b894',
                        backgroundColor: '#00b89433',
                        yAxisID: 'y1'
                    }
                ];
            } else if (queryType === 'subsidy_impact') {
                datasets = [{
                    label: 'Estimated ROI',
                    data: results.map(r => parseFloat(r.estimated_roi.replace('x', ''))),
                    backgroundColor: results.map(r => r.recommended ? '#00b894' : '#fdcb6e'),
                    borderColor: '#2d3436'
                }];
            }
            
            currentChart = new Chart(ctx, {
                type: 'bar',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Aggregate Analysis Results'
                        }
                    }
                }
            });
        }
        
        // Initialize with example query
        setTimeout(() => runQuery('risk_analysis'), 1000);
    </script>
</body>
</html>
''')

@app.route("/set_role/<role>")
def set_role(role):
    session['role'] = role
    return jsonify({"status": "role_updated", "role": role})

@app.route("/run_query/<query_type>")
def run_query(query_type):
    start_time = datetime.now()
    
    if query_type == "risk_analysis":
        results = calculate_combined_risk(datasets["bank"]["data"], datasets["insurer"]["data"])
    elif query_type == "subsidy_impact":
        results = calculate_subsidy_impact(datasets["bank"]["data"], datasets["insurer"]["data"])
    elif query_type == "spending_correlation":
        results = calculate_spending_correlation(datasets["bank"]["data"], datasets["retailer"]["data"])
    else:
        results = []
    
    processing_time = f"{(datetime.now() - start_time).total_seconds()*1000:.0f}ms"
    
    return jsonify({
        "query_name": query_templates[query_type]["name"],
        "query_type": query_type,
        "results": results,
        "explanation": generate_ai_explanation(results, query_type),
        "processing_time": processing_time,
        "organizations": query_templates[query_type]["organizations"],
        "data_sources": len(query_templates[query_type]["organizations"])
    })

@app.route("/process_query", methods=["POST"])
def process_query():
    data = request.json
    user_query = data.get("query", "")
    
    query_type = process_natural_language(user_query)
    
    return run_query(query_type)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Privacy-Safe Analytics Platform...")
    print("🔐 Access the application at: http://127.0.0.1:5000")
    print("📊 Features:")
    print("   • Natural Language Query Processing")
    print("   • Multi-Organization Analytics")
    print("   • Role-Based Views (Analyst/Executive/Compliance)")
    print("   • Privacy-Safe Aggregate Output Only")
    print("   • AI-Generated Insights")
    print("   • Interactive Visualizations")
    app.run(debug=True, host='0.0.0.0', port=5000)
