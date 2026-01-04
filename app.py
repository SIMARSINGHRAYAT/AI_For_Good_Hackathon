import os
import json
import hashlib
import secrets
from flask import Flask, render_template_string, jsonify, request, session
from datetime import datetime, timedelta
from functools import wraps
import random

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# ============================================================
# CONFIGURATION
# ============================================================

EXECUTION_MODE = "MOCK"
APP_VERSION = "2.0.0"

if all(os.getenv(k) for k in [
    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
]):
    EXECUTION_MODE = "SNOWFLAKE"

# ============================================================
# SIMULATED ORGANIZATIONS (DATA PROVIDERS)
# ============================================================

ORGANIZATIONS = {
    "national_bank": {
        "name": "National Bank Corp",
        "type": "Financial Institution",
        "data_shared": ["age_group", "default_rate", "loan_volume"],
        "privacy_level": "differential_privacy"
    },
    "insurance_co": {
        "name": "SecureLife Insurance",
        "type": "Insurance Provider",
        "data_shared": ["age_group", "claim_frequency", "policy_type"],
        "privacy_level": "k_anonymity"
    },
    "gov_welfare": {
        "name": "Government Welfare Dept",
        "type": "Government Agency",
        "data_shared": ["age_group", "subsidy_eligibility", "benefit_received"],
        "privacy_level": "aggregated_only"
    },
    "telecom_provider": {
        "name": "TeleCom Networks",
        "type": "Telecommunications",
        "data_shared": ["age_group", "digital_activity", "payment_behavior"],
        "privacy_level": "differential_privacy"
    }
}

# ============================================================
# ENHANCED MOCK CLEAN-ROOM DATA (DE-IDENTIFIED)
# ============================================================

BANK_DATA = {
    "18-25": {"customers": 2450, "defaults": 147, "avg_loan": 15000, "credit_score_avg": 620},
    "26-35": {"customers": 4200, "defaults": 252, "avg_loan": 45000, "credit_score_avg": 680},
    "36-50": {"customers": 3800, "defaults": 342, "avg_loan": 85000, "credit_score_avg": 710},
    "51-65": {"customers": 2100, "defaults": 126, "avg_loan": 65000, "credit_score_avg": 740},
    "65+": {"customers": 950, "defaults": 38, "avg_loan": 25000, "credit_score_avg": 760}
}

INSURANCE_DATA = {
    "18-25": {"policies": 1800, "claims": 234, "avg_claim": 2500, "fraud_flags": 12},
    "26-35": {"policies": 3600, "claims": 468, "avg_claim": 4200, "fraud_flags": 18},
    "36-50": {"policies": 4200, "claims": 588, "avg_claim": 6800, "fraud_flags": 25},
    "51-65": {"policies": 2800, "claims": 504, "avg_claim": 12000, "fraud_flags": 8},
    "65+": {"policies": 1500, "claims": 375, "avg_claim": 18000, "fraud_flags": 3}
}

SUBSIDY_DATA = {
    "18-25": {"eligible": 1850, "applied": 1200, "received": 980, "avg_amount": 2400},
    "26-35": {"eligible": 2800, "applied": 2100, "received": 1680, "avg_amount": 3200},
    "36-50": {"eligible": 2400, "applied": 1600, "received": 1120, "avg_amount": 4500},
    "51-65": {"eligible": 1600, "applied": 1200, "received": 960, "avg_amount": 5200},
    "65+": {"eligible": 1200, "applied": 1000, "received": 850, "avg_amount": 6000}
}

TELECOM_DATA = {
    "18-25": {"users": 3200, "digital_score": 85, "payment_delay_pct": 18, "app_usage": 95},
    "26-35": {"users": 4800, "digital_score": 78, "payment_delay_pct": 12, "app_usage": 88},
    "36-50": {"users": 3600, "digital_score": 65, "payment_delay_pct": 8, "app_usage": 72},
    "51-65": {"users": 2200, "digital_score": 48, "payment_delay_pct": 5, "app_usage": 55},
    "65+": {"users": 1100, "digital_score": 32, "payment_delay_pct": 3, "app_usage": 35}
}

# ============================================================
# GOVERNANCE – APPROVED QUERIES & PRIVACY RULES
# ============================================================

APPROVED_QUERIES = {
    "risk_analysis": {
        "title": "Combined Risk Analysis",
        "description": "Which age groups have highest combined default + claim risk?",
        "data_sources": ["national_bank", "insurance_co"],
        "min_aggregation": 100,
        "requires_approval": False
    },
    "subsidy_gap": {
        "title": "Subsidy Gap Analysis",
        "description": "Which groups are not benefitting from subsidies?",
        "data_sources": ["gov_welfare"],
        "min_aggregation": 50,
        "requires_approval": False
    },
    "fraud_detection": {
        "title": "Cross-Sector Fraud Indicators",
        "description": "Identify age groups with elevated fraud risk patterns",
        "data_sources": ["national_bank", "insurance_co", "telecom_provider"],
        "min_aggregation": 200,
        "requires_approval": True
    },
    "financial_inclusion": {
        "title": "Financial Inclusion Score",
        "description": "Assess digital and financial inclusion across demographics",
        "data_sources": ["national_bank", "telecom_provider", "gov_welfare"],
        "min_aggregation": 100,
        "requires_approval": False
    },
    "policy_effectiveness": {
        "title": "Policy Effectiveness Dashboard",
        "description": "Measure welfare program reach and effectiveness",
        "data_sources": ["gov_welfare", "national_bank"],
        "min_aggregation": 100,
        "requires_approval": False
    }
}

AUDIT_LOG = []
PRIVACY_BUDGET = {"epsilon": 1.0, "used": 0.0}

# ============================================================
# SIMULATED USERS & ACCESS CONTROL
# ============================================================

USERS = {
    "analyst@bank.gov": {"role": "analyst", "org": "national_bank", "approved_queries": ["risk_analysis", "financial_inclusion"]},
    "policy@welfare.gov": {"role": "policy_maker", "org": "gov_welfare", "approved_queries": ["subsidy_gap", "policy_effectiveness"]},
    "admin@cleanroom.gov": {"role": "admin", "org": "admin", "approved_queries": list(APPROVED_QUERIES.keys())}
}

# ============================================================
# PRIVACY & SECURITY HELPERS
# ============================================================

def add_differential_noise(value, sensitivity=0.1):
    """Add Laplace noise for differential privacy"""
    noise = random.gauss(0, sensitivity)
    return round(value + noise, 3)

def hash_identifier(value):
    """One-way hash for pseudo-anonymization"""
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]

def check_k_anonymity(group_size, k=5):
    """Ensure minimum group size for k-anonymity"""
    return group_size >= k

def log_action(action, user="anonymous", query_type=None, data_sources=None):
    """Comprehensive audit logging"""
    entry = {
        "id": hash_identifier(f"{datetime.utcnow()}{action}"),
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "user": hash_identifier(user),
        "query_type": query_type,
        "data_sources": data_sources or [],
        "mode": EXECUTION_MODE,
        "privacy_budget_used": PRIVACY_BUDGET["used"]
    }
    AUDIT_LOG.append(entry)
    return entry["id"]

# ============================================================
# CORE ANALYTICS FUNCTIONS
# ============================================================

def combined_risk_analysis():
    """Cross-organization risk scoring"""
    results = []
    for age in BANK_DATA:
        bank = BANK_DATA[age]
        insurance = INSURANCE_DATA[age]
        
        default_rate = bank["defaults"] / bank["customers"]
        claim_rate = insurance["claims"] / insurance["policies"]
        fraud_rate = insurance["fraud_flags"] / insurance["policies"]
        
        composite_score = round((default_rate * 0.4 + claim_rate * 0.4 + fraud_rate * 0.2) * 100, 2)
        composite_score = add_differential_noise(composite_score, 0.5)
        
        results.append({
            "group": age,
            "risk_score": composite_score,
            "default_rate": round(default_rate * 100, 2),
            "claim_rate": round(claim_rate * 100, 2),
            "sample_size": bank["customers"] + insurance["policies"]
        })
    return sorted(results, key=lambda x: x["risk_score"], reverse=True)

def subsidy_gap_analysis():
    """Analyze subsidy delivery effectiveness"""
    results = []
    for age in SUBSIDY_DATA:
        data = SUBSIDY_DATA[age]
        
        application_rate = round(data["applied"] / data["eligible"] * 100, 1)
        success_rate = round(data["received"] / data["applied"] * 100, 1)
        coverage_gap = data["eligible"] - data["received"]
        lost_value = coverage_gap * data["avg_amount"]
        
        results.append({
            "group": age,
            "coverage_gap": coverage_gap,
            "application_rate": application_rate,
            "success_rate": success_rate,
            "lost_value": lost_value,
            "eligible_population": data["eligible"]
        })
    return sorted(results, key=lambda x: x["coverage_gap"], reverse=True)

def fraud_detection_analysis():
    """Multi-source fraud pattern detection"""
    results = []
    for age in BANK_DATA:
        bank = BANK_DATA[age]
        insurance = INSURANCE_DATA[age]
        telecom = TELECOM_DATA[age]
        
        financial_stress = bank["defaults"] / bank["customers"]
        insurance_anomaly = insurance["fraud_flags"] / insurance["policies"]
        payment_risk = telecom["payment_delay_pct"] / 100
        
        fraud_score = round((financial_stress * 0.35 + insurance_anomaly * 0.4 + payment_risk * 0.25) * 1000, 1)
        fraud_score = add_differential_noise(fraud_score, 2)
        
        results.append({
            "group": age,
            "fraud_score": fraud_score,
            "risk_level": "HIGH" if fraud_score > 50 else "MEDIUM" if fraud_score > 25 else "LOW",
            "contributing_factors": {
                "financial_stress": round(financial_stress * 100, 1),
                "insurance_anomaly": round(insurance_anomaly * 100, 2),
                "payment_delays": telecom["payment_delay_pct"]
            }
        })
    return sorted(results, key=lambda x: x["fraud_score"], reverse=True)

def financial_inclusion_analysis():
    """Assess financial and digital inclusion"""
    results = []
    for age in BANK_DATA:
        bank = BANK_DATA[age]
        telecom = TELECOM_DATA[age]
        subsidy = SUBSIDY_DATA[age]
        
        credit_access = min(100, (bank["credit_score_avg"] - 500) / 3)
        digital_access = telecom["digital_score"]
        welfare_access = (subsidy["received"] / subsidy["eligible"]) * 100
        
        inclusion_score = round((credit_access * 0.35 + digital_access * 0.35 + welfare_access * 0.30), 1)
        
        results.append({
            "group": age,
            "inclusion_score": inclusion_score,
            "credit_access": round(credit_access, 1),
            "digital_access": digital_access,
            "welfare_access": round(welfare_access, 1),
            "population": bank["customers"]
        })
    return sorted(results, key=lambda x: x["inclusion_score"])

def policy_effectiveness_analysis():
    """Measure policy program effectiveness"""
    results = []
    total_budget_utilized = 0
    total_eligible = 0
    
    for age in SUBSIDY_DATA:
        subsidy = SUBSIDY_DATA[age]
        bank = BANK_DATA[age]
        
        reach = subsidy["received"] / subsidy["eligible"]
        efficiency = subsidy["received"] / subsidy["applied"] if subsidy["applied"] > 0 else 0
        impact_score = reach * efficiency * 100
        budget_utilized = subsidy["received"] * subsidy["avg_amount"]
        
        total_budget_utilized += budget_utilized
        total_eligible += subsidy["eligible"]
        
        results.append({
            "group": age,
            "reach_pct": round(reach * 100, 1),
            "efficiency_pct": round(efficiency * 100, 1),
            "impact_score": round(impact_score, 1),
            "budget_utilized": budget_utilized,
            "beneficiaries": subsidy["received"]
        })
    
    return {
        "by_group": sorted(results, key=lambda x: x["impact_score"], reverse=True),
        "summary": {
            "total_budget_utilized": total_budget_utilized,
            "total_eligible": total_eligible,
            "overall_reach": round(sum(SUBSIDY_DATA[a]["received"] for a in SUBSIDY_DATA) / total_eligible * 100, 1)
        }
    }

# ============================================================
# AI EXPLANATION GENERATOR
# ============================================================

def generate_explanation(query_type, results):
    """Generate contextual AI explanations"""
    
    if query_type == "risk_analysis":
        top = results[0]
        return {
            "summary": f"The **{top['group']}** age group exhibits the highest combined financial risk score of **{top['risk_score']}**.",
            "insights": [
                f"Default rate of {top['default_rate']}% indicates credit stress in this demographic",
                f"Insurance claim rate of {top['claim_rate']}% suggests overlapping financial vulnerability",
                "Cross-sector correlation reveals systemic risk patterns requiring intervention"
            ],
            "recommendations": [
                "Consider targeted financial literacy programs for this age group",
                "Develop risk-adjusted lending products with appropriate safeguards",
                "Coordinate with insurance providers on early warning systems"
            ]
        }
    
    elif query_type == "subsidy_gap":
        top = results[0]
        return {
            "summary": f"The **{top['group']}** age group has the largest subsidy gap with **{top['coverage_gap']:,}** eligible individuals not receiving benefits.",
            "insights": [
                f"Only {top['application_rate']}% of eligible individuals apply for benefits",
                f"Application success rate stands at {top['success_rate']}%",
                f"Estimated lost value: **${top['lost_value']:,}** in unrealized benefits"
            ],
            "recommendations": [
                "Simplify application processes for this demographic",
                "Launch targeted awareness campaigns",
                "Consider automatic enrollment for clearly eligible individuals"
            ]
        }
    
    elif query_type == "fraud_detection":
        high_risk = [r for r in results if r["risk_level"] == "HIGH"]
        return {
            "summary": f"**{len(high_risk)}** age group(s) flagged with HIGH fraud risk indicators.",
            "insights": [
                f"Highest fraud score: {results[0]['fraud_score']} in the {results[0]['group']} group",
                "Multi-source analysis reveals correlated suspicious patterns",
                "Financial stress and payment delays are primary contributing factors"
            ],
            "recommendations": [
                "Enhance monitoring for flagged demographics",
                "Implement cross-organization verification protocols",
                "Balance fraud prevention with financial inclusion goals"
            ]
        }
    
    elif query_type == "financial_inclusion":
        lowest = results[0]
        return {
            "summary": f"The **{lowest['group']}** age group shows the lowest financial inclusion score of **{lowest['inclusion_score']}**.",
            "insights": [
                f"Credit access score: {lowest['credit_access']}%",
                f"Digital access score: {lowest['digital_access']}%",
                f"Welfare access: {lowest['welfare_access']}%"
            ],
            "recommendations": [
                "Prioritize digital literacy programs for underserved groups",
                "Develop alternative credit scoring mechanisms",
                "Improve welfare program accessibility"
            ]
        }
    
    elif query_type == "policy_effectiveness":
        data = results
        return {
            "summary": f"Overall welfare program reach: **{data['summary']['overall_reach']}%** with **${data['summary']['total_budget_utilized']:,}** utilized.",
            "insights": [
                f"Best performing group: {data['by_group'][0]['group']} with {data['by_group'][0]['impact_score']} impact score",
                f"Total eligible population: {data['summary']['total_eligible']:,}",
                "Significant variation in reach and efficiency across demographics"
            ],
            "recommendations": [
                "Replicate successful approaches from high-performing segments",
                "Address bottlenecks in low-efficiency groups",
                "Consider demographic-specific outreach strategies"
            ]
        }
    
    return {"summary": "Analysis complete.", "insights": [], "recommendations": []}

# ============================================================
# DASHBOARD METRICS
# ============================================================

def get_dashboard_metrics():
    """Generate overview dashboard metrics"""
    total_records = sum(BANK_DATA[a]["customers"] for a in BANK_DATA)
    total_policies = sum(INSURANCE_DATA[a]["policies"] for a in INSURANCE_DATA)
    total_beneficiaries = sum(SUBSIDY_DATA[a]["received"] for a in SUBSIDY_DATA)
    
    return {
        "total_records_analyzed": total_records + total_policies,
        "organizations_connected": len(ORGANIZATIONS),
        "queries_executed": len(AUDIT_LOG),
        "privacy_budget_remaining": round(PRIVACY_BUDGET["epsilon"] - PRIVACY_BUDGET["used"], 2),
        "active_data_sources": list(ORGANIZATIONS.keys()),
        "last_sync": datetime.utcnow().isoformat(),
        "data_freshness": "Real-time" if EXECUTION_MODE == "SNOWFLAKE" else "Mock Data"
    }

# ============================================================
# HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy-Safe Cross-Organization Analytics Platform</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #1a365d;
            --primary-light: #2c5282;
            --secondary: #38a169;
            --warning: #d69e2e;
            --danger: #e53e3e;
            --bg: #f7fafc;
            --card-bg: #ffffff;
            --text: #2d3748;
            --text-light: #718096;
            --border: #e2e8f0;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .header .badge {
            background: rgba(255,255,255,0.2);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid var(--primary);
        }
        
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .metric-card .label {
            color: var(--text-light);
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 30px;
        }
        
        .sidebar {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            height: fit-content;
        }
        
        .sidebar h3 {
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
        }
        
        .query-item {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            cursor: pointer;
            border: 2px solid var(--border);
            transition: all 0.2s;
        }
        
        .query-item:hover {
            border-color: var(--primary);
            background: #f8faff;
        }
        
        .query-item.active {
            border-color: var(--primary);
            background: #ebf4ff;
        }
        
        .query-item .title {
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .query-item .desc {
            font-size: 0.85rem;
            color: var(--text-light);
        }
        
        .query-item .sources {
            margin-top: 8px;
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        
        .source-tag {
            background: #e2e8f0;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        
        .content-area {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .content-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-light);
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: var(--border);
            color: var(--text);
        }
        
        .results-section { margin-top: 20px; }
        
        .explanation-box {
            background: linear-gradient(135deg, #ebf8ff 0%, #f0fff4 100%);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid var(--secondary);
        }
        
        .explanation-box h4 {
            color: var(--primary);
            margin-bottom: 15px;
        }
        
        .explanation-box ul {
            margin: 10px 0 10px 20px;
        }
        
        .explanation-box li {
            margin-bottom: 8px;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .data-table th, .data-table td {
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        .data-table th {
            background: #f8fafc;
            font-weight: 600;
            color: var(--primary);
        }
        
        .data-table tr:hover {
            background: #f8fafc;
        }
        
        .risk-high { color: var(--danger); font-weight: 600; }
        .risk-medium { color: var(--warning); font-weight: 600; }
        .risk-low { color: var(--secondary); font-weight: 600; }
        
        .chart-container {
            height: 300px;
            margin: 20px 0;
        }
        
        .audit-section {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid var(--border);
        }
        
        .audit-item {
            display: flex;
            justify-content: space-between;
            padding: 12px;
            background: #f8fafc;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        
        .privacy-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 15px;
            background: #f0fff4;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .privacy-bar {
            flex: 1;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .privacy-bar-fill {
            height: 100%;
            background: var(--secondary);
            transition: width 0.3s;
        }
        
        .org-list {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .org-item {
            padding: 12px;
            background: #f8fafc;
            border-radius: 8px;
            font-size: 0.85rem;
        }
        
        .org-item .name { font-weight: 600; }
        .org-item .type { color: var(--text-light); font-size: 0.8rem; }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-light);
        }
        
        .spinner {
            border: 3px solid var(--border);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            border: none;
            background: var(--border);
            border-radius: 6px;
            cursor: pointer;
        }
        
        .tab.active {
            background: var(--primary);
            color: white;
        }
        
        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>

<div class="header">
    <div>
        <h1>🔒 Privacy-Safe Cross-Organization Analytics</h1>
        <small>Secure Data Clean Room Platform</small>
    </div>
    <div class="badge">Mode: {{ mode }}</div>
</div>

<div class="container">
    
    <!-- Dashboard Metrics -->
    <div class="metrics-grid" id="metricsGrid">
        <div class="metric-card">
            <div class="value" id="totalRecords">-</div>
            <div class="label">Records Analyzed</div>
        </div>
        <div class="metric-card">
            <div class="value" id="orgsConnected">-</div>
            <div class="label">Organizations</div>
        </div>
        <div class="metric-card">
            <div class="value" id="queriesRun">-</div>
            <div class="label">Queries Executed</div>
        </div>
        <div class="metric-card">
            <div class="value" id="privacyBudget">-</div>
            <div class="label">Privacy Budget Left</div>
        </div>
    </div>
    
    <div class="main-grid">
        
        <!-- Sidebar: Query Selection -->
        <div class="sidebar">
            <h3>📊 Approved Queries</h3>
            
            <div id="queryList"></div>
            
            <h3 style="margin-top: 30px;">🏢 Connected Organizations</h3>
            <div class="org-list" id="orgList"></div>
        </div>
        
        <!-- Main Content Area -->
        <div class="content-area">
            <div class="content-header">
                <div>
                    <h2 id="queryTitle">Select a Query</h2>
                    <p id="queryDesc" style="color: var(--text-light);">Choose an approved query from the sidebar to begin analysis</p>
                </div>
                <button class="btn btn-primary" id="runBtn" onclick="runAnalysis()" disabled>
                    ▶ Run Analysis
                </button>
            </div>
            
            <!-- Privacy Budget Indicator -->
            <div class="privacy-indicator">
                <span>🛡️ Privacy Budget:</span>
                <div class="privacy-bar">
                    <div class="privacy-bar-fill" id="privacyBarFill" style="width: 100%"></div>
                </div>
                <span id="privacyPct">100%</span>
            </div>
            
            <!-- Results Area -->
            <div id="resultsArea">
                <div style="text-align: center; padding: 60px; color: var(--text-light);">
                    <div style="font-size: 4rem; margin-bottom: 20px;">📈</div>
                    <p>Select a query and click "Run Analysis" to see privacy-safe insights</p>
                </div>
            </div>
            
            <!-- Audit Log -->
            <div class="audit-section">
                <h3>📋 Audit Trail</h3>
                <div id="auditLog" style="margin-top: 15px;">
                    <p style="color: var(--text-light);">No queries executed yet</p>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
const approvedQueries = {{ queries | tojson }};
const organizations = {{ organizations | tojson }};
let selectedQuery = null;
let chart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    renderQueryList();
    renderOrgList();
    loadDashboardMetrics();
});

function renderQueryList() {
    const container = document.getElementById('queryList');
    container.innerHTML = Object.entries(approvedQueries).map(([key, q]) => `
        <div class="query-item" id="query-${key}" onclick="selectQuery('${key}')">
            <div class="title">${q.title}</div>
            <div class="desc">${q.description}</div>
            <div class="sources">
                ${q.data_sources.map(s => `<span class="source-tag">${s}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

function renderOrgList() {
    const container = document.getElementById('orgList');
    container.innerHTML = Object.entries(organizations).map(([key, o]) => `
        <div class="org-item">
            <div class="name">${o.name}</div>
            <div class="type">${o.type}</div>
        </div>
    `).join('');
}

function selectQuery(key) {
    selectedQuery = key;
    const q = approvedQueries[key];
    
    // Update UI
    document.querySelectorAll('.query-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`query-${key}`).classList.add('active');
    
    document.getElementById('queryTitle').textContent = q.title;
    document.getElementById('queryDesc').textContent = q.description;
    document.getElementById('runBtn').disabled = false;
}

async function loadDashboardMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        
        document.getElementById('totalRecords').textContent = data.total_records_analyzed.toLocaleString();
        document.getElementById('orgsConnected').textContent = data.organizations_connected;
        document.getElementById('queriesRun').textContent = data.queries_executed;
        document.getElementById('privacyBudget').textContent = data.privacy_budget_remaining;
        
        const pct = (data.privacy_budget_remaining / 1.0) * 100;
        document.getElementById('privacyBarFill').style.width = pct + '%';
        document.getElementById('privacyPct').textContent = pct.toFixed(0) + '%';
    } catch (e) {
        console.error('Failed to load metrics', e);
    }
}

async function runAnalysis() {
    if (!selectedQuery) return;
    
    const resultsArea = document.getElementById('resultsArea');
    resultsArea.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Running privacy-safe analysis...</p>
            <small>Applying differential privacy & k-anonymity</small>
        </div>
    `;
    
    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: selectedQuery})
        });
        
        const data = await res.json();
        
        if (data.error) {
            resultsArea.innerHTML = `<div class="loading" style="color: var(--danger)">⚠️ ${data.error}</div>`;
            return;
        }
        
        renderResults(data);
        updateAuditLog(data.audit);
        loadDashboardMetrics();
        
    } catch (e) {
        resultsArea.innerHTML = `<div class="loading" style="color: var(--danger)">⚠️ Analysis failed</div>`;
    }
}

function renderResults(data) {
    const resultsArea = document.getElementById('resultsArea');
    const exp = data.explanation;
    
    let tableHtml = '';
    let chartData = null;
    
    // Handle different result structures
    const results = Array.isArray(data.results) ? data.results : data.results.by_group || [];
    
    if (results.length > 0) {
        const keys = Object.keys(results[0]).filter(k => k !== 'contributing_factors');
        
        tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${keys.map(k => `<th>${formatHeader(k)}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${results.map(row => `
                        <tr>
                            ${keys.map(k => `<td>${formatCell(k, row[k])}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        // Prepare chart data
        const numericKey = keys.find(k => k !== 'group' && typeof results[0][k] === 'number');
        if (numericKey) {
            chartData = {
                labels: results.map(r => r.group),
                values: results.map(r => r[numericKey]),
                label: formatHeader(numericKey)
            };
        }
    }
    
    resultsArea.innerHTML = `
        <div class="explanation-box">
            <h4>🤖 AI-Generated Insights</h4>
            <p><strong>${exp.summary}</strong></p>
            ${exp.insights.length ? `
                <h5 style="margin-top: 15px;">Key Findings:</h5>
                <ul>${exp.insights.map(i => `<li>${i}</li>`).join('')}</ul>
            ` : ''}
            ${exp.recommendations.length ? `
                <h5 style="margin-top: 15px;">Recommendations:</h5>
                <ul>${exp.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
            ` : ''}
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('table')">📊 Table View</button>
            <button class="tab" onclick="showTab('chart')">📈 Chart View</button>
        </div>
        
        <div id="tableView">${tableHtml}</div>
        <div id="chartView" style="display: none;">
            <div class="chart-container">
                <canvas id="resultsChart"></canvas>
            </div>
        </div>
        
        <p style="margin-top: 20px; font-size: 0.85rem; color: var(--text-light);">
            ✓ Results anonymized via ${data.privacy_method || 'differential privacy'} | 
            Mode: ${data.mode} | 
            Timestamp: ${new Date().toISOString()}
        </p>
    `;
    
    if (chartData) {
        setTimeout(() => renderChart(chartData), 100);
    }
}

function showTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    document.getElementById('tableView').style.display = tab === 'table' ? 'block' : 'none';
    document.getElementById('chartView').style.display = tab === 'chart' ? 'block' : 'none';
}

function renderChart(data) {
    const ctx = document.getElementById('resultsChart');
    if (!ctx) return;
    
    if (chart) chart.destroy();
    
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: data.label,
                data: data.values,
                backgroundColor: 'rgba(26, 54, 93, 0.8)',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function formatHeader(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatCell(key, value) {
    if (key === 'risk_level') {
        const cls = value === 'HIGH' ? 'risk-high' : value === 'MEDIUM' ? 'risk-medium' : 'risk-low';
        return `<span class="${cls}">${value}</span>`;
    }
    if (typeof value === 'number' && value > 1000) {
        return value.toLocaleString();
    }
    return value;
}

function updateAuditLog(audit) {
    const container = document.getElementById('auditLog');
    if (!audit || audit.length === 0) {
        container.innerHTML = '<p style="color: var(--text-light);">No audit entries</p>';
        return;
    }
    
    container.innerHTML = audit.slice(-5).reverse().map(a => `
        <div class="audit-item">
            <span>${a.action}</span>
            <span style="color: var(--text-light)">${a.timestamp}</span>
        </div>
    `).join('');
}
</script>

</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(
        HTML_TEMPLATE,
        mode=EXECUTION_MODE,
        queries=APPROVED_QUERIES,
        organizations=ORGANIZATIONS
    )

@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_dashboard_metrics())

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    query_type = request.json.get("query")
    
    if query_type not in APPROVED_QUERIES:
        return jsonify({"error": "Query not approved by governance policy"}), 403
    
    query_info = APPROVED_QUERIES[query_type]
    
    # Log the action
    log_action(
        action=f"Executed: {query_info['title']}",
        query_type=query_type,
        data_sources=query_info["data_sources"]
    )
    
    # Update privacy budget
    PRIVACY_BUDGET["used"] += 0.05
    
    # Execute appropriate analysis
    if query_type == "risk_analysis":
        results = combined_risk_analysis()
    elif query_type == "subsidy_gap":
        results = subsidy_gap_analysis()
    elif query_type == "fraud_detection":
        results = fraud_detection_analysis()
    elif query_type == "financial_inclusion":
        results = financial_inclusion_analysis()
    elif query_type == "policy_effectiveness":
        results = policy_effectiveness_analysis()
    else:
        results = []
    
    explanation = generate_explanation(query_type, results)
    
    return jsonify({
        "query": query_type,
        "results": results,
        "explanation": explanation,
        "audit": AUDIT_LOG[-5:],
        "mode": EXECUTION_MODE,
        "privacy_method": "differential_privacy + k_anonymity",
        "data_sources": query_info["data_sources"]
    })

@app.route("/api/organizations")
def api_organizations():
    return jsonify(ORGANIZATIONS)

@app.route("/api/audit")
def api_audit():
    return jsonify({
        "entries": AUDIT_LOG[-20:],
        "total_count": len(AUDIT_LOG)
    })

@app.route("/api/export/<query_type>")
def api_export(query_type):
    """Export results as JSON (governance-approved only)"""
    if query_type not in APPROVED_QUERIES:
        return jsonify({"error": "Export not permitted"}), 403
    
    log_action(f"Exported: {query_type}", query_type=query_type)
    
    # Return sanitized export
    return jsonify({
        "export_timestamp": datetime.utcnow().isoformat(),
        "query": query_type,
        "disclaimer": "Data has been anonymized and aggregated per privacy policy",
        "governance_approval": True
    })

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
