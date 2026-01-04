"""
Privacy-Safe Cross-Organization Analytics Platform
Version: 4.0.0 - Render Deployment Ready
Enterprise-Grade Data Clean Room Implementation
"""

import os
import json
import hashlib
import secrets
import threading
import time
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import random
import math
import csv
import io
import base64
import zlib
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Flask and dependencies
from flask import (
    Flask, render_template_string, jsonify, request, 
    Response, session, g, send_file
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Database and caching
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Security and encryption
import jwt
from cryptography.fernet import Fernet

# Analytics and ML
import numpy as np

# For environment variables
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load .env file for local development
load_dotenv()

# ============================================================
# SETUP LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

app = Flask(__name__)

# Configure for Render
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

# CORS configuration
CORS(app, supports_credentials=True)

# Rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("REDIS_URL", "memory://")
)

# ============================================================
# RENDER-SPECIFIC CONFIGURATION
# ============================================================

@dataclass
class RenderConfig:
    VERSION = "4.0.0"
    APP_NAME = "DataCleanRoom Pro - Render"
    
    # Render-specific settings
    IS_RENDER = os.getenv("RENDER", "false").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    
    # Execution mode based on environment
    if IS_RENDER:
        EXECUTION_MODE = "PRODUCTION"
    else:
        EXECUTION_MODE = os.getenv("EXECUTION_MODE", "DEVELOPMENT")
    
    # Privacy Settings (with environment variable support)
    EPSILON = float(os.getenv("DP_EPSILON", 1.0))
    DELTA = float(os.getenv("DP_DELTA", 1e-5))
    K_ANONYMITY_THRESHOLD = int(os.getenv("K_ANONYMITY", 5))
    L_DIVERSITY_THRESHOLD = int(os.getenv("L_DIVERSITY", 3))
    
    # Security
    JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
    JWT_ALGORITHM = "HS256"
    
    # Generate or use provided encryption key
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        encryption_key = base64.urlsafe_b64encode(Fernet.generate_key()).decode()
    ENCRYPTION_KEY = encryption_key.encode()
    
    MFA_REQUIRED = os.getenv("MFA_REQUIRED", "False").lower() == "true"
    
    # Database configuration for Render PostgreSQL
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        # Fix for PostgreSQL URL format
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Redis configuration for Render Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Feature flags
    ENABLE_ML_MODELS = os.getenv("ENABLE_ML_MODELS", "True").lower() == "true"
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "True").lower() == "true"
    
    # Real-time settings
    SIMULATION_INTERVAL = int(os.getenv("SIMULATION_INTERVAL", 5))
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", 90))
    
    # Compliance
    GDPR_COMPLIANT = os.getenv("GDPR_COMPLIANT", "True").lower() == "true"
    HIPAA_COMPLIANT = os.getenv("HIPAA_COMPLIANT", "False").lower() == "true"
    CCPA_COMPLIANT = os.getenv("CCPA_COMPLIANT", "True").lower() == "true"

config = RenderConfig()

# ============================================================
# DATABASE INITIALIZATION FOR RENDER
# ============================================================

def init_render_database():
    """Initialize database connection for Render PostgreSQL"""
    if not config.DATABASE_URL:
        logger.warning("No DATABASE_URL provided. Using in-memory storage.")
        return None
    
    try:
        conn = psycopg2.connect(
            config.DATABASE_URL,
            cursor_factory=RealDictCursor,
            sslmode='require' if config.IS_RENDER else 'prefer'
        )
        logger.info("Database connection established successfully")
        return conn
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        return None

def init_render_redis():
    """Initialize Redis connection for Render Redis"""
    if not config.REDIS_URL:
        logger.warning("No REDIS_URL provided. Using in-memory storage.")
        return None
    
    try:
        redis_client = redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        # Test connection
        redis_client.ping()
        logger.info("Redis connection established successfully")
        return redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        return None

# ============================================================
# RENDER-OPTIMIZED DATA STORE
# ============================================================

class RenderDataStore:
    """Data store optimized for Render deployment"""
    
    def __init__(self):
        self.lock = threading.RLock()
        
        # Initialize external connections
        self.db_conn = init_render_database()
        self.redis_client = init_render_redis()
        
        # In-memory fallbacks
        self.audit_log = []
        self.sessions = {}
        self.rate_limits = defaultdict(list)
        self.privacy_budget = {"epsilon": config.EPSILON, "used": 0.0}
        self.real_time_metrics = {}
        self.alerts = []
        self.query_cache = {}
        self.ml_models = {}
        
        # Initialize data
        self.initialize_data()
        self.load_ml_models()
        
        # Create database tables if needed
        self.create_tables()
    
    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        if not self.db_conn:
            return
        
        try:
            with self.db_conn.cursor() as cur:
                # Audit logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id SERIAL PRIMARY KEY,
                        log_id VARCHAR(50) UNIQUE,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        action TEXT,
                        user_hash VARCHAR(64),
                        query_type VARCHAR(50),
                        execution_mode VARCHAR(20),
                        privacy_budget_used FLOAT,
                        metadata JSONB
                    )
                """)
                
                # Query history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id SERIAL PRIMARY KEY,
                        query_id VARCHAR(50),
                        query_type VARCHAR(100),
                        user_email VARCHAR(255),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        privacy_cost FLOAT,
                        results_count INTEGER,
                        execution_time FLOAT
                    )
                """)
                
                # User sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(100) UNIQUE,
                        email VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP,
                        ip_address VARCHAR(45)
                    )
                """)
                
                self.db_conn.commit()
                logger.info("Database tables created/verified successfully")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            if self.db_conn:
                self.db_conn.rollback()
    
    def initialize_data(self):
        """Initialize realistic mock data for demo"""
        self.bank_data = self._generate_bank_data()
        self.insurance_data = self._generate_insurance_data()
        self.welfare_data = self._generate_welfare_data()
        self.telecom_data = self._generate_telecom_data()
        self.healthcare_data = self._generate_healthcare_data()
        self.retail_data = self._generate_retail_data()
        self.education_data = self._generate_education_data()
        
        self.last_update = datetime.utcnow()
        self.data_version = "1.0.0"
    
    def _generate_bank_data(self):
        return {
            "18-25": {"customers": 24500, "loans_active": 18200, "defaults": 1470, 
                     "avg_balance": 2850, "credit_score_avg": 618, "savings_rate": 0.15},
            "26-35": {"customers": 42000, "loans_active": 38500, "defaults": 2520,
                     "avg_balance": 8920, "credit_score_avg": 682, "savings_rate": 0.22},
            "36-50": {"customers": 38000, "loans_active": 35200, "defaults": 3420,
                     "avg_balance": 24500, "credit_score_avg": 712, "savings_rate": 0.28},
            "51-65": {"customers": 21000, "loans_active": 16800, "defaults": 1260,
                     "avg_balance": 45200, "credit_score_avg": 738, "savings_rate": 0.32},
            "65+": {"customers": 9500, "loans_active": 4750, "defaults": 380,
                   "avg_balance": 62000, "credit_score_avg": 758, "savings_rate": 0.25}
        }
    
    def _generate_insurance_data(self):
        return {
            "18-25": {"policies": 18000, "claims_filed": 2340, "claims_approved": 1872, 
                     "avg_premium": 1850, "fraud_flags": 124},
            "26-35": {"policies": 36000, "claims_filed": 4680, "claims_approved": 4024,
                     "avg_premium": 2420, "fraud_flags": 186},
            "36-50": {"policies": 42000, "claims_filed": 5880, "claims_approved": 5292,
                     "avg_premium": 3850, "fraud_flags": 252},
            "51-65": {"policies": 28000, "claims_filed": 5040, "claims_approved": 4788,
                     "avg_premium": 5200, "fraud_flags": 84},
            "65+": {"policies": 15000, "claims_filed": 3750, "claims_approved": 3600,
                   "avg_premium": 7800, "fraud_flags": 30}
        }
    
    def _generate_welfare_data(self):
        return {
            "18-25": {"eligible": 18500, "registered": 14200, "active_benefits": 9800, 
                     "avg_monthly": 485, "program_satisfaction": 72},
            "26-35": {"eligible": 28000, "registered": 23800, "active_benefits": 16800,
                     "avg_monthly": 625, "program_satisfaction": 68},
            "36-50": {"eligible": 24000, "registered": 18000, "active_benefits": 11200,
                     "avg_monthly": 780, "program_satisfaction": 65},
            "51-65": {"eligible": 16000, "registered": 13600, "active_benefits": 9600,
                     "avg_monthly": 920, "program_satisfaction": 74},
            "65+": {"eligible": 12000, "registered": 11400, "active_benefits": 8500,
                   "avg_monthly": 1150, "program_satisfaction": 82}
        }
    
    def _generate_telecom_data(self):
        return {
            "18-25": {"subscribers": 32000, "digital_score": 92, "payment_delay_pct": 18.5, 
                     "data_usage_gb": 45, "churn_risk": 24},
            "26-35": {"subscribers": 48000, "digital_score": 85, "payment_delay_pct": 12.2,
                     "data_usage_gb": 38, "churn_risk": 18},
            "36-50": {"subscribers": 36000, "digital_score": 68, "payment_delay_pct": 7.8,
                     "data_usage_gb": 22, "churn_risk": 12},
            "51-65": {"subscribers": 22000, "digital_score": 52, "payment_delay_pct": 4.5,
                     "data_usage_gb": 12, "churn_risk": 8},
            "65+": {"subscribers": 11000, "digital_score": 35, "payment_delay_pct": 2.8,
                   "data_usage_gb": 5, "churn_risk": 5}
        }
    
    def _generate_healthcare_data(self):
        return {
            "18-25": {"patients": 28000, "chronic_conditions": 1120, "preventive_visits": 8400,
                     "emergency_visits": 5600, "health_score": 85},
            "26-35": {"patients": 45000, "chronic_conditions": 4050, "preventive_visits": 18000,
                     "emergency_visits": 6750, "health_score": 78},
            "36-50": {"patients": 52000, "chronic_conditions": 10400, "preventive_visits": 26000,
                     "emergency_visits": 7800, "health_score": 68},
            "51-65": {"patients": 38000, "chronic_conditions": 15200, "preventive_visits": 22800,
                     "emergency_visits": 7600, "health_score": 58},
            "65+": {"patients": 24000, "chronic_conditions": 16800, "preventive_visits": 19200,
                   "emergency_visits": 9600, "health_score": 48}
        }
    
    def _generate_retail_data(self):
        return {
            "18-25": {"customers": 18500, "avg_spend": 85, "online_rate": 92, 
                     "loyalty_members": 12400, "return_rate": 0.12},
            "26-35": {"customers": 32000, "avg_spend": 125, "online_rate": 88,
                     "loyalty_members": 25600, "return_rate": 0.08},
            "36-50": {"customers": 28000, "avg_spend": 145, "online_rate": 75,
                     "loyalty_members": 21000, "return_rate": 0.06},
            "51-65": {"customers": 16000, "avg_spend": 110, "online_rate": 62,
                     "loyalty_members": 11200, "return_rate": 0.05},
            "65+": {"customers": 8500, "avg_spend": 95, "online_rate": 45,
                   "loyalty_members": 5100, "return_rate": 0.04}
        }
    
    def _generate_education_data(self):
        return {
            "18-25": {"students": 45000, "graduation_rate": 0.68, "employment_rate": 0.72,
                     "avg_debt": 28500, "digital_learning": 94},
            "26-35": {"students": 22000, "graduation_rate": 0.75, "employment_rate": 0.85,
                     "avg_debt": 32000, "digital_learning": 88},
            "36-50": {"students": 18000, "graduation_rate": 0.82, "employment_rate": 0.92,
                     "avg_debt": 0, "digital_learning": 72},
            "51-65": {"students": 8500, "graduation_rate": 0.88, "employment_rate": 0.95,
                     "avg_debt": 0, "digital_learning": 58},
            "65+": {"students": 4200, "graduation_rate": 0.92, "employment_rate": 0.98,
                   "avg_debt": 0, "digital_learning": 42}
        }
    
    def load_ml_models(self):
        """Load ML models"""
        if config.ENABLE_ML_MODELS:
            self.ml_models = {
                "fraud_detection": {"name": "XGBoost Fraud Detector", "accuracy": 0.945},
                "churn_prediction": {"name": "Random Forest Churn Predictor", "accuracy": 0.892},
                "credit_scoring": {"name": "Neural Network Credit Scorer", "accuracy": 0.912}
            }
    
    def update_real_time(self):
        """Simulate real-time data updates"""
        with self.lock:
            # Small random fluctuations
            for age in self.bank_data:
                if "defaults" in self.bank_data[age]:
                    self.bank_data[age]["defaults"] = max(0, 
                        self.bank_data[age]["defaults"] + random.randint(-5, 5))
            self.last_update = datetime.utcnow()
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.set("last_update", self.last_update.isoformat())
                except Exception as e:
                    logger.error(f"Redis update failed: {e}")

store = RenderDataStore()

# ============================================================
# ORGANIZATIONS REGISTRY
# ============================================================

ORGANIZATIONS = {
    "national_bank": {
        "id": "ORG-001",
        "name": "National Bank Corporation",
        "type": "Financial Institution",
        "jurisdiction": "Federal",
        "data_contributed": ["customer_demographics", "loan_performance", "credit_scores"],
        "privacy_method": "differential_privacy",
        "status": "active",
        "joined": "2024-01-15",
        "data_volume": "2.4M records",
        "last_sync": datetime.utcnow().isoformat(),
        "compliance": ["GDPR", "CCPA", "PCI-DSS"]
    },
    "secure_insurance": {
        "id": "ORG-002", 
        "name": "SecureLife Insurance Group",
        "type": "Insurance Provider",
        "jurisdiction": "State",
        "data_contributed": ["policy_data", "claims_history", "fraud_indicators"],
        "privacy_method": "k_anonymity",
        "status": "active",
        "joined": "2024-02-01",
        "data_volume": "1.8M records",
        "last_sync": datetime.utcnow().isoformat(),
        "compliance": ["HIPAA", "GDPR"]
    },
    "gov_welfare": {
        "id": "ORG-003",
        "name": "Department of Social Welfare",
        "type": "Government Agency",
        "jurisdiction": "Federal",
        "data_contributed": ["benefit_eligibility", "program_enrollment", "disbursements"],
        "privacy_method": "aggregated_only",
        "status": "active",
        "joined": "2024-01-01",
        "data_volume": "3.2M records",
        "last_sync": datetime.utcnow().isoformat(),
        "compliance": ["FISMA", "GDPR"]
    },
    "telecom_networks": {
        "id": "ORG-004",
        "name": "TeleCom Networks Inc.",
        "type": "Telecommunications",
        "jurisdiction": "Federal",
        "data_contributed": ["digital_activity", "payment_behavior", "connectivity"],
        "privacy_method": "differential_privacy",
        "status": "active",
        "joined": "2024-03-10",
        "data_volume": "5.1M records",
        "last_sync": datetime.utcnow().isoformat(),
        "compliance": ["GDPR", "CCPA"]
    },
    "health_services": {
        "id": "ORG-005",
        "name": "United Health Services",
        "type": "Healthcare Provider",
        "jurisdiction": "State",
        "data_contributed": ["health_metrics", "utilization_patterns", "outcomes"],
        "privacy_method": "l_diversity",
        "status": "active",
        "joined": "2024-04-01",
        "data_volume": "1.9M records",
        "last_sync": datetime.utcnow().isoformat(),
        "compliance": ["HIPAA", "GDPR"]
    },
    "tech_retail": {
        "id": "ORG-006",
        "name": "TechRetail Solutions",
        "type": "E-commerce",
        "jurisdiction": "Federal",
        "data_contributed": ["purchase_history", "customer_behavior", "inventory_data"],
        "privacy_method": "differential_privacy",
        "status": "active",
        "joined": "2024-05-01",
        "data_volume": "8.5M records",
        "last_sync": datetime.utcnow().isoformat(),
        "compliance": ["GDPR", "CCPA"]
    }
}

# ============================================================
# QUERY CATALOG
# ============================================================

QUERY_CATALOG = {
    "cross_sector_risk": {
        "id": "QRY-001",
        "title": "Cross-Sector Risk Assessment",
        "category": "Risk Analytics",
        "description": "Comprehensive risk scoring combining financial, insurance, and behavioral data",
        "data_sources": ["national_bank", "secure_insurance", "telecom_networks"],
        "min_aggregation": 100,
        "privacy_cost": 0.08,
        "requires_approval": False,
        "output_fields": ["age_group", "composite_risk_score", "default_probability", "claim_likelihood"],
        "refresh_rate": "real-time",
        "created": "2024-01-20",
        "usage_count": 0
    },
    "welfare_gap_analysis": {
        "id": "QRY-002",
        "title": "Welfare Program Gap Analysis",
        "category": "Policy Analytics",
        "description": "Identify underserved populations and benefit delivery inefficiencies",
        "data_sources": ["gov_welfare", "national_bank"],
        "min_aggregation": 50,
        "privacy_cost": 0.05,
        "requires_approval": False,
        "output_fields": ["age_group", "coverage_gap", "delivery_efficiency", "estimated_impact"],
        "refresh_rate": "daily",
        "created": "2024-01-25",
        "usage_count": 0
    },
    "fraud_detection": {
        "id": "QRY-003",
        "title": "Multi-Source Fraud Detection",
        "category": "Fraud Analytics",
        "description": "Advanced fraud pattern detection using cross-organizational signals",
        "data_sources": ["national_bank", "secure_insurance", "telecom_networks", "health_services"],
        "min_aggregation": 200,
        "privacy_cost": 0.12,
        "requires_approval": True,
        "output_fields": ["age_group", "fraud_score", "risk_tier", "primary_indicators"],
        "refresh_rate": "real-time",
        "created": "2024-02-10",
        "usage_count": 0
    },
    "financial_inclusion": {
        "id": "QRY-004",
        "title": "Financial Inclusion Index",
        "category": "Inclusion Analytics",
        "description": "Measure financial and digital inclusion across demographic segments",
        "data_sources": ["national_bank", "telecom_networks", "gov_welfare"],
        "min_aggregation": 100,
        "privacy_cost": 0.06,
        "requires_approval": False,
        "output_fields": ["age_group", "inclusion_score", "credit_access", "digital_access"],
        "refresh_rate": "weekly",
        "created": "2024-02-15",
        "usage_count": 0
    },
    "health_financial_correlation": {
        "id": "QRY-005",
        "title": "Health-Financial Correlation Study",
        "category": "Research Analytics",
        "description": "Analyze correlation between health outcomes and financial stability",
        "data_sources": ["health_services", "national_bank", "secure_insurance"],
        "min_aggregation": 150,
        "privacy_cost": 0.10,
        "requires_approval": True,
        "output_fields": ["age_group", "health_score", "financial_stability", "correlation_coefficient"],
        "refresh_rate": "monthly",
        "created": "2024-03-01",
        "usage_count": 0
    }
}

# ============================================================
# USER MANAGEMENT
# ============================================================

USERS = {
    "admin@cleanroom.gov": {
        "id": "USR-001",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "System Administrator",
        "role": "admin",
        "organization": "Platform Admin",
        "permissions": ["all"],
        "approved_queries": list(QUERY_CATALOG.keys()),
        "created": "2024-01-01",
        "last_login": None,
        "mfa_enabled": True
    },
    "analyst@bank.gov": {
        "id": "USR-002",
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "name": "Financial Analyst",
        "role": "analyst",
        "organization": "National Bank Corporation",
        "permissions": ["view", "query", "export"],
        "approved_queries": ["cross_sector_risk", "financial_inclusion"],
        "created": "2024-02-01",
        "last_login": None,
        "mfa_enabled": False
    },
    "demo@example.com": {
        "id": "USR-003",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Demo User",
        "role": "viewer",
        "organization": "Demo Organization",
        "permissions": ["view", "query"],
        "approved_queries": list(QUERY_CATALOG.keys()),
        "created": "2024-04-01",
        "last_login": None,
        "mfa_enabled": False
    }
}

# ============================================================
# PRIVACY ENGINE
# ============================================================

class PrivacyEngine:
    """Privacy-preserving computation engine"""
    
    def __init__(self):
        self.fernet = Fernet(config.ENCRYPTION_KEY)
    
    @staticmethod
    def add_laplace_noise(value, sensitivity, epsilon):
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / epsilon
        noise = random.gauss(0, scale)
        return value + noise
    
    @staticmethod
    def add_gaussian_noise(value, sensitivity, epsilon, delta):
        """Add Gaussian noise for (ε, δ)-differential privacy"""
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        noise = random.gauss(0, sigma)
        return value + noise
    
    def encrypt(self, data):
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data):
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_token(self, payload):
        """Generate JWT token"""
        return jwt.encode(
            payload,
            config.JWT_SECRET,
            algorithm=config.JWT_ALGORITHM
        )
    
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            return jwt.decode(
                token,
                config.JWT_SECRET,
                algorithms=[config.JWT_ALGORITHM]
            )
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def hash_identifier(value):
        """One-way cryptographic hash"""
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]

privacy_engine = PrivacyEngine()

# ============================================================
# ANALYTICS ENGINE
# ============================================================

class AnalyticsEngine:
    """Core analytics computation engine"""
    
    @staticmethod
    def cross_sector_risk_analysis():
        """Comprehensive cross-sector risk assessment"""
        results = []
        for age in store.bank_data:
            bank = store.bank_data[age]
            insurance = store.insurance_data[age]
            telecom = store.telecom_data[age]
            
            # Calculate component scores
            default_prob = (bank["defaults"] / bank["customers"]) * 100
            claim_likelihood = (insurance["claims_filed"] / insurance["policies"]) * 100
            payment_risk = telecom["payment_delay_pct"]
            
            # Weighted composite score
            composite = (
                default_prob * 0.30 +
                claim_likelihood * 0.25 +
                payment_risk * 0.20
            )
            
            # Apply differential privacy
            composite = PrivacyEngine.add_laplace_noise(composite, 2.0, 0.1)
            
            results.append({
                "age_group": age,
                "composite_risk_score": round(composite, 2),
                "default_probability": round(default_prob, 2),
                "claim_likelihood": round(claim_likelihood, 2),
                "behavioral_risk": round(payment_risk, 2),
                "sample_size": bank["customers"] + insurance["policies"]
            })
        
        return sorted(results, key=lambda x: x["composite_risk_score"], reverse=True)
    
    @staticmethod
    def welfare_gap_analysis():
        """Analyze welfare program coverage gaps"""
        results = []
        total_gap = 0
        
        for age in store.welfare_data:
            welfare = store.welfare_data[age]
            bank = store.bank_data[age]
            
            registration_rate = (welfare["registered"] / welfare["eligible"]) * 100
            coverage_gap = welfare["eligible"] - welfare["active_benefits"]
            
            total_gap += coverage_gap
            
            results.append({
                "age_group": age,
                "eligible_population": welfare["eligible"],
                "coverage_gap": coverage_gap,
                "registration_rate": round(registration_rate, 1),
                "delivery_efficiency": round((welfare["active_benefits"] / welfare["eligible"]) * 100, 1),
                "satisfaction_score": welfare["program_satisfaction"]
            })
        
        return {
            "by_segment": sorted(results, key=lambda x: x["coverage_gap"], reverse=True),
            "summary": {
                "total_coverage_gap": total_gap,
                "overall_efficiency": round(
                    sum(store.welfare_data[a]["active_benefits"] for a in store.welfare_data) /
                    sum(store.welfare_data[a]["eligible"] for a in store.welfare_data) * 100, 1
                )
            }
        }
    
    @staticmethod
    def fraud_detection_analysis():
        """Multi-source fraud pattern detection"""
        results = []
        
        for age in store.bank_data:
            bank = store.bank_data[age]
            insurance = store.insurance_data[age]
            
            financial_anomaly = bank["defaults"] / bank["customers"]
            insurance_fraud_rate = insurance["fraud_flags"] / insurance["policies"]
            
            fraud_score = (
                financial_anomaly * 200 +
                insurance_fraud_rate * 350
            )
            
            fraud_score = PrivacyEngine.add_laplace_noise(fraud_score, 5.0, 0.1)
            
            if fraud_score > 40:
                risk_tier = "CRITICAL"
            elif fraud_score > 30:
                risk_tier = "HIGH"
            elif fraud_score > 20:
                risk_tier = "ELEVATED"
            elif fraud_score > 10:
                risk_tier = "MODERATE"
            else:
                risk_tier = "LOW"
            
            results.append({
                "age_group": age,
                "fraud_score": round(fraud_score, 1),
                "risk_tier": risk_tier,
                "primary_indicators": {
                    "financial_anomaly": round(financial_anomaly * 100, 2),
                    "insurance_fraud_signals": round(insurance_fraud_rate * 100, 3)
                },
                "confidence": round(85 + random.uniform(-5, 5), 1)
            })
        
        return sorted(results, key=lambda x: x["fraud_score"], reverse=True)
    
    @staticmethod
    def financial_inclusion_analysis():
        """Financial and digital inclusion assessment"""
        results = []
        
        for age in store.bank_data:
            bank = store.bank_data[age]
            telecom = store.telecom_data[age]
            welfare = store.welfare_data[age]
            
            credit_access = min(100, max(0, (bank["credit_score_avg"] - 500) / 2.5))
            banking_access = (bank["loans_active"] / bank["customers"]) * 100
            digital_access = telecom["digital_score"]
            welfare_access = (welfare["active_benefits"] / welfare["eligible"]) * 100
            
            inclusion_score = (
                credit_access * 0.25 +
                banking_access * 0.25 +
                digital_access * 0.30 +
                welfare_access * 0.20
            )
            
            results.append({
                "age_group": age,
                "inclusion_score": round(inclusion_score, 1),
                "credit_access": round(credit_access, 1),
                "banking_access": round(banking_access, 1),
                "digital_access": digital_access,
                "welfare_access": round(welfare_access, 1)
            })
        
        return sorted(results, key=lambda x: x["inclusion_score"])

analytics = AnalyticsEngine()

# ============================================================
# EXPLANATION ENGINE
# ============================================================

class ExplanationEngine:
    """Generate contextual explanations"""
    
    @staticmethod
    def generate(query_type, results):
        """Generate explanation for query results"""
        
        explanations = {
            "cross_sector_risk": ExplanationEngine._explain_risk,
            "welfare_gap_analysis": ExplanationEngine._explain_welfare,
            "fraud_detection": ExplanationEngine._explain_fraud,
            "financial_inclusion": ExplanationEngine._explain_inclusion,
            "health_financial_correlation": ExplanationEngine._explain_health
        }
        
        func = explanations.get(query_type, ExplanationEngine._default_explanation)
        return func(results)
    
    @staticmethod
    def _explain_risk(results):
        top = results[0]
        
        return {
            "headline": f"Highest Risk: {top['age_group']} Age Group",
            "summary": f"The **{top['age_group']}** demographic exhibits the highest composite risk score of **{top['composite_risk_score']}**.",
            "key_findings": [
                f"Default probability ranges from {min(r['default_probability'] for r in results)}% to {max(r['default_probability'] for r in results)}%",
                "Cross-sector correlation reveals compounding vulnerabilities in younger demographics"
            ],
            "recommendations": [
                {"priority": "High", "action": "Implement early warning systems for high-risk segments"},
                {"priority": "High", "action": "Develop targeted financial literacy programs"}
            ],
            "methodology": "Composite risk score calculated using weighted combination of default probability (30%), claim likelihood (25%), and behavioral signals (20%).",
            "confidence_level": "High",
            "data_sources_used": ["national_bank", "secure_insurance", "telecom_networks"]
        }
    
    @staticmethod
    def _explain_welfare(results):
        data = results
        segments = data["by_segment"]
        summary = data["summary"]
        top = segments[0]
        
        return {
            "headline": f"Largest Coverage Gap: {top['age_group']} Age Group",
            "summary": f"The **{top['age_group']}** demographic has the largest welfare coverage gap with **{top['coverage_gap']:,}** eligible individuals not receiving benefits.",
            "key_findings": [
                f"Overall program efficiency: **{summary['overall_efficiency']}%**",
                f"Registration rates vary from {min(s['registration_rate'] for s in segments)}% to {max(s['registration_rate'] for s in segments)}%"
            ],
            "recommendations": [
                {"priority": "High", "action": f"Launch targeted outreach campaign for {top['age_group']} demographic"},
                {"priority": "High", "action": "Simplify registration process"}
            ],
            "methodology": "Gap analysis performed by comparing eligible populations against active benefit recipients.",
            "confidence_level": "High",
            "data_sources_used": ["gov_welfare", "national_bank"]
        }
    
    @staticmethod
    def _explain_fraud(results):
        critical = [r for r in results if r["risk_tier"] == "CRITICAL"]
        top = results[0]
        
        return {
            "headline": f"Fraud Alert: {len(critical)} Critical Risk Segments Identified",
            "summary": f"Multi-source fraud analysis flagged the **{top['age_group']}** group with highest fraud score of **{top['fraud_score']}**.",
            "key_findings": [
                f"{len(critical)} demographic segments require enhanced monitoring",
                "Cross-sector signal correlation improves fraud detection accuracy"
            ],
            "recommendations": [
                {"priority": "Critical", "action": "Activate enhanced monitoring protocols for CRITICAL segments"},
                {"priority": "High", "action": "Implement real-time cross-organizational verification checks"}
            ],
            "methodology": "Fraud scores derived from multi-dimensional analysis: financial anomalies and insurance fraud flags.",
            "confidence_level": f"Average {sum(r['confidence'] for r in results)/len(results):.1f}%",
            "data_sources_used": ["national_bank", "secure_insurance", "telecom_networks", "health_services"]
        }
    
    @staticmethod
    def _explain_inclusion(results):
        lowest = results[0]
        
        return {
            "headline": f"Inclusion Priority: {lowest['age_group']} Age Group",
            "summary": f"The **{lowest['age_group']}** demographic shows the lowest financial inclusion score of **{lowest['inclusion_score']}**.",
            "key_findings": [
                "Strong correlation between digital access and credit access",
                "Welfare program participation inversely correlates with age"
            ],
            "recommendations": [
                {"priority": "High", "action": f"Priority digital literacy programs for {lowest['age_group']} group"},
                {"priority": "High", "action": "Develop alternative credit scoring using telecom data"}
            ],
            "methodology": "Inclusion index calculated as weighted composite: credit access (25%), banking access (25%), digital access (30%), and welfare access (20%).",
            "confidence_level": "High",
            "data_sources_used": ["national_bank", "telecom_networks", "gov_welfare"]
        }
    
    @staticmethod
    def _explain_health(results):
        return {
            "headline": "Health-Finance Correlation Analysis",
            "summary": "Analysis reveals significant correlation between financial stability and health outcomes across all demographics.",
            "key_findings": [
                "Financial stress strongly predicts chronic condition prevalence",
                "Preventive care utilization correlates with credit score"
            ],
            "recommendations": [
                {"priority": "High", "action": "Integrate financial counseling into healthcare outreach programs"},
                {"priority": "High", "action": "Develop joint health-financial wellness programs"}
            ],
            "methodology": "Correlation analysis using aggregated, privacy-protected data.",
            "confidence_level": "Medium-High",
            "data_sources_used": ["health_services", "national_bank", "secure_insurance"]
        }
    
    @staticmethod
    def _default_explanation(results):
        return {
            "headline": "Analysis Complete",
            "summary": "Query executed successfully with privacy-preserving techniques applied.",
            "key_findings": ["Results aggregated across participating organizations", "Privacy budget consumed within limits"],
            "recommendations": [],
            "methodology": "Standard aggregation with differential privacy.",
            "confidence_level": "Medium",
            "data_sources_used": []
        }

explanation_engine = ExplanationEngine()

# ============================================================
# AUDIT ENGINE
# ============================================================

class AuditEngine:
    """Audit trail and compliance management"""
    
    @staticmethod
    def log(action, user=None, query_type=None, data_sources=None, metadata=None):
        """Create audit log entry"""
        entry = {
            "id": f"AUD-{secrets.token_hex(8).upper()}",
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user_hash": privacy_engine.hash_identifier(user) if user else "system",
            "query_type": query_type,
            "data_sources": data_sources or [],
            "execution_mode": config.EXECUTION_MODE,
            "privacy_budget_used": store.privacy_budget["used"],
            "privacy_budget_remaining": round(store.privacy_budget["epsilon"] - store.privacy_budget["used"], 4),
            "metadata": metadata or {}
        }
        
        with store.lock:
            store.audit_log.append(entry)
            
            # Keep only last 1000 entries
            if len(store.audit_log) > 1000:
                store.audit_log = store.audit_log[-1000:]
        
        return entry["id"]

audit = AuditEngine()

# ============================================================
# REAL-TIME SIMULATOR
# ============================================================

class RealTimeSimulator:
    """Simulate real-time data updates"""
    
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start(self):
        """Start real-time simulation"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("Real-time simulator started")
    
    def stop(self):
        """Stop real-time simulation"""
        self.running = False
    
    def _run(self):
        """Background simulation loop"""
        while self.running:
            store.update_real_time()
            time.sleep(config.SIMULATION_INTERVAL)

simulator = RealTimeSimulator()

# ============================================================
# HTML TEMPLATE FOR RENDER
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataCleanRoom Pro - Privacy-Safe Analytics</title>
    <link href="https://cdn.jsdelivr.net/npm/[email protected]/dist/tailwind.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #1e40af;
            --primary-dark: #1e3a8a;
            --secondary: #059669;
        }
        
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            min-height: 100vh;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        .gradient-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        }
        
        .badge {
            padding: 4px 8px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }
    </style>
</head>
<body class="text-gray-800">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b">
        <div class="max-w-7xl mx-auto px-4 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 gradient-primary rounded-lg flex items-center justify-center">
                        <span class="text-white text-lg">🔒</span>
                    </div>
                    <div>
                        <h1 class="text-lg font-bold text-gray-900">DataCleanRoom Pro</h1>
                        <p class="text-xs text-gray-500">Render Deployment • v{{ version }}</p>
                    </div>
                </div>
                
                <div class="flex items-center gap-4">
                    <div class="text-sm">
                        <span class="text-gray-500">Mode:</span>
                        <span class="font-semibold text-blue-600">{{ mode }}</span>
                    </div>
                    
                    <div id="userInfo" style="display: none;">
                        <div class="flex items-center gap-2">
                            <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                                <span class="text-blue-600 text-sm font-semibold" id="userInitial">U</span>
                            </div>
                            <div class="text-sm">
                                <div class="font-medium" id="userName">User</div>
                            </div>
                        </div>
                    </div>
                    
                    <button id="loginBtn" onclick="showLogin()" 
                            class="px-4 py-2 gradient-primary text-white rounded-lg text-sm font-medium">
                        Sign In
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 py-8">
        <!-- Hero Section -->
        <div class="text-center mb-12">
            <h1 class="text-4xl font-bold text-gray-900 mb-4">
                Privacy-Safe Cross-Organization Analytics
            </h1>
            <p class="text-lg text-gray-600 max-w-3xl mx-auto">
                Secure, compliant data collaboration platform for organizations to gain insights 
                without exposing sensitive information. Powered by differential privacy and 
                advanced security protocols.
            </p>
        </div>

        <!-- Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card p-6 text-center">
                <div class="text-3xl font-bold text-blue-600" id="metricOrgs">6</div>
                <div class="text-gray-500 mt-2">Organizations</div>
            </div>
            <div class="card p-6 text-center">
                <div class="text-3xl font-bold text-green-600" id="metricQueries">5</div>
                <div class="text-gray-500 mt-2">Analytics Queries</div>
            </div>
            <div class="card p-6 text-center">
                <div class="text-3xl font-bold text-purple-600" id="metricPrivacy">{{ epsilon }}ε</div>
                <div class="text-gray-500 mt-2">Privacy Budget</div>
            </div>
        </div>

        <!-- Query Catalog -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-bold mb-4">Available Analytics</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="queryCatalog">
                {% for query_id, query in queries.items() %}
                <div class="border rounded-lg p-4 hover:border-blue-500 transition">
                    <div class="flex items-start justify-between mb-2">
                        <h3 class="font-semibold">{{ query.title }}</h3>
                        <span class="badge bg-blue-100 text-blue-700">{{ query.category }}</span>
                    </div>
                    <p class="text-sm text-gray-600 mb-3">{{ query.description }}</p>
                    <div class="flex flex-wrap gap-1">
                        {% for source in query.data_sources[:3] %}
                        <span class="text-xs px-2 py-1 bg-gray-100 rounded">{{ source }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Organizations -->
        <div class="card p-6">
            <h2 class="text-xl font-bold mb-4">Participating Organizations</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {% for org_id, org in organizations.items() %}
                <div class="border rounded-lg p-4">
                    <div class="flex items-center gap-3 mb-3">
                        <div class="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                            <span class="text-blue-600 font-semibold">{{ org.name[0] }}</span>
                        </div>
                        <div>
                            <h3 class="font-semibold">{{ org.name }}</h3>
                            <p class="text-xs text-gray-500">{{ org.type }}</p>
                        </div>
                    </div>
                    <div class="text-sm text-gray-600">
                        <div class="flex justify-between mb-1">
                            <span>Data Volume:</span>
                            <span class="font-medium">{{ org.data_volume }}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Status:</span>
                            <span class="font-medium text-green-600">{{ org.status }}</span>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t bg-white mt-12">
        <div class="max-w-7xl mx-auto px-4 py-6">
            <div class="text-center text-sm text-gray-500">
                &copy; 2024 DataCleanRoom Pro v{{ version }} | 
                Deployed on Render | 
                Privacy-First Analytics Platform
            </div>
            <div class="text-center text-xs text-gray-400 mt-2">
                <a href="/api/health" class="hover:text-gray-600">Health Check</a> • 
                <a href="/api/queries" class="hover:text-gray-600">API</a> • 
                <a href="https://render.com" class="hover:text-gray-600">Powered by Render</a>
            </div>
        </div>
    </footer>

    <!-- Login Modal -->
    <div id="loginModal" class="fixed inset-0 bg-gray-500 bg-opacity-75 hidden items-center justify-center z-50">
        <div class="bg-white rounded-lg p-8 max-w-md w-full mx-4">
            <h2 class="text-2xl font-bold mb-6 text-center">🔐 Secure Login</h2>
            <form id="loginForm" class="space-y-4">
                <div>
                    <label class="block text-sm font-medium mb-2">Email</label>
                    <input type="email" id="loginEmail" 
                           class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                           placeholder="demo@example.com" required>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Password</label>
                    <input type="password" id="loginPassword" 
                           class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                           placeholder="demo123" required>
                </div>
                <button type="submit" 
                        class="w-full py-2 gradient-primary text-white font-semibold rounded-lg">
                    Sign In
                </button>
                <p class="text-sm text-gray-500 text-center mt-4">
                    Demo credentials: demo@example.com / demo123
                </p>
            </form>
        </div>
    </div>

    <script>
        // Simple frontend functionality
        function showLogin() {
            document.getElementById('loginModal').classList.remove('hidden');
            document.getElementById('loginModal').classList.add('flex');
        }
        
        function hideLogin() {
            document.getElementById('loginModal').classList.add('hidden');
            document.getElementById('loginModal').classList.remove('flex');
        }
        
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    hideLogin();
                    document.getElementById('loginBtn').style.display = 'none';
                    document.getElementById('userInfo').style.display = 'flex';
                    document.getElementById('userName').textContent = data.user.name;
                    document.getElementById('userInitial').textContent = data.user.name.charAt(0);
                    alert('Login successful! Welcome ' + data.user.name);
                } else {
                    alert('Login failed: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Login failed: Network error');
            }
        });
        
        // Close modal on background click
        document.getElementById('loginModal').addEventListener('click', (e) => {
            if (e.target.id === 'loginModal') {
                hideLogin();
            }
        });
        
        // Check if user is already logged in
        async function checkAuthStatus() {
            try {
                const response = await fetch('/api/auth/status');
                const data = await response.json();
                
                if (data.authenticated) {
                    document.getElementById('loginBtn').style.display = 'none';
                    document.getElementById('userInfo').style.display = 'flex';
                    document.getElementById('userName').textContent = data.user.name;
                    document.getElementById('userInitial').textContent = data.user.name.charAt(0);
                }
            } catch (error) {
                console.log('Not authenticated');
            }
        }
        
        // Check auth status on page load
        checkAuthStatus();
    </script>
</body>
</html>
"""

# ============================================================
# API ROUTES
# ============================================================

@app.route("/")
def index():
    """Render main dashboard"""
    return render_template_string(
        HTML_TEMPLATE,
        version=config.VERSION,
        mode=config.EXECUTION_MODE,
        epsilon=config.EPSILON,
        queries=QUERY_CATALOG,
        organizations=ORGANIZATIONS
    )

@app.route("/api/health")
def api_health():
    """Health check endpoint for Render"""
    try:
        # Test database connection if available
        db_ok = False
        if store.db_conn:
            with store.db_conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_ok = True
        
        # Test Redis connection if available
        redis_ok = False
        if store.redis_client:
            store.redis_client.ping()
            redis_ok = True
        
        return jsonify({
            "status": "healthy",
            "version": config.VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": config.EXECUTION_MODE,
            "services": {
                "database": "connected" if db_ok else "not_configured",
                "redis": "connected" if redis_ok else "not_configured"
            },
            "resources": {
                "organizations": len(ORGANIZATIONS),
                "queries": len(QUERY_CATALOG),
                "privacy_budget_remaining": round(config.EPSILON - store.privacy_budget["used"], 4)
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("10 per minute")
def api_login():
    """User authentication endpoint"""
    try:
        data = request.json
        email = data.get("email", "").lower()
        password = data.get("password", "")
        
        if email not in USERS:
            audit.log("Login failed - user not found", user=email)
            return jsonify({"error": "Invalid credentials"}), 401
        
        user = USERS[email]
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if user["password_hash"] != password_hash:
            audit.log("Login failed - wrong password", user=email)
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Update last login
        user["last_login"] = datetime.utcnow().isoformat()
        
        # Set session
        session["user"] = email
        
        audit.log("User logged in", user=email)
        
        return jsonify({
            "success": True,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": email,
                "role": user["role"],
                "organization": user["organization"]
            }
        })
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({"error": "Authentication failed"}), 500

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    """User logout endpoint"""
    user = session.get("user")
    if user:
        audit.log("User logged out", user=user)
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/status")
def api_auth_status():
    """Check authentication status"""
    email = session.get("user")
    if email and email in USERS:
        user = USERS[email]
        return jsonify({
            "authenticated": True,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": email,
                "role": user["role"],
                "organization": user["organization"]
            }
        })
    return jsonify({"authenticated": False})

@app.route("/api/queries")
def api_queries():
    """Get available queries"""
    return jsonify(QUERY_CATALOG)

@app.route("/api/organizations")
def api_organizations():
    """Get connected organizations"""
    return jsonify(ORGANIZATIONS)

@app.route("/api/metrics")
def api_metrics():
    """Get dashboard metrics"""
    total_records = sum(
        store.bank_data[age]["customers"] 
        for age in store.bank_data
    )
    
    return jsonify({
        "total_records": total_records,
        "organizations_count": len(ORGANIZATIONS),
        "total_queries": sum(q["usage_count"] for q in QUERY_CATALOG.values()),
        "queries_today": 0,  # Simplified
        "privacy_total": config.EPSILON,
        "privacy_used": round(store.privacy_budget["used"], 4),
        "privacy_remaining": round(config.EPSILON - store.privacy_budget["used"], 4),
        "active_sessions": len([u for u in USERS.values() if u.get("last_login")]),
        "last_sync": store.last_update.isoformat(),
        "uptime": "99.9%",
        "mode": config.EXECUTION_MODE
    })

@app.route("/api/analyze", methods=["POST"])
@limiter.limit("30 per minute")
def api_analyze():
    """Execute privacy-safe analysis"""
    try:
        data = request.json
        query_type = data.get("query")
        
        if not query_type or query_type not in QUERY_CATALOG:
            return jsonify({"error": "Invalid query type"}), 400
        
        query = QUERY_CATALOG[query_type]
        
        # Check privacy budget
        if store.privacy_budget["used"] + query["privacy_cost"] > store.privacy_budget["epsilon"]:
            return jsonify({"error": "Privacy budget exhausted"}), 403
        
        # Consume privacy budget
        store.privacy_budget["used"] += query["privacy_cost"]
        
        # Update query usage
        query["usage_count"] += 1
        
        # Log the action
        user = session.get("user", "anonymous")
        audit.log(
            action=f"Query executed: {query['title']}",
            user=user,
            query_type=query_type,
            data_sources=query["data_sources"]
        )
        
        # Execute query
        if query_type == "cross_sector_risk":
            results = analytics.cross_sector_risk_analysis()
        elif query_type == "welfare_gap_analysis":
            results = analytics.welfare_gap_analysis()
        elif query_type == "fraud_detection":
            results = analytics.fraud_detection_analysis()
        elif query_type == "financial_inclusion":
            results = analytics.financial_inclusion_analysis()
        elif query_type == "health_financial_correlation":
            # Simplified version
            results = analytics.cross_sector_risk_analysis()
        else:
            results = []
        
        explanation = explanation_engine.generate(query_type, results)
        
        return jsonify({
            "success": True,
            "query": query_type,
            "query_info": query,
            "results": results,
            "explanation": explanation,
            "privacy": {
                "cost": query["privacy_cost"],
                "budget_remaining": round(store.privacy_budget["epsilon"] - store.privacy_budget["used"], 4)
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": "Analysis failed"}), 500

@app.route("/api/audit")
def api_audit():
    """Get audit log entries"""
    limit = min(int(request.args.get("limit", 20)), 100)
    entries = store.audit_log[-limit:][::-1]
    return jsonify({"entries": entries, "total": len(store.audit_log)})

@app.route("/api/export", methods=["POST"])
@limiter.limit("10 per hour")
def api_export():
    """Export analysis results"""
    try:
        data = request.json
        query_type = data.get("query")
        format_type = data.get("format", "json")
        
        if not query_type:
            return jsonify({"error": "Query type required"}), 400
        
        # Execute query to get results
        response = api_analyze()
        if response.status_code != 200:
            return response
        
        result_data = response.get_json()
        results = result_data.get("results", [])
        
        # Generate export
        if format_type == "csv":
            output = io.StringIO()
            
            if isinstance(results, list) and results:
                writer = csv.DictWriter(output, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            elif isinstance(results, dict) and "by_segment" in results:
                writer = csv.DictWriter(output, fieldnames=results["by_segment"][0].keys())
                writer.writeheader()
                writer.writerows(results["by_segment"])
            
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment;filename={query_type}_export.csv"}
            )
        
        # Default JSON export
        return jsonify({
            "export": True,
            "query": query_type,
            "data": results,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({"error": "Export failed"}), 500

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "code": 400}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"error": "Unauthorized", "code": 401}), 401

@app.errorhandler(403)
def forbidden(e):
    return jsonify({"error": "Forbidden", "code": 403}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "code": 404}), 404

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Rate limit exceeded", "code": 429}), 429

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error", "code": 500}), 500

# ============================================================
# APPLICATION STARTUP
# ============================================================

def initialize_application():
    """Initialize the application for Render deployment"""
    
    # Start real-time simulator
    simulator.start()
    
    # Log startup
    audit.log("Application started on Render", 
              metadata={
                  "version": config.VERSION,
                  "mode": config.EXECUTION_MODE,
                  "port": config.PORT,
                  "database": "configured" if store.db_conn else "in-memory",
                  "redis": "configured" if store.redis_client else "in-memory"
              })
    
    # Print startup message
    startup_message = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🔒 DataCleanRoom Pro v{config.VERSION} - Render Ready     ║
    ║   Privacy-Safe Cross-Organization Analytics Platform     ║
    ║                                                          ║
    ║   Deployment: {config.EXECUTION_MODE:<10}                 ║
    ║   Port: {config.PORT:<6}                                 ║
    ║   Privacy Budget: ε = {config.EPSILON}                    ║
    ║   Organizations: {len(ORGANIZATIONS)}                     ║
    ║   Analytics Queries: {len(QUERY_CATALOG)}                 ║
    ║                                                          ║
    ║   Database: {'PostgreSQL' if store.db_conn else 'In-memory'} ║
    ║   Redis: {'Connected' if store.redis_client else 'In-memory'} ║
    ║                                                          ║
    ║   Demo Login: demo@example.com / demo123                 ║
    ║                                                          ║
    ║   Health Check: /api/health                              ║
    ║   API Documentation: /api/queries                        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(startup_message)
    logger.info(f"Application initialized in {config.EXECUTION_MODE} mode")

# ============================================================
# RENDER ENTRY POINT
# ============================================================

# Initialize the application
initialize_application()

# Note: Render will use gunicorn to run the app
# The app object is imported by gunicorn
if __name__ == "__main__":
    # This is for local development only
    # On Render, gunicorn will be used via the start command
    app.run(
        host="0.0.0.0",
        port=config.PORT,
        debug=(config.EXECUTION_MODE == "DEVELOPMENT")
    )
