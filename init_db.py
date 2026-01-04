import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def init_database():
    """Initialize PostgreSQL database tables"""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL not set. Using in-memory storage.")
        return
    
    try:
        # Fix for Render's PostgreSQL URL
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        conn = psycopg2.connect(database_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Create tables
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
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) UNIQUE,
            email VARCHAR(255) UNIQUE,
            name VARCHAR(255),
            role VARCHAR(50),
            organization VARCHAR(255),
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
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
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS privacy_budget (
            id SERIAL PRIMARY KEY,
            organization VARCHAR(100),
            epsilon_total FLOAT,
            epsilon_used FLOAT,
            last_reset TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        print("Database tables created successfully!")
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
