import os
import sqlite3
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database path from environment
DB_PATH = os.getenv("SQLITE_DB_PATH", "violations.db")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def setup_database():
    """Create necessary tables if they don't exist"""
    
    try:
        # Check if we should use PostgreSQL
        database_url = os.getenv("DATABASE_URL")
        using_postgres = False
        print(database_url)
        if database_url:
            try:
                import psycopg2
                logging.info(f"Setting up PostgreSQL database using connection URL")
                conn = psycopg2.connect(database_url)
                using_postgres = True
            except (ImportError, psycopg2.Error) as e:
                logging.error(f"Failed to connect to PostgreSQL: {str(e)}, falling back to SQLite")
                using_postgres = False
                
        if not using_postgres:
            logging.info(f"Setting up SQLite database at {DB_PATH}...")
            conn = sqlite3.connect(DB_PATH)
            
        cursor = conn.cursor()
        
        # Create violations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            violation_timestamp REAL NOT NULL,
            face_id TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            violation_image_path TEXT,
            details TEXT
        )
        """)
        
        # Create videos table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            is_live INTEGER NOT NULL DEFAULT 0,
            filename TEXT,
            stream_url TEXT,
            size INTEGER DEFAULT 0,
            fps REAL DEFAULT 0,
            total_frames INTEGER DEFAULT 0,
            duration REAL DEFAULT 0,
            status TEXT DEFAULT 'uploaded',
            frames_processed INTEGER DEFAULT 0,
            violations_detected INTEGER DEFAULT 0,
            UNIQUE(tenant_id, camera_id)
        )
        """)
        
        # Create faces table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            face_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            metadata TEXT,
            image_path TEXT,
            s3_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        
        # Create tenant_config table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tenant_config (
            tenant_id TEXT PRIMARY KEY,
            tenant_name TEXT NOT NULL,
            similarity_threshold REAL DEFAULT 0.5,
            mask_threshold_minutes INTEGER DEFAULT 5,
            vest_threshold_minutes INTEGER DEFAULT 5,
            hardhat_threshold_minutes INTEGER DEFAULT 5,
            external_trigger_url TEXT,
            is_active BOOLEAN DEFAULT true  -- Correct boolean default
        );  -- Closing parenthesis was missing
    """)


        
        conn.commit()
        logging.info("Database setup completed successfully")
    except Exception as e:
        logging.error(f"Error setting up database: {str(e)}")
    finally:
        if conn:
            conn.close()
    
    # Create directories
    os.makedirs("violation_images", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    logging.info("Created necessary directories")

if __name__ == "__main__":
    setup_database() 