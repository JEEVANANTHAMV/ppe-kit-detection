import os
import sqlite3
import psycopg2
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Get database path from environment
DB_PATH = os.getenv("SQLITE_DB_PATH", "violations.db")

def get_connection():
    """
    Returns a tuple (conn, db_type). Uses Postgres if DATABASE_URL is set and psycopg2 is available;
    otherwise connects to local SQLite.
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url and psycopg2 is not None:
        try:
            conn = psycopg2.connect(database_url)
            return conn, "postgres"
        except Exception as e:
            print(f"Error connecting to Postgres: {e}")
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    return conn, "sqlite"

def format_query(query, db_type):
    """
    Converts '?' placeholders to '%s' for Postgres.
    """
    if db_type == "postgres":
        return query.replace("?", "%s")
    return query

def migrate():
    """Run database migrations"""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if s3_url column exists in videos table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'videos' AND column_name = 's3_url'
        """)
        if not cursor.fetchone():
            # Add s3_url column to videos table
            cursor.execute("""
                ALTER TABLE videos 
                ADD COLUMN s3_url TEXT
            """)
            logging.info("Added s3_url column to videos table")
        
        # Check if camera_id column exists in faces table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'faces' AND column_name = 'camera_id'
        """)
        if cursor.fetchone():
            # Remove camera_id column from faces table
            cursor.execute("""
                ALTER TABLE faces 
                DROP COLUMN camera_id
            """)
            logging.info("Removed camera_id column from faces table")
        
        conn.commit()
        logging.info("Database migration completed successfully")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error during migration: {str(e)}")
        raise
    finally:
        conn.close()

def create_tables():
    # Database connection already established with environment variables
    conn, db_type = get_connection()
    c = conn.cursor()

    # Create tables if they don't exist
    tables = {
        "tenant_config": """
        CREATE TABLE IF NOT EXISTS tenant_config (
            tenant_id TEXT PRIMARY KEY,
            similarity_threshold REAL,
            no_mask_threshold INTEGER,
            no_safety_vest_threshold INTEGER,
            no_hardhat_threshold INTEGER,
            external_trigger_url TEXT,
            is_active BOOLEAN DEFAULT TRUE
        )
        """,
        "videos": """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            is_live INTEGER NOT NULL,
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
        """,
        "faces": """
        CREATE TABLE IF NOT EXISTS faces (
            face_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            name TEXT,
            embedding TEXT,
            metadata TEXT
        )
        """,
        "violations": """
        CREATE TABLE IF NOT EXISTS violations (
            id TEXT PRIMARY KEY,
            tenant_id TEXT,
            camera_id TEXT,
            violation_timestamp REAL,
            face_id TEXT,
            violation_type TEXT,
            violation_image_path TEXT,
            details TEXT
        )
        """
    }

    # Execute table creation queries
    for table_name, query in tables.items():
        try:
            c.execute(format_query(query, db_type))
            print(f"Created/verified table: {table_name}")
        except Exception as e:
            print(f"Error creating table {table_name}: {e}")

    # Check and add missing columns
    if db_type == "postgres":
        # Check for is_active column in tenant_config
        c.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'tenant_config' AND column_name = 'is_active'
        """)
        if not c.fetchone():
            c.execute("""
                ALTER TABLE tenant_config 
                ADD COLUMN is_active BOOLEAN DEFAULT TRUE
            """)
            print("Added is_active column to tenant_config table in Postgres")

        # Check for status column in videos
        c.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'videos' AND column_name = 'status'
        """)
        if not c.fetchone():
            c.execute("""
                ALTER TABLE videos 
                ADD COLUMN status TEXT DEFAULT 'uploaded'
            """)
            print("Added status column to videos table in Postgres")

        # Check for frames_processed and violations_detected columns
        for col in ['frames_processed', 'violations_detected']:
            c.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'videos' AND column_name = '{col}'
            """)
            if not c.fetchone():
                c.execute(f"""
                    ALTER TABLE videos 
                    ADD COLUMN {col} INTEGER DEFAULT 0
                """)
                print(f"Added {col} column to videos table in Postgres")

    else:  # SQLite
        # Check for is_active column in tenant_config
        c.execute("PRAGMA table_info(tenant_config)")
        columns = [col[1] for col in c.fetchall()]
        if 'is_active' not in columns:
            c.execute("""
                ALTER TABLE tenant_config 
                ADD COLUMN is_active BOOLEAN DEFAULT TRUE
            """)
            print("Added is_active column to tenant_config table in SQLite")

        # Check for status column in videos
        c.execute("PRAGMA table_info(videos)")
        columns = [col[1] for col in c.fetchall()]
        if 'status' not in columns:
            c.execute("""
                ALTER TABLE videos 
                ADD COLUMN status TEXT DEFAULT 'uploaded'
            """)
            print("Added status column to videos table in SQLite")

        # Check for frames_processed and violations_detected columns
        for col in ['frames_processed', 'violations_detected']:
            if col not in columns:
                c.execute(f"""
                    ALTER TABLE videos 
                    ADD COLUMN {col} INTEGER DEFAULT 0
                """)
                print(f"Added {col} column to videos table in SQLite")

    # Create indexes for better performance
    indexes = [
        ("CREATE INDEX IF NOT EXISTS idx_videos_tenant_camera ON videos(tenant_id, camera_id)"),
        ("CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)"),
        ("CREATE INDEX IF NOT EXISTS idx_faces_tenant ON faces(tenant_id)"),
        ("CREATE INDEX IF NOT EXISTS idx_violations_tenant_camera ON violations(tenant_id, camera_id)"),
        ("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(violation_timestamp)")
    ]

    for index_query in indexes:
        try:
            c.execute(format_query(index_query, db_type))
            print(f"Created/verified index: {index_query}")
        except Exception as e:
            print(f"Error creating index: {e}")

    conn.commit()
    conn.close()
    print("Database migration completed successfully!")

if __name__ == "__main__":
    migrate() 