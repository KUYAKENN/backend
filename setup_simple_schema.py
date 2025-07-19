"""
Simple Database Schema Setup - Just the essential tables
"""
import os
import sys
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# Load environment variables
load_dotenv()

def setup_simple_schema():
    """Set up simple database schema without stored procedures"""
    
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'face_recognition_db')
    }
    
    try:
        print("🔧 Setting up Simple Database Schema...")
        print(f"Connecting to: {config['user']}@{config['host']}:{config['port']}/{config['database']}")
        
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Drop existing tables
        print("🗑️ Dropping existing tables...")
        drop_tables = [
            "DROP TABLE IF EXISTS recognition_logs",
            "DROP TABLE IF EXISTS face_encodings", 
            "DROP TABLE IF EXISTS attendance",
            "DROP TABLE IF EXISTS persons",
            "DROP TABLE IF EXISTS locations",
            "DROP TABLE IF EXISTS system_settings"
        ]
        
        for drop_sql in drop_tables:
            cursor.execute(drop_sql)
        
        print("🏗️ Creating new tables...")
        
        # Create persons table
        cursor.execute("""
        CREATE TABLE persons (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL UNIQUE,
            email VARCHAR(255) UNIQUE,
            phone VARCHAR(50),
            department VARCHAR(100),
            position VARCHAR(100),
            status ENUM('active', 'inactive') DEFAULT 'active',
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            notes TEXT,
            
            INDEX idx_name (name),
            INDEX idx_email (email),
            INDEX idx_status (status)
        )
        """)
        print("✅ persons table created")
        
        # Create face_encodings table
        cursor.execute("""
        CREATE TABLE face_encodings (
            id INT PRIMARY KEY AUTO_INCREMENT,
            person_id INT NOT NULL,
            encoding_data TEXT NOT NULL,
            encoding_type ENUM('standard', 'enhanced', 'multi_angle') DEFAULT 'standard',
            face_angle ENUM('front', 'left', 'right', 'up', 'down') DEFAULT 'front',
            confidence_score DECIMAL(3,2) DEFAULT 0.80,
            image_path VARCHAR(500),
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_primary BOOLEAN DEFAULT FALSE,
            
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
            INDEX idx_person_id (person_id)
        )
        """)
        print("✅ face_encodings table created")
        
        # Create attendance table
        cursor.execute("""
        CREATE TABLE attendance (
            id INT PRIMARY KEY AUTO_INCREMENT,
            person_id INT NOT NULL,
            person_name VARCHAR(255) NOT NULL,
            check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            check_out_time TIMESTAMP NULL,
            date_recorded DATE NOT NULL,
            confidence_score DECIMAL(3,2) NOT NULL,
            detection_method ENUM('auto', 'manual', 'override') DEFAULT 'auto',
            image_path VARCHAR(500),
            location_id INT,
            status ENUM('present', 'late', 'early_leave', 'absent') DEFAULT 'present',
            notes TEXT,
            created_by VARCHAR(100) DEFAULT 'system',
            
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
            UNIQUE KEY unique_person_date (person_id, date_recorded),
            INDEX idx_person_id (person_id),
            INDEX idx_date_recorded (date_recorded)
        )
        """)
        print("✅ attendance table created")
        
        # Create locations table
        cursor.execute("""
        CREATE TABLE locations (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            camera_id VARCHAR(100),
            coordinates TEXT,
            status ENUM('active', 'inactive', 'maintenance') DEFAULT 'active',
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_name (name)
        )
        """)
        print("✅ locations table created")
        
        # Create recognition_logs table
        cursor.execute("""
        CREATE TABLE recognition_logs (
            id INT PRIMARY KEY AUTO_INCREMENT,
            person_id INT,
            person_name VARCHAR(255),
            recognition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_score DECIMAL(3,2) NOT NULL,
            detection_status ENUM('recognized', 'unknown', 'low_confidence', 'failed') NOT NULL,
            image_path VARCHAR(500),
            location_id INT,
            processing_time_ms INT,
            face_coordinates TEXT,
            notes TEXT,
            
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL,
            FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE SET NULL,
            INDEX idx_person_id (person_id),
            INDEX idx_recognition_time (recognition_time)
        )
        """)
        print("✅ recognition_logs table created")
        
        # Insert default locations
        cursor.execute("""
        INSERT INTO locations (name, description, status) VALUES 
        ('Main Entrance', 'Primary entrance camera', 'active'),
        ('Office Floor 1', 'Office area monitoring', 'active')
        ON DUPLICATE KEY UPDATE description = VALUES(description)
        """)
        print("✅ default locations inserted")
        
        connection.commit()
        
        # Verify tables
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"\n✅ Created tables: {tables}")
        
        cursor.close()
        connection.close()
        
        print("\n🎉 Database schema setup completed successfully!")
        print("Your system is now ready to use MySQL database.")
        
    except Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    setup_simple_schema()
