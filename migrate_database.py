#!/usr/bin/env python3
"""
Database Migration Script - Add Image Storage Support
This script will add image storage columns to your existing MySQL database
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection from environment variables"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'face_recognition_db')
        )
        
        if connection.is_connected():
            print(f"✅ Connected to MySQL database: {os.getenv('DB_NAME', 'face_recognition_db')}")
            return connection
        
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    try:
        cursor.execute(f"""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = %s 
            AND COLUMN_NAME = %s
        """, (os.getenv('DB_NAME', 'face_recognition_db'), table_name, column_name))
        
        result = cursor.fetchone()
        return result[0] > 0
    except Error as e:
        print(f"❌ Error checking column existence: {e}")
        return False

def run_migration():
    """Run the database migration"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        print("\n🔄 Starting database migration...")
        
        # Migration queries
        migrations = [
            {
                'description': 'Add image storage columns to face_encodings table',
                'table': 'face_encodings',
                'columns_to_add': [
                    ('image_data', 'ADD COLUMN image_data LONGBLOB AFTER confidence_score'),
                    ('image_filename', 'ADD COLUMN image_filename VARCHAR(255) AFTER image_data'),
                    ('image_mime_type', 'ADD COLUMN image_mime_type VARCHAR(100) DEFAULT "image/jpeg" AFTER image_filename'),
                    ('image_size', 'ADD COLUMN image_size INT AFTER image_mime_type')
                ]
            },
            {
                'description': 'Add image storage columns to attendance table',
                'table': 'attendance',
                'columns_to_add': [
                    ('detection_image_data', 'ADD COLUMN detection_image_data LONGBLOB AFTER detection_method'),
                    ('detection_image_filename', 'ADD COLUMN detection_image_filename VARCHAR(255) AFTER detection_image_data'),
                    ('detection_image_mime_type', 'ADD COLUMN detection_image_mime_type VARCHAR(100) DEFAULT "image/jpeg" AFTER detection_image_filename')
                ]
            },
            {
                'description': 'Add image storage columns to recognition_logs table',
                'table': 'recognition_logs',
                'columns_to_add': [
                    ('recognition_image_data', 'ADD COLUMN recognition_image_data LONGBLOB AFTER detection_status'),
                    ('recognition_image_filename', 'ADD COLUMN recognition_image_filename VARCHAR(255) AFTER recognition_image_data'),
                    ('recognition_image_mime_type', 'ADD COLUMN recognition_image_mime_type VARCHAR(100) DEFAULT "image/jpeg" AFTER recognition_image_filename')
                ]
            }
        ]
        
        # Execute migrations
        for migration in migrations:
            print(f"\n📝 {migration['description']}")
            
            for column_name, sql_command in migration['columns_to_add']:
                # Check if column already exists
                if check_column_exists(cursor, migration['table'], column_name):
                    print(f"   ⏭️  Column '{column_name}' already exists in {migration['table']}")
                    continue
                
                try:
                    # Execute the ALTER TABLE command
                    alter_sql = f"ALTER TABLE {migration['table']} {sql_command}"
                    cursor.execute(alter_sql)
                    print(f"   ✅ Added column '{column_name}' to {migration['table']}")
                    
                except Error as e:
                    print(f"   ❌ Error adding column '{column_name}' to {migration['table']}: {e}")
                    # Continue with other columns even if one fails
        
        # Commit all changes
        connection.commit()
        print(f"\n🎉 Migration completed successfully!")
        print(f"📊 Database updated with image storage support")
        
        # Show updated table structures
        print(f"\n📋 Updated table structures:")
        for migration in migrations:
            cursor.execute(f"DESCRIBE {migration['table']}")
            columns = cursor.fetchall()
            
            print(f"\n{migration['table']}:")
            for column in columns:
                col_name, col_type = column[0], column[1]
                if any(col_name.endswith(suffix) for suffix in ['_data', '_filename', '_mime_type', '_size']):
                    print(f"   🆕 {col_name}: {col_type}")
        
        return True
        
    except Error as e:
        print(f"❌ Error during migration: {e}")
        connection.rollback()
        return False
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print(f"\n🔌 Database connection closed")

def main():
    """Main function"""
    print("🚀 Face Recognition Database Migration Tool")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📝 Please create a .env file with your database credentials:")
        print("""
DB_HOST=your_mysql_host
DB_PORT=3306
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=face_recognition_db
        """)
        return False
    
    # Show current environment
    print(f"🔗 Database Host: {os.getenv('DB_HOST', 'localhost')}")
    print(f"🔗 Database Name: {os.getenv('DB_NAME', 'face_recognition_db')}")
    print(f"👤 Database User: {os.getenv('DB_USER', 'root')}")
    
    # Confirm before proceeding
    response = input(f"\n⚠️  This will modify your database structure. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled")
        return False
    
    # Run the migration
    success = run_migration()
    
    if success:
        print(f"\n✨ Your database is now ready for image storage!")
        print(f"💡 You can now:")
        print(f"   - Store face images directly in MySQL")
        print(f"   - Deploy to any cloud platform (Vercel, Railway, etc.)")
        print(f"   - No need for file system storage")
    else:
        print(f"\n💥 Migration failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
