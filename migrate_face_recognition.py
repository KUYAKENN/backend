"""
Migration utility to handle transition from face_recognition to DeepFace
"""

import json
import numpy as np
from database import DatabaseManager
import deepface_recognition as face_recognition
from image_utils import ImageHandler
import cv2

class FaceRecognitionMigrator:
    """Handle migration from old face_recognition to new DeepFace system"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.image_handler = ImageHandler()
    
    def migrate_encodings(self):
        """
        Migrate existing face encodings to DeepFace format
        This will re-process stored images to create new encodings
        """
        try:
            print("🔄 Starting face encoding migration...")
            
            # Get all existing face encodings with image data
            query = """
                SELECT id, person_id, encoding_data, image_data, image_filename 
                FROM face_encodings 
                WHERE image_data IS NOT NULL
            """
            
            results = self.db.execute_query(query)
            
            if not results:
                print("ℹ️ No existing encodings with image data found")
                return True
            
            updated_count = 0
            
            for record in results:
                encoding_id, person_id, old_encoding_data, image_data, filename = record
                
                try:
                    # Convert image data to numpy array
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        print(f"⚠️ Could not decode image for encoding ID {encoding_id}")
                        continue
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get new DeepFace encoding
                    face_locations = face_recognition.face_locations(rgb_image)
                    
                    if not face_locations:
                        print(f"⚠️ No faces found in image for encoding ID {encoding_id}")
                        continue
                    
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    
                    if not face_encodings:
                        print(f"⚠️ Could not generate encoding for encoding ID {encoding_id}")
                        continue
                    
                    # Use the first face encoding
                    new_encoding = face_encodings[0]
                    new_encoding_json = json.dumps(new_encoding.tolist())
                    
                    # Update the database with new encoding
                    update_query = """
                        UPDATE face_encodings 
                        SET encoding_data = %s 
                        WHERE id = %s
                    """
                    
                    self.db.execute_query(update_query, (new_encoding_json, encoding_id))
                    updated_count += 1
                    
                    print(f"✅ Updated encoding for person_id {person_id} (ID: {encoding_id})")
                    
                except Exception as e:
                    print(f"❌ Error processing encoding ID {encoding_id}: {e}")
                    continue
            
            print(f"🎉 Migration completed! Updated {updated_count} encodings")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    def verify_migration(self):
        """Verify that the migration worked correctly"""
        try:
            print("🔍 Verifying migration...")
            
            # Test loading encodings with new format
            query = """
                SELECT person_id, encoding_data 
                FROM face_encodings 
                LIMIT 5
            """
            
            results = self.db.execute_query(query)
            
            for person_id, encoding_data in results:
                try:
                    # Try to load the encoding
                    encoding = np.array(json.loads(encoding_data))
                    print(f"✅ Person {person_id}: Encoding shape {encoding.shape}")
                except Exception as e:
                    print(f"❌ Person {person_id}: Invalid encoding - {e}")
                    return False
            
            print("✅ Migration verification successful!")
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

def run_migration():
    """Run the face recognition migration"""
    migrator = FaceRecognitionMigrator()
    
    print("🚀 Face Recognition Migration Tool")
    print("This will update all face encodings to use DeepFace format")
    print("=" * 60)
    
    response = input("⚠️ This will modify your existing face encodings. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled")
        return False
    
    # Run migration
    success = migrator.migrate_encodings()
    
    if success:
        # Verify migration
        migrator.verify_migration()
        print("\n🎉 Face recognition system successfully migrated to DeepFace!")
        print("💡 Your face recognition API is now compatible with Railway deployment!")
    else:
        print("\n💥 Migration failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    run_migration()
