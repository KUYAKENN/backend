"""
Migration utility to handle transition from face_recognition to DeepFace
"""

import json
import numpy as np
from database import DatabaseManager, FaceEncoding
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
            
            if not self.db.is_connected():
                print("❌ Database connection not available")
                return False
            
            session = self.db.get_session()
            if not session:
                print("❌ Could not create database session")
                return False
            
            try:
                # Get all existing face encodings with image data
                encodings = session.query(FaceEncoding).filter(
                    FaceEncoding.image_data.isnot(None)
                ).all()
                
                if not encodings:
                    print("ℹ️ No existing encodings with image data found")
                    return True
                
                updated_count = 0
                
                for encoding_record in encodings:
                    try:
                        print(f"🔄 Processing encoding ID {encoding_record.id} for person_id {encoding_record.person_id}")
                        
                        # Convert image data to numpy array
                        nparr = np.frombuffer(encoding_record.image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is None:
                            print(f"⚠️ Could not decode image for encoding ID {encoding_record.id}")
                            continue
                        
                        # Convert BGR to RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Get new DeepFace encoding
                        face_locations = face_recognition.face_locations(rgb_image)
                        
                        if not face_locations:
                            print(f"⚠️ No faces found in image for encoding ID {encoding_record.id}")
                            continue
                        
                        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        
                        if not face_encodings:
                            print(f"⚠️ Could not generate encoding for encoding ID {encoding_record.id}")
                            continue
                        
                        # Use the first face encoding
                        new_encoding = face_encodings[0]
                        new_encoding_json = json.dumps(new_encoding.tolist())
                        
                        # Update the database with new encoding
                        encoding_record.encoding_data = new_encoding_json
                        updated_count += 1
                        
                        print(f"✅ Updated encoding for person_id {encoding_record.person_id} (ID: {encoding_record.id})")
                        
                    except Exception as e:
                        print(f"❌ Error processing encoding ID {encoding_record.id}: {e}")
                        continue
                
                # Commit all changes
                session.commit()
                print(f"🎉 Migration completed! Updated {updated_count} encodings")
                return True
                
            finally:
                session.close()
                
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    def verify_migration(self):
        """Verify that the migration worked correctly"""
        try:
            print("🔍 Verifying migration...")
            
            if not self.db.is_connected():
                print("❌ Database connection not available")
                return False
            
            session = self.db.get_session()
            if not session:
                print("❌ Could not create database session")
                return False
            
            try:
                # Test loading encodings with new format
                encodings = session.query(FaceEncoding).limit(5).all()
                
                for encoding_record in encodings:
                    try:
                        # Try to load the encoding
                        encoding = np.array(json.loads(encoding_record.encoding_data))
                        print(f"✅ Person {encoding_record.person_id}: Encoding shape {encoding.shape}")
                    except Exception as e:
                        print(f"❌ Person {encoding_record.person_id}: Invalid encoding - {e}")
                        return False
                
                print("✅ Migration verification successful!")
                return True
                
            finally:
                session.close()
                
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
