import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ENHANCED MULTI-ANGLE FACE REGISTRATION BACKEND
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import threading
import time
import gc
from PIL import Image
import math
import calendar
import queue

# Import database manager
from database import init_database, get_db

app = Flask(__name__)

# Configure CORS for production
CORS(app, origins=[
    "http://localhost:4200",
    "https://your-frontend-domain.com",  # Replace with your actual frontend domain
    "*"  # Remove this in production for security
])

# Environment configuration
ENV = os.getenv('FLASK_ENV', 'development')
PORT = int(os.getenv('PORT', 5000))

# Initialize database
print("🔄 Initializing database connection...")
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'face_recognition_db')
}

# Try to initialize database
try:
    db_manager = init_database(db_config)
    USE_DATABASE = db_manager.is_connected()
    print(f"📊 Database status: {'Connected' if USE_DATABASE else 'Disabled - using file storage'}")
except Exception as e:
    print(f"⚠️ Database initialization failed: {e}")
    print("📁 Falling back to file-based storage")
    USE_DATABASE = False
    db_manager = None

class EnhancedAttendanceSystem:
    def __init__(self, known_faces_dir="known_faces", attendance_file="attendance.csv", 
                 captured_faces_dir="captured_faces"):
        self.known_faces_dir = known_faces_dir
        self.attendance_file = attendance_file
        self.captured_faces_dir = captured_faces_dir
        
        # Database integration
        self.use_database = USE_DATABASE
        self.db_manager = db_manager
        
        # Create directories
        os.makedirs(known_faces_dir, exist_ok=True)
        os.makedirs(captured_faces_dir, exist_ok=True)
        os.makedirs(f"{captured_faces_dir}/registered", exist_ok=True)
        os.makedirs(f"{captured_faces_dir}/recognized", exist_ok=True)
        
        # Enhanced face recognition data
        self.known_face_encodings = []  # Will store multiple encodings per person
        self.known_face_names = []
        self.known_face_metadata = []  # Store pose information and confidence scores
        self.attendance_today = set()
        
        # Camera and processing
        self.video_capture = None
        self.is_running = False
        self.current_mode = "recognition"
        
        # ENHANCED: Multi-angle registration system
        self.registration_name = ""
        self.registration_samples = []
        self.current_pose_index = 0
        self.required_poses = [
            {"name": "front", "description": "Look straight at the camera", "icon": "👤"},
            {"name": "left", "description": "Turn your head slightly to the left", "icon": "👤⬅️"},
            {"name": "right", "description": "Turn your head slightly to the right", "icon": "➡️👤"},
            {"name": "up", "description": "Tilt your head slightly up", "icon": "👤⬆️"},
            {"name": "down", "description": "Tilt your head slightly down", "icon": "👤⬇️"}
        ]
        self.samples_per_pose = 2  # 2 samples per pose = 10 total samples
        self.current_pose_samples = 0
        
        # Performance optimizations
        self.frame_skip_counter = 0
        self.face_detection_interval = 3
        self.recognition_interval = 15
        self.last_frame_time = 0
        self.target_fps = 20
        
        # Caching
        self.cached_frame = None
        self.cached_faces = []
        self.last_recognition_results = []
        self.last_recognition_time = 0
        
        # Recent recognitions
        self.recent_recognitions = {}
        self.recognition_cooldown = 5
        
        # Load existing known faces
        self.load_known_faces()
        self.load_today_attendance()
        
        # Migrate existing data if using database for first time
        if self.use_database:
            self.migrate_existing_data()
    
    def migrate_existing_data(self):
        """Migrate existing CSV and JSON data to MySQL database"""
        try:
            print("🔄 Checking for existing data to migrate...")
            
            # Migrate CSV attendance data
            if os.path.exists(self.attendance_file):
                print("📊 Migrating attendance data from CSV...")
                self.db_manager.migrate_from_csv(self.attendance_file)
            
            # Migrate JSON face data
            enhanced_file = os.path.join(self.known_faces_dir, "enhanced_encodings.json")
            legacy_file = os.path.join(self.known_faces_dir, "encodings.json")
            
            if os.path.exists(enhanced_file):
                print("👤 Migrating enhanced face data from JSON...")
                self.db_manager.migrate_from_json(enhanced_file)
            elif os.path.exists(legacy_file):
                print("👤 Migrating legacy face data from JSON...")
                self.db_manager.migrate_from_json(legacy_file)
            
            print("✅ Data migration completed")
            
        except Exception as e:
            print(f"⚠️ Data migration warning: {e}")
    
    def load_known_faces(self):
        """Load all known faces with enhanced metadata from database or files"""
        print("Loading enhanced face database...")
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = []
        
        if self.use_database:
            # Load from MySQL database
            try:
                face_encodings, face_names, face_metadata = self.db_manager.get_all_face_data()
                self.known_face_encodings = face_encodings
                self.known_face_names = face_names
                self.known_face_metadata = face_metadata
                
                unique_people = len(set(face_names))
                total_encodings = len(face_encodings)
                print(f"✅ Loaded {unique_people} people with {total_encodings} face encodings from database")
                return
                
            except Exception as e:
                print(f"❌ Error loading from database: {e}")
                print("📁 Falling back to file-based storage")
        
        # Fallback to file-based storage
        # Load from enhanced encodings file
        encodings_file = os.path.join(self.known_faces_dir, "enhanced_encodings.json")
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'r') as f:
                    data = json.load(f)
                
                for person_name, person_data in data.items():
                    if 'poses' in person_data:
                        # Enhanced format with multiple poses
                        for pose_data in person_data['poses']:
                            encoding = np.array(pose_data['encoding'])
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(person_name)
                            self.known_face_metadata.append({
                                'pose': pose_data['pose'],
                                'confidence': pose_data.get('confidence', 1.0),
                                'quality_score': pose_data.get('quality_score', 0.8)
                            })
                
                unique_people = len(set(self.known_face_names))
                total_encodings = len(self.known_face_encodings)
                print(f"✅ Loaded {unique_people} people with {total_encodings} face encodings from files")
                return
            except Exception as e:
                print(f"Error loading enhanced encodings: {str(e)}")
        
        # Fallback to legacy format
        legacy_file = os.path.join(self.known_faces_dir, "encodings.json")
        if os.path.exists(legacy_file):
            try:
                with open(legacy_file, 'r') as f:
                    encodings_data = json.load(f)
                
                for name, encoding in encodings_data.items():
                    encoding_array = np.array(encoding)
                    self.known_face_encodings.append(encoding_array)
                    self.known_face_names.append(name)
                    self.known_face_metadata.append({
                        'pose': 'front',
                        'confidence': 1.0,
                        'quality_score': 0.8
                    })
                
                print(f"✅ Loaded {len(encodings_data)} people from legacy format")
            except Exception as e:
                print(f"Error loading legacy encodings: {str(e)}")
    
    def load_today_attendance(self):
        """Load today's attendance from database or CSV"""
        if self.use_database:
            try:
                self.attendance_today = self.db_manager.get_todays_attendance()
                return
            except Exception as e:
                print(f"❌ Error loading attendance from database: {e}")
        
        # Fallback to CSV
        if os.path.exists(self.attendance_file):
            try:
                df = pd.read_csv(self.attendance_file)
                today = datetime.now().strftime("%Y-%m-%d")
                today_records = df[df['Date'] == today]
                self.attendance_today = set(today_records['Name'].tolist())
            except:
                self.attendance_today = set()
        else:
            self.attendance_today = set()
        self.load_today_attendance()
    
    def load_known_faces(self):
        """Load all known faces with enhanced metadata"""
        print("Loading enhanced face database...")
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = []
        
        # Load from enhanced encodings file
        encodings_file = os.path.join(self.known_faces_dir, "enhanced_encodings.json")
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'r') as f:
                    data = json.load(f)
                
                for person_name, person_data in data.items():
                    if 'poses' in person_data:
                        # New enhanced format with multiple poses
                        for pose_data in person_data['poses']:
                            encoding = np.array(pose_data['encoding'])
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(person_name)
                            self.known_face_metadata.append({
                                'pose': pose_data.get('pose', 'front'),
                                'confidence': pose_data.get('confidence', 1.0),
                                'timestamp': pose_data.get('timestamp', ''),
                                'quality_score': pose_data.get('quality_score', 1.0)
                            })
                        print(f"Loaded {len(person_data['poses'])} poses for: {person_name}")
                    else:
                        # Legacy format compatibility
                        encoding = np.array(person_data)
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(person_name)
                        self.known_face_metadata.append({
                            'pose': 'front',
                            'confidence': 1.0,
                            'timestamp': '',
                            'quality_score': 1.0
                        })
                        print(f"Loaded legacy encoding for: {person_name}")
                return
            except Exception as e:
                print(f"Error loading enhanced encodings: {str(e)}")
        
        # Fallback to legacy format
        legacy_file = os.path.join(self.known_faces_dir, "encodings.json")
        if os.path.exists(legacy_file):
            try:
                with open(legacy_file, 'r') as f:
                    encodings_data = json.load(f)
                
                for name, encoding in encodings_data.items():
                    self.known_face_encodings.append(np.array(encoding))
                    self.known_face_names.append(name)
                    self.known_face_metadata.append({
                        'pose': 'front',
                        'confidence': 1.0,
                        'timestamp': '',
                        'quality_score': 1.0
                    })
                    print(f"Loaded legacy face for: {name}")
            except Exception as e:
                print(f"Error loading legacy encodings: {str(e)}")
    
    def load_today_attendance(self):
        """Load today's attendance"""
        if os.path.exists(self.attendance_file):
            try:
                df = pd.read_csv(self.attendance_file)
                today = datetime.now().strftime("%Y-%m-%d")
                today_records = df[df['Date'] == today]
                self.attendance_today = set(today_records['Name'].tolist())
            except:
                self.attendance_today = set()
    
    def calculate_face_quality(self, face_image, face_location):
        """Calculate face quality score based on size, clarity, and pose"""
        try:
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top
            
            # Size score (larger faces generally better)
            size_score = min(1.0, (face_width * face_height) / (150 * 150))
            
            # Aspect ratio score (faces should be roughly rectangular)
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            aspect_score = 1.0 - abs(aspect_ratio - 0.75) * 2  # Ideal ratio ~0.75
            aspect_score = max(0.0, min(1.0, aspect_score))
            
            # Blur detection using Laplacian variance
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_score / 1000)  # Normalize
            
            # Lighting score (avoid over/under exposed faces)
            mean_brightness = np.mean(gray)
            lighting_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Combined quality score
            quality = (size_score * 0.3 + aspect_score * 0.2 + 
                      blur_score * 0.3 + lighting_score * 0.2)
            
            return min(1.0, max(0.0, quality))
        except:
            return 0.5  # Default score if calculation fails
    
    def analyze_face_pose(self, face_landmarks):
        """Analyze face pose based on facial landmarks"""
        try:
            if not face_landmarks:
                return "unknown"
            
            landmarks = face_landmarks[0]  # First face
            
            # Get key points
            nose_tip = landmarks['nose_tip'][2]  # Center of nose tip
            left_eye = np.mean(landmarks['left_eye'], axis=0)
            right_eye = np.mean(landmarks['right_eye'], axis=0)
            
            # Calculate eye distance and nose position
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            nose_offset_x = nose_tip[0] - eye_center[0]
            nose_offset_y = nose_tip[1] - eye_center[1]
            
            # Determine pose based on nose position relative to eyes
            if abs(nose_offset_x) < 5 and abs(nose_offset_y) < 5:
                return "front"
            elif nose_offset_x > 10:
                return "left"  # Person's left (appears on right in image)
            elif nose_offset_x < -10:
                return "right"  # Person's right (appears on left in image)
            elif nose_offset_y > 8:
                return "down"
            elif nose_offset_y < -8:
                return "up"
            else:
                return "front"
        except:
            return "unknown"
    
    def save_enhanced_face_image(self, face_image, name, pose, quality_score, category="registered"):
        """Save face image with enhanced metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{pose}_{quality_score:.2f}_{timestamp}.jpg"
            
            if category == "registered":
                save_path = os.path.join(self.captured_faces_dir, "registered", filename)
            else:
                save_path = os.path.join(self.captured_faces_dir, "recognized", filename)
            
            if face_image is not None and face_image.size > 0:
                # Enhance and resize face image
                face_image = cv2.resize(face_image, (200, 200))  # Larger size for better quality
                
                # Apply slight sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                face_image = cv2.filter2D(face_image, -1, kernel)
                
                cv2.imwrite(save_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return save_path
            return None
        except Exception as e:
            print(f"Error saving enhanced face image: {str(e)}")
            return None
    
    def mark_attendance(self, name, face_image=None):
        """Mark attendance for a person using database or CSV fallback"""
        if name not in self.attendance_today and name != "Unknown":
            try:
                now = datetime.now()
                
                # Save face image if provided
                saved_path = None
                if face_image is not None:
                    saved_path = self.save_enhanced_face_image(face_image, name, "recognized", 1.0, "recognized")
                    if saved_path:
                        print(f"Saved recognized face: {saved_path}")
                
                # Try to save to database first
                if self.use_database:
                    try:
                        success = self.db_manager.mark_attendance(
                            person_name=name,
                            confidence=1.0,
                            image_path=saved_path
                        )
                        
                        if success:
                            self.attendance_today.add(name)
                            print(f"✅ Attendance marked for {name} in database")
                            return True
                        
                    except Exception as e:
                        print(f"❌ Database attendance error: {e}")
                        print("📁 Falling back to CSV storage")
                
                # Fallback to CSV
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                
                record = {
                    'Name': name,
                    'Date': date_str,
                    'Time': time_str,
                    'Status': 'Present'
                }
                
                df = pd.DataFrame([record])
                if os.path.exists(self.attendance_file):
                    df.to_csv(self.attendance_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(self.attendance_file, index=False)
                
                self.attendance_today.add(name)
                print(f"📝 Attendance marked for {name} in CSV at {time_str}")
                return True
                
            except Exception as e:
                print(f"Error marking attendance: {str(e)}")
                return False
        return False
    
    def enhanced_face_recognition(self, face_encoding):
        """Enhanced face recognition with better debugging and lower threshold"""
        if not self.known_face_encodings:
            print("ERROR: No known face encodings loaded!")
            return "Unknown", 0.0
        
        print(f"\n=== FACE RECOGNITION DEBUG ===")
        print(f"Comparing against {len(self.known_face_encodings)} known encodings")
        print(f"Known people: {list(set(self.known_face_names))}")
        
        # Calculate distances to all known encodings
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        print(f"Face distances: {[f'{d:.3f}' for d in face_distances]}")
        
        # Group distances by person name
        person_scores = {}
        for i, distance in enumerate(face_distances):
            name = self.known_face_names[i]
            metadata = self.known_face_metadata[i]
            
            if name not in person_scores:
                person_scores[name] = []
            
            # Convert distance to confidence (1 - distance)
            confidence = 1 - distance
            
            # Weight the score by pose confidence and quality
            weighted_score = confidence * metadata['confidence'] * metadata['quality_score']
            person_scores[name].append({
                'confidence': confidence,
                'weighted_score': weighted_score,
                'distance': distance,
                'pose': metadata['pose']
            })
            
            print(f"  {name} ({metadata['pose']}): distance={distance:.3f}, confidence={confidence:.3f}, weighted={weighted_score:.3f}")
        
        # Find best match using average of top scores per person
        best_name = "Unknown"
        best_confidence = 0.0
        
        print(f"\n=== PERSON ANALYSIS ===")
        for name, scores in person_scores.items():
            # Use average of top 3 scores (or all if less than 3)
            top_scores = sorted([s['weighted_score'] for s in scores], reverse=True)[:3]
            avg_confidence = sum(top_scores) / len(top_scores)
            
            # Also get the best individual score for this person
            best_individual = max([s['confidence'] for s in scores])
            
            print(f"  {name}:")
            print(f"    avg_weighted: {avg_confidence:.3f}")
            print(f"    best_individual: {best_individual:.3f}")
            print(f"    poses: {[s['pose'] for s in scores]}")
            
            # LOWERED THRESHOLD: Use 0.32 instead of 0.4 for better recognition
            # Use either weighted average or best individual score
            final_confidence = max(avg_confidence, best_individual * 0.8)  # Slight penalty for individual vs average
            
            print(f"    final_confidence: {final_confidence:.3f}")
            
            if final_confidence > best_confidence and final_confidence > 0.32:  # LOWERED from 0.4 to 0.32
                best_confidence = final_confidence
                best_name = name
                print(f"    *** NEW BEST MATCH ***")
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Best match: {best_name} with confidence {best_confidence:.3f}")
        print(f"Threshold: 0.32")
        print(f"Passed threshold: {'YES' if best_confidence > 0.32 else 'NO'}")
        
        return best_name, best_confidence
    def process_frame_optimized(self, frame):
        """Enhanced frame processing with multi-angle recognition - COMPLETE VERSION"""
        original_height, original_width = frame.shape[:2]
        resized_width, resized_height = 320, 240
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        try:
            current_time = time.time()
            
            # FPS limiting
            if current_time - self.last_frame_time < (1.0 / self.target_fps):
                return self.last_recognition_results
            
            self.last_frame_time = current_time
            self.frame_skip_counter += 1
            
            # Use smaller frame for face detection
            small_frame = cv2.resize(frame, (320, 240))
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Only detect faces every Nth frame
            if self.frame_skip_counter % self.face_detection_interval == 0:
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                # Scale back up face locations
                face_locations = [
                (int(top * scale_y), int(right * scale_x), int(bottom * scale_y), int(left * scale_x))
                            for (top, right, bottom, left) in face_locations
                        ]
                self.cached_faces = face_locations
            else:
                face_locations = self.cached_faces
            
            results = []
            
            # Full recognition processing every 15th frame
            if self.frame_skip_counter % self.recognition_interval == 0 and face_locations:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for i, face_encoding in enumerate(face_encodings):
                    # USE THE ENHANCED FACE RECOGNITION METHOD
                    name, confidence = self.enhanced_face_recognition(face_encoding)
                    
                    if name != "Unknown" and name not in self.recent_recognitions or \
                    (name in self.recent_recognitions and 
                        (datetime.now() - self.recent_recognitions[name]).seconds > self.recognition_cooldown):
                        
                        top, right, bottom, left = face_locations[i]
                        face_image = frame[top:bottom, left:right]
                        
                        if self.mark_attendance(name, face_image):
                            self.recent_recognitions[name] = datetime.now()
                    
                    top, right, bottom, left = face_locations[i]
                    results.append({
                        'name': name,
                        'confidence': float(confidence),
                        'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                        'present': name in self.attendance_today
                    })
                
                self.last_recognition_results = results
                self.last_recognition_time = current_time
                
            else:
                # Use cached results with updated locations
                if face_locations and self.last_recognition_results:
                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        if i < len(self.last_recognition_results):
                            result = self.last_recognition_results[i].copy()
                            result['location'] = {'top': top, 'right': right, 'bottom': bottom, 'left': left}
                            results.append(result)
                        else:
                            results.append({
                                'name': 'Detecting...',
                                'confidence': 0,
                                'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                                'present': False
                            })
                elif self.last_recognition_results:
                    results = []
                else:
                    for top, right, bottom, left in face_locations:
                        results.append({
                            'name': 'Detecting...',
                            'confidence': 0,
                            'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                            'present': False
                        })
            
            return results
            
        except Exception as e:
            print(f"Error in optimized frame processing: {str(e)}")
            return []
    
    
    def process_enhanced_registration_sample(self, frame):
        """Process enhanced registration sample for current pose"""
        try:
            if self.current_mode != "registration":
                return False, "Not in registration mode", {}
            
            # Use smaller frame for faster processing
            small_frame = cv2.resize(frame, (320, 240))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
            
            if face_locations:
                face_location = face_locations[0]
                face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
                
                if face_encodings:
                    # Scale face location back to original frame
                    top, right, bottom, left = face_location
                    top = int(top * (frame.shape[0] / 240))
                    right = int(right * (frame.shape[1] / 320))
                    bottom = int(bottom * (frame.shape[0] / 240))
                    left = int(left * (frame.shape[1] / 320))
                    
                    face_image = frame[top:bottom, left:right]
                    
                    # Calculate quality score
                    quality_score = self.calculate_face_quality(face_image, (top, right, bottom, left))
                    
                    # Analyze pose
                    detected_pose = self.analyze_face_pose(face_landmarks)
                    current_pose = self.required_poses[self.current_pose_index]['name']
                    
                    # Check if quality is acceptable
                    if quality_score < 0.3:
                        return False, "Face quality too low - move closer or improve lighting", {
                            'current_pose': current_pose,
                            'pose_description': self.required_poses[self.current_pose_index]['description'],
                            'pose_icon': self.required_poses[self.current_pose_index]['icon'],
                            'quality_score': quality_score,
                            'detected_pose': detected_pose
                        }
                    
                    # Save the sample
                    sample_id = len(self.registration_samples)
                    saved_path = self.save_enhanced_face_image(
                        face_image, 
                        f"{self.registration_name}_sample_{sample_id}", 
                        current_pose,
                        quality_score,
                        "registered"
                    )
                    
                    if saved_path:
                        self.registration_samples.append({
                            'encoding': face_encodings[0],
                            'image_path': saved_path,
                            'face_location': (top, right, bottom, left),
                            'pose': current_pose,
                            'quality_score': quality_score,
                            'detected_pose': detected_pose,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        self.current_pose_samples += 1
                        total_samples = len(self.registration_samples)
                        
                        print(f"Captured sample {self.current_pose_samples}/{self.samples_per_pose} for pose '{current_pose}' (Total: {total_samples})")
                        
                        # Check if we need to move to next pose
                        if self.current_pose_samples >= self.samples_per_pose:
                            self.current_pose_index += 1
                            self.current_pose_samples = 0
                            
                            if self.current_pose_index >= len(self.required_poses):
                                # All poses completed
                                return True, "All poses captured successfully!", {
                                    'completed': True,
                                    'total_samples': total_samples
                                }
                            else:
                                # Move to next pose
                                next_pose = self.required_poses[self.current_pose_index]
                                return True, f"Pose '{current_pose}' completed! Now: {next_pose['description']}", {
                                    'current_pose': next_pose['name'],
                                    'pose_description': next_pose['description'],
                                    'pose_icon': next_pose['icon'],
                                    'samples_for_pose': self.current_pose_samples,
                                    'total_samples': total_samples,
                                    'quality_score': quality_score,
                                    'detected_pose': detected_pose
                                }
                        else:
                            # Need more samples for current pose
                            return True, f"Sample {self.current_pose_samples}/{self.samples_per_pose} captured for '{current_pose}'", {
                                'current_pose': current_pose,
                                'pose_description': self.required_poses[self.current_pose_index]['description'],
                                'pose_icon': self.required_poses[self.current_pose_index]['icon'],
                                'samples_for_pose': self.current_pose_samples,
                                'total_samples': total_samples,
                                'quality_score': quality_score,
                                'detected_pose': detected_pose
                            }
            
            # No face detected
            current_pose = self.required_poses[self.current_pose_index]['name']
            return False, "No face detected in frame", {
                'current_pose': current_pose,
                'pose_description': self.required_poses[self.current_pose_index]['description'],
                'pose_icon': self.required_poses[self.current_pose_index]['icon'],
                'quality_score': 0.0,
                'detected_pose': 'none'
            }
            
        except Exception as e:
            print(f"Enhanced registration processing error: {str(e)}")
            return False, f"Registration error: {str(e)}", {}
    
    def complete_enhanced_registration(self, name):
        """Complete enhanced registration with multiple poses"""
        try:
            if len(self.registration_samples) < (len(self.required_poses) * self.samples_per_pose):
                return False, f"Not enough samples. Need {len(self.required_poses) * self.samples_per_pose}, got {len(self.registration_samples)}"
            
            # Organize samples by pose
            poses_data = {}
            for sample in self.registration_samples:
                pose = sample['pose']
                if pose not in poses_data:
                    poses_data[pose] = []
                poses_data[pose].append(sample)
            
            # Create enhanced encoding data
            enhanced_data = {
                'name': name,
                'registration_date': datetime.now().isoformat(),
                'total_samples': len(self.registration_samples),
                'poses': []
            }
            
            for pose_name, pose_samples in poses_data.items():
                # Calculate average encoding for this pose
                encodings = [sample['encoding'] for sample in pose_samples]
                average_encoding = np.mean(encodings, axis=0)
                
                # Calculate average quality
                avg_quality = np.mean([sample['quality_score'] for sample in pose_samples])
                
                pose_data = {
                    'pose': pose_name,
                    'encoding': average_encoding.tolist(),
                    'confidence': min(1.0, avg_quality * 1.2),  # Boost confidence slightly
                    'quality_score': avg_quality,
                    'sample_count': len(pose_samples),
                    'timestamp': datetime.now().isoformat()
                }
                
                enhanced_data['poses'].append(pose_data)
            
            # Save to database first, then files as backup
            saved_to_db = False
            if self.use_database:
                try:
                    # Save person
                    person_id = self.db_manager.save_person(name, is_enhanced=True, total_samples=len(self.registration_samples))
                    
                    # Save each pose encoding
                    for pose_name, pose_samples in poses_data.items():
                        # Calculate average encoding for this pose
                        encodings = [sample['encoding'] for sample in pose_samples]
                        average_encoding = np.mean(encodings, axis=0)
                        avg_quality = np.mean([sample['quality_score'] for sample in pose_samples])
                        confidence = min(1.0, avg_quality * 1.2)
                        
                        # Save to database
                        self.db_manager.save_face_encoding(
                            person_name=name,
                            pose=pose_name,
                            encoding=average_encoding,
                            confidence=confidence,
                            quality_score=avg_quality,
                            image_path=pose_samples[0].get('image_path')  # Use first sample's image path
                        )
                    
                    saved_to_db = True
                    print(f"✅ Saved enhanced registration to database")
                    
                except Exception as e:
                    print(f"❌ Database save failed: {e}")
                    print("📁 Falling back to file storage")
            
            # Also save to file as backup (or primary if database failed)
            try:
                # Save to enhanced encodings file
                encodings_file = os.path.join(self.known_faces_dir, "enhanced_encodings.json")
                
                # Load existing data
                existing_data = {}
                if os.path.exists(encodings_file):
                    try:
                        with open(encodings_file, 'r') as f:
                            existing_data = json.load(f)
                    except:
                        existing_data = {}
                
                # Add/update person data
                existing_data[name] = enhanced_data
                
                # Save updated data
                with open(encodings_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
                print(f"✅ Saved enhanced registration to files")
                
            except Exception as e:
                print(f"❌ File save failed: {e}")
                if not saved_to_db:
                    # Both database and file failed
                    self.cleanup_registration()
                    return False, f"Registration failed: Could not save data - {str(e)}"
            
            # Reload face database
            self.load_known_faces()
            
            # Cleanup
            self.cleanup_registration()
            
            print(f"Successfully completed enhanced registration for {name}")
            print(f"Captured {len(enhanced_data['poses'])} different poses")
            return True, f"Enhanced registration completed! Captured {len(enhanced_data['poses'])} poses with {enhanced_data['total_samples']} samples."
            
        except Exception as e:
            print(f"Enhanced registration completion error: {str(e)}")
            self.cleanup_registration()
            return False, f"Registration failed: {str(e)}"
    
    def get_registration_progress(self):
        """Get current registration progress"""
        if self.current_mode != "registration":
            return {}
        
        total_required = len(self.required_poses) * self.samples_per_pose
        current_samples = len(self.registration_samples)
        progress_percentage = (current_samples / total_required) * 100
        
        current_pose = self.required_poses[self.current_pose_index] if self.current_pose_index < len(self.required_poses) else None
        
        return {
            'current_pose_index': self.current_pose_index,
            'current_pose': current_pose['name'] if current_pose else 'completed',
            'pose_description': current_pose['description'] if current_pose else 'All poses completed',
            'pose_icon': current_pose['icon'] if current_pose else '✅',
            'samples_for_pose': self.current_pose_samples,
            'samples_per_pose': self.samples_per_pose,
            'total_samples': current_samples,
            'total_required': total_required,
            'progress_percentage': progress_percentage,
            'completed': current_samples >= total_required
        }
    
    def cleanup_registration(self):
        """Clean up registration data"""
        self.registration_samples = []
        self.current_mode = "recognition"
        self.registration_name = ""
        self.current_pose_index = 0
        self.current_pose_samples = 0
        gc.collect()

    def start_enhanced_registration(self, name):
        """Start enhanced multi-angle registration for a person"""
        try:
            if self.current_mode == "registration":
                return False, "Registration already in progress"
            
            if not name or name.strip() == "":
                return False, "Name cannot be empty"
            
            # Clean up any previous registration
            self.cleanup_registration()
            
            # Initialize registration
            self.current_mode = "registration"
            self.registration_name = name.strip()
            self.registration_samples = []
            self.current_pose_index = 0
            self.current_pose_samples = 0
            
            print(f"🎯 Started enhanced registration for: {self.registration_name}")
            print(f"📸 Required poses: {[pose['name'] for pose in self.required_poses]}")
            print(f"🔢 Samples per pose: {self.samples_per_pose}")
            
            return True, f"Enhanced registration started for {self.registration_name}. Position yourself for the '{self.required_poses[0]['name']}' pose."
            
        except Exception as e:
            print(f"Error starting enhanced registration: {str(e)}")
            self.cleanup_registration()
            return False, f"Error starting registration: {str(e)}"

    # Keep all existing attendance deletion methods...
    def delete_attendance_record(self, name, date, time_str):
        try:
            if not os.path.exists(self.attendance_file):
                return False, "No attendance file found"
            
            df = pd.read_csv(self.attendance_file)
            mask = (df['Name'] == name) & (df['Date'] == date) & (df['Time'] == time_str)
            
            if not mask.any():
                return False, "Record not found"
            
            df = df[~mask]
            df.to_csv(self.attendance_file, index=False)
            
            today = datetime.now().strftime("%Y-%m-%d")
            if date == today:
                self.load_today_attendance()
            
            return True, "Record deleted successfully"
            
        except Exception as e:
            print(f"Error deleting attendance record: {str(e)}")
            return False, str(e)
    
    def delete_all_attendance_for_person(self, name):
        try:
            if not os.path.exists(self.attendance_file):
                return False, "No attendance file found"
            
            df = pd.read_csv(self.attendance_file)
            original_count = len(df)
            df = df[df['Name'] != name]
            deleted_count = original_count - len(df)
            
            if deleted_count == 0:
                return False, "No records found for this person"
            
            df.to_csv(self.attendance_file, index=False)
            self.load_today_attendance()
            
            return True, f"Deleted {deleted_count} records for {name}"
            
        except Exception as e:
            print(f"Error deleting attendance records: {str(e)}")
            return False, str(e)
    
    def delete_attendance_by_date(self, date):
        try:
            if not os.path.exists(self.attendance_file):
                return False, "No attendance file found"
            
            df = pd.read_csv(self.attendance_file)
            original_count = len(df)
            df = df[df['Date'] != date]
            deleted_count = original_count - len(df)
            
            if deleted_count == 0:
                return False, "No records found for this date"
            
            df.to_csv(self.attendance_file, index=False)
            
            today = datetime.now().strftime("%Y-%m-%d")
            if date == today:
                self.load_today_attendance()
            
            return True, f"Deleted {deleted_count} records for {date}"
            
        except Exception as e:
            print(f"Error deleting attendance records: {str(e)}")
            return False, str(e)

    # ===================================
    # REAL-TIME DETECTION SYSTEM
    # ===================================
    
    def __init_realtime_system__(self):
        """Initialize real-time detection system"""
        self.realtime_clients = set()  # SSE clients
        self.realtime_running = False
        self.realtime_thread = None
        self.detection_queue = queue.Queue()
        self.realtime_lock = threading.Lock()
        
    def start_realtime_detection(self):
        """Start real-time face detection in background thread"""
        with self.realtime_lock:
            if self.realtime_running:
                return True, "Real-time detection already running"
            
            try:
                # Initialize realtime system if not done
                if not hasattr(self, 'realtime_clients'):
                    self.__init_realtime_system__()
                
                self.realtime_running = True
                self.realtime_thread = threading.Thread(target=self._realtime_detection_loop, daemon=True)
                self.realtime_thread.start()
                
                print("🚀 Real-time detection started")
                return True, "Real-time detection started successfully"
                
            except Exception as e:
                self.realtime_running = False
                print(f"❌ Error starting real-time detection: {str(e)}")
                return False, str(e)
    
    def stop_realtime_detection(self):
        """Stop real-time face detection"""
        with self.realtime_lock:
            if not self.realtime_running:
                return True, "Real-time detection not running"
            
            try:
                self.realtime_running = False
                
                # Wait for thread to finish
                if self.realtime_thread and self.realtime_thread.is_alive():
                    self.realtime_thread.join(timeout=2)
                
                # Clear clients
                self.realtime_clients.clear()
                
                print("⏹️ Real-time detection stopped")
                return True, "Real-time detection stopped successfully"
                
            except Exception as e:
                print(f"❌ Error stopping real-time detection: {str(e)}")
                return False, str(e)
    
    def _realtime_detection_loop(self):
        """Main real-time detection loop"""
        print("🔄 Real-time detection loop started")
        last_detection_time = {}  # Track last detection per person
        detection_cooldown = 30  # 30 seconds between detections for same person
        
        while self.realtime_running:
            try:
                if not self.is_running or self.video_capture is None:
                    time.sleep(1)
                    continue
                
                # Capture frame
                ret, frame = self.video_capture.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Process frame for face detection
                faces_detected = self.process_frame_optimized(frame)
                
                if faces_detected:
                    current_time = time.time()
                    
                    for face_data in faces_detected:
                        name = face_data.get('name')
                        confidence = face_data.get('confidence', 0)
                        
                        if name and name != 'Unknown' and confidence > 0.7:
                            # Check cooldown
                            if name in last_detection_time:
                                time_diff = current_time - last_detection_time[name]
                                if time_diff < detection_cooldown:
                                    continue
                            
                            # Update last detection time
                            last_detection_time[name] = current_time
                            
                            # Emit face detected event
                            self._emit_realtime_event({
                                'type': 'face_detected',
                                'data': {
                                    'name': name,
                                    'confidence': confidence,
                                    'timestamp': datetime.now().isoformat(),
                                    'location': face_data.get('location')
                                }
                            })
                            
                            # Mark attendance automatically
                            attendance_result = self.mark_attendance(name)
                            if attendance_result:
                                # Emit attendance marked event
                                self._emit_realtime_event({
                                    'type': 'attendance_marked',
                                    'data': {
                                        'name': name,
                                        'status': 'present',
                                        'message': f'{name} automatically marked present',
                                        'timestamp': datetime.now().isoformat(),
                                        'confidence': confidence
                                    }
                                })
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error in real-time detection loop: {str(e)}")
                time.sleep(1)
    
    def _emit_realtime_event(self, event_data):
        """Emit event to all connected SSE clients"""
        if not self.realtime_clients:
            return
        
        try:
            # Add to queue for SSE clients
            self.detection_queue.put(event_data)
            print(f"📡 Emitted event: {event_data['type']} for {event_data.get('data', {}).get('name', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error emitting real-time event: {str(e)}")
    
    def add_realtime_client(self, client_id):
        """Add SSE client"""
        self.realtime_clients.add(client_id)
        print(f"👤 Added realtime client: {client_id}")
    
    def remove_realtime_client(self, client_id):
        """Remove SSE client"""
        self.realtime_clients.discard(client_id)
        print(f"👋 Removed realtime client: {client_id}")
    
    def get_realtime_status(self):
        """Get real-time detection status"""
        return {
            'running': self.realtime_running,
            'clients_connected': len(getattr(self, 'realtime_clients', [])),
            'camera_active': self.is_running,
            'thread_alive': self.realtime_thread.is_alive() if hasattr(self, 'realtime_thread') and self.realtime_thread else False
        }

# Initialize the enhanced attendance system
attendance_system = EnhancedAttendanceSystem()

# Health check endpoint for Render
@app.route('/')
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Face Recognition Attendance System API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'environment': ENV
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    try:
        if not attendance_system.is_running:
            attendance_system.video_capture = cv2.VideoCapture(0)
            if attendance_system.video_capture.isOpened():
                attendance_system.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                attendance_system.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                attendance_system.video_capture.set(cv2.CAP_PROP_FPS, 30)
                attendance_system.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                attendance_system.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                
                attendance_system.is_running = True
                return jsonify({'success': True, 'message': 'Camera started'})
            else:
                return jsonify({'success': False, 'message': 'Could not open camera'})
        else:
            return jsonify({'success': True, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    try:
        attendance_system.is_running = False
        if attendance_system.video_capture:
            attendance_system.video_capture.release()
        attendance_system.cleanup_registration()
        return jsonify({'success': True, 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/camera/frame', methods=['GET'])
def get_camera_frame():
    try:
        if not attendance_system.is_running or not attendance_system.video_capture:
            return jsonify({'success': False, 'message': 'Camera not running'})
        
        ret, frame = attendance_system.video_capture.read()
        if not ret:
            return jsonify({'success': False, 'message': 'Could not read frame'})
        
        # Use enhanced processing
        results = attendance_system.process_frame_optimized(frame)
        
        # Draw enhanced bounding boxes
        for result in results:
            loc = result['location']
            
            # Enhanced color coding
            if result['name'] not in ["Unknown", "Detecting..."]:
                if result['present']:
                    color = (0, 255, 0)  # Green for present
                else:
                    color = (0, 150, 255)  # Blue for registered but not present
            else:
                color = (0, 100, 255)  # Orange for unknown
            
            # Draw rectangle with confidence-based thickness
            thickness = 3 if result['confidence'] > 0.7 else 2
            cv2.rectangle(frame, (loc['left'], loc['top']), (loc['right'], loc['bottom']), color, thickness)
            
            # Enhanced label
            label = result['name']
            if result['name'] not in ["Unknown", "Detecting..."] and result['confidence'] > 0:
                label += f" ({result['confidence']:.2f})"
            
            # Draw label with enhanced styling
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Background for text
            cv2.rectangle(frame, 
                         (loc['left'], loc['bottom'] + 5), 
                         (loc['left'] + label_width + 6, loc['bottom'] + label_height + 10), 
                         color, cv2.FILLED)
            
            cv2.putText(frame, label, (loc['left'] + 3, loc['bottom'] + label_height + 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Enhanced status indicators
            if result['name'] not in ["Unknown", "Detecting..."] and result['present']:
                cv2.circle(frame, (loc['right'] - 15, loc['top'] + 15), 8, (0, 255, 0), -1)
                cv2.circle(frame, (loc['right'] - 15, loc['top'] + 15), 8, (255, 255, 255), 2)
        
        # Encode with good quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True, 
            'frame': frame_base64,
            'faces': results
        })
    except Exception as e:
        print(f"Frame capture error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

# Enhanced registration endpoints
@app.route('/api/registration/start', methods=['POST'])
def start_enhanced_registration():
    try:
        data = request.json
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'})
        
        if not attendance_system.is_running:
            return jsonify({'success': False, 'message': 'Camera must be running first'})
        
        success, message = attendance_system.start_enhanced_registration(name)
        
        if success:
            progress = attendance_system.get_registration_progress()
            return jsonify({
                'success': True, 
                'message': message,
                'progress': progress
            })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        print(f"Enhanced registration start error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/registration/capture', methods=['POST'])
def capture_enhanced_registration_sample():
    try:
        if attendance_system.current_mode != "registration":
            return jsonify({'success': False, 'message': 'Not in registration mode'})
        
        if not attendance_system.is_running or not attendance_system.video_capture:
            return jsonify({'success': False, 'message': 'Camera not running'})
        
        # Read frame
        for attempt in range(3):
            ret, frame = attendance_system.video_capture.read()
            if ret:
                break
            time.sleep(0.1)
        
        if not ret:
            return jsonify({'success': False, 'message': 'Could not read frame'})
        
        success, message, sample_data = attendance_system.process_enhanced_registration_sample(frame)
        progress = attendance_system.get_registration_progress()
        
        if success and sample_data.get('completed', False):
            # Registration completed
            success_reg, completion_message = attendance_system.complete_enhanced_registration(attendance_system.registration_name)
            return jsonify({
                'success': success_reg,
                'message': completion_message,
                'progress': progress,
                'sample_data': sample_data,
                'completed': True
            })
        else:
            return jsonify({
                'success': success,
                'message': message,
                'progress': progress,
                'sample_data': sample_data,
                'completed': False
            })
            
    except Exception as e:
        print(f"Enhanced registration capture error: {str(e)}")
        attendance_system.cleanup_registration()
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/registration/progress', methods=['GET'])
def get_registration_progress():
    try:
        progress = attendance_system.get_registration_progress()
        return jsonify({'success': True, 'progress': progress})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Keep all existing endpoints for people, statistics, attendance management...
@app.route('/api/people', methods=['GET'])
def get_people():
    try:
        # Get unique people from the enhanced database with metadata
        unique_people = {}
        
        # Check if we have enhanced encodings file
        enhanced_file = os.path.join(attendance_system.known_faces_dir, "enhanced_encodings.json")
        enhanced_data = {}
        
        if os.path.exists(enhanced_file):
            try:
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
            except:
                enhanced_data = {}
        
        for name in attendance_system.known_face_names:
            if name not in unique_people:
                # Check if person has enhanced registration
                is_enhanced = name in enhanced_data and 'poses' in enhanced_data[name]
                
                person_data = {
                    'name': name,
                    'present': name in attendance_system.attendance_today,
                    'enhanced': is_enhanced
                }
                
                if is_enhanced:
                    person_info = enhanced_data[name]
                    person_data.update({
                        'poses': len(person_info.get('poses', [])),
                        'quality': 'Enhanced',
                        'registration_date': person_info.get('registration_date', 'Unknown')
                    })
                else:
                    person_data.update({
                        'poses': 1,
                        'quality': 'Basic',
                        'registration_date': 'Legacy'
                    })
                
                unique_people[name] = person_data
        
        people = list(unique_people.values())
        
        # Sort by enhanced first, then by name
        people.sort(key=lambda x: (not x['enhanced'], x['name']))
        
        return jsonify({'success': True, 'people': people})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/people/<name>', methods=['DELETE'])
def delete_person(name):
    try:
        deleted = False
        
        # Try to delete from enhanced database first
        enhanced_file = os.path.join(attendance_system.known_faces_dir, "enhanced_encodings.json")
        if os.path.exists(enhanced_file):
            try:
                with open(enhanced_file, 'r') as f:
                    data = json.load(f)
                
                if name in data:
                    del data[name]
                    
                    with open(enhanced_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    deleted = True
                    print(f"Deleted {name} from enhanced database")
            except Exception as e:
                print(f"Error updating enhanced database: {str(e)}")
        
        # Try to delete from legacy database if not found in enhanced
        legacy_file = os.path.join(attendance_system.known_faces_dir, "encodings.json")
        if not deleted and os.path.exists(legacy_file):
            try:
                with open(legacy_file, 'r') as f:
                    data = json.load(f)
                
                if name in data:
                    del data[name]
                    
                    with open(legacy_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    deleted = True
                    print(f"Deleted {name} from legacy database")
            except Exception as e:
                print(f"Error updating legacy database: {str(e)}")
        
        if deleted:
            # Reload the face database
            attendance_system.load_known_faces()
            return jsonify({'success': True, 'message': f'{name} deleted successfully'})
        else:
            # Check if person exists in memory but not in files
            if name in attendance_system.known_face_names:
                return jsonify({'success': False, 'message': f'{name} found in memory but not in database files. Try reloading the system.'})
            else:
                return jsonify({'success': False, 'message': f'Person "{name}" not found in database'})
    
    except Exception as e:
        print(f"Delete person error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error deleting person: {str(e)}'})

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        # Count unique people
        unique_people = set(attendance_system.known_face_names)
        registered_count = len(unique_people)
        present_count = len(attendance_system.attendance_today)
        absent_count = registered_count - present_count
        
        last_recognition = "None"
        if attendance_system.recent_recognitions:
            last_name = max(attendance_system.recent_recognitions.keys(), 
                           key=lambda x: attendance_system.recent_recognitions[x])
            last_time = attendance_system.recent_recognitions[last_name].strftime("%H:%M")
            last_recognition = f"{last_name} ({last_time})"
        
        return jsonify({
            'success': True,
            'statistics': {
                'registered_count': registered_count,
                'present_count': present_count,
                'absent_count': absent_count,
                'last_recognition': last_recognition
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/<date>', methods=['GET'])
def get_attendance(date):
    try:
        if os.path.exists(attendance_system.attendance_file):
            df = pd.read_csv(attendance_system.attendance_file)
            date_records = df[df['Date'] == date]
            
            records = []
            for _, row in date_records.iterrows():
                records.append({
                    'name': row['Name'],
                    'time': row['Time'],
                    'status': row['Status']
                })
            
            return jsonify({
                'success': True,
                'date': date,
                'records': records,
                'total': len(records)
            })
        else:
            return jsonify({'success': True, 'records': [], 'total': 0})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Attendance deletion endpoints
@app.route('/api/attendance/delete/record', methods=['DELETE'])
def delete_attendance_record():
    try:
        data = request.json
        name = data.get('name')
        date = data.get('date')
        time_str = data.get('time')
        
        if not all([name, date, time_str]):
            return jsonify({'success': False, 'message': 'Name, date, and time are required'})
        
        success, message = attendance_system.delete_attendance_record(name, date, time_str)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/delete/person/<name>', methods=['DELETE'])
def delete_all_attendance_for_person(name):
    try:
        success, message = attendance_system.delete_all_attendance_for_person(name)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/delete/date/<date>', methods=['DELETE'])
def delete_attendance_by_date(date):
    try:
        success, message = attendance_system.delete_attendance_by_date(date)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/export', methods=['GET'])
def export_attendance():
    try:
        if not os.path.exists(attendance_system.attendance_file):
            return jsonify({'success': False, 'message': 'No attendance data found'})
        
        return send_file(attendance_system.attendance_file, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system/reload', methods=['POST'])
def reload_system():
    try:
        attendance_system.load_known_faces()
        attendance_system.load_today_attendance()
        return jsonify({'success': True, 'message': 'Enhanced system reloaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/database/status', methods=['GET'])
def get_database_status():
    """Get database connection status and info"""
    try:
        status = {
            'connected': USE_DATABASE,
            'type': 'MySQL' if USE_DATABASE else 'File-based',
            'host': db_config.get('host', 'N/A') if USE_DATABASE else 'N/A',
            'database': db_config.get('database', 'N/A') if USE_DATABASE else 'N/A'
        }
        
        if USE_DATABASE:
            try:
                persons = db_manager.get_all_persons()
                status['persons_count'] = len(persons)
                status['persons'] = [p['name'] for p in persons]
                
                # Get today's attendance count
                today_attendance = db_manager.get_todays_attendance()
                status['todays_attendance'] = len(today_attendance)
                
            except Exception as e:
                status['error'] = str(e)
        
        return jsonify({'success': True, 'database': status})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/database/migrate', methods=['POST'])
def migrate_to_database():
    """Manually trigger migration from files to database"""
    try:
        if not USE_DATABASE:
            return jsonify({'success': False, 'message': 'Database not connected'})
        
        # Force migration
        attendance_system.migrate_existing_data()
        
        # Reload data
        attendance_system.load_known_faces()
        attendance_system.load_today_attendance()
        
        return jsonify({
            'success': True, 
            'message': 'Migration completed successfully',
            'persons_count': len(set(attendance_system.known_face_names)),
            'attendance_today': len(attendance_system.attendance_today)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/database/<date>', methods=['GET'])
def get_attendance_from_database(date):
    """Get attendance from database for specific date"""
    try:
        if not USE_DATABASE:
            return jsonify({'success': False, 'message': 'Database not connected'})
        
        records = db_manager.get_attendance(date=date)
        
        return jsonify({
            'success': True,
            'date': date,
            'records': records,
            'total': len(records),
            'source': 'database'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/locations/save', methods=['POST'])
def save_location():
    """Save location data to database"""
    try:
        data = request.json
        person_name = data.get('name')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        address = data.get('address')
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'})
        
        if USE_DATABASE:
            success = db_manager.save_location(person_name, latitude, longitude, address)
            if success:
                return jsonify({'success': True, 'message': 'Location saved to database'})
        
        # Fallback to file storage (your existing implementation)
        return record_attendance_location()
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/locations/get/<name>', methods=['GET'])
def get_locations_from_database(name):
    """Get location history from database"""
    try:
        if USE_DATABASE:
            locations = db_manager.get_locations(name, limit=10)
            return jsonify({
                'success': True,
                'name': name,
                'locations': locations,
                'source': 'database'
            })
        
        # Fallback to file storage
        return get_person_locations(name)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/debug/database-status', methods=['GET'])
def debug_database_status():
    """Debug endpoint to check database file status"""
    try:
        status = {
            'known_faces_dir': attendance_system.known_faces_dir,
            'dir_exists': os.path.exists(attendance_system.known_faces_dir),
            'files_in_dir': [],
            'enhanced_file_exists': False,
            'legacy_file_exists': False,
            'people_in_memory': len(set(attendance_system.known_face_names)),
            'unique_people_names': list(set(attendance_system.known_face_names)),
            'total_encodings_loaded': len(attendance_system.known_face_encodings)
        }
        
        # Check directory contents
        if os.path.exists(attendance_system.known_faces_dir):
            status['files_in_dir'] = os.listdir(attendance_system.known_faces_dir)
        
        # Check enhanced file
        enhanced_file = os.path.join(attendance_system.known_faces_dir, "enhanced_encodings.json")
        status['enhanced_file_exists'] = os.path.exists(enhanced_file)
        status['enhanced_file_path'] = enhanced_file
        
        if status['enhanced_file_exists']:
            try:
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                status['enhanced_people'] = list(enhanced_data.keys())
                status['enhanced_count'] = len(enhanced_data)
            except Exception as e:
                status['enhanced_error'] = str(e)
        
        # Check legacy file
        legacy_file = os.path.join(attendance_system.known_faces_dir, "encodings.json")
        status['legacy_file_exists'] = os.path.exists(legacy_file)
        status['legacy_file_path'] = legacy_file
        
        if status['legacy_file_exists']:
            try:
                with open(legacy_file, 'r') as f:
                    legacy_data = json.load(f)
                status['legacy_people'] = list(legacy_data.keys())
                status['legacy_count'] = len(legacy_data)
            except Exception as e:
                status['legacy_error'] = str(e)
        
        return jsonify({'success': True, 'status': status})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/debug/create-database', methods=['POST'])
def debug_create_database():
    """Create the enhanced database file if it doesn't exist"""
    try:
        # Ensure directory exists
        os.makedirs(attendance_system.known_faces_dir, exist_ok=True)
        
        enhanced_file = os.path.join(attendance_system.known_faces_dir, "enhanced_encodings.json")
        
        if not os.path.exists(enhanced_file):
            # Create empty enhanced database
            with open(enhanced_file, 'w') as f:
                json.dump({}, f, indent=2)
            
            return jsonify({'success': True, 'message': 'Enhanced database file created'})
        else:
            return jsonify({'success': True, 'message': 'Enhanced database file already exists'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/debug/migrate-legacy', methods=['POST'])
def debug_migrate_legacy():
    """Migrate legacy encodings to enhanced format"""
    try:
        legacy_file = os.path.join(attendance_system.known_faces_dir, "encodings.json")
        enhanced_file = os.path.join(attendance_system.known_faces_dir, "enhanced_encodings.json")
        
        if not os.path.exists(legacy_file):
            return jsonify({'success': False, 'message': 'No legacy database found to migrate'})
        
        # Load legacy data
        with open(legacy_file, 'r') as f:
            legacy_data = json.load(f)
        
        # Load or create enhanced data
        enhanced_data = {}
        if os.path.exists(enhanced_file):
            with open(enhanced_file, 'r') as f:
                enhanced_data = json.load(f)
        
        migrated_count = 0
        
        # Convert legacy format to enhanced format
        for name, encoding in legacy_data.items():
            if name not in enhanced_data:
                enhanced_data[name] = {
                    'name': name,
                    'registration_date': datetime.now().isoformat(),
                    'total_samples': 1,
                    'poses': [{
                        'pose': 'front',
                        'encoding': encoding if isinstance(encoding, list) else encoding.tolist(),
                        'confidence': 1.0,
                        'quality_score': 0.8,
                        'sample_count': 1,
                        'timestamp': datetime.now().isoformat()
                    }]
                }
                migrated_count += 1
        
        # Save enhanced data
        with open(enhanced_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        # Reload face database
        attendance_system.load_known_faces()
        
        return jsonify({
            'success': True, 
            'message': f'Migrated {migrated_count} people from legacy to enhanced format'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/debug/face-recognition-test', methods=['POST'])
def debug_face_recognition():
    """Test face recognition with current frame - WITH DETAILED DEBUGGING"""
    try:
        if not attendance_system.is_running or not attendance_system.video_capture:
            return jsonify({'success': False, 'message': 'Camera not running'})
        
        # Get current frame
        ret, frame = attendance_system.video_capture.read()
        if not ret:
            return jsonify({'success': False, 'message': 'Could not capture frame'})
        
        # Process frame for debugging
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No faces detected in current frame'})
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        results = []
        for i, face_encoding in enumerate(face_encodings):
            # DETAILED DEBUGGING VERSION
            print(f"\n=== DEBUGGING FACE {i+1} ===")
            print(f"Known face encodings: {len(attendance_system.known_face_encodings)}")
            print(f"Known face names: {attendance_system.known_face_names}")
            
            if not attendance_system.known_face_encodings:
                print("ERROR: No known face encodings loaded!")
                results.append({
                    'face_index': i,
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'error': 'No face encodings loaded',
                    'location': face_locations[i]
                })
                continue
            
            # Calculate distances
            face_distances = face_recognition.face_distance(attendance_system.known_face_encodings, face_encoding)
            print(f"Face distances: {face_distances}")
            
            # Group by person
            person_scores = {}
            for j, distance in enumerate(face_distances):
                name = attendance_system.known_face_names[j]
                metadata = attendance_system.known_face_metadata[j]
                
                if name not in person_scores:
                    person_scores[name] = []
                
                confidence = 1 - distance
                weighted_score = confidence * metadata['confidence'] * metadata['quality_score']
                person_scores[name].append(weighted_score)
                
                print(f"  {name} ({metadata['pose']}): distance={distance:.3f}, confidence={confidence:.3f}, weighted={weighted_score:.3f}")
            
            # Find best match
            best_name = "Unknown"
            best_confidence = 0.0
            
            for name, scores in person_scores.items():
                top_scores = sorted(scores, reverse=True)[:3]
                avg_confidence = sum(top_scores) / len(top_scores)
                
                print(f"  {name}: avg_confidence={avg_confidence:.3f}")
                
                # LOWERED THRESHOLD for testing
                if avg_confidence > best_confidence and avg_confidence > 0.30:  # Lowered from 0.4 to 0.30
                    best_confidence = avg_confidence
                    best_name = name
            
            print(f"RESULT: {best_name} with confidence {best_confidence:.3f}")
            
            results.append({
                'face_index': i,
                'name': best_name,
                'confidence': best_confidence,
                'location': face_locations[i],
                'person_scores': {name: max(scores) for name, scores in person_scores.items()},
                'known_people_count': len(set(attendance_system.known_face_names)),
                'total_encodings': len(attendance_system.known_face_encodings)
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'debug_info': {
                'total_known_people': len(set(attendance_system.known_face_names)),
                'total_encodings': len(attendance_system.known_face_encodings),
                'known_names': list(set(attendance_system.known_face_names))
            }
        })
        
    except Exception as e:
        print(f"Debug face recognition error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/debug/lower-threshold', methods=['POST'])
def debug_lower_threshold():
    """Temporarily lower recognition threshold for testing"""
    try:
        # This will modify the enhanced_face_recognition method threshold
        # We'll return current settings
        return jsonify({
            'success': True, 
            'message': 'Use the face recognition test endpoint - it uses a lowered threshold of 0.30',
            'current_threshold': 0.4,
            'test_threshold': 0.30
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/location', methods=['POST'])
def record_attendance_location():
    """Record location information for attendance"""
    try:
        data = request.json
        name = data.get('name')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        address = data.get('address')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'})
        
        # Create location record
        location_data = {
            'name': name,
            'latitude': latitude,
            'longitude': longitude,
            'address': address,
            'timestamp': timestamp
        }
        
        # Save to location file (you could enhance this to use a proper database)
        location_file = os.path.join(attendance_system.known_faces_dir, "locations.json")
        
        # Load existing location data
        locations = {}
        if os.path.exists(location_file):
            try:
                with open(location_file, 'r') as f:
                    locations = json.load(f)
            except:
                locations = {}
        
        # Update location for person
        if name not in locations:
            locations[name] = []
        
        locations[name].append(location_data)
        
        # Keep only last 10 locations per person
        locations[name] = locations[name][-10:]
        
        # Save updated locations
        with open(location_file, 'w') as f:
            json.dump(locations, f, indent=2)
        
        return jsonify({
            'success': True, 
            'message': f'Location recorded for {name}',
            'location': location_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/locations/<name>', methods=['GET'])
def get_person_locations(name):
    """Get location history for a specific person"""
    try:
        location_file = os.path.join(attendance_system.known_faces_dir, "locations.json")
        
        if not os.path.exists(location_file):
            return jsonify({'success': True, 'locations': []})
        
        with open(location_file, 'r') as f:
            locations = json.load(f)
        
        person_locations = locations.get(name, [])
        
        return jsonify({
            'success': True, 
            'name': name,
            'locations': person_locations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Enhanced Export Endpoints
@app.route('/api/attendance/export/range', methods=['GET'])
def export_attendance_by_range():
    """Export attendance data for a specific date range"""
    try:
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        
        if not start_date or not end_date:
            return jsonify({'success': False, 'message': 'Start date and end date are required'})
        
        if not os.path.exists(attendance_system.attendance_file):
            return jsonify({'success': False, 'message': 'No attendance data found'})
        
        df = pd.read_csv(attendance_system.attendance_file)
        
        # Filter by date range
        df['Date'] = pd.to_datetime(df['Date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
        
        if filtered_df.empty:
            return jsonify({'success': False, 'message': 'No data found for the specified date range'})
        
        # Create enhanced export with location data if available
        enhanced_df = enhance_attendance_data(filtered_df)
        
        # Create temporary file
        temp_file = f"attendance_export_{start_date}_to_{end_date}.csv"
        enhanced_df.to_csv(temp_file, index=False)
        
        return send_file(temp_file, as_attachment=True, download_name=temp_file, mimetype='text/csv')
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/export/day/<date>', methods=['GET'])
def export_attendance_by_day(date):
    """Export attendance data for a specific day"""
    try:
        if not os.path.exists(attendance_system.attendance_file):
            return jsonify({'success': False, 'message': 'No attendance data found'})
        
        df = pd.read_csv(attendance_system.attendance_file)
        day_records = df[df['Date'] == date]
        
        if day_records.empty:
            return jsonify({'success': False, 'message': f'No attendance data found for {date}'})
        
        # Create enhanced export
        enhanced_df = enhance_attendance_data(day_records)
        
        # Create temporary file
        temp_file = f"attendance_{date}.csv"
        enhanced_df.to_csv(temp_file, index=False)
        
        return send_file(temp_file, as_attachment=True, download_name=temp_file, mimetype='text/csv')
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/export/month/<year>/<month>', methods=['GET'])
def export_attendance_by_month(year, month):
    """Export attendance data for a specific month"""
    try:
        if not os.path.exists(attendance_system.attendance_file):
            return jsonify({'success': False, 'message': 'No attendance data found'})
        
        df = pd.read_csv(attendance_system.attendance_file)
        
        # Filter by month and year
        df['Date'] = pd.to_datetime(df['Date'])
        year_int = int(year)
        month_int = int(month)
        
        filtered_df = df[(df['Date'].dt.year == year_int) & (df['Date'].dt.month == month_int)]
        
        if filtered_df.empty:
            month_name = calendar.month_name[month_int]
            return jsonify({'success': False, 'message': f'No attendance data found for {month_name} {year}'})
        
        # Create enhanced export
        enhanced_df = enhance_attendance_data(filtered_df)
        
        # Create summary statistics
        summary_stats = calculate_monthly_summary(enhanced_df, year_int, month_int)
        
        # Create temporary file
        month_name = calendar.month_name[month_int]
        temp_file = f"attendance_{month_name}_{year}.csv"
        
        # Write both detailed data and summary
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            # Write summary first
            f.write(f"ATTENDANCE SUMMARY FOR {month_name.upper()} {year}\n")
            f.write("="*50 + "\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("DETAILED ATTENDANCE RECORDS\n")
            f.write("="*50 + "\n\n")
            
            # Write detailed data
            enhanced_df.to_csv(f, index=False)
        
        return send_file(temp_file, as_attachment=True, download_name=temp_file, mimetype='text/csv')
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance/export/detailed', methods=['POST'])
def export_detailed_attendance():
    """Export detailed attendance with custom options"""
    try:
        options = request.json or {}
        start_date = options.get('startDate')
        end_date = options.get('endDate')
        include_location = options.get('includeLocation', True)
        include_time_spent = options.get('includeTimeSpent', True)
        export_format = options.get('format', 'csv')
        
        if not os.path.exists(attendance_system.attendance_file):
            return jsonify({'success': False, 'message': 'No attendance data found'})
        
        df = pd.read_csv(attendance_system.attendance_file)
        
        # Filter by date range if provided
        if start_date and end_date:
            df['Date'] = pd.to_datetime(df['Date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
        
        if df.empty:
            return jsonify({'success': False, 'message': 'No data found for the specified criteria'})
        
        # Create enhanced export
        enhanced_df = enhance_attendance_data(df, include_location, include_time_spent)
        
        # Create temporary file
        file_extension = 'csv' if export_format == 'csv' else 'xlsx'
        temp_file = f"detailed_attendance_export.{file_extension}"
        
        if export_format == 'csv':
            enhanced_df.to_csv(temp_file, index=False)
            mimetype = 'text/csv'
        else:
            enhanced_df.to_excel(temp_file, index=False)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        return send_file(temp_file, as_attachment=True, download_name=temp_file, mimetype=mimetype)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def enhance_attendance_data(df, include_location=True, include_time_spent=True):
    """Enhance attendance dataframe with additional information"""
    try:
        enhanced_df = df.copy()
        
        # Add day of week
        enhanced_df['Date'] = pd.to_datetime(enhanced_df['Date'])
        enhanced_df['Day_of_Week'] = enhanced_df['Date'].dt.day_name()
        
        # Add location information if requested
        if include_location:
            location_file = os.path.join(attendance_system.known_faces_dir, "locations.json")
            locations = {}
            
            if os.path.exists(location_file):
                try:
                    with open(location_file, 'r') as f:
                        locations = json.load(f)
                except:
                    locations = {}
            
            # Add location columns
            enhanced_df['Location'] = ''
            enhanced_df['Latitude'] = ''
            enhanced_df['Longitude'] = ''
            
            for idx, row in enhanced_df.iterrows():
                person_locations = locations.get(row['Name'], [])
                if person_locations:
                    # Find location closest to attendance time
                    attendance_datetime = pd.to_datetime(f"{row['Date'].date()} {row['Time']}")
                    closest_location = find_closest_location(person_locations, attendance_datetime)
                    
                    if closest_location:
                        enhanced_df.at[idx, 'Location'] = closest_location.get('address', 'Unknown')
                        enhanced_df.at[idx, 'Latitude'] = closest_location.get('latitude', '')
                        enhanced_df.at[idx, 'Longitude'] = closest_location.get('longitude', '')
        
        # Add time spent calculation if requested
        if include_time_spent:
            enhanced_df = calculate_time_spent(enhanced_df)
        
        # Reorder columns for better readability
        column_order = ['Name', 'Date', 'Day_of_Week', 'Time', 'Status']
        
        if include_location:
            column_order.extend(['Location', 'Latitude', 'Longitude'])
        
        if include_time_spent:
            column_order.extend(['Time_Spent_Hours', 'First_Seen', 'Last_Seen'])
        
        # Add any remaining columns
        remaining_cols = [col for col in enhanced_df.columns if col not in column_order]
        column_order.extend(remaining_cols)
        
        # Filter to only existing columns
        column_order = [col for col in column_order if col in enhanced_df.columns]
        enhanced_df = enhanced_df[column_order]
        
        return enhanced_df
        
    except Exception as e:
        print(f"Error enhancing attendance data: {str(e)}")
        return df

def find_closest_location(person_locations, target_datetime):
    """Find the location record closest to the target datetime"""
    try:
        closest_location = None
        min_time_diff = float('inf')
        
        for location in person_locations:
            location_datetime = pd.to_datetime(location['timestamp'])
            time_diff = abs((target_datetime - location_datetime).total_seconds())
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_location = location
        
        # Only return location if it's within 1 hour of the attendance time
        if min_time_diff <= 3600:  # 1 hour in seconds
            return closest_location
        
        return None
        
    except Exception as e:
        print(f"Error finding closest location: {str(e)}")
        return None

def calculate_time_spent(df):
    """Calculate time spent for each person per day"""
    try:
        enhanced_df = df.copy()
        enhanced_df['Time_Spent_Hours'] = ''
        enhanced_df['First_Seen'] = ''
        enhanced_df['Last_Seen'] = ''
        
        # Group by person and date
        for (name, date), group in enhanced_df.groupby(['Name', enhanced_df['Date'].dt.date]):
            if len(group) > 1:
                times = pd.to_datetime(group['Time'], format='%H:%M:%S')
                first_time = times.min()
                last_time = times.max()
                
                time_spent = (last_time - first_time).total_seconds() / 3600  # Convert to hours
                
                # Update all records for this person on this date
                mask = (enhanced_df['Name'] == name) & (enhanced_df['Date'].dt.date == date)
                enhanced_df.loc[mask, 'Time_Spent_Hours'] = f"{time_spent:.2f}"
                enhanced_df.loc[mask, 'First_Seen'] = first_time.strftime('%H:%M:%S')
                enhanced_df.loc[mask, 'Last_Seen'] = last_time.strftime('%H:%M:%S')
            else:
                # Single attendance record
                mask = (enhanced_df['Name'] == name) & (enhanced_df['Date'].dt.date == date)
                enhanced_df.loc[mask, 'Time_Spent_Hours'] = '0.00'
                enhanced_df.loc[mask, 'First_Seen'] = group.iloc[0]['Time']
                enhanced_df.loc[mask, 'Last_Seen'] = group.iloc[0]['Time']
        
        return enhanced_df
        
    except Exception as e:
        print(f"Error calculating time spent: {str(e)}")
        return df

def calculate_monthly_summary(df, year, month):
    """Calculate monthly attendance summary statistics"""
    try:
        month_name = calendar.month_name[month]
        total_days_in_month = calendar.monthrange(year, month)[1]
        
        # Basic statistics
        total_attendance_records = len(df)
        unique_people = df['Name'].nunique()
        unique_days_with_attendance = df['Date'].dt.date.nunique()
        
        # Person-wise statistics
        person_stats = df.groupby('Name').agg({
            'Date': 'nunique',  # Days attended
            'Time': 'count'     # Total attendance records
        }).rename(columns={'Date': 'Days_Attended', 'Time': 'Total_Records'})
        
        # Most active person
        most_active_person = person_stats.loc[person_stats['Days_Attended'].idxmax(), :]
        
        summary = {
            'Month': f"{month_name} {year}",
            'Total Working Days in Month': total_days_in_month,
            'Days with Attendance': unique_days_with_attendance,
            'Total Attendance Records': total_attendance_records,
            'Unique People': unique_people,
            'Average Attendance per Day': f"{total_attendance_records / unique_days_with_attendance:.2f}" if unique_days_with_attendance > 0 else "0",
            'Most Active Person': f"{most_active_person.name} ({most_active_person['Days_Attended']} days)",
            'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
        
    except Exception as e:
        print(f"Error calculating monthly summary: {str(e)}")
        return {'Error': str(e)}

# ===================================
# REAL-TIME DETECTION ENDPOINTS
# ===================================

@app.route('/api/realtime/start', methods=['POST'])
def start_realtime_detection():
    """Start real-time face detection"""
    try:
        success, message = attendance_system.start_realtime_detection()
        return jsonify({
            'success': success,
            'message': message,
            'status': attendance_system.get_realtime_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/stop', methods=['POST'])
def stop_realtime_detection():
    """Stop real-time face detection"""
    try:
        success, message = attendance_system.stop_realtime_detection()
        return jsonify({
            'success': success,
            'message': message,
            'status': attendance_system.get_realtime_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/status', methods=['GET'])
def get_realtime_status():
    """Get real-time detection status"""
    try:
        return jsonify({
            'success': True,
            'status': attendance_system.get_realtime_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/events')
def realtime_events():
    """Server-Sent Events endpoint for real-time detection"""
    def event_generator():
        # Initialize realtime system if needed
        if not hasattr(attendance_system, 'realtime_clients'):
            attendance_system.__init_realtime_system__()
        
        # Generate unique client ID
        client_id = f"client_{int(time.time() * 1000)}"
        attendance_system.add_realtime_client(client_id)
        
        try:
            while True:
                try:
                    # Get event from queue (blocking with timeout)
                    event_data = attendance_system.detection_queue.get(timeout=30)
                    
                    # Format as SSE event
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                except Exception as e:
                    print(f"Error in SSE event generator: {str(e)}")
                    break
                    
        except GeneratorExit:
            print(f"Client {client_id} disconnected")
        finally:
            attendance_system.remove_realtime_client(client_id)
    
    return Response(
        event_generator(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )

if __name__ == '__main__':
    print("Starting ENHANCED Multi-Angle Face Recognition Attendance System...")
    
    if ENV == 'production':
        print(f"Production mode - Backend running on port {PORT}")
    else:
        print("Development mode - Backend running on http://localhost:5000")
        
    print("ENHANCED FEATURES:")
    print("- Multi-angle face registration (front, left, right, up, down)")
    print("- Quality-based sample validation")
    print("- Pose detection and analysis")
    print("- Enhanced recognition with multiple reference points")
    print("- Improved accuracy through diverse face angles")
    print("- Location tracking for attendance records")
    
    # Create necessary directories
    try:
        os.makedirs('known_faces', exist_ok=True)
        os.makedirs('captured_faces/registered', exist_ok=True)
        os.makedirs('captured_faces/recognized', exist_ok=True)
        print("✅ Directories created/verified")
    except Exception as e:
        print(f"⚠️ Directory creation warning: {e}")
    
    if ENV == 'production':
        # Production configuration
        app.run(host='0.0.0.0', port=PORT, debug=False)
    else:
        # Development configuration
        app.run(debug=True, host='0.0.0.0', port=5000)