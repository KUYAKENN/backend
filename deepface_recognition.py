"""
DeepFace Face Recognition Module
Replacement for face_recognition library using DeepFace + TensorFlow
Compatible with Railway deployment (no compilation needed)
"""

import cv2
import numpy as np
from deepface import DeepFace
import base64
import io
from PIL import Image
import json
import logging

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class DeepFaceRecognizer:
    """DeepFace-based face recognition system"""
    
    def __init__(self, model_name='VGG-Face', distance_metric='cosine'):
        """
        Initialize DeepFace recognizer
        
        Args:
            model_name: Face recognition model ('VGG-Face', 'Facenet', 'OpenFace', 'DeepFace')
            distance_metric: Distance metric ('cosine', 'euclidean', 'euclidean_l2')
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = self._get_threshold()
        
        # Pre-load the model
        try:
            DeepFace.build_model(model_name)
            print(f"✅ DeepFace model '{model_name}' loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning loading model: {e}")
    
    def _get_threshold(self):
        """Get optimal threshold for the selected model"""
        thresholds = {
            'VGG-Face': 0.68,
            'Facenet': 10,
            'OpenFace': 0.10,
            'DeepFace': 0.64,
            'ArcFace': 6.12
        }
        return thresholds.get(self.model_name, 0.68)
    
    def face_locations(self, image, model="hog"):
        """
        Find face locations in an image
        
        Args:
            image: RGB image array
            model: Detection model (kept for compatibility)
            
        Returns:
            List of face locations [(top, right, bottom, left), ...]
        """
        try:
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            # Use DeepFace to detect faces
            faces = DeepFace.extract_faces(
                bgr_image, 
                detector_backend='opencv',
                enforce_detection=False
            )
            
            if not faces:
                return []
            
            # Get face locations using OpenCV cascade (faster for detection only)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Convert to face_recognition format: (top, right, bottom, left)
            locations = []
            for (x, y, w, h) in detected_faces:
                locations.append((y, x + w, y + h, x))
            
            return locations
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def face_encodings(self, image, known_face_locations=None, num_jitters=1, model="small"):
        """
        Get face encodings for faces in an image
        
        Args:
            image: RGB image array
            known_face_locations: Optional face locations
            num_jitters: Number of times to re-sample (kept for compatibility)
            model: Model size (kept for compatibility)
            
        Returns:
            List of face encodings
        """
        try:
            # Convert RGB to BGR for OpenCV/DeepFace
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            encodings = []
            
            if known_face_locations:
                # Extract encodings for specific face locations
                for (top, right, bottom, left) in known_face_locations:
                    # Crop face region
                    face_region = bgr_image[top:bottom, left:right]
                    
                    if face_region.size > 0:
                        try:
                            # Get embedding using DeepFace
                            embedding = DeepFace.represent(
                                face_region,
                                model_name=self.model_name,
                                enforce_detection=False
                            )
                            
                            if embedding and len(embedding) > 0:
                                encodings.append(np.array(embedding[0]['embedding']))
                                
                        except Exception as e:
                            print(f"Error getting encoding for face region: {e}")
                            # Add zero embedding as placeholder
                            encodings.append(np.zeros(512))  # VGG-Face embedding size
            else:
                # Get all face encodings in the image
                try:
                    embeddings = DeepFace.represent(
                        bgr_image,
                        model_name=self.model_name,
                        enforce_detection=False
                    )
                    
                    for embedding_data in embeddings:
                        encodings.append(np.array(embedding_data['embedding']))
                        
                except Exception as e:
                    print(f"Error getting face encodings: {e}")
            
            return encodings
            
        except Exception as e:
            print(f"Error in face_encodings: {e}")
            return []
    
    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=None):
        """
        Compare a face encoding against known face encodings
        
        Args:
            known_face_encodings: List of known face encodings
            face_encoding_to_check: Face encoding to compare
            tolerance: Optional distance tolerance
            
        Returns:
            List of boolean matches
        """
        if tolerance is None:
            tolerance = self.threshold
        
        try:
            distances = self.face_distance(known_face_encodings, face_encoding_to_check)
            
            if self.distance_metric == 'cosine':
                return [distance <= tolerance for distance in distances]
            else:
                return [distance <= tolerance for distance in distances]
                
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return [False] * len(known_face_encodings)
    
    def face_distance(self, known_face_encodings, face_encoding_to_check):
        """
        Calculate distance between face encodings
        
        Args:
            known_face_encodings: List of known face encodings
            face_encoding_to_check: Face encoding to compare against
            
        Returns:
            List of distances
        """
        try:
            if len(known_face_encodings) == 0:
                return np.array([])
            
            distances = []
            
            for known_encoding in known_face_encodings:
                if self.distance_metric == 'cosine':
                    # Cosine distance
                    distance = 1 - np.dot(known_encoding, face_encoding_to_check) / (
                        np.linalg.norm(known_encoding) * np.linalg.norm(face_encoding_to_check)
                    )
                elif self.distance_metric == 'euclidean':
                    # Euclidean distance
                    distance = np.linalg.norm(known_encoding - face_encoding_to_check)
                else:
                    # Default to cosine
                    distance = 1 - np.dot(known_encoding, face_encoding_to_check) / (
                        np.linalg.norm(known_encoding) * np.linalg.norm(face_encoding_to_check)
                    )
                
                distances.append(distance)
            
            return np.array(distances)
            
        except Exception as e:
            print(f"Error calculating face distances: {e}")
            return np.array([float('inf')] * len(known_face_encodings))
    
    def face_landmarks(self, image, face_locations=None, model="large"):
        """
        Get face landmarks (kept for compatibility, basic implementation)
        
        Args:
            image: RGB image array
            face_locations: Optional face locations
            model: Model size (kept for compatibility)
            
        Returns:
            List of landmark dictionaries
        """
        # Basic implementation - DeepFace doesn't have direct landmark detection
        # You could integrate with MediaPipe or dlib for landmarks if needed
        landmarks = []
        
        if face_locations:
            for location in face_locations:
                # Return empty landmark dict for compatibility
                landmarks.append({
                    'chin': [],
                    'left_eyebrow': [],
                    'right_eyebrow': [],
                    'nose_bridge': [],
                    'nose_tip': [],
                    'left_eye': [],
                    'right_eye': [],
                    'top_lip': [],
                    'bottom_lip': []
                })
        
        return landmarks

# Create global instance (replaces face_recognition module)
_recognizer = DeepFaceRecognizer()

# Module-level functions that mimic face_recognition API
def face_locations(image, number_of_times_to_upsample=1, model="hog"):
    """Find face locations in an image"""
    return _recognizer.face_locations(image, model)

def face_encodings(image, known_face_locations=None, num_jitters=1, model="small"):
    """Get face encodings for faces in an image"""
    return _recognizer.face_encodings(image, known_face_locations, num_jitters, model)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """Compare a face encoding against known face encodings"""
    return _recognizer.compare_faces(known_face_encodings, face_encoding_to_check, tolerance)

def face_distance(known_face_encodings, face_encoding_to_check):
    """Calculate distance between face encodings"""
    return _recognizer.face_distance(known_face_encodings, face_encoding_to_check)

def face_landmarks(image, face_locations=None, model="large"):
    """Get face landmarks"""
    return _recognizer.face_landmarks(image, face_locations, model)

# Additional utility functions
def load_image_file(file_path):
    """Load an image file into a numpy array"""
    try:
        image = cv2.imread(file_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128):
    """Process multiple images (kept for compatibility)"""
    results = []
    for image in images:
        results.append(face_locations(image, number_of_times_to_upsample))
    return results
