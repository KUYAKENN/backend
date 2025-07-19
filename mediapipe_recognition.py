"""
MediaPipe Face Recognition Module
Pure Python implementation compatible with Railway deployment
No compilation dependencies (dlib/cmake free)
"""

import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from PIL import Image
import json
import logging

# Configure logging to reduce noise
logging.getLogger('mediapipe').setLevel(logging.WARNING)

class MediaPipeFaceRecognizer:
    """MediaPipe-based face recognition system"""
    
    def __init__(self):
        """Initialize MediaPipe face detection and recognition"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Initialize face mesh for landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        print("✅ MediaPipe face recognition initialized")
    
    def face_locations(self, image, model="hog"):
        """
        Find face locations in an image using MediaPipe
        
        Args:
            image: RGB image array
            model: Detection model (kept for compatibility)
            
        Returns:
            List of face locations [(top, right, bottom, left), ...]
        """
        try:
            # Convert RGB to BGR for MediaPipe
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            # Detect faces
            results = self.face_detection.process(bgr_image)
            
            locations = []
            if results.detections:
                h, w = bgr_image.shape[:2]
                
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert normalized coordinates to pixel coordinates
                    left = int(bbox.xmin * w)
                    top = int(bbox.ymin * h)
                    right = int((bbox.xmin + bbox.width) * w)
                    bottom = int((bbox.ymin + bbox.height) * h)
                    
                    # Return in face_recognition format: (top, right, bottom, left)
                    locations.append((top, right, bottom, left))
            
            return locations
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def face_encodings(self, image, known_face_locations=None, num_jitters=1, model="small"):
        """
        Get face encodings using MediaPipe face mesh landmarks
        
        Args:
            image: RGB image array
            known_face_locations: Optional face locations
            num_jitters: Number of times to re-sample (kept for compatibility)
            model: Model size (kept for compatibility)
            
        Returns:
            List of face encodings (468-dimensional landmark vectors)
        """
        try:
            # Convert RGB to BGR for MediaPipe
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            encodings = []
            
            if known_face_locations:
                # Process specific face locations
                h, w = bgr_image.shape[:2]
                
                for (top, right, bottom, left) in known_face_locations:
                    # Crop face region
                    face_region = bgr_image[top:bottom, left:right]
                    
                    if face_region.size > 0:
                        # Get landmarks for this face region
                        results = self.face_mesh.process(face_region)
                        
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0]
                            # Convert landmarks to encoding vector
                            encoding = self._landmarks_to_encoding(landmarks, face_region.shape)
                            encodings.append(encoding)
                        else:
                            # Add zero encoding as placeholder
                            encodings.append(np.zeros(204))  # 68 landmarks × 3 coordinates
            else:
                # Get all face encodings in the image
                results = self.face_mesh.process(bgr_image)
                
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        encoding = self._landmarks_to_encoding(landmarks, bgr_image.shape)
                        encodings.append(encoding)
            
            return encodings
            
        except Exception as e:
            print(f"Error in face_encodings: {e}")
            return []
    
    def _landmarks_to_encoding(self, landmarks, image_shape):
        """Convert MediaPipe landmarks to face encoding vector"""
        try:
            h, w = image_shape[:2]
            
            # Extract key landmark coordinates and create encoding
            key_landmarks = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  # Face outline
                17, 18, 19, 20, 21,  # Right eyebrow
                22, 23, 24, 25, 26,  # Left eyebrow
                27, 28, 29, 30, 31, 32, 33, 34, 35,  # Nose
                36, 37, 38, 39, 40, 41,  # Right eye
                42, 43, 44, 45, 46, 47,  # Left eye
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67  # Mouth
            ]
            
            # Extract coordinates for key landmarks
            coords = []
            landmark_list = list(landmarks.landmark)
            
            # Use a subset of landmarks for encoding (first 68 for compatibility)
            for i in range(min(68, len(landmark_list))):
                landmark = landmark_list[i]
                # Normalize coordinates relative to image size
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z if hasattr(landmark, 'z') else 0.0
                coords.extend([x/w, y/h, z])  # Normalize to 0-1 range
            
            # Pad to fixed size if necessary
            while len(coords) < 204:  # 68 landmarks × 3 coordinates
                coords.append(0.0)
            
            return np.array(coords[:204], dtype=np.float32)
            
        except Exception as e:
            print(f"Error converting landmarks to encoding: {e}")
            return np.zeros(204)
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors using NumPy"""
        try:
            # Normalize vectors
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            
            if a_norm == 0 or b_norm == 0:
                return 0.0
            
            # Calculate cosine similarity
            return np.dot(a, b) / (a_norm * b_norm)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):
        """
        Compare a face encoding against known face encodings using cosine similarity
        
        Args:
            known_face_encodings: List of known face encodings
            face_encoding_to_check: Face encoding to compare
            tolerance: Similarity tolerance (higher = more strict)
            
        Returns:
            List of boolean matches
        """
        try:
            if len(known_face_encodings) == 0:
                return []
            
            # Calculate cosine similarities
            similarities = []
            for known_encoding in known_face_encodings:
                similarity = self._cosine_similarity(known_encoding, face_encoding_to_check)
                similarities.append(similarity)
            
            # Convert similarities to boolean matches
            matches = [sim >= tolerance for sim in similarities]
            return matches
            
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return [False] * len(known_face_encodings)
    
    def face_distance(self, known_face_encodings, face_encoding_to_check):
        """
        Calculate distance between face encodings using cosine distance
        
        Args:
            known_face_encodings: List of known face encodings
            face_encoding_to_check: Face encoding to compare against
            
        Returns:
            List of distances (lower = more similar)
        """
        try:
            if len(known_face_encodings) == 0:
                return np.array([])
            
            distances = []
            for known_encoding in known_face_encodings:
                # Cosine distance = 1 - cosine similarity
                similarity = self._cosine_similarity(known_encoding, face_encoding_to_check)
                distance = 1 - similarity
                distances.append(distance)
            
            return np.array(distances)
            
        except Exception as e:
            print(f"Error calculating face distances: {e}")
            return np.array([float('inf')] * len(known_face_encodings))
    
    def face_landmarks(self, image, face_locations=None, model="large"):
        """
        Get face landmarks using MediaPipe
        
        Args:
            image: RGB image array
            face_locations: Optional face locations
            model: Model size (kept for compatibility)
            
        Returns:
            List of landmark dictionaries
        """
        try:
            # Convert RGB to BGR for MediaPipe
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            landmarks_list = []
            
            # Get face mesh results
            results = self.face_mesh.process(bgr_image)
            
            if results.multi_face_landmarks:
                h, w = bgr_image.shape[:2]
                
                for landmarks in results.multi_face_landmarks:
                    # Convert to face_recognition format
                    landmark_dict = {
                        'chin': [],
                        'left_eyebrow': [],
                        'right_eyebrow': [],
                        'nose_bridge': [],
                        'nose_tip': [],
                        'left_eye': [],
                        'right_eye': [],
                        'top_lip': [],
                        'bottom_lip': []
                    }
                    
                    # MediaPipe has 468 landmarks, map key ones to face_recognition format
                    for i, landmark in enumerate(landmarks.landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        
                        # Simplified mapping (MediaPipe uses different landmark indices)
                        if i < 17:  # Approximate chin area
                            landmark_dict['chin'].append((x, y))
                        elif i < 27:  # Approximate eyebrow area
                            landmark_dict['left_eyebrow'].append((x, y))
                        # Add more mappings as needed...
                    
                    landmarks_list.append(landmark_dict)
            
            return landmarks_list
            
        except Exception as e:
            print(f"Error getting face landmarks: {e}")
            return []

# Create global instance (replaces face_recognition module)
_recognizer = MediaPipeFaceRecognizer()

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
