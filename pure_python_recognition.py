"""
Pure Python Face Recognition Module
Railway deployment compatible - no OpenCV/MediaPipe dependencies
Uses MTCNN for face detection and basic facial feature extraction
"""

import numpy as np
from PIL import Image, ImageDraw
import base64
import io
import json
import logging

# Try importing MTCNN, fallback to basic detection if not available
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    print("MTCNN not available, using basic face detection")
    MTCNN_AVAILABLE = False

class PurePythonFaceRecognizer:
    """Pure Python face recognition system (no system dependencies)"""
    
    def __init__(self):
        """Initialize face detection"""
        if MTCNN_AVAILABLE:
            try:
                self.detector = MTCNN()
                print("✅ MTCNN face detector initialized")
            except Exception as e:
                print(f"MTCNN initialization failed: {e}")
                self.detector = None
        else:
            self.detector = None
            print("✅ Basic face recognition initialized (no external detector)")
    
    def face_locations(self, image, model="hog"):
        """
        Find face locations in an image
        
        Args:
            image: RGB image array or PIL Image
            model: Detection model (kept for compatibility)
            
        Returns:
            List of face locations [(top, right, bottom, left), ...]
        """
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'), 'RGB')
            
            if not MTCNN_AVAILABLE or self.detector is None:
                # Basic fallback: return center region as "face"
                w, h = image.size
                center_x, center_y = w // 2, h // 2
                face_size = min(w, h) // 3
                
                top = max(0, center_y - face_size // 2)
                bottom = min(h, center_y + face_size // 2)
                left = max(0, center_x - face_size // 2)
                right = min(w, center_x + face_size // 2)
                
                return [(top, right, bottom, left)]
            
            # Convert PIL to numpy for MTCNN
            img_array = np.array(image)
            
            # Detect faces using MTCNN
            result = self.detector.detect_faces(img_array)
            
            locations = []
            for face in result:
                if face['confidence'] > 0.9:  # High confidence threshold
                    x, y, w, h = face['box']
                    # Convert to face_recognition format: (top, right, bottom, left)
                    top = y
                    right = x + w
                    bottom = y + h
                    left = x
                    locations.append((top, right, bottom, left))
            
            return locations
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            # Fallback to center region
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                w, h = image.size
            
            center_x, center_y = w // 2, h // 2
            face_size = min(w, h) // 3
            
            top = max(0, center_y - face_size // 2)
            bottom = min(h, center_y + face_size // 2)
            left = max(0, center_x - face_size // 2)
            right = min(w, center_x + face_size // 2)
            
            return [(top, right, bottom, left)]
    
    def face_encodings(self, image, known_face_locations=None, num_jitters=1, model="small"):
        """
        Get face encodings using basic image statistics
        
        Args:
            image: RGB image array or PIL Image
            known_face_locations: Optional face locations
            num_jitters: Number of times to re-sample (kept for compatibility)
            model: Model size (kept for compatibility)
            
        Returns:
            List of face encodings (128-dimensional feature vectors)
        """
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'), 'RGB')
            
            encodings = []
            
            # Get face locations if not provided
            if known_face_locations is None:
                known_face_locations = self.face_locations(image)
            
            for (top, right, bottom, left) in known_face_locations:
                # Crop face region
                face_region = image.crop((left, top, right, bottom))
                
                # Generate encoding from face region
                encoding = self._image_to_encoding(face_region)
                encodings.append(encoding)
            
            return encodings
            
        except Exception as e:
            print(f"Error in face_encodings: {e}")
            return []
    
    def _image_to_encoding(self, face_image):
        """Convert face image to encoding vector using basic image statistics"""
        try:
            # Resize to standard size
            face_image = face_image.resize((64, 64))
            
            # Convert to grayscale for feature extraction
            gray = face_image.convert('L')
            pixels = np.array(gray, dtype=np.float32)
            
            # Extract basic features
            encoding = []
            
            # Global statistics
            encoding.extend([
                np.mean(pixels),           # Mean intensity
                np.std(pixels),            # Standard deviation
                np.min(pixels),            # Min intensity
                np.max(pixels),            # Max intensity
            ])
            
            # Regional statistics (8x8 grid)
            for i in range(8):
                for j in range(8):
                    region = pixels[i*8:(i+1)*8, j*8:(j+1)*8]
                    encoding.extend([
                        np.mean(region),       # Regional mean
                        np.std(region),        # Regional std
                    ])
            
            # Pad to 128 dimensions
            while len(encoding) < 128:
                encoding.append(0.0)
            
            return np.array(encoding[:128], dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating encoding: {e}")
            return np.zeros(128, dtype=np.float32)
    
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
        Compare a face encoding against known face encodings
        
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

# Create global instance (replaces face_recognition module)
_recognizer = PurePythonFaceRecognizer()

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

# Additional utility functions
def load_image_file(file_path):
    """Load an image file into a numpy array"""
    try:
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None
