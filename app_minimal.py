"""
Minimal Flask App for Railway Deployment
Pure Python face recognition attendance system (OpenCV-free)
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import mediapipe_recognition as face_recognition
import numpy as np
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from PIL import Image
import threading

# Import database modules
from database import db, FaceEncoding, Attendance, RecognitionLog

# Create Flask app
app = Flask(__name__)
CORS(app)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('mysql://'):
    DATABASE_URL = DATABASE_URL.replace('mysql://', 'mysql+pymysql://')

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL or 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'status': 'success',
        'message': 'Face Recognition Attendance System API',
        'version': '2.0',
        'features': ['face_registration', 'attendance_tracking', 'pure_python'],
        'deployment': 'Railway Ready'
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'face_recognition': 'pure_python',
        'database': 'connected'
    })

@app.route('/api/register-face', methods=['POST'])
def register_face():
    """Register a new face (simplified version)"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'image' not in data:
            return jsonify({'error': 'Missing name or image data'}), 400
        
        name = data['name']
        image_data = data['image']
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get face encodings
        face_locations = face_recognition.face_locations(img_array)
        if not face_locations:
            return jsonify({'error': 'No face detected in image'}), 400
        
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        if not face_encodings:
            return jsonify({'error': 'Could not generate face encoding'}), 400
        
        # Save to database
        encoding_data = face_encodings[0].tolist()
        
        # Store in database
        with app.app_context():
            face_encoding = FaceEncoding(
                name=name,
                encoding=json.dumps(encoding_data),
                image_data=image_bytes
            )
            db.session.add(face_encoding)
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Face registered successfully for {name}',
            'encoding_length': len(encoding_data)
        })
        
    except Exception as e:
        print(f"Error in register_face: {e}")
        return jsonify({'error': 'Face registration failed'}), 500

@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():
    """Recognize a face and mark attendance"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        image_data = data['image']
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get face encodings
        face_locations = face_recognition.face_locations(img_array)
        if not face_locations:
            return jsonify({'error': 'No face detected in image'}), 400
        
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        if not face_encodings:
            return jsonify({'error': 'Could not generate face encoding'}), 400
        
        unknown_encoding = face_encodings[0]
        
        # Load known faces from database
        with app.app_context():
            known_faces = FaceEncoding.query.all()
            
            if not known_faces:
                return jsonify({'error': 'No registered faces found'}), 404
            
            # Compare with known faces
            known_encodings = []
            known_names = []
            
            for face in known_faces:
                encoding = json.loads(face.encoding)
                known_encodings.append(np.array(encoding))
                known_names.append(face.name)
            
            # Find matches
            matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.6)
            
            if any(matches):
                # Find best match
                face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    
                    # Record attendance
                    attendance = Attendance(
                        name=name,
                        timestamp=datetime.now(),
                        confidence=float(confidence),
                        image_data=image_bytes
                    )
                    db.session.add(attendance)
                    db.session.commit()
                    
                    return jsonify({
                        'success': True,
                        'name': name,
                        'confidence': f"{confidence:.2%}",
                        'timestamp': datetime.now().isoformat(),
                        'message': f'Welcome, {name}!'
                    })
            
            return jsonify({
                'success': False,
                'message': 'Face not recognized',
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        print(f"Error in recognize_face: {e}")
        return jsonify({'error': 'Face recognition failed'}), 500

@app.route('/api/attendance-records')
def get_attendance_records():
    """Get attendance records"""
    try:
        with app.app_context():
            records = Attendance.query.order_by(Attendance.timestamp.desc()).limit(100).all()
            
            attendance_list = []
            for record in records:
                attendance_list.append({
                    'id': record.id,
                    'name': record.name,
                    'timestamp': record.timestamp.isoformat(),
                    'confidence': f"{record.confidence:.2%}" if record.confidence else 'N/A'
                })
            
            return jsonify({
                'success': True,
                'records': attendance_list,
                'count': len(attendance_list)
            })
            
    except Exception as e:
        print(f"Error getting attendance records: {e}")
        return jsonify({'error': 'Failed to fetch attendance records'}), 500

@app.route('/api/registered-faces')
def get_registered_faces():
    """Get list of registered faces"""
    try:
        with app.app_context():
            faces = FaceEncoding.query.all()
            
            face_list = []
            for face in faces:
                face_list.append({
                    'id': face.id,
                    'name': face.name,
                    'registered_date': face.created_date.isoformat() if face.created_date else 'N/A'
                })
            
            return jsonify({
                'success': True,
                'faces': face_list,
                'count': len(face_list)
            })
            
    except Exception as e:
        print(f"Error getting registered faces: {e}")
        return jsonify({'error': 'Failed to fetch registered faces'}), 500

# Create tables
with app.app_context():
    db.create_all()
    print("✅ Database tables created")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
