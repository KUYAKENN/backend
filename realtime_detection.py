# Real-time Face Detection with Server-Sent Events
import json
import time
import threading
from queue import Queue
from flask import Flask, Response, request
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
from datetime import datetime

class RealtimeDetectionSystem:
    def __init__(self):
        self.clients = []  # Connected SSE clients
        self.detection_queue = Queue()
        self.is_running = False
        self.camera = None
        self.known_faces = {}
        self.attendance_records = set()  # Track today's attendance
        
    def add_client(self, client_queue):
        """Add a new SSE client"""
        self.clients.append(client_queue)
        print(f"✅ New client connected. Total clients: {len(self.clients)}")
        
    def remove_client(self, client_queue):
        """Remove disconnected SSE client"""
        if client_queue in self.clients:
            self.clients.remove(client_queue)
            print(f"🔌 Client disconnected. Total clients: {len(self.clients)}")
    
    def broadcast_event(self, event_type, data):
        """Send event to all connected clients"""
        event_data = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Remove disconnected clients
        disconnected_clients = []
        for client_queue in self.clients:
            try:
                client_queue.put(event_data)
            except:
                disconnected_clients.append(client_queue)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.remove_client(client)
    
    def start_detection(self):
        """Start real-time face detection"""
        if self.is_running:
            return
            
        self.is_running = True
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        print("🎥 Real-time detection started")
    
    def stop_detection(self):
        """Stop real-time face detection"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("⏹️ Real-time detection stopped")
    
    def _detection_loop(self):
        """Main detection loop"""
        try:
            self.camera = cv2.VideoCapture(0)
            frame_skip = 0
            
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Process every 3rd frame for performance
                frame_skip += 1
                if frame_skip % 3 != 0:
                    continue
                
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_frame)
                if not face_locations:
                    continue
                
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        list(self.known_faces.values()), 
                        face_encoding, 
                        tolerance=0.6
                    )
                    
                    face_distances = face_recognition.face_distance(
                        list(self.known_faces.values()), 
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = list(self.known_faces.keys())[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            
                            # Broadcast face detection
                            self.broadcast_event('face_detected', {
                                'name': name,
                                'confidence': float(confidence),
                                'location': face_locations[0]  # [top, right, bottom, left]
                            })
                            
                            # Mark attendance if not already marked today
                            if name not in self.attendance_records and confidence > 0.7:
                                self.mark_attendance(name, confidence)
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Detection error: {e}")
        finally:
            if self.camera:
                self.camera.release()
    
    def mark_attendance(self, name, confidence):
        """Mark attendance for a person"""
        try:
            # Add to today's records
            self.attendance_records.add(name)
            
            # Get current time and location (if available)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Here you would typically save to database
            # For now, we'll just broadcast the event
            
            self.broadcast_event('attendance_marked', {
                'name': name,
                'status': 'present',
                'time': current_time,
                'confidence': float(confidence),
                'message': f'{name} marked as present'
            })
            
            print(f"✅ Attendance marked for {name} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Error marking attendance: {e}")
    
    def load_known_faces(self):
        """Load known faces from your existing system"""
        # This should integrate with your existing face loading logic
        # For now, placeholder
        try:
            # Load from your encodings.json or database
            # self.known_faces = your_existing_load_function()
            print("📸 Known faces loaded")
        except Exception as e:
            print(f"❌ Error loading faces: {e}")

# Global detection system instance
detection_system = RealtimeDetectionSystem()

# Flask SSE endpoint
def create_sse_app():
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/realtime/events')
    def stream_events():
        """Server-Sent Events endpoint"""
        def event_stream():
            client_queue = Queue()
            detection_system.add_client(client_queue)
            
            try:
                # Send initial connection event
                yield f"event: connected\ndata: {{\"message\": \"Connected to real-time detection\"}}\n\n"
                
                while True:
                    try:
                        # Wait for new event (timeout every 30 seconds for heartbeat)
                        event_data = client_queue.get(timeout=30)
                        yield f"event: {event_data['type']}\ndata: {json.dumps(event_data['data'])}\n\n"
                    except:
                        # Send heartbeat
                        yield f"event: heartbeat\ndata: {{\"timestamp\": \"{datetime.now().isoformat()}\"}}\n\n"
                        
            except GeneratorExit:
                detection_system.remove_client(client_queue)
            except Exception as e:
                print(f"❌ SSE stream error: {e}")
                detection_system.remove_client(client_queue)
        
        return Response(
            event_stream(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
    
    @app.route('/api/realtime/start', methods=['POST'])
    def start_detection():
        """Start real-time detection"""
        detection_system.load_known_faces()
        detection_system.start_detection()
        return {'success': True, 'message': 'Real-time detection started'}
    
    @app.route('/api/realtime/stop', methods=['POST'])
    def stop_detection():
        """Stop real-time detection"""
        detection_system.stop_detection()
        return {'success': True, 'message': 'Real-time detection stopped'}
    
    @app.route('/api/realtime/status')
    def get_status():
        """Get real-time detection status"""
        return {
            'running': detection_system.is_running,
            'clients': len(detection_system.clients),
            'attendance_count': len(detection_system.attendance_records)
        }
    
    return app

if __name__ == '__main__':
    app = create_sse_app()
    print("🚀 Starting real-time detection server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
