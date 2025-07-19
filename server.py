"""
Backend API Server for Face Recognition Attendance System
Optimized for Render.com deployment with CORS support for separate frontend
"""
import os
import sys
from datetime import datetime
import threading
import time

# Import your main Flask app
from app import app as main_app, attendance_system

# Import your real-time detection system
from realtime_detection import detection_system, create_sse_app

# Environment configuration
ENV = os.getenv('FLASK_ENV', 'development')
PORT = int(os.getenv('PORT', 5000))
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')  # Set your frontend URL here

def setup_cors():
    """Configure CORS for separate frontend deployment"""
    from flask_cors import CORS
    
    if ENV == 'production':
        # In production, restrict to your frontend domain
        frontend_url = os.getenv('FRONTEND_URL', '*')
        CORS(main_app, origins=[frontend_url] if frontend_url != '*' else '*')
    else:
        # In development, allow all origins
        CORS(main_app, origins='*')
    
    print(f"✅ CORS configured for: {CORS_ORIGINS}")

def setup_health_checks():
    """Add health check endpoints for Render.com"""
    
    @main_app.route('/health')
    @main_app.route('/')
    def health_check():
        """Health check endpoint for Render.com"""
        return {
            'status': 'healthy',
            'service': 'Face Recognition API',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'database': 'connected' if hasattr(attendance_system, 'db_manager') else 'fallback',
            'realtime': 'available'
        }
    
    @main_app.route('/api/health')
    def api_health():
        """API-specific health check"""
        return {
            'api_status': 'healthy',
            'endpoints': {
                'attendance': '/api/attendance',
                'registration': '/api/register',
                'realtime': '/api/realtime/events',
                'people': '/api/people'
            },
            'database_status': 'connected' if hasattr(attendance_system, 'db_manager') else 'fallback_mode'
        }

def setup_unified_routes():
    """Add real-time detection routes to main app"""
    
    @main_app.route('/api/realtime/events')
    def stream_events():
        """Server-Sent Events endpoint for real-time detection"""
        from flask import Response
        import json
        from queue import Queue
        
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
    
    @main_app.route('/api/realtime/start', methods=['POST'])
    def start_realtime_detection():
        """Start real-time detection"""
        try:
            detection_system.load_known_faces()
            detection_system.start_detection()
            return {'success': True, 'message': 'Real-time detection started'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    @main_app.route('/api/realtime/stop', methods=['POST'])
    def stop_realtime_detection():
        """Stop real-time detection"""
        try:
            detection_system.stop_detection()
            return {'success': True, 'message': 'Real-time detection stopped'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    @main_app.route('/api/realtime/status')
    def get_realtime_status():
        """Get real-time detection status"""
        return {
            'running': detection_system.is_running,
            'clients': len(detection_system.clients),
            'attendance_count': len(detection_system.attendance_records)
        }

# Set up CORS and health checks
setup_cors()
setup_health_checks()

# Set up unified routes
setup_unified_routes()

# Export the unified app
app = main_app

if __name__ == '__main__':
    print("🚀 Starting Face Recognition Backend API...")
    print(f"🌍 Environment: {ENV}")
    print(f"🔗 CORS Origins: {CORS_ORIGINS}")
    
    if ENV == 'production':
        print(f"🌐 Production mode - API running on port {PORT}")
        print("📡 Frontend should connect to this API URL")
    else:
        print("🔧 Development mode - API running on http://localhost:5000")
        print("🔗 Frontend can connect to: http://localhost:5000")
        
    print("\n📡 API Endpoints:")
    print("- Health Check: /health")
    print("- API Status: /api/health") 
    print("- Attendance: /api/attendance")
    print("- Registration: /api/register")
    print("- People Management: /api/people")
    print("- Real-time Events: /api/realtime/events")
    print("- Real-time Control: /api/realtime/start, /api/realtime/stop")
    
    # Create necessary directories
    try:
        os.makedirs('known_faces', exist_ok=True)
        os.makedirs('captured_faces/registered', exist_ok=True)
        os.makedirs('captured_faces/recognized', exist_ok=True)
        print("✅ Directories created/verified")
    except Exception as e:
        print(f"⚠️ Directory creation warning: {e}")
    
    if ENV == 'production':
        # Production configuration for Render
        print("🔄 Starting backend API in production mode...")
        app.run(host='0.0.0.0', port=PORT, debug=False)
    else:
        # Development configuration
        print("🔄 Starting backend API in development mode...")
        app.run(debug=True, host='0.0.0.0', port=5000)
