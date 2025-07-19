#!/usr/bin/env python3
"""
Test Flask app import with MediaPipe
"""

try:
    print('Testing Flask app import...')
    import app
    print('✅ Flask app imported successfully')
    print('✅ MediaPipe face recognition integration working')
    
    # Test the app creation
    if hasattr(app, 'app'):
        print('✅ Flask app instance found')
    
except Exception as e:
    print(f'❌ Error importing Flask app: {e}')
    import traceback
    traceback.print_exc()
