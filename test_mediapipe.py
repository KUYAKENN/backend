#!/usr/bin/env python3
"""
Test script for MediaPipe face recognition module
"""

try:
    print('Starting import test...')
    import mediapipe_recognition as mp_rec
    print('✅ MediaPipe module loaded')
    
    print('Testing face_locations function...')
    import numpy as np
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    locations = mp_rec.face_locations(test_img)
    print(f'✅ face_locations returned: {locations}')
    
    print('Testing face_encodings function...')
    encodings = mp_rec.face_encodings(test_img)
    print(f'✅ face_encodings returned: {len(encodings)} encodings')
    
    print('✅ All tests passed!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
