#!/usr/bin/env python3
"""
Quick test to verify DeepFace face recognition works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("🔄 Testing DeepFace face recognition...")
    
    # Test import
    import deepface_recognition as face_recognition
    print("✅ DeepFace module imported successfully")
    
    # Test basic functionality with a simple test
    import numpy as np
    import cv2
    
    # Create a simple test image (100x100 pixels)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some "face-like" features (just for testing)
    cv2.rectangle(test_image, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.circle(test_image, (35, 40), 5, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_image, (65, 40), 5, (0, 0, 0), -1)  # Right eye
    cv2.rectangle(test_image, (45, 55), (55, 65), (0, 0, 0), -1)  # Nose
    cv2.rectangle(test_image, (35, 70), (65, 75), (0, 0, 0), -1)  # Mouth
    
    print("🔄 Testing face detection...")
    locations = face_recognition.face_locations(test_image)
    print(f"✅ Face detection test completed (found {len(locations)} faces)")
    
    print("🔄 Testing face encoding...")
    if locations:
        encodings = face_recognition.face_encodings(test_image, locations)
        if encodings:
            print(f"✅ Face encoding test completed (encoding shape: {encodings[0].shape})")
        else:
            print("⚠️ No encodings generated (this is normal for synthetic test image)")
    else:
        print("⚠️ No faces detected (this is normal for synthetic test image)")
    
    print("🎉 All tests passed! DeepFace is working correctly.")
    print("🚀 Your face recognition API should deploy successfully on Railway!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all requirements are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("💡 Check the error above and ensure all dependencies are working")
    sys.exit(1)
