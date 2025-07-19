"""
Minimal Server for Railway Deployment
Pure Python face recognition attendance system (OpenCV-free)
"""

import os
from app_minimal import app

# This is the app that gunicorn will serve
# The variable name must match what's in the Procfile/start command

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
