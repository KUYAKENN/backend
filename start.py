#!/usr/bin/env python3
"""
Railway deployment startup script
Handles PORT environment variable properly
"""

import os
import subprocess
import sys

def main():
    # Get port from environment or use default
    port = os.environ.get('PORT', '5000')
    
    print(f"Starting application on port {port}")
    
    # Build gunicorn command
    cmd = [
        'gunicorn',
        'server_minimal:app',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '1',
        '--timeout', '120',
        '--preload',
        '--log-level', 'info'
    ]
    
    # Start the server
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
