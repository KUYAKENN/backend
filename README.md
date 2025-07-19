# Face Recognition Backend API

🚀 **Production-ready backend for Render.com deployment**

## Quick Deploy to Render.com

1. **Push this repo to GitHub**
2. **Connect to Render.com**
3. **Set Environment Variables:**
   ```
   DB_HOST=your-mysql-host
   DB_PORT=3306
   DB_USER=your-mysql-user
   DB_PASSWORD=your-mysql-password
   DB_NAME=face_recognition_db
   FLASK_ENV=production
   CORS_ORIGINS=https://your-frontend-domain.com
   ```

## API Endpoints

- `GET /health` - Health check
- `GET /api/health` - API status
- `POST /api/register` - Register new person
- `GET /api/people` - Get all people
- `POST /api/attendance` - Mark attendance
- `GET /api/attendance` - Get attendance records
- `GET /api/realtime/events` - Real-time detection stream

## Database Setup

The backend uses MySQL database. Make sure to:
1. Create a MySQL database
2. Set the database connection environment variables
3. The app will automatically create tables on first run

## Files in this deployment:
- ✅ 12 essential files
- ✅ Clean directory structure
- ✅ Production configuration
- ✅ Environment template

Ready for deployment! 🎉