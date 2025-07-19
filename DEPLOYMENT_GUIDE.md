# Face Recognition Backend API - Render.com Deployment Guide

## 📋 Backend Repository Structure

Your backend should contain these essential files:

```
backend/
├── app.py                 # Main Flask application
├── server.py              # Production server entry point
├── database.py            # Database management
├── realtime_detection.py  # Real-time face detection
├── requirements.txt       # Python dependencies
├── Procfile              # Render.com process file
├── render.yaml           # Render.com configuration
├── .env                  # Environment variables (local only)
├── face_recognition_schema.sql  # Database schema
└── known_faces/          # Face encodings directory
```

## 🚀 Render.com Deployment Steps

### Step 1: Set Up MySQL Database

#### Option A: Use Your Existing MySQL Database (Recommended)
If you already have a MySQL database running locally or on your server:
1. Create a new database: `face_recognition_db`
2. Run your `face_recognition_schema.sql` to create the required tables
3. Make sure your MySQL server is accessible from the internet (configure firewall/port forwarding if needed)
4. Note down your connection details:
   - Host (your server IP or domain)
   - Port (usually 3306)
   - Username and password
   - Database name

#### Option B: Use Cloud MySQL Providers (If you need a cloud database)

**PlanetScale**
1. Go to [planetscale.com](https://planetscale.com)
2. Create free account and database
3. Run your `face_recognition_schema.sql`

**Railway**
1. Go to [railway.app](https://railway.app)
2. Create new project and add MySQL service

**Aiven**
1. Go to [aiven.io](https://aiven.io)
2. Create MySQL service

### Step 2: Deploy to Render.com

1. **Push Backend to New Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial backend commit"
   git remote add origin https://github.com/yourusername/face-recognition-backend.git
   git push -u origin main
   ```

2. **Connect to Render.com**
   - Go to [render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your backend repository
   - Use these settings:

3. **Environment Variables** (Set in Render Dashboard)
   ```
   FLASK_ENV=production
   DB_HOST=your-mysql-host
   DB_PORT=3306
   DB_USER=your-mysql-user
   DB_PASSWORD=your-mysql-password
   DB_NAME=face_recognition_db
   SECRET_KEY=auto-generated
   CORS_ORIGINS=https://your-frontend-url.com
   ```

4. **Build Settings**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn server:app --bind 0.0.0.0:$PORT`

### Step 3: Test Your API

Once deployed, your API will be available at:
`https://your-backend-name.onrender.com`

Test endpoints:
- Health: `GET /health`
- API Status: `GET /api/health`
- People: `GET /api/people`

## 🔗 Frontend Integration

Update your Angular frontend to use the Render.com API:

```typescript
// environment.prod.ts
export const environment = {
  production: true,
  apiUrl: 'https://your-backend-name.onrender.com/api'
};

// services/attendance.service.ts
constructor(private http: HttpClient) {
  this.apiUrl = environment.apiUrl;
}
```

## 🛠️ Backend Features

### API Endpoints

#### Attendance Management
- `GET /api/attendance` - Get attendance records
- `POST /api/attendance` - Mark attendance
- `GET /api/attendance/today` - Today's attendance
- `GET /api/attendance/summary` - Attendance summary

#### People Management
- `GET /api/people` - Get all registered people
- `POST /api/register` - Register new person
- `DELETE /api/people/{id}` - Remove person

#### Real-time Detection
- `GET /api/realtime/events` - SSE stream for real-time updates
- `POST /api/realtime/start` - Start detection
- `POST /api/realtime/stop` - Stop detection

#### Health & Status
- `GET /health` - Service health check
- `GET /api/health` - API status with details

### Database Features
- MySQL integration with fallback to CSV files
- Automatic table creation and migration
- Face encoding storage and management
- Attendance tracking with timestamps
- Real-time detection logging

### Security Features
- CORS configuration for frontend integration
- Environment-based configuration
- Secure database connections
- Input validation and sanitization

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your local MySQL credentials

# Run development server
python server.py
```

## 📊 Monitoring

### Render.com Dashboard
- View logs and metrics
- Monitor uptime and performance
- Configure auto-deploy from GitHub

### Health Checks
- `/health` - Basic service status
- `/api/health` - Detailed API status
- Database connection monitoring

## 🎯 Next Steps

1. ✅ Deploy backend to Render.com
2. ✅ Set up MySQL database
3. 🔄 Update frontend to use new API URL
4. 🔄 Deploy frontend separately (Vercel/Netlify)
5. 🔄 Update CORS_ORIGINS with frontend URL
6. 🔄 Test end-to-end functionality

Your backend is now ready for production deployment! 🚀
