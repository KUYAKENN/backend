-- Face Recognition System Database Schema
-- Created: July 19, 2025
-- Database: face_recognition_db

-- Create database (run this first)
CREATE DATABASE IF NOT EXISTS face_recognition_db;
USE face_recognition_db;

-- ============================================================================
-- PERSONS TABLE - Store registered people information
-- ============================================================================
CREATE TABLE IF NOT EXISTS persons (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(50),
    department VARCHAR(100),
    position VARCHAR(100),
    status ENUM('active', 'inactive') DEFAULT 'active',
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    notes TEXT,
    
    -- Indexes for better performance
    INDEX idx_name (name),
    INDEX idx_email (email),
    INDEX idx_status (status),
    INDEX idx_department (department)
);

-- ============================================================================
-- FACE_ENCODINGS TABLE - Store face recognition encodings
-- ============================================================================
CREATE TABLE IF NOT EXISTS face_encodings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id INT NOT NULL,
    encoding_data TEXT NOT NULL,  -- JSON string of face encoding array
    encoding_type ENUM('standard', 'enhanced', 'multi_angle') DEFAULT 'standard',
    face_angle ENUM('front', 'left', 'right', 'up', 'down') DEFAULT 'front',
    confidence_score DECIMAL(3,2) DEFAULT 0.80,
    image_path VARCHAR(500),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_primary BOOLEAN DEFAULT FALSE,
    
    -- Foreign key constraint
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_person_id (person_id),
    INDEX idx_encoding_type (encoding_type),
    INDEX idx_face_angle (face_angle),
    INDEX idx_is_primary (is_primary)
);

-- ============================================================================
-- ATTENDANCE TABLE - Store attendance records
-- ============================================================================
CREATE TABLE IF NOT EXISTS attendance (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id INT NOT NULL,
    person_name VARCHAR(255) NOT NULL,  -- Denormalized for faster queries
    check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_out_time TIMESTAMP NULL,
    date_recorded DATE NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    detection_method ENUM('auto', 'manual', 'override') DEFAULT 'auto',
    image_path VARCHAR(500),
    location_id INT,
    status ENUM('present', 'late', 'early_leave', 'absent') DEFAULT 'present',
    notes TEXT,
    created_by VARCHAR(100) DEFAULT 'system',
    
    -- Foreign key constraint
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    
    -- Composite unique constraint to prevent duplicate entries per day
    UNIQUE KEY unique_person_date (person_id, date_recorded),
    
    -- Indexes for better performance
    INDEX idx_person_id (person_id),
    INDEX idx_date_recorded (date_recorded),
    INDEX idx_person_name (person_name),
    INDEX idx_status (status),
    INDEX idx_check_in_time (check_in_time),
    INDEX idx_confidence_score (confidence_score)
);

-- ============================================================================
-- LOCATIONS TABLE - Store detection locations/cameras
-- ============================================================================
CREATE TABLE IF NOT EXISTS locations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    camera_id VARCHAR(100),
    coordinates JSON,  -- Store lat/lng or x/y coordinates
    status ENUM('active', 'inactive', 'maintenance') DEFAULT 'active',
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_name (name),
    INDEX idx_status (status)
);

-- ============================================================================
-- RECOGNITION_LOGS TABLE - Detailed logs of all recognition attempts
-- ============================================================================
CREATE TABLE IF NOT EXISTS recognition_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id INT,
    person_name VARCHAR(255),
    recognition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2) NOT NULL,
    detection_status ENUM('recognized', 'unknown', 'low_confidence', 'failed') NOT NULL,
    image_path VARCHAR(500),
    location_id INT,
    processing_time_ms INT,  -- How long recognition took
    face_coordinates JSON,   -- Bounding box of detected face
    notes TEXT,
    
    -- Foreign key constraints (nullable for unknown persons)
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL,
    FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE SET NULL,
    
    -- Indexes
    INDEX idx_person_id (person_id),
    INDEX idx_recognition_time (recognition_time),
    INDEX idx_detection_status (detection_status),
    INDEX idx_confidence_score (confidence_score)
);

-- ============================================================================
-- SYSTEM_SETTINGS TABLE - Store application configuration
-- ============================================================================
CREATE TABLE IF NOT EXISTS system_settings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value TEXT NOT NULL,
    data_type ENUM('string', 'number', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    category VARCHAR(50) DEFAULT 'general',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'system',
    
    -- Index
    INDEX idx_setting_key (setting_key),
    INDEX idx_category (category)
);

-- ============================================================================
-- INSERT DEFAULT DATA
-- ============================================================================

-- Insert default location
INSERT INTO locations (name, description, status) VALUES 
('Main Entrance', 'Primary entrance camera', 'active'),
('Office Floor 1', 'Office area monitoring', 'active')
ON DUPLICATE KEY UPDATE description = VALUES(description);

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, data_type, description, category) VALUES 
('confidence_threshold', '0.6', 'number', 'Minimum confidence score for face recognition', 'recognition'),
('max_faces_per_person', '5', 'number', 'Maximum face encodings per person', 'recognition'),
('attendance_grace_period', '15', 'number', 'Grace period in minutes for late arrival', 'attendance'),
('auto_checkout_time', '18:00', 'string', 'Automatic checkout time if not manually checked out', 'attendance'),
('image_retention_days', '30', 'number', 'Number of days to keep captured images', 'storage'),
('enable_realtime_detection', 'true', 'boolean', 'Enable real-time face detection', 'features'),
('working_hours_start', '09:00', 'string', 'Standard working hours start time', 'attendance'),
('working_hours_end', '17:00', 'string', 'Standard working hours end time', 'attendance')
ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value);

-- ============================================================================
-- CREATE VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Daily attendance summary view
CREATE OR REPLACE VIEW daily_attendance_summary AS
SELECT 
    date_recorded,
    COUNT(*) as total_present,
    COUNT(CASE WHEN status = 'late' THEN 1 END) as late_arrivals,
    COUNT(CASE WHEN status = 'early_leave' THEN 1 END) as early_leaves,
    AVG(confidence_score) as avg_confidence
FROM attendance 
GROUP BY date_recorded
ORDER BY date_recorded DESC;

-- Person attendance history view
CREATE OR REPLACE VIEW person_attendance_history AS
SELECT 
    p.id,
    p.name,
    p.department,
    a.date_recorded,
    a.check_in_time,
    a.check_out_time,
    a.status,
    a.confidence_score,
    CASE 
        WHEN a.check_out_time IS NOT NULL 
        THEN TIMESTAMPDIFF(HOUR, a.check_in_time, a.check_out_time)
        ELSE NULL 
    END as hours_worked
FROM persons p
LEFT JOIN attendance a ON p.id = a.person_id
ORDER BY p.name, a.date_recorded DESC;

-- Recent recognition activity view
CREATE OR REPLACE VIEW recent_recognition_activity AS
SELECT 
    rl.recognition_time,
    COALESCE(p.name, 'Unknown Person') as person_name,
    rl.confidence_score,
    rl.detection_status,
    l.name as location_name,
    rl.processing_time_ms
FROM recognition_logs rl
LEFT JOIN persons p ON rl.person_id = p.id
LEFT JOIN locations l ON rl.location_id = l.id
ORDER BY rl.recognition_time DESC
LIMIT 100;

-- ============================================================================
-- STORED PROCEDURES
-- ============================================================================

DELIMITER //

-- Procedure to mark attendance
CREATE PROCEDURE MarkAttendance(
    IN p_person_id INT,
    IN p_person_name VARCHAR(255),
    IN p_confidence_score DECIMAL(3,2),
    IN p_image_path VARCHAR(500),
    IN p_location_id INT
)
BEGIN
    DECLARE today_date DATE DEFAULT CURDATE();
    DECLARE existing_count INT DEFAULT 0;
    
    -- Check if already marked today
    SELECT COUNT(*) INTO existing_count 
    FROM attendance 
    WHERE person_id = p_person_id AND date_recorded = today_date;
    
    IF existing_count = 0 THEN
        -- Insert new attendance record
        INSERT INTO attendance (
            person_id, person_name, date_recorded, 
            confidence_score, image_path, location_id
        ) VALUES (
            p_person_id, p_person_name, today_date,
            p_confidence_score, p_image_path, p_location_id
        );
        
        -- Log the recognition
        INSERT INTO recognition_logs (
            person_id, person_name, confidence_score, 
            detection_status, image_path, location_id
        ) VALUES (
            p_person_id, p_person_name, p_confidence_score,
            'recognized', p_image_path, p_location_id
        );
        
        SELECT 'SUCCESS' as result, 'Attendance marked successfully' as message;
    ELSE
        SELECT 'DUPLICATE' as result, 'Attendance already marked today' as message;
    END IF;
END //

-- Procedure to get attendance summary
CREATE PROCEDURE GetAttendanceSummary(
    IN start_date DATE,
    IN end_date DATE
)
BEGIN
    SELECT 
        date_recorded,
        COUNT(*) as total_present,
        COUNT(CASE WHEN TIME(check_in_time) > '09:15:00' THEN 1 END) as late_count,
        AVG(confidence_score) as avg_confidence,
        MIN(check_in_time) as earliest_arrival,
        MAX(check_in_time) as latest_arrival
    FROM attendance 
    WHERE date_recorded BETWEEN start_date AND end_date
    GROUP BY date_recorded
    ORDER BY date_recorded DESC;
END //

DELIMITER ;

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Additional composite indexes for common query patterns
CREATE INDEX idx_attendance_date_person ON attendance(date_recorded, person_id);
CREATE INDEX idx_recognition_time_status ON recognition_logs(recognition_time, detection_status);
CREATE INDEX idx_person_status_name ON persons(status, name);

-- ============================================================================
-- DATABASE COMPLETE MESSAGE
-- ============================================================================

-- Show completion message
SELECT 
'✅ Face Recognition Database Schema Created Successfully!' as Status,
'All tables, views, procedures, and indexes have been created.' as Message,
'You can now run your Flask application to start using the database.' as NextStep;

-- Show basic table list (avoiding information_schema access issues)
SHOW TABLES;
