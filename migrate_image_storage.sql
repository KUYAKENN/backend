-- Migration script to add image storage columns
-- Run this to update existing database to store images directly

USE face_recognition_db;

-- Add image storage columns to face_encodings table
ALTER TABLE face_encodings 
ADD COLUMN image_data LONGBLOB AFTER confidence_score,
ADD COLUMN image_filename VARCHAR(255) AFTER image_data,
ADD COLUMN image_mime_type VARCHAR(100) DEFAULT 'image/jpeg' AFTER image_filename,
ADD COLUMN image_size INT AFTER image_mime_type;

-- Add image storage columns to attendance table
ALTER TABLE attendance 
ADD COLUMN detection_image_data LONGBLOB AFTER detection_method,
ADD COLUMN detection_image_filename VARCHAR(255) AFTER detection_image_data,
ADD COLUMN detection_image_mime_type VARCHAR(100) DEFAULT 'image/jpeg' AFTER detection_image_filename;

-- Optional: Remove old image_path columns after migrating data
-- ALTER TABLE face_encodings DROP COLUMN image_path;
-- ALTER TABLE attendance DROP COLUMN image_path;

SHOW WARNINGS;
