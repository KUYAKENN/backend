# database.py - MySQL Database Configuration and Models
import os
import json
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import Column, Integer, String, Text, DateTime, Date, Time, Float, Boolean, ForeignKey, create_engine, TIMESTAMP, DECIMAL, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from flask_sqlalchemy import SQLAlchemy

Base = declarative_base()

# Database Models
class Person(Base):
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    phone = Column(String(50))
    department = Column(String(100))
    position = Column(String(100))
    status = Column(Enum('active', 'inactive'), default='active')
    registration_date = Column(TIMESTAMP, default=datetime.now)
    last_updated = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now)
    notes = Column(Text)

class FaceEncoding(Base):
    __tablename__ = 'face_encodings'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    encoding_data = Column(Text, nullable=False)
    encoding_type = Column(Enum('standard', 'enhanced', 'multi_angle'), default='standard')
    face_angle = Column(Enum('front', 'left', 'right', 'up', 'down'), default='front')
    confidence_score = Column(DECIMAL(3, 2), default=1.0)
    image_path = Column(String(500))
    created_date = Column(TIMESTAMP, default=datetime.now)
    is_primary = Column(Boolean, default=False)
    
    # Relationship
    person = relationship("Person", backref="face_encodings")

class Attendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    person_name = Column(String(255))
    check_in_time = Column(TIMESTAMP, default=datetime.now)
    check_out_time = Column(TIMESTAMP)
    date_recorded = Column(Date, default=datetime.now().date)
    confidence_score = Column(DECIMAL(3, 2), default=1.0)
    detection_method = Column(Enum('auto', 'manual', 'override'), default='auto')
    image_path = Column(String(500))
    location_id = Column(Integer, ForeignKey('locations.id'))
    status = Column(Enum('present', 'late', 'early_leave', 'absent'), default='present')
    notes = Column(Text)
    created_by = Column(String(100))
    
    # Relationship
    person = relationship("Person", backref="attendance_records")

class Location(Base):
    __tablename__ = 'locations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    camera_id = Column(String(100))
    coordinates = Column(Text)
    status = Column(Enum('active', 'inactive', 'maintenance'), default='active')
    created_date = Column(TIMESTAMP, default=datetime.now)

class RecognitionLog(Base):
    __tablename__ = 'recognition_logs'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'))
    person_name = Column(String(255))
    recognition_time = Column(TIMESTAMP, default=datetime.now)
    confidence_score = Column(DECIMAL(3, 2), nullable=False)
    detection_status = Column(Enum('recognized', 'unknown', 'low_confidence', 'failed'), nullable=False)
    image_path = Column(String(500))
    location_id = Column(Integer, ForeignKey('locations.id'))
    
    # Relationships
    person = relationship("Person", backref="recognition_logs")
    location = relationship("Location", backref="recognition_logs")

class DatabaseManager:
    def __init__(self, db_config=None):
        """Initialize database manager with configuration"""
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
                'database': os.getenv('DB_NAME', 'face_recognition_db')
            }
        
        self.db_config = db_config
        self.connection = None
        self.engine = None
        self.Session = None
        
        # Create SQLAlchemy engine
        db_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        try:
            self.engine = create_engine(db_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            self.create_database_if_not_exists()
            self.create_tables()
            print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("💡 Falling back to file-based storage")
            self.engine = None
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Connect without specifying database
            temp_config = self.db_config.copy()
            temp_config.pop('database', None)
            
            connection = mysql.connector.connect(**temp_config)
            cursor = connection.cursor()
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            cursor.close()
            connection.close()
            
            print(f"✅ Database '{self.db_config['database']}' ready")
            
        except Error as e:
            print(f"❌ Error creating database: {e}")
            raise
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            if self.engine:
                Base.metadata.create_all(self.engine)
                print("✅ Database tables created/verified")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        if self.Session:
            return self.Session()
        return None
    
    def is_connected(self):
        """Check if database is connected"""
        return self.engine is not None
    
    # Person Management
    def save_person(self, name, is_enhanced=False, total_samples=0):
        """Save or update person"""
        if not self.is_connected():
            return False
        
        session = self.get_session()
        try:
            # Check if person exists
            person = session.query(Person).filter_by(name=name).first()
            
            if person:
                # Update existing person
                person.last_updated = datetime.now()
            else:
                # Create new person
                person = Person(
                    name=name
                )
                session.add(person)
            
            session.commit()
            return person.id
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving person: {e}")
            return False
        finally:
            session.close()
    
    def get_person(self, name):
        """Get person by name"""
        if not self.is_connected():
            return None
        
        session = self.get_session()
        try:
            person = session.query(Person).filter_by(name=name).first()
            return person
        except Exception as e:
            print(f"❌ Error getting person: {e}")
            return None
        finally:
            session.close()
    
    def get_all_persons(self):
        """Get all persons"""
        if not self.is_connected():
            return []
        
        session = self.get_session()
        try:
            persons = session.query(Person).all()
            return [{'id': p.id, 'name': p.name, 'status': p.status} for p in persons]
        except Exception as e:
            print(f"❌ Error getting persons: {e}")
            return []
        finally:
            session.close()
    
    def delete_person(self, name):
        """Delete person and all related data"""
        if not self.is_connected():
            return False
        
        session = self.get_session()
        try:
            # Get person
            person = session.query(Person).filter_by(name=name).first()
            if not person:
                return False
            
            # Delete related data using person_id
            session.query(FaceEncoding).filter_by(person_id=person.id).delete()
            session.query(Attendance).filter_by(person_id=person.id).delete()
            session.query(Location).filter_by(person_id=person.id).delete()
            
            # Delete person
            session.delete(person)
            session.commit()
            
            print(f"✅ Deleted person '{name}' and all related data")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error deleting person: {e}")
            return False
        finally:
            session.close()
    
    # Face Encoding Management
    def save_face_encoding(self, person_name, pose, encoding, confidence=1.0, quality_score=0.8, image_path=None):
        """Save face encoding"""
        if not self.is_connected():
            return False
        
        session = self.get_session()
        try:
            # Get person ID
            person = session.query(Person).filter_by(name=person_name).first()
            if not person:
                # Create person if doesn't exist
                person_id = self.save_person(person_name)
            else:
                person_id = person.id
            
            # Convert numpy array to JSON string
            encoding_json = json.dumps(encoding.tolist() if isinstance(encoding, np.ndarray) else encoding)
            
            # Create face encoding record
            face_encoding = FaceEncoding(
                person_id=person_id,
                face_angle=pose,
                encoding_data=encoding_json,
                confidence_score=confidence,
                image_path=image_path
            )
            
            session.add(face_encoding)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving face encoding: {e}")
            return False
        finally:
            session.close()
    
    def get_face_encodings(self, person_name=None):
        """Get face encodings, optionally filtered by person"""
        if not self.is_connected():
            return []
        
        session = self.get_session()
        try:
            query = session.query(FaceEncoding).join(Person)
            if person_name:
                query = query.filter(Person.name == person_name)
            
            encodings = query.all()
            
            result = []
            for enc in encodings:
                result.append({
                    'id': enc.id,
                    'person_name': enc.person.name,  # Get name from related Person
                    'face_angle': enc.face_angle,
                    'encoding': json.loads(enc.encoding_data),
                    'confidence_score': enc.confidence_score,
                    'image_path': enc.image_path,
                    'created_date': enc.created_date
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting face encodings: {e}")
            return []
        finally:
            session.close()
    
    def get_all_face_data(self):
        """Get all face data for recognition system"""
        if not self.is_connected():
            return [], [], []
        
        encodings = self.get_face_encodings()
        
        face_encodings = []
        face_names = []
        face_metadata = []
        
        for enc in encodings:
            face_encodings.append(np.array(enc['encoding']))
            face_names.append(enc['person_name'])
            face_metadata.append({
                'pose': enc['pose'],
                'confidence': enc['confidence'],
                'quality_score': enc['quality_score']
            })
        
        return face_encodings, face_names, face_metadata
    
    # Attendance Management
    def mark_attendance(self, person_name, confidence=None, image_path=None, location_data=None):
        """Mark attendance for a person"""
        if not self.is_connected():
            return False
        
        session = self.get_session()
        try:
            # Get person
            person = session.query(Person).filter_by(name=person_name).first()
            if not person:
                person_id = self.save_person(person_name)
            else:
                person_id = person.id
            
            # Check if already marked today
            today = datetime.now().date()
            existing = session.query(Attendance).filter(
                Attendance.person_id == person_id,
                Attendance.date_recorded >= today,
                Attendance.date_recorded < today + timedelta(days=1)
            ).first()
            
            if existing:
                return False  # Already marked today
            
            # Create attendance record
            now = datetime.now()
            attendance = Attendance(
                person_id=person_id,
                person_name=person_name,
                check_in_time=now,
                date_recorded=now.date(),
                confidence_score=confidence,
                image_path=image_path
            )
            
            session.add(attendance)
            session.commit()
            
            print(f"✅ Attendance marked for {person_name}")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error marking attendance: {e}")
            return False
        finally:
            session.close()
    
    def get_attendance(self, date=None, person_name=None):
        """Get attendance records"""
        if not self.is_connected():
            return []
        
        session = self.get_session()
        try:
            query = session.query(Attendance).join(Person)
            
            if date:
                if isinstance(date, str):
                    date = datetime.strptime(date, '%Y-%m-%d').date()
                query = query.filter(Attendance.date_recorded == date)
            
            if person_name:
                query = query.filter(Person.name == person_name)
            
            records = query.order_by(Attendance.check_in_time.desc()).all()
            
            result = []
            for record in records:
                result.append({
                    'id': record.id,
                    'name': record.person.name,
                    'date': record.date_recorded.strftime('%Y-%m-%d'),
                    'time': record.check_in_time.strftime('%H:%M:%S'),
                    'status': record.status,
                    'confidence': record.confidence_score
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting attendance: {e}")
            return []
        finally:
            session.close()
    
    def get_todays_attendance(self):
        """Get today's attendance as set of names"""
        today = datetime.now().date()
        records = self.get_attendance(date=today)
        return set([record['name'] for record in records])
    
    def delete_attendance_record(self, record_id):
        """Delete specific attendance record"""
        if not self.is_connected():
            return False
        
        session = self.get_session()
        try:
            record = session.query(Attendance).filter_by(id=record_id).first()
            if record:
                session.delete(record)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error deleting attendance record: {e}")
            return False
        finally:
            session.close()
    
    # Location Management
    def save_location(self, person_name, latitude, longitude, address=None):
        """Save location data"""
        if not self.is_connected():
            return False
        
        session = self.get_session()
        try:
            # Get person
            person = session.query(Person).filter_by(name=person_name).first()
            if not person:
                person_id = self.save_person(person_name)
            else:
                person_id = person.id
            
            # Create location record
            location = Location(
                person_id=person_id,
                person_name=person_name,
                latitude=latitude,
                longitude=longitude,
                address=address
            )
            
            session.add(location)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving location: {e}")
            return False
        finally:
            session.close()
    
    def get_locations(self, person_name, limit=10):
        """Get recent locations for a person"""
        if not self.is_connected():
            return []
        
        session = self.get_session()
        try:
            locations = session.query(Location).filter_by(person_name=person_name)\
                             .order_by(Location.timestamp.desc()).limit(limit).all()
            
            result = []
            for loc in locations:
                result.append({
                    'latitude': loc.latitude,
                    'longitude': loc.longitude,
                    'address': loc.address,
                    'timestamp': loc.timestamp.isoformat()
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting locations: {e}")
            return []
        finally:
            session.close()
    
    # Migration Methods
    def migrate_from_csv(self, csv_file_path):
        """Migrate existing CSV attendance data to MySQL"""
        if not self.is_connected():
            return False
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_file_path)
            
            session = self.get_session()
            migrated_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Parse date and time
                    date_obj = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                    time_str = row['Time']
                    datetime_obj = datetime.strptime(f"{row['Date']} {time_str}", '%Y-%m-%d %H:%M:%S')
                    
                    # Check if record already exists
                    person = session.query(Person).filter_by(name=row['Name']).first()
                    if person:
                        existing = session.query(Attendance).filter(
                            Attendance.person_id == person.id,
                            Attendance.date_recorded == date_obj,
                            Attendance.check_in_time == datetime_obj
                        ).first()
                    else:
                        existing = None
                    
                    if not existing:
                        # Get or create person
                        if not person:
                            person = Person(name=row['Name'])
                            session.add(person)
                            session.flush()  # Get the ID
                        
                        # Create attendance record
                        attendance = Attendance(
                            person_id=person.id,
                            person_name=row['Name'],
                            date_recorded=date_obj,
                            check_in_time=datetime_obj,
                            status=row.get('Status', 'present')
                        )
                        
                        session.add(attendance)
                        migrated_count += 1
                
                except Exception as e:
                    print(f"⚠️ Error migrating row {row['Name']}: {e}")
                    continue
            
            session.commit()
            print(f"✅ Migrated {migrated_count} attendance records from CSV")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error migrating CSV data: {e}")
            return False
        finally:
            session.close()
    
    def migrate_from_json(self, json_file_path):
        """Migrate existing JSON face data to MySQL"""
        if not self.is_connected():
            return False
        
        try:
            with open(json_file_path, 'r') as f:
                face_data = json.load(f)
            
            migrated_count = 0
            
            for person_name, person_data in face_data.items():
                try:
                    # Save person
                    is_enhanced = 'poses' in person_data
                    total_samples = person_data.get('total_samples', 1)
                    person_id = self.save_person(person_name, is_enhanced, total_samples)
                    
                    if is_enhanced:
                        # Enhanced format with multiple poses
                        for pose_data in person_data['poses']:
                            self.save_face_encoding(
                                person_name=person_name,
                                pose=pose_data['pose'],
                                encoding=pose_data['encoding'],
                                confidence=pose_data.get('confidence', 1.0),
                                quality_score=pose_data.get('quality_score', 0.8)
                            )
                            migrated_count += 1
                    else:
                        # Legacy format - single encoding
                        self.save_face_encoding(
                            person_name=person_name,
                            pose='front',
                            encoding=person_data,
                            confidence=1.0,
                            quality_score=0.8
                        )
                        migrated_count += 1
                
                except Exception as e:
                    print(f"⚠️ Error migrating person {person_name}: {e}")
                    continue
            
            print(f"✅ Migrated {migrated_count} face encodings from JSON")
            return True
            
        except Exception as e:
            print(f"❌ Error migrating JSON data: {e}")
            return False

# Global database manager instance
db_manager = None

def init_database(db_config=None):
    """Initialize database manager"""
    global db_manager
    db_manager = DatabaseManager(db_config)
    return db_manager

def get_db():
    """Get database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = init_database()
    return db_manager
