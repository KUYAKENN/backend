"""
Image utilities for storing and retrieving images from MySQL database
"""
import base64
import io
from PIL import Image
import mimetypes

class ImageHandler:
    """Handles image storage and retrieval for MySQL BLOB fields"""
    
    @staticmethod
    def prepare_image_for_storage(image_file):
        """
        Prepare image file for database storage
        
        Args:
            image_file: File object or file path
            
        Returns:
            dict: Contains image_data, filename, mime_type, size
        """
        try:
            if isinstance(image_file, str):
                # File path provided
                with open(image_file, 'rb') as f:
                    image_data = f.read()
                filename = image_file.split('/')[-1]
                mime_type = mimetypes.guess_type(image_file)[0] or 'image/jpeg'
            else:
                # File object provided (from Flask request)
                image_data = image_file.read()
                filename = getattr(image_file, 'filename', 'unknown.jpg')
                mime_type = getattr(image_file, 'content_type', 'image/jpeg')
            
            size = len(image_data)
            
            return {
                'image_data': image_data,
                'filename': filename,
                'mime_type': mime_type,
                'size': size
            }
        except Exception as e:
            print(f"Error preparing image for storage: {e}")
            return None
    
    @staticmethod
    def get_image_as_base64(image_data, mime_type='image/jpeg'):
        """
        Convert binary image data to base64 string for frontend
        
        Args:
            image_data: Binary image data from database
            mime_type: MIME type of the image
            
        Returns:
            str: Base64 encoded image data URI
        """
        try:
            if image_data:
                b64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:{mime_type};base64,{b64_data}"
            return None
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    @staticmethod
    def save_image_from_base64(base64_string, output_path):
        """
        Save base64 image string to file
        
        Args:
            base64_string: Base64 encoded image
            output_path: Where to save the file
        """
        try:
            # Remove data URI prefix if present
            if base64_string.startswith('data:'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            return True
        except Exception as e:
            print(f"Error saving image from base64: {e}")
            return False
    
    @staticmethod
    def resize_image_for_storage(image_data, max_width=800, max_height=600, quality=85):
        """
        Resize image to reduce storage size while maintaining quality
        
        Args:
            image_data: Binary image data
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            quality: JPEG quality (1-100)
            
        Returns:
            bytes: Resized image data
        """
        try:
            # Open image from binary data
            image = Image.open(io.BytesIO(image_data))
            
            # Calculate new size maintaining aspect ratio
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Save to bytes
            output = io.BytesIO()
            format = image.format if image.format else 'JPEG'
            if format == 'JPEG':
                image.save(output, format=format, quality=quality, optimize=True)
            else:
                image.save(output, format=format)
            
            return output.getvalue()
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image_data  # Return original if resize fails

# Utility functions for backward compatibility
def store_image_in_db(image_file, resize=True):
    """Store image in database format"""
    handler = ImageHandler()
    image_info = handler.prepare_image_for_storage(image_file)
    
    if image_info and resize:
        image_info['image_data'] = handler.resize_image_for_storage(
            image_info['image_data']
        )
        image_info['size'] = len(image_info['image_data'])
    
    return image_info

def get_image_for_display(image_data, mime_type='image/jpeg'):
    """Get image as base64 for frontend display"""
    return ImageHandler.get_image_as_base64(image_data, mime_type)
