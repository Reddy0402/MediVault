import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from datetime import datetime
import os

class MedicalOCR:
    def __init__(self):
        # Configure pytesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Common medical terms and patterns
        self.medical_patterns = {
            'medications': r'(?:tablet|capsule|syrup|injection|mg|ml|dose|prescription|Rx)[\s\:]*([^\n\r]*)',
            'diagnosis': r'(?:diagnosis|condition|disease|findings)[\s\:]*([^\n\r]*)',
            'doctor': r'(?:Dr\.|Doctor|Physician)[\s\.]*([A-Za-z\s\.]+)',
            'date': r'(?:date|dated|on)[\s\:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
            'hospital': r'(?:hospital|clinic|medical center|healthcare)[\s\:]*([^\n\r]*)',
            'patient': r'(?:patient|name)[\s\:]*([^\n\r]*)',
            'age': r'(?:age|years old)[\s\:]*(\d+)',
            'gender': r'(?:gender|sex)[\s\:]*([MF]|Male|Female)',
            'symptoms': r'(?:symptoms|complaints)[\s\:]*([^\n\r]*)',
            'allergies': r'(?:allergies|allergic to)[\s\:]*([^\n\r]*)'
        }
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Apply dilation to connect text components
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            return dilated
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return None
    
    def extract_text(self, image_path, lang='eng'):
        """Extract text from medical document"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                raise ValueError("Image preprocessing failed")
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(processed_image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence text (confidence < 60)
            text_blocks = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 60 and data['text'][i].strip():
                    text_blocks.append(data['text'][i])
            
            # Join text blocks and clean
            text = ' '.join(text_blocks)
            return self.clean_text(text)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and format extracted text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms and important punctuation
        text = re.sub(r'[^\w\s\-\.\,\:\;\'\"\(\)\[\]\{\}\+\=\*\/]', '', text)
        
        # Fix common OCR mistakes
        text = text.replace('l', 'I')  # Common OCR mistake for capital I
        text = re.sub(r'(?<=\d)I(?=\d)', '1', text)  # Fix I to 1 in numbers
        
        return text.strip()
    
    def extract_medical_info(self, text):
        """Extract specific medical information from text"""
        medical_info = {
            'medications': [],
            'diagnosis': [],
            'doctor_name': '',
            'date': '',
            'hospital': '',
            'patient_name': '',
            'age': '',
            'gender': '',
            'symptoms': [],
            'allergies': [],
            'confidence': {}
        }
        
        # Extract information using patterns
        for key, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if key in ['medications', 'diagnosis', 'symptoms', 'allergies']:
                    medical_info[key] = [m.strip() for m in matches if m.strip()]
                else:
                    medical_info[key] = matches[0].strip()
                medical_info['confidence'][key] = True
        
        # Try to parse date if found
        if medical_info['date']:
            try:
                # Try different date formats
                for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d %B %Y', '%d %b %Y']:
                    try:
                        parsed_date = datetime.strptime(medical_info['date'], fmt)
                        medical_info['date'] = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        return medical_info
    
    def process_document(self, image_path, lang='eng'):
        """Process a medical document and return both text and structured information"""
        try:
            # Extract text
            text = self.extract_text(image_path, lang)
            if not text:
                return {
                    'success': False,
                    'error': 'No text could be extracted from the document',
                    'text': '',
                    'medical_info': {}
                }
            
            # Extract medical information
            medical_info = self.extract_medical_info(text)
            
            return {
                'success': True,
                'text': text,
                'medical_info': medical_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'medical_info': {}
            }