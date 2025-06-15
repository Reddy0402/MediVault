import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

class MedicalClassifier:
    def __init__(self):
        self.pipeline = None
        self.categories = [
            'Prescription', 'Lab Report', 'X-Ray', 'MRI', 'CT Scan', 
            'Discharge Summary', 'Consultation Notes', 'Vaccination Record'
        ]
    
    def create_sample_data(self):
        """Create sample training data for demonstration"""
        data = {
            'text': [
                'Take this tablet twice daily after meals blood pressure medication',
                'Complete blood count hemoglobin levels normal range',
                'X-ray shows no fracture bone density normal chest clear',
                'MRI scan brain shows no abnormalities contrast imaging',
                'Patient discharged in stable condition follow up required',
                'Consultation with cardiologist heart rate normal ECG',
                'Vaccination record hepatitis B completed immunization',
                'CT scan abdomen liver function normal no stones detected'
            ],
            'category': [
                'Prescription', 'Lab Report', 'X-Ray', 'MRI', 
                'Discharge Summary', 'Consultation Notes', 'Vaccination Record', 'CT Scan'
            ]
        }
        return pd.DataFrame(data)
    
    def train_model(self):
        """Train the classification model"""
        # Create sample data (in production, use real medical data)
        df = self.create_sample_data()
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['category'], test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(self.pipeline, 'medical_classifier.pkl')
        
        return self.pipeline.score(X_test, y_test)
    
    def load_model(self):
        """Load trained model"""
        try:
            self.pipeline = joblib.load('medical_classifier.pkl')
        except FileNotFoundError:
            print("Model not found. Training new model...")
            self.train_model()
    
    def classify_document(self, text):
        """Classify medical document"""
        if not self.pipeline:
            self.load_model()
        
        prediction = self.pipeline.predict([text])[0]
        confidence = max(self.pipeline.predict_proba([text])[0])
        
        return {
            'category': prediction,
            'confidence': float(confidence)
        }