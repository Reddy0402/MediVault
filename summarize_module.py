from transformers import pipeline
import re
from collections import Counter

class MedicalSummarizer:
    def __init__(self):
        try:
            # Load pre-trained summarization model
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            self.summarizer = None
    
    def extractive_summary(self, text, num_sentences=3):
        """Create extractive summary using sentence scoring"""
        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()] # Clean up empty strings

        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences based on medical keywords
        medical_keywords = [
            'diagnosis', 'treatment', 'medication', 'symptoms', 'condition',
            'patient', 'doctor', 'hospital', 'prescription', 'therapy'
        ]
        
        sentence_scores = []
        for sentence in sentences:
            score = 0
            words = sentence.lower().split()
            for keyword in medical_keywords:
                score += words.count(keyword)
            sentence_scores.append(score)
        
        # Get top sentences
        top_indices = sorted(range(len(sentence_scores)), 
                           key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
        top_indices.sort()
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def simple_summary(self, text, num_sentences=3):
        """Simple summary by taking first few sentences"""
        sentences = text.split('.')
        return '. '.join(sentences[:num_sentences]) + '.'
    
    def abstractive_summary(self, text, max_length=150):
        """Create abstractive summary using transformer model"""
        if not self.summarizer:
            return self.simple_summary(text)
        
        try:
            if len(text) < 50:
                return text
            
            # Limit input length for the model
            if len(text) > 1024:
                text = text[:1024]
            
            summary = self.summarizer(text, max_length=max_length, 
                                    min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return self.simple_summary(text)
    
    def extract_key_points(self, text):
        """Extract key medical information"""
        key_points = {
            'medications': [],
            'conditions': [],
            'procedures': [],
            'recommendations': []
        }
        
        # Simple pattern matching for key information
        med_patterns = r'(?:take|prescribed|medication|tablet|capsule|mg|ml)\s+[\w\s]*'
        condition_patterns = r'(?:diagnosed|condition|disease|syndrome)\s+[\w\s]*'
        
        medications = re.findall(med_patterns, text, re.IGNORECASE)
        conditions = re.findall(condition_patterns, text, re.IGNORECASE)
        
        key_points['medications'] = medications[:5]  # Limit to top 5
        key_points['conditions'] = conditions[:3]    # Limit to top 3
        
        return key_points