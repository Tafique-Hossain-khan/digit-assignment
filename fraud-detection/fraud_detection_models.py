import pandas as pd
import numpy as np
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import os

class FraudDetectionModels:
    """Class containing all fraud detection models and utilities"""
    
    def __init__(self):
        self.bert_model = None
        self.random_forest_model = None
        self.vectorizer = None
        
    def load_bert_model(self):
        """Load pre-trained BERT model for sentiment analysis (can be fine-tuned for fraud detection)"""
        try:
            # Using a general sentiment model as a proxy for fraud detection
            # In practice, you would fine-tune BERT on insurance fraud data
            classifier = pipeline("sentiment-analysis", 
                                model="nlptown/bert-base-multilingual-uncased-sentiment",
                                return_all_scores=True)
            self.bert_model = classifier
            return classifier
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            return None

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_claim_info(self, text):
        """Extract relevant information from claim text using regex patterns"""
        claim_info = {
            'claim_amount': None,
            'date': None,
            'location': None,
            'description': text[:500] if text else ""  # First 500 chars as description
        }

        # Extract monetary amounts
        amount_pattern = r'\$([0-9,]+(?:\.[0-9]{2})?)|([0-9,]+(?:\.[0-9]{2})?)\s*(?:dollars?|USD)'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        if amounts:
            # Take the largest amount found
            amounts_clean = []
            for match in amounts:
                amount_str = match[0] if match[0] else match[1]
                amount_clean = float(amount_str.replace(',', ''))
                amounts_clean.append(amount_clean)
            claim_info['claim_amount'] = max(amounts_clean)

        # Extract dates
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|\b(\w+ \d{1,2}, \d{4})\b'
        dates = re.findall(date_pattern, text)
        if dates:
            claim_info['date'] = dates[0][0] if dates[0][0] else dates[0][1]

        # Extract locations (simple approach)
        location_keywords = ['highway', 'street', 'avenue', 'road', 'parking', 'intersection']
        for keyword in location_keywords:
            if keyword in text.lower():
                # Find sentence containing the keyword
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        claim_info['location'] = sentence.strip()[:100]
                        break

        return claim_info

    def predict_fraud_bert(self, claim_text):
        """Predict fraud using BERT model (simplified approach)"""
        if not self.bert_model or not claim_text:
            return {'fraud_probability': 0.5, 'prediction': 'Unknown', 'confidence': 0}

        try:
            # Use sentiment analysis as a proxy for fraud detection
            # Negative sentiment might correlate with suspicious claims
            result = self.bert_model(claim_text)

            # Convert sentiment scores to fraud probability
            # This is a simplified approach - in practice, you'd fine-tune BERT on fraud data
            if isinstance(result[0], list):
                scores = {item['label']: item['score'] for item in result[0]}
            else:
                scores = {result[0]['label']: result[0]['score']}

            # Simple heuristic: negative sentiment = higher fraud probability
            neg_score = scores.get('NEGATIVE', scores.get('1 star', 0))
            pos_score = scores.get('POSITIVE', scores.get('5 stars', 0))

            fraud_probability = (neg_score + (1 - pos_score)) / 2

            return {
                'fraud_probability': fraud_probability,
                'prediction': 'Fraud' if fraud_probability > 0.6 else 'Legitimate',
                'confidence': max(neg_score, pos_score),
                'sentiment_scores': scores
            }
        except Exception as e:
            print(f"Error in BERT prediction: {e}")
            return {'fraud_probability': 0.5, 'prediction': 'Unknown', 'confidence': 0}

    def train_random_forest_model(self, df):
        """Train a Random Forest model on the dataset"""
        # Prepare features
        # For this example, we'll use TF-IDF on claim descriptions plus numerical features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        text_features = vectorizer.fit_transform(df['claim_description']).toarray()

        # Numerical features
        numerical_cols = ['claim_amount', 'age_of_driver', 'annual_income', 
                         'vehicle_age', 'witness_present', 'police_report', 'past_claims']
        numerical_features = df[numerical_cols].values

        # Combine features
        X = np.hstack([text_features, numerical_features])
        y = df['fraud_reported'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store models
        self.random_forest_model = model
        self.vectorizer = vectorizer

        # Save models
        joblib.dump(model, 'fraud_detection_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }

    def load_sample_dataset(self):
        """Load the sample insurance fraud dataset"""
        try:
            df = pd.read_csv('insurance_fraud_dataset.csv')
            return df
        except FileNotFoundError:
            print("Sample dataset not found. Please ensure 'insurance_fraud_dataset.csv' is available.")
            return None

    def get_dataset_statistics(self, df):
        """Get basic statistics from the dataset"""
        if df is None:
            return None
            
        return {
            'total_claims': len(df),
            'fraudulent_claims': df['fraud_reported'].sum(),
            'legitimate_claims': (df['fraud_reported'] == 0).sum(),
            'fraud_rate': df['fraud_reported'].mean(),
            'avg_claim_amount': df['claim_amount'].mean(),
            'max_claim_amount': df['claim_amount'].max(),
            'min_claim_amount': df['claim_amount'].min()
        }

    def get_feature_correlations(self, df):
        """Calculate correlation between features and fraud"""
        if df is None:
            return None
            
        numerical_cols = ['claim_amount', 'age_of_driver', 'annual_income', 
                         'vehicle_age', 'witness_present', 'police_report', 'past_claims']

        correlations = []
        for col in numerical_cols:
            corr = df[col].corr(df['fraud_reported'])
            correlations.append({'Feature': col, 'Correlation': corr})

        return pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)

# Global instance for easy access
fraud_detector = FraudDetectionModels()
