# Insurance Fraud Detection System

A comprehensive web application for detecting fraudulent insurance claims using Natural Language Processing (NLP) and Machine Learning techniques, including BERT transformers.

## 🎯 Features

- **PDF Document Processing**: Upload and extract text from multiple PDF claim documents
- **Multi-Modal Fraud Detection**: 
  - Keyword-based analysis for quick screening
  - BERT transformer model for advanced NLP analysis
- **Real-time Predictions**: Instant fraud probability scoring
- **Interactive Dashboard**: Professional UI with data visualizations
- **Model Training**: Train custom models on your insurance data
- **Dataset Analysis**: Comprehensive exploratory data analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run fraud_detection_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
insurance-fraud-detection/
├── fraud_detection_app.py      # Main Streamlit application
├── insurance_fraud_dataset.csv # Sample dataset
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 How to Use

### 1. Fraud Detection Page
- **Upload PDF Files**: Drag and drop or browse to upload claim documents
- **Manual Input**: Type claim descriptions directly
- **Choose Detection Method**: Select keyword-based, BERT model, or both
- **Analyze**: Click the analyze button to get fraud predictions

### 2. Model Training Page
- **View Dataset Statistics**: See overview of the training data
- **Train Models**: Train Random Forest models on the sample dataset
- **Performance Metrics**: View accuracy, precision, recall, and confusion matrices

### 3. Dataset Analysis Page
- **Explore Data**: Interactive visualizations of claim patterns
- **Feature Analysis**: Correlation analysis with fraud indicators
- **Statistical Insights**: Distribution plots and summary statistics

## 🤖 Technical Architecture

### Data Processing Pipeline
1. **PDF Text Extraction**: Uses `pdfplumber` for accurate text extraction
2. **Information Extraction**: Regex patterns to extract amounts, dates, locations
3. **Text Preprocessing**: Cleaning and normalization for ML models

### Machine Learning Models

#### Keyword-Based Detection
- Rule-based approach using predefined fraud indicators
- Fast processing for real-time screening
- Interpretable results with clear reasoning

#### BERT Transformer Model
- Pre-trained language model for contextual understanding
- Can be fine-tuned on insurance-specific data
- Captures complex linguistic patterns and relationships

#### Traditional ML Models
- Random Forest classifier with TF-IDF features
- Combines text and numerical features
- Ensemble methods for improved accuracy

### Performance Optimization
- **Caching**: Models and data loading optimization
- **Batch Processing**: Handle multiple documents efficiently
- **Session State**: Maintain application state across interactions

## 📊 Sample Dataset

The application includes a synthetic insurance fraud dataset with:
- **1000 claims** with realistic features
- **15% fraud rate** (150 fraudulent claims)
- **Features include**:
  - Claim descriptions (text data for NLP)
  - Driver demographics (age, gender, marital status)
  - Financial information (income, claim amount)
  - Vehicle details (age, category)
  - Incident details (location, witnesses, police report)

## 🔍 Fraud Detection Methodology

### Keyword-Based Approach
Analyzes claim text for suspicious patterns:
- High-value claims with vague descriptions
- Multiple injuries in minor accidents
- Unusual timing or circumstances
- Inconsistent witness information

### BERT-Based Approach
Leverages transformer architecture for:
- Contextual understanding of claim narratives
- Semantic analysis of suspicious language patterns
- Transfer learning from pre-trained models
- Fine-tuning capability on domain-specific data

## 📈 Model Performance

The system provides comprehensive evaluation metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: Fraud detection accuracy (minimize false positives)
- **Recall**: Fraud capture rate (minimize false negatives)
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Detailed prediction analysis

## 🎨 User Interface

- **Modern Design**: Clean, professional interface
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Charts**: Plotly visualizations
- **Color-Coded Results**: Easy interpretation of predictions
- **Progress Indicators**: Real-time processing feedback

## 🔄 Extensibility

The application is designed for easy extension:

### Adding New Models
```python
def custom_fraud_detector(claim_text):
    # Your custom logic here
    return {
        'fraud_probability': probability,
        'prediction': 'Fraud' or 'Legitimate',
        'confidence': confidence_score
    }
```

### Custom Feature Extraction
```python
def extract_custom_features(text):
    # Extract domain-specific features
    return features_dict
```

### Integration with External APIs
```python
def validate_with_external_db(claim_id):
    # Validate against external databases
    return validation_result
```

## 📚 Educational Value

This project demonstrates:
- **End-to-end ML pipeline** development
- **Production-ready** application architecture
- **Modern NLP techniques** with transformers
- **Interactive web application** development
- **Best practices** in fraud detection

## 🛠️ Development Setup

For developers wanting to extend the application:

1. **Set up virtual environment**:
   ```bash
   python -m venv fraud_detection_env
   source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install jupyter notebook ipykernel  # For data exploration
   ```

3. **Run in development mode**:
   ```bash
   streamlit run fraud_detection_app.py --server.runOnSave true
   ```

## 🚀 Deployment Options

### Local Deployment
- Run directly with Streamlit
- Suitable for development and testing

### Cloud Deployment
- **Streamlit Cloud**: Easy deployment with GitHub integration
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Scalable cloud solutions

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "fraud_detection_app.py"]
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional NLP models (GPT, T5, etc.)
- Enhanced feature engineering
- Real-time data integration
- Advanced visualization techniques
- Multi-language support

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Transformers Library**: Hugging Face for BERT models
- **Streamlit**: Amazing framework for ML web apps
- **scikit-learn**: Comprehensive ML library
- **Plotly**: Interactive visualization library

## 📞 Support

For questions or issues:
1. Check the application's "About" page for technical details
2. Review the code comments for implementation details
3. Test with the provided sample dataset

---

**Built for Machine Learning Engineers and Data Scientists**

This application showcases modern techniques in fraud detection, natural language processing, and web application development. Perfect for learning, experimentation, and production deployment.
