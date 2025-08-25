
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

# Import the fraud detection models
from fraud_detection_models import fraud_detector

# Set page config
st.set_page_config(
    page_title="Insurance Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .fraud-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .legitimate-claim {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'processed_claims' not in st.session_state:
    st.session_state.processed_claims = []

@st.cache_data
def load_sample_dataset():
    """Load the sample insurance fraud dataset"""
    return fraud_detector.load_sample_dataset()

@st.cache_resource
def load_bert_model():
    """Load pre-trained BERT model for sentiment analysis"""
    return fraud_detector.load_bert_model()

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 Insurance Fraud Detection System</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Fraud Detection", "Model Training", "Dataset Analysis", "About"])

    if page == "Fraud Detection":
        fraud_detection_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Dataset Analysis":
        dataset_analysis_page()
    else:
        about_page()

def fraud_detection_page():
    st.markdown('<h2 class="sub-header">📄 Claim Analysis</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # File upload section
        st.subheader("Upload Claim Document")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files containing insurance claim documents"
        )

        # Manual text input option
        st.subheader("Or Enter Claim Description Manually")
        manual_text = st.text_area(
            "Claim Description",
            height=150,
            placeholder="Enter the insurance claim description here..."
        )

    with col2:
        # Model selection
        st.subheader("Detection Method")
        st.info("Using BERT Model for fraud detection")
        
        # Processing button
        if st.button("🔍 Analyze Claims", type="primary"):
            process_claims(uploaded_files, manual_text)

def process_claims(uploaded_files, manual_text):
    """Process uploaded files and manual text for fraud detection"""

    # Load BERT model
    with st.spinner("Loading BERT model..."):
        bert_model = load_bert_model()

    claims_to_process = []

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                text = fraud_detector.extract_text_from_pdf(uploaded_file)
                if text.strip():
                    claim_info = fraud_detector.extract_claim_info(text)
                    claims_to_process.append({
                        'source': uploaded_file.name,
                        'text': text,
                        'info': claim_info
                    })

    # Process manual text
    if manual_text.strip():
        claim_info = fraud_detector.extract_claim_info(manual_text)
        claims_to_process.append({
            'source': 'Manual Input',
            'text': manual_text,
            'info': claim_info
        })

    # Analyze claims
    if claims_to_process:
        st.markdown('<h3 class="sub-header">📊 Analysis Results</h3>', unsafe_allow_html=True)

        for i, claim in enumerate(claims_to_process):
            with st.expander(f"📋 Claim Analysis - {claim['source']}", expanded=True):

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Display extracted information
                    st.write("**Extracted Information:**")
                    info = claim['info']
                    if info['claim_amount']:
                        st.write(f"💰 **Claim Amount:** ${info['claim_amount']:,.2f}")
                    if info['date']:
                        st.write(f"📅 **Date:** {info['date']}")
                    if info['location']:
                        st.write(f"📍 **Location:** {info['location']}")

                    st.write("**Claim Description:**")
                    st.write(claim['text'][:500] + "..." if len(claim['text']) > 500 else claim['text'])

                with col2:
                    # Fraud detection results
                    st.write("**🤖 BERT Model Analysis:**")
                    bert_result = fraud_detector.predict_fraud_bert(claim['text'])
                    display_prediction_result(bert_result, "bert")

                # Store processed claim
                st.session_state.processed_claims.append({
                    'timestamp': datetime.now(),
                    'source': claim['source'],
                    'bert_result': fraud_detector.predict_fraud_bert(claim['text'])
                })

def display_prediction_result(result, model_type):
    """Display fraud prediction results"""
    fraud_prob = result['fraud_probability']
    prediction = result['prediction']

    # Color coding
    if prediction == 'Fraud':
        color = "red"
        emoji = "🚨"
    else:
        color = "green"
        emoji = "✅"

    # Display prediction
    st.markdown(f"{emoji} **Prediction:** <span style='color: {color}'>{prediction}</span>", 
                unsafe_allow_html=True)

    # Probability bar
    st.progress(fraud_prob)
    st.write(f"Fraud Probability: {fraud_prob:.1%}")

    # Additional metrics
    if 'confidence' in result:
        st.write(f"Confidence: {result['confidence']:.1%}")

def model_training_page():
    st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)

    # Load sample dataset
    df = load_sample_dataset()
    if df is None:
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dataset Overview")
        stats = fraud_detector.get_dataset_statistics(df)
        if stats:
            st.write(f"**Total Claims:** {stats['total_claims']:,}")
            st.write(f"**Fraudulent Claims:** {stats['fraudulent_claims']:,}")
            st.write(f"**Legitimate Claims:** {stats['legitimate_claims']:,}")
            st.write(f"**Fraud Rate:** {stats['fraud_rate']:.1%}")

        # Feature importance (simplified)
        if st.button("Train Random Forest Model"):
            train_random_forest_model(df)

    with col2:
        st.subheader("Model Performance")
        if st.session_state.model_trained:
            display_model_performance()
        else:
            st.info("Train a model to see performance metrics")

def train_random_forest_model(df):
    """Train a Random Forest model on the dataset"""
    with st.spinner("Training model..."):
        results = fraud_detector.train_random_forest_model(df)
        
        # Store results in session state
        st.session_state.model_trained = True
        st.session_state.model_accuracy = results['accuracy']
        st.session_state.classification_report = results['classification_report']
        st.session_state.confusion_matrix = results['confusion_matrix']

        st.success(f"Model trained successfully! Accuracy: {results['accuracy']:.1%}")

def display_model_performance():
    """Display model performance metrics"""
    st.write(f"**Accuracy:** {st.session_state.model_accuracy:.1%}")

    # Classification report
    report = st.session_state.classification_report

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Precision:**")
        st.write(f"- Legitimate: {report['0']['precision']:.3f}")
        st.write(f"- Fraud: {report['1']['precision']:.3f}")

    with col2:
        st.write("**Recall:**")
        st.write(f"- Legitimate: {report['0']['recall']:.3f}")
        st.write(f"- Fraud: {report['1']['recall']:.3f}")

    # Confusion matrix
    cm = st.session_state.confusion_matrix
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Legitimate', 'Fraud'],
                    y=['Legitimate', 'Fraud'],
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

def dataset_analysis_page():
    st.markdown('<h2 class="sub-header">📊 Dataset Analysis</h2>', unsafe_allow_html=True)

    df = load_sample_dataset()
    if df is None:
        return

    # Basic statistics
    stats = fraud_detector.get_dataset_statistics(df)
    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Claims", f"{stats['total_claims']:,}")

        with col2:
            st.metric("Fraudulent Claims", f"{stats['fraudulent_claims']:,}")

        with col3:
            st.metric("Average Claim", f"${stats['avg_claim_amount']:,.0f}")

        with col4:
            st.metric("Fraud Rate", f"{stats['fraud_rate']:.1%}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Claim amount distribution
        fig = px.histogram(df, x='claim_amount', color='fraud_reported',
                          title='Claim Amount Distribution',
                          labels={'fraud_reported': 'Fraud Status'},
                          nbins=30)
        fig.update_layout(xaxis_title="Claim Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Age distribution
        fig = px.box(df, x='fraud_reported', y='age_of_driver',
                     title='Age Distribution by Fraud Status',
                     labels={'fraud_reported': 'Fraud Status', 'age_of_driver': 'Driver Age'})
        st.plotly_chart(fig, use_container_width=True)

    # Feature correlation with fraud
    st.subheader("Feature Analysis")

    # Calculate correlation with fraud
    corr_df = fraud_detector.get_feature_correlations(df)
    if corr_df is not None:
        fig = px.bar(corr_df, x='Feature', y='Correlation',
                     title='Feature Correlation with Fraud',
                     color='Correlation',
                     color_continuous_scale='RdBu_r')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)

    st.markdown("""
    ## Insurance Fraud Detection System

    This application demonstrates advanced techniques for detecting fraudulent insurance claims using:

    ### 🔧 Technologies Used:
    - **Streamlit**: Web application framework
    - **BERT/Transformers**: Natural Language Processing
    - **scikit-learn**: Machine Learning models
    - **pdfplumber**: PDF text extraction
    - **Plotly**: Interactive visualizations

    ### 🎯 Features:
    1. **PDF Processing**: Upload and extract text from claim documents
    2. **BERT Model Analysis**: Advanced language understanding for fraud detection
    3. **Real-time Predictions**: Instant fraud probability scoring
    4. **Model Training**: Train custom models on your data
    5. **Data Visualization**: Comprehensive dataset analysis

    ### 🔍 Detection Methods:

    #### BERT Model Detection:
    - Leverages transformer-based language understanding
    - Can be fine-tuned on insurance-specific data
    - Captures complex linguistic patterns
    - Uses sentiment analysis as a proxy for fraud detection

    ### 📊 Model Performance:
    The system provides detailed performance metrics including:
    - Accuracy, Precision, Recall
    - Confusion matrices
    - Feature importance analysis
    - ROC curves and other evaluation metrics

    ### 🚀 Future Enhancements:
    - Integration with real insurance databases
    - Advanced ensemble methods
    - Explainable AI techniques (SHAP, LIME)
    - Real-time monitoring dashboards
    - Multi-language support

    ### 💡 Use Cases:
    - **Insurance Companies**: Automated claim screening
    - **Fraud Investigators**: Decision support tool
    - **Researchers**: Fraud detection methodology testing
    - **Regulators**: Compliance and oversight

    ---

    **Built for ML Engineers and Data Scientists**

    This application demonstrates best practices in:
    - End-to-end ML pipeline development
    - Production-ready model deployment
    - User-friendly interface design
    - Scalable architecture patterns
    """)

    # Technical details
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        ### Architecture Overview:

        1. **Data Ingestion Layer**:
           - PDF text extraction using pdfplumber
           - Text preprocessing and cleaning
           - Feature extraction from unstructured data

        2. **ML Pipeline**:
           - TF-IDF vectorization for text features
           - Random Forest for traditional ML approach
           - BERT for deep learning approach
           - Ensemble methods for improved accuracy

        3. **Deployment Layer**:
           - Streamlit for web interface
           - Session state management
           - Caching for model loading optimization
           - Real-time prediction serving

        ### Model Training Process:

        ```python
        # Feature Engineering
        text_features = TfidfVectorizer().fit_transform(descriptions)
        numerical_features = StandardScaler().fit_transform(numerical_data)
        X = hstack([text_features, numerical_features])

        # Model Training
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        # BERT Fine-tuning (pseudo-code)
        model = AutoModelForSequenceClassification.from_pretrained('bert-base')
        trainer = Trainer(model=model, train_dataset=train_data)
        trainer.train()
        ```

        ### Performance Optimization:

        - **Caching**: Model loading and data processing
        - **Batch Processing**: Multiple file handling
        - **Lazy Loading**: Models loaded only when needed
        - **Memory Management**: Efficient data structures
        """)

if __name__ == "__main__":
    main()
