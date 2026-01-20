import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
        text-align: center;
    }
    .status-yes {
        color: #ff4b4b;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .status-no {
        color: #00d4ff;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff 0%, #0027ff 100%);
        color: white;
        border: none;
        padding: 0.8rem 1rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('assets/model.keras')
    preprocessor = joblib.load('assets/preprocessor.joblib')
    categorical_cols = joblib.load('assets/categorical_cols.joblib')
    numerical_cols = joblib.load('assets/numerical_cols.joblib')
    classes = joblib.load('assets/label_encoder_classes.joblib')
    return model, preprocessor, categorical_cols, numerical_cols, classes

try:
    model, preprocessor, categorical_cols, numerical_cols, classes = load_assets()
except Exception as e:
    st.error(f"Error loading model or assets: {e}. Please run 'train_model.py' first.")
    st.stop()

# Header
st.title("🏢 Employee Attrition Predictor")
st.markdown("### Predict the likelihood of an employee leaving the organization using Deep Learning.")

st.divider()

# Sidebar for inputs
st.sidebar.header("Employee Details")

def user_input_features():
    inputs = {}
    
    # Numerical Inputs
    with st.expander("📊 Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inputs['Age'] = st.number_input("Age", 18, 65, 30)
            inputs['MonthlyIncome'] = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
            inputs['DistanceFromHome'] = st.number_input("Distance From Home (km)", 1, 30, 5)
        with col2:
            inputs['Gender'] = st.selectbox("Gender", ["Female", "Male"])
            inputs['MaritalStatus'] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            inputs['Education'] = st.slider("Education Level (1-5)", 1, 5, 3)

    with st.expander("💼 Job & Performance"):
        col1, col2 = st.columns(2)
        with col1:
            inputs['Department'] = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            inputs['JobRole'] = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager", 
                "Sales Representative", "Research Director", "Human Resources"
            ])
            inputs['JobLevel'] = st.slider("Job Level (1-5)", 1, 5, 2)
            inputs['BusinessTravel'] = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        with col2:
            inputs['OverTime'] = st.selectbox("Over Time", ["Yes", "No"])
            inputs['PercentSalaryHike'] = st.slider("Percent Salary Hike", 10, 30, 15)
            inputs['PerformanceRating'] = st.selectbox("Performance Rating", [3, 4])
            inputs['StockOptionLevel'] = st.slider("Stock Option Level", 0, 3, 1)

    with st.expander("📈 Experience & Satisfaction"):
        col1, col2 = st.columns(2)
        with col1:
            inputs['TotalWorkingYears'] = st.number_input("Total Working Years", 0, 45, 10)
            inputs['YearsAtCompany'] = st.number_input("Years At Company", 0, 45, 5)
            inputs['YearsInCurrentRole'] = st.number_input("Years In Current Role", 0, 20, 2)
            inputs['NumCompaniesWorked'] = st.number_input("No. of Companies Worked", 0, 10, 1)
        with col2:
            inputs['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            inputs['JobSatisfaction'] = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            inputs['JobInvolvement'] = st.slider("Job Involvement (1-4)", 1, 4, 3)
            inputs['RelationshipSatisfaction'] = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
            inputs['WorkLifeBalance'] = st.slider("Work-Life Balance (1-4)", 1, 4, 3)

    with st.expander("🗓 History"):
        col1, col2 = st.columns(2)
        with col1:
            inputs['YearsSinceLastPromotion'] = st.number_input("Years Since Last Promotion", 0, 20, 1)
            inputs['YearsWithCurrManager'] = st.number_input("Years With Current Manager", 0, 20, 2)
        with col2:
            inputs['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", 0, 10, 2)
            inputs['EducationField'] = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
            inputs['DailyRate'] = st.number_input("Daily Rate", 100, 1500, 800)
            inputs['HourlyRate'] = st.number_input("Hourly Rate", 30, 100, 65)
            inputs['MonthlyRate'] = st.number_input("Monthly Rate", 2000, 30000, 14000)

    return pd.DataFrame([inputs])

# Get input
input_df = user_input_features()

# Display input summary (optional)
# st.subheader("Summary of Input Data")
# st.write(input_df)

# Prediction
if st.button("🚀 Predict Attrition"):
    with st.spinner("Analyzing employee data..."):
        # Preprocess the input
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction_prob = model.predict(processed_input)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        result_text = classes[prediction_class]
        
        # Display Result
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if result_text == "Yes":
            st.markdown(f'<h3>Prediction: <span class="status-yes">Likely to Leave</span></h3>', unsafe_allow_html=True)
            st.warning(f"Probability of Attrition: {prediction_prob*100:.2f}%")
            st.markdown("⚠️ High risk of turnover. Consider retention strategies and employee feedback.")
        else:
            st.markdown(f'<h3>Prediction: <span class="status-no">Likely to Stay</span></h3>', unsafe_allow_html=True)
            st.success(f"Confidence of Stability: {(1-prediction_prob)*100:.2f}%")
            st.markdown("✅ Low risk of turnover. Maintain current engagement and growth opportunities.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for Human Resource Insights.")
