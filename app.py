import streamlit as st
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import time  # Import time for simulating loading duration

# Load saved models
min_salary_model = load_model('min_salary_model.h5')
max_salary_model = load_model('max_salary_model.h5')

# Load the label encoder and scaler used during model training
label_encoder = load('label_encoder.pkl')
scaler = load('scaler.pkl')

# Function to set page background color
def set_background_color(color):
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the page configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="📊",
    layout="wide"
)

# Set the background color for the entire app (default)
set_background_color("#f0f2f5")  # Light grey background

# Sidebar for navigation
with st.sidebar:
    st.title("Prediction App")
    page = option_menu(
        menu_title=None,
        options=["Home", "About", "Salary Prediction"],
        icons=["house", "info-circle", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": 'transparent'},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"color": "black", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
            "nav-link-selected": {"background-color": "#7B06A6", "font-size": "15px"},
        }
    )
    st.write("***")

if page == "Home":
    st.title("Welcome to the Salary Prediction App")
    st.write("This app predicts the minimum and maximum salary based on job title and minimum years of experience.")
    st.write("Navigate to the About page to learn more or to the Salary Prediction page to make predictions.")

elif page == "About":
    st.title("About This App")
    st.write("""\
        The Salary Prediction App is designed to provide minimum and maximum salary based on the following parameters:
        - Job Title: The title of the job position.
        - Minimum Years of Experience: The minimum years of relevant work experience ranges from 1 to 10.

        This app uses a machine learning model trained on historical salary data to make predictions.
    """)

elif page == "Salary Prediction":
    st.title('Salary Prediction')

    # Input fields for user with default 'Select' and experience as 0
    job_title = st.selectbox('Job Title', [
        'Select', 'Data Scientist', 'Business Analyst', 'Data Analyst',
        'Data Engineer', 'Senior Data Scientist',
        'Senior Business Analyst', 'Senior Data Analyst',
        'Senior Data Engineer', 'Machine Learning Engineer',
        'Data Architect'
    ], index=0)  # Default index set to 'Select'

    # Limiting the experience input to a range of 1 to 10, with default value 0
    min_experience = st.number_input('Minimum Years of Experience', min_value=0, max_value=10, step=1, value=0)  # Default value set to 0

    # Prepare the input data for prediction only if conditions are met
    if job_title != 'Select' and min_experience > 0:
        job_title_encoded = label_encoder.transform([job_title])[0]
        input_data = pd.DataFrame([[min_experience, job_title_encoded]], columns=['min_experience', 'job_title_encoded'])
        input_data_scaled = scaler.transform(input_data)

        st.markdown(
            """
            <style>
            div.stButton > button {
                background-color: #7B06A6;  /* Purple background */
                color: white;  /* White text */
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            div.stButton > button:hover {
                background-color: #7B06A6 !important; /* Purple background on hover (no change) */
                color: white !important;
                border: none !important;
            }
            div.stButton > button:focus {
                background-color: #7B06A6 !important;  /* Maintain purple on focus */
                color: white !important;
                border: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Make predictions
        if st.button('Predict Salary'):
            # Display loading image
            loading_image = st.image("Spinner-3.gif", use_column_width=False, width=50)  # Adjust width to 100 pixels

            # Simulating a delay for loading (remove this in production)
            time.sleep(2)  # Simulating delay for demonstration purposes

            # Predictions
            min_salary_pred = min_salary_model.predict(input_data_scaled)[0][0]
            max_salary_pred = max_salary_model.predict(input_data_scaled)[0][0]

            # Remove loading image
            loading_image.empty()

            # Display the predicted salary range in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.image("minimum_img.png", use_column_width=False, width=70)
                st.markdown(f"### Predicted Minimum Annual Salary")
                st.write(f'**₹{min_salary_pred:,.2f} INR**')

            with col2:
                st.image("maximum_image.png", use_column_width=False, width=70)
                st.markdown(f"### Predicted Maximum Annual Salary")
                st.write(f'**₹{max_salary_pred:,.2f} INR**')


