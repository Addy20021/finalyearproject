import streamlit as st
from login import login
from setup_db import create_user_table, populate_users
import base64
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import csv

# Initialize database
create_user_table()
populate_users()

# Set page configuration
st.set_page_config(page_title="Thyroid Detection", layout="wide")

# Function to encode an image file to base64
def get_base64_image(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# Base64 strings for background and logo
background = get_base64_image("background.jpg")  # Update with your image file
logo_image = get_base64_image("logo.jpg")  # Update with your image file

# Define custom CSS styles
custom_css = f"""
<style>
body {{
  margin: 0;
  padding: 0;
  font-family: 'Georgia', serif;
}}

.navbar {{
  position: absolute;
  top: 10px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: center;
  gap: 30px;
  z-index: 10;
  padding: 10px;
  background-color: rgba(0, 0, 0, 0.5);
  border-radius: 10px;
}}

.navbar a {{
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
  color: white;
  font-weight: bold;
  text-decoration: none;
}}

.navbar a:hover {{
  color: #3498db;
}}

.banner {{
  width: 100%;
  height: 100vh;
  background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.6)), url({background});
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  background-attachment: fixed;
  text-align: center;
  color: white;
  display: flex;
  flex-direction: column;
  justify-content: center;
}}

.content {{
  margin-top: 50px;
}}
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state for login and navigation
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Show the login page or the main content
if not st.session_state.logged_in:
    login()
else:
    # Navbar
    col1, col2, col3 = st.columns([6, 0.5, 1])

    with col1:
        if st.button("Home"):
            st.session_state.page = "Home"
    with col2:
        if st.button("Contact"):
            st.session_state.page = "Contact"
    with col3:
        if st.button("Thyroid Detection"):
            st.session_state.page = "Thyroid Detection"

    # Home Page Content
    def home_page():
        st.markdown(
            f"""
            <div class="banner">
                <h1 style="font-size: 64px; margin: 0;">ThyroNet : Thyroid Nodule Analysis</h1>
                <p style="font-size: 20px; margin-top: 20px;">
                    Discover an advanced way to assess thyroid health with our dual diagnostic system <br> by 
                    Image recognition and Text-based analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Contact Page Content
    def contact_page():
        st.markdown(""" 
        <div style="background-color: #101010; padding: 20px; border-radius: 10px;">
            <h1 style="color: white; font-size: 48px; text-align: center; font-family: 'Georgia', serif;">
                We would love to hear from you <br> Get in Touch!
            </h1>
        </div>
        """, unsafe_allow_html=True)

        with st.form("contact_form"):
            name = st.text_input("Name", placeholder="Enter your name")
            email = st.text_input("Email", placeholder="Enter your email")
            contact_number = st.text_input("Contact Number", placeholder="Enter your contact number")
            role = st.selectbox(
                "Select your role",
                options=["Patient", "Relative", "Caretaker", "Other"]
            )
            message = st.text_area("Message", placeholder="Enter your message here")
            consent = st.checkbox("I consent to the facility using my personal data to contact me.")

            submitted = st.form_submit_button("Submit")
            if consent:
                # Save the form data to a CSV file
                with open("contacts.csv", mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, email, contact_number, role, message])
                
                st.success(f"Thank you, {name}! Your message has been submitted successfully.")
            else:
                st.error("Please consent to using your personal data.")

        st.markdown(""" 
        <hr>
        <p style="text-align: center;">Thyroid Detection Facility ©ï¸ 2024</p>
        """, unsafe_allow_html=True)
    image_model = load_model("Resnet_model.h5")
    finalized_model = pickle.load(open('thyroid_model.pkl', 'rb'))
    # Thyroid Detection Page
    def thyroid_detection_page():
        st.title("Thyroid Detection System")
        
        # Text-based prediction form
        with st.form("thyroid_form"):
            st.header("Enter Thyroid Parameters")
            age = st.number_input("Age (years)", min_value=0.0, step=1.0, key="age")
            sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
            TSH = st.number_input("TSH (mIU/L)", min_value=0.0, step=0.1, key="tsh")
            T3 = st.number_input("T3 (ng/dL)", min_value=0.0, step=0.1, key="t3")
            TT4 = st.number_input("TT4 (ng/dL)", min_value=0.0, step=0.1, key="tt4")
            T4U = st.number_input("T4U (mg/dL)", min_value=0.0, step=0.1, key="t4u")
            FTI = st.number_input("FTI (Free Thyroxine Index)", min_value=0.0, step=0.1, key="fti")
            
            # Radio buttons for binary features
            onthyroxine = st.radio("On Thyroxine", ["Yes", "No"], key="onthyroxine")
            queryonthyroxine = st.radio("Query On Thyroxine", ["Yes", "No"], key="queryonthyroxine")
            onantithyroidmedication = st.radio("On Antithyroid Medication", ["Yes", "No"], key="onantithyroid")
            sick = st.radio("Sick", ["Yes", "No"], key="sick")
            pregnant = st.radio("Pregnant", ["Yes", "No"], key="pregnant")
            thyroidsurgery = st.radio("Thyroid Surgery", ["Yes", "No"], key="thyroidsurgery")
            I131treatment = st.radio("I131 Treatment", ["Yes", "No"], key="i131treatment")
            queryhypothyroid = st.radio("Query Hypothyroid", ["Yes", "No"], key="queryhypothyroid")
            queryhyperthyroid = st.radio("Query Hyperthyroid", ["Yes", "No"], key="queryhyperthyroid")
            lithium = st.radio("Lithium", ["Yes", "No"], key="lithium")
            goitre = st.radio("Goitre", ["Yes", "No"], key="goitre")
            tumor = st.radio("Tumor", ["Yes", "No"], key="tumor")
            hypopituitary = st.radio("Hypopituitary", ["Yes", "No"], key="hypopituitary")
            psych = st.radio("Psychological Symptoms", ["Yes", "No"], key="psych")
            
            predict_button = st.form_submit_button("Predict")

            if predict_button:
                # Input validation: Ensure no field is left empty or zero where required
                if (
                    age == 0.0 or TSH == 0.0 or T3 == 0.0 or TT4 == 0.0 or T4U == 0.0 or FTI == 0.0
                ):
                    st.error("Error: Please enter valid non-zero values for all numeric fields.")
                else:
                    # Convert input values to numerical format
                    sex = 1 if sex == "Male" else 0
                    onthyroxine = 1 if onthyroxine == "Yes" else 0
                    queryonthyroxine = 1 if queryonthyroxine == "Yes" else 0
                    onantithyroidmedication = 1 if onantithyroidmedication == "Yes" else 0
                    sick = 1 if sick == "Yes" else 0
                    pregnant = 1 if pregnant == "Yes" else 0
                    thyroidsurgery = 1 if thyroidsurgery == "Yes" else 0
                    I131treatment = 1 if I131treatment == "Yes" else 0
                    queryhypothyroid = 1 if queryhypothyroid == "Yes" else 0
                    queryhyperthyroid = 1 if queryhyperthyroid == "Yes" else 0
                    lithium = 1 if lithium == "Yes" else 0
                    goitre = 1 if goitre == "Yes" else 0
                    tumor = 1 if tumor == "Yes" else 0
                    hypopituitary = 1 if hypopituitary == "Yes" else 0
                    psych = 1 if psych == "Yes" else 0

                    # Prepare input data
                    input_data = {
                        "age": age,
                        "sex": sex,
                        "TSH": TSH,
                        "T3": T3,
                        "TT4": TT4,
                        "T4U": T4U,
                        "FTI": FTI,
                        "on_thyroxine": onthyroxine,
                        "query_on_thyroxine": queryonthyroxine,
                        "on_antithyroid_medication": onantithyroidmedication,
                        "sick": sick,
                        "pregnant": pregnant,
                        "thyroid_surgery": thyroidsurgery,
                        "I131_treatment": I131treatment,
                        "query_hypothyroid": queryhypothyroid,
                        "query_hyperthyroid": queryhyperthyroid,
                        "lithium": lithium,
                        "goitre": goitre,
                        "tumor": tumor,
                        "hypopituitary": hypopituitary,
                        "psych": psych,
                        "referral_source_SVHC": 0,
                        "referral_source_SVHD": 0,
                        "referral_source_SVI": 0,
                        "referral_source_other": 0,
                    }

                    # Apply necessary transformations
                    df_transform = pd.DataFrame([input_data])

                    # Ensure the feature order matches the training data
                    training_features = [
                        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 
                        'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
                        'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
                        'T3', 'TT4', 'T4U', 'FTI', 'referral_source_SVHC', 'referral_source_SVHD', 
                        'referral_source_SVI', 'referral_source_other'
                    ]
                    df_transform = df_transform[training_features]

                    # Convert to array and predict
                    arr = df_transform.values
                    prediction = finalized_model.predict(arr)[0]

                    # Display prediction
                    if prediction == 0:
                        st.success("Prediction: Negative")
                    elif prediction == 1:
                        st.success("Prediction: Compensated Hypothyroid")
                    elif prediction == 2:
                        st.success("Prediction: Primary Hypothyroid")
                    else:
                        st.success("Prediction: Secondary Hypothyroid")
        # Image-based prediction
        st.header("Upload Thyroid Scan Image")
        uploaded_image = st.file_uploader("Upload a thyroid scan image (JPEG/PNG)", type=["jpeg", "jpg", "png"])

        if uploaded_image:  
        # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", width=500)

        # Check if the file name starts with 'screenshot' or 'image'
            if uploaded_image.name.lower().startswith("screenshot") or uploaded_image.name.lower().startswith("image"):
                st.warning("The uploaded image appears to be a screenshot or a generic image. This may affect prediction accuracy.")
                st.stop()
            def preprocess_image(image, target_size=(200, 200)):
                img = load_img(image, target_size=target_size)
                img_array = img_to_array(img) / 255.0
                return np.expand_dims(img_array, axis=0)

            try:
            # Preprocess the uploaded image
                preprocessed_image = preprocess_image(uploaded_image)

            # Make a prediction
                prediction = image_model.predict(preprocessed_image)
                classes = ['Benign', 'Malignant', 'Normal Thyroid']
                predicted_class = np.argmax(prediction)

            # Display the prediction result
                st.success(f"Prediction: {classes[predicted_class]} with confidence {prediction[0][predicted_class]:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")


    # Page Rendering Logic
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Contact":
        contact_page()
    elif st.session_state.page == "Thyroid Detection":
        thyroid_detection_page()
    else:
        st.error("Page not found!")