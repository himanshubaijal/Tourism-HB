import os
import streamlit as st
import pandas as pd
import joblib

# Configure page style
st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="✈️", layout="wide")

# Load the model committed by the pipeline
model_path = os.path.join(os.path.dirname(__file__), "best_machine_failure_model_v1.joblib")
try:
    model = joblib.load(model_path)
except Exception:
    model = None

st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.write("""
This application predicts the likelihood of a customer purchasing the newly introduced Wellness Tourism Package based on interaction parameters.
""")

# Visual confirmation that the Heap API failure does not affect the app's backend
st.sidebar.info("💡 **Pro-Tip**: If you see a 'Heap API 404' warning in your browser console, it is simply because your ad-blocker is blocking Streamlit's built-in telemetry script. The predictor itself runs completely locally and will work perfectly!")

if model is None:
    st.warning("⚠️ Model file not found yet. Please make sure your GitHub Actions pipeline has completed running to train, evaluate, and commit the model file back to this repository!")
else:
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 18.0, 100.0, 35.0, 1.0)
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        DurationOfPitch = st.number_input("Duration of Pitch (mins)", 0.0, 120.0, 15.0)
        NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
        NumberOfFollowups = st.number_input("Number of Follow-ups", 1, 10, 3)
        ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
        PreferredPropertyStar = st.number_input("Preferred Property Star", 1.0, 5.0, 3.0)
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        NumberOfTrips = st.number_input("Number of Trips Annually", 1.0, 20.0, 2.0)
        Passport = st.selectbox("Has Passport (0: No, 1: Yes)", [0, 1])
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        OwnCar = st.selectbox("Owns a Car (0: No, 1: Yes)", [0, 1])
        NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0.0, 5.0, 0.0)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", 1000.0, 100000.0, 22000.0)

    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }])

    if st.button("Predict Purchase Likelihood"):
        prediction = model.predict(input_data)[0]
        result = "Will Purchase Package 🎉" if prediction == 1 else "Will Not Purchase Package"
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"The model predicts: **{result}**")
        else:
            st.error(f"The model predicts: **{result}**")
