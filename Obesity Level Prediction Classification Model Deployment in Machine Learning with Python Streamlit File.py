# Import Libraries
import streamlit as st
import requests

# Set Streamlit Page
st.set_page_config(page_title = "Obesity Level Prediction Classification Model Deployment in Machine Learning with Python",
                   page_icon = "⚖️",
                   layout = "centered")

st.markdown("<h1 style = 'text-align: center; color: #FF4B4B;'>⚖️ OBESITY CLASSIFIER ⚖️</h1>",
            unsafe_allow_html = True)
st.markdown("---")

def centered_subheader(text):
    st.markdown(f"<h3 style = 'text-align: center;'>{text}</h3>",
                unsafe_allow_html = True)
    
def make_prediction(features):
    try:
        response = requests.post("http://127.0.0.1:7000/predict", 
                                 json = features)
        if response.status_code == 200:
            result = response.json()
            return result["prediction"]
        else:
            st.error(f"API Error: {response.status_code}")
            return "❌ API Error"
    except Exception as e:
        st.error(f"Request failed: {e}")
        return "❌ Request Exception"

# Form
with st.form("obesity-form"):
    centered_subheader("🧑 Personal Information")
    age = st.number_input("Age", 
                          min_value = 1, 
                          max_value = 150, 
                          value = 24)
    gender = st.radio("Gender", 
                      ["Male", "Female"])
    height = st.number_input("Height (meters)", 
                             min_value = 1.0, 
                             max_value = 2.5, 
                             step = 0.01, 
                             value = 1.70)
    weight = st.number_input("Weight (kg)", 
                             min_value = 20.0, 
                             max_value = 200.0, 
                             step = 0.01, 
                             value = 86.7)
    family_history = st.radio("Do you have family with overweight history?", 
                              ["Yes", "No"])
    
    centered_subheader("👤 Habits")
    favc = st.radio("Do you eat high caloric food frequently? (FAVC)", 
                    ["Yes", "No"])
    smoke = st.radio("Do you smoke?", 
                     ["Yes", "No"])
    scc = st.radio("Do you monitor your calories? (SCC)", 
                   ["Yes", "No"])

    centered_subheader("🍞 Eating Patterns")
    caec = st.selectbox("Do you eat between meals? (CAEC)", 
                        ["No", "Sometimes", "Frequently", "Always"])
    calc = st.selectbox("How often do you drink alcohol? (CALC)", 
                        ["No", "Sometimes", "Frequently", "Always"])

    centered_subheader("🍅 Nutrition and Activity")
    fcvc = st.slider("Frequency of consumption of vegetables", 
                     min_value = 1.0, 
                     max_value = 3.0, 
                     step = 0.1, 
                     value = 3.0)
    ncp = st.slider("Number of main meals", 
                    min_value = 1.0, 
                    max_value = 4.0, 
                    step = 0.1, 
                    value = 3.0)
    ch2o = st.slider("Daily water intake (liters)", 
                     min_value = 1.0, 
                     max_value = 3.0, 
                     step = 0.1, 
                     value = 2.0)
    faf = st.slider("Physical activity frequency", 
                    min_value = 0.0, 
                    max_value = 3.0, 
                    step = 0.1, 
                    value = 3.0)
    tue = st.slider("Time using technology devices", 
                    min_value = 0.0, 
                    max_value = 3.0, 
                    step = 0.01, 
                    value = 0.6)
    mtrans_display = [
        "Automobile", 
        "Bike", 
        "Motorbike", 
        "Public Transportation", 
        "Walking"
    ]
    mtrans = st.selectbox("Preferred transportation method", 
                          mtrans_display)
    submitted = st.form_submit_button("🎯 PREDICT YOUR OBESITY CLASS")

# Predict
if submitted:
    bmi = weight / (height ** 2)
    features = {
        "Gender": 0 if gender == "Male" else 1, 
        "Age": age, 
        "Height": height, 
        "Weight": weight, 
        "family_history_with_overweight": 1 if family_history == "Yes" else 0,
        "FAVC": 1 if favc == "Yes" else 0, 
        "FCVC": fcvc, 
        "NCP": ncp,
        "CAEC": {"No": 0, 
                "Sometimes": 1, 
                "Frequently": 2, 
                "Always": 3}[caec], 
        "SMOKE": 1 if smoke == "Yes" else 0, 
        "CH2O": ch2o, 
        "SCC": 1 if scc == "Yes" else 0, 
        "FAF": faf, 
        "TUE": tue, 
        "CALC": {"No": 0, 
                "Sometimes": 1, 
                "Frequently": 2, 
                "Always": 3}[calc],
        "MTRANS": 1 if mtrans == "Walking" else 0, 
        "BMI": bmi
    }

    result = make_prediction(features)

    class_map = {
        0: "🦴 Insufficient Weight 🦴",
        1: "😎 Normal Weight 😎",
        2: "⚠️ Overweight Level I ⚠️",
        3: "⚠️ Overweight Level II ⚠️",
        4: "‼️ Obesity Type I ‼️",
        5: "⁉️ Obesity Type II ⁉️",
        6: "🚨 Obesity Type III 🚨"
    }
    st.success(f"Predicted Obesity Class: **{class_map[result]}** (BMI: {bmi:.2f})")
