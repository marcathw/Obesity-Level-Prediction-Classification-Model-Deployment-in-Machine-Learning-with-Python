# ⚖️ Obesity Level Prediction with Streamlit, XGBoost, and FastAPI

This project builds a classification model using XGBoost to predict an individual's obesity level based on physical, demographic, and lifestyle attributes. The trained model is deployed using a FastAPI backend, with an interactive Streamlit web app providing a user-friendly interface for real-time prediction.

---
## 🔧 Features

#### Frontend Web App (Streamlit)
- **Interactive Form**: Intuitive sliders, dropdowns, and number inputs for users to enter their data.
- **Real-time Prediction**: Instantly displays the predicted obesity level upon submission.
- **User-Friendly Interface**: Clean and simple design for a seamless user experience.

#### Backend API (FastAPI)
- **High-Performance API**: Serves the XGBoost model efficiently.
- **`/predict` Endpoint**: A dedicated endpoint for real-time obesity level classification.
- **JSON-based Communication**: Standard format for requests and responses.

#### Machine Learning Model
- **XGBoost Classification Model**: Predicts one of seven obesity level categories.
- **Model Evaluation**:
  - `Accuracy`
  - `Precision`
  - `Recall`
  - `F1-Score`
  - `Classification Report` & `Confusion Matrix`

---
## 🧠 Concepts Used

- **Frontend**: Interactive web app development with `Streamlit`
- **Backend**: REST API development with `FastAPI`
- **Machine Learning**:
  - Classification modeling with `XGBClassifier`
  - Feature Engineering (e.g., creating a BMI feature)
  - Data Preprocessing (`RobustScaler`, Ordinal & Binary Encoding)
- **Data Serialization**: Using `Pydantic` for data validation in FastAPI
- **Python Ecosystem**: `pandas`, `scikit-learn`, `xgboost`, `streamlit`, `uvicorn`, `fastapi`

---
## 🏁 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites
- Python 3.8+
- pip

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/obesity-prediction-project.git](https://github.com/your-username/obesity-prediction-project.git)
    cd obesity-prediction-project
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the FastAPI backend:**
    Open a terminal and run the following command. The API will be available at `http://127.0.0.1:8000`.
    ```bash
    uvicorn api.main:app --reload
    ```
    *(Note: Replace `api.main:app` with the actual path to your FastAPI app instance)*

4.  **Run the Streamlit frontend:**
    Open a **new** terminal and run this command. The web app will open in your browser.
    ```bash
    streamlit run app.py
    ```
    *(Note: Replace `app.py` with the actual name of your Streamlit script)*
