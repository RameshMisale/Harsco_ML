import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your trained model
model = joblib.load('your_model_path.pkl')

# Create a Streamlit web app
st.title("Classification Model Deployment")

# Allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Assuming 'Resubmit_binary' is your target variable
    X = df.drop(columns=['Resubmit_binary'])

    # Make predictions on the target variable
    y_pred = model.predict(X)

    # Add predictions as a new column
    df['Prediction'] = y_pred

    # Display the DataFrame with predictions
    st.subheader("DataFrame with Predictions")
    st.write(df)

    # Display model evaluation on the uploaded data
    st.subheader("Model Evaluation on Uploaded Data")

    # Assuming 'Resubmit_binary' is your target variable
    y_true = df['Resubmit_binary']

    # Evaluate the model
    accuracy = accuracy_score(y_true, y_pred)
    classification_report_str = classification_report(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Display metrics
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(confusion_mat)
    st.write("Classification Report:")
    st.text(classification_report_str)
