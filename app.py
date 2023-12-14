import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your trained model 
model = joblib.load('Harsco_mpdel.pkl')

# Create a Streamlit web app
st.title("Classification Model Deployment")

# Custom CSS to add a background image
st.markdown(
    """
    <style>
        body {
            background-image: url('Logo.png');  # Replace 'your_image_url.jpg' with the URL or local path of your image
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Try reading the uploaded CSV file with different encodings
        for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding if decoding fails

        # Assuming 'Resubmit_binary' is your target variable
        X = df.drop(columns=['Resubmit'])

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
        y_true = df['Resubmit']

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

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please make sure the file is encoded in UTF-8, latin-1, or ISO-8859-1.")
