import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Page configuration
st.set_page_config(page_title="Sonar Rock vs Mine Detector", layout="wide")

# App title
st.title("🛰️ Sonar Signal Classifier")
st.markdown("**This fun AI app detects whether the sonar signal is a Rock 🪨 or a Mine 💣!**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("sonar data.csv", header=None)
    return df

sonar_data = load_data()

# Show data info
with st.expander("🔍 Explore Dataset"):
    st.write("**Dataset Preview:**")
    st.dataframe(sonar_data.head())

    st.write("**Class Distribution:**")
    st.bar_chart(sonar_data[60].value_counts())

    st.write("**Statistical Summary:**")
    st.dataframe(sonar_data.describe())

# Preprocessing
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

# Show accuracy
st.sidebar.header("📈 Model Performance")
st.sidebar.metric("Training Accuracy", f"{train_acc*100:.2f}%")
st.sidebar.metric("Test Accuracy", f"{test_acc*100:.2f}%")

# Prediction interface
st.subheader("🔮 Predict using Sonar Signal Values")
st.markdown("Enter 60 sonar values (between 0 and 1):")

# Generate 60 sliders dynamically
user_input = []
with st.form(key="prediction_form"):
    cols = st.columns(6)
    for i in range(60):
        with cols[i % 6]:
            val = st.slider(f"F{i+1}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            user_input.append(val)
    submit_button = st.form_submit_button(label="🚀 Classify Signal")

# Prediction
if submit_button:
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)

    st.success("Prediction complete!")

    with st.spinner("Analyzing signal..."):
        time.sleep(1.5)

    if prediction[0] == 'R':

        st.markdown("### 🪨 The object is a **Rock**!")
    else:

        st.markdown("### 💣 The object is a **Mine**!")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")
